from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import Config


OPENAI_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(slots=True)
class TransformContext:
    requested_model: str
    upstream_model: str
    tool_name_to_upstream: dict[str, str] = field(default_factory=dict)
    tool_name_from_upstream: dict[str, str] = field(default_factory=dict)

    def to_upstream_tool_name(self, name: str) -> str:
        if name in self.tool_name_to_upstream:
            return self.tool_name_to_upstream[name]
        clean = OPENAI_TOOL_NAME_RE.sub("_", name or "tool")
        if not clean:
            clean = "tool"
        if clean[0] in "-_":
            clean = "tool_" + clean.lstrip("-_")

        candidate = clean[:64]
        suffix = 2
        existing = set(self.tool_name_from_upstream)
        while candidate in existing and self.tool_name_from_upstream[candidate] != name:
            tail = f"_{suffix}"
            candidate = (clean[: 64 - len(tail)] + tail)[:64]
            suffix += 1

        self.tool_name_to_upstream[name] = candidate
        self.tool_name_from_upstream[candidate] = name
        return candidate

    def from_upstream_tool_name(self, name: str) -> str:
        return self.tool_name_from_upstream.get(name, name)


def transform_anthropic_to_openai(request: dict[str, Any], config: Config) -> tuple[dict[str, Any], TransformContext]:
    requested_model = str(request.get("model") or "")
    upstream_model = config.resolve_model(requested_model)
    ctx = TransformContext(requested_model=requested_model, upstream_model=upstream_model)

    messages: list[dict[str, Any]] = []
    system_content = request.get("system")
    if system_content:
        messages.append({"role": "system", "content": anthropic_blocks_to_text(system_content)})

    for message in request.get("messages", []):
        messages.extend(convert_message(message, ctx))

    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(request.get("stream")),
    }

    if config.force_stream:
        payload["stream"] = True

    if config.max_tokens_field and config.max_tokens_field.lower() != "omit":
        if "max_tokens" in request:
            payload[config.max_tokens_field] = request["max_tokens"]

    if not config.omit_temperature and "temperature" in request:
        payload["temperature"] = request["temperature"]

    if not config.omit_top_p and "top_p" in request:
        payload["top_p"] = request["top_p"]

    if "stop_sequences" in request and request["stop_sequences"]:
        payload["stop"] = request["stop_sequences"]

    tools = request.get("tools") or []
    if tools:
        payload["tools"] = [convert_tool(tool, ctx) for tool in tools]
        payload["tool_choice"] = convert_tool_choice(request.get("tool_choice"), ctx)

    if payload.get("stream") and config.stream_include_usage:
        payload["stream_options"] = {"include_usage": True}

    # These OpenAI-compatible switches are accepted by many relay services,
    # and ignored by providers that do not support them.
    if "metadata" in request and isinstance(request["metadata"], dict):
        payload["metadata"] = request["metadata"]

    return payload, ctx


def convert_message(message: dict[str, Any], ctx: TransformContext) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content", "")

    if role == "assistant":
        return [convert_assistant_message(content, ctx)]

    if role == "user":
        return convert_user_message(content)

    return [{"role": role or "user", "content": anthropic_blocks_to_openai_content(content)}]


def convert_assistant_message(content: Any, ctx: TransformContext) -> dict[str, Any]:
    blocks = normalize_blocks(content)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            if text:
                text_parts.append(str(text))
        elif block_type == "tool_use":
            name = str(block.get("name") or "tool")
            upstream_name = ctx.to_upstream_tool_name(name)
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": upstream_name,
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        elif block_type in {"thinking", "redacted_thinking"}:
            thinking = convert_thinking_block(block)
            if thinking:
                reasoning_parts.append(thinking)
        else:
            text = block_to_text(block)
            if text:
                text_parts.append(text)

    content_text = "\n".join(text_parts) if text_parts else ""
    result: dict[str, Any] = {"role": "assistant", "content": content_text if content_text or not tool_calls else None}
    if reasoning_parts:
        result["reasoning_content"] = "\n".join(reasoning_parts)
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def convert_user_message(content: Any) -> list[dict[str, Any]]:
    blocks = normalize_blocks(content)
    tool_messages: list[dict[str, Any]] = []
    user_blocks: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "tool_result":
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": tool_result_to_text(block),
                }
            )
        elif block_type in {"thinking", "redacted_thinking"}:
            thinking = convert_thinking_block(block)
            if thinking:
                user_blocks.append({"type": "text", "text": thinking})
        else:
            user_blocks.append(block)

    result = tool_messages[:]
    if user_blocks:
        result.append({"role": "user", "content": blocks_to_openai_user_content(user_blocks)})
    if not result:
        result.append({"role": "user", "content": ""})
    return result


def convert_tool(tool: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
    name = str(tool.get("name") or "tool")
    upstream_name = ctx.to_upstream_tool_name(name)
    description = tool.get("description") or ""
    if upstream_name != name:
        description = f"Original Claude Code tool name: {name}\n{description}".strip()

    parameters = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": upstream_name,
            "description": description,
            "parameters": parameters,
        },
    }


def convert_tool_choice(tool_choice: Any, ctx: TransformContext) -> Any:
    if not tool_choice:
        return "auto"
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice in {"any", "required"}:
            return "required"
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none"}:
            return choice_type
        if choice_type in {"any", "required"}:
            return "required"
        if choice_type == "tool":
            name = str(tool_choice.get("name") or "")
            return {"type": "function", "function": {"name": ctx.to_upstream_tool_name(name)}}
    return "auto"


def normalize_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        result: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                result.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
        return result
    if isinstance(content, dict):
        return [content]
    return [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]


def blocks_to_openai_user_content(blocks: list[dict[str, Any]]) -> Any:
    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []
    has_non_text = False

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text") or "")
            text_parts.append(text)
            parts.append({"type": "text", "text": text})
        elif block_type == "image":
            image_url = image_block_to_data_url(block)
            if image_url:
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
                has_non_text = True
        else:
            text = block_to_text(block)
            if text:
                text_parts.append(text)
                parts.append({"type": "text", "text": text})

    if not has_non_text:
        return "\n".join(part for part in text_parts if part)
    return parts


def anthropic_blocks_to_openai_content(content: Any) -> Any:
    return blocks_to_openai_user_content(normalize_blocks(content))


def anthropic_blocks_to_text(content: Any) -> str:
    blocks = normalize_blocks(content)
    text = [block_to_text(block) for block in blocks]
    return "\n".join(part for part in text if part)



def convert_thinking_block(block: dict[str, Any]) -> str:
    if block.get("type") == "redacted_thinking":
        data = block.get("data") or block.get("thinking") or ""
        return json.dumps({"type": "redacted_thinking", "data": data}, ensure_ascii=False)
    thinking = block.get("thinking") or block.get("text") or ""
    signature = block.get("signature")
    if signature:
        return json.dumps({"type": "thinking", "thinking": thinking, "signature": signature}, ensure_ascii=False)
    return str(thinking)


def block_to_text(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    if block_type == "text":
        return str(block.get("text") or "")
    if block_type == "tool_result":
        return tool_result_to_text(block)
    if block_type == "image":
        return "[image omitted]"
    if block_type in {"thinking", "redacted_thinking"}:
        return ""
    if "text" in block:
        return str(block.get("text") or "")
    return json.dumps(block, ensure_ascii=False)


def tool_result_to_text(block: dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        text = content
    else:
        text = anthropic_blocks_to_text(content)
    if block.get("is_error"):
        return f"[tool_error]\n{text}"
    return text


def image_block_to_data_url(block: dict[str, Any]) -> str:
    source = block.get("source") or {}
    if source.get("type") != "base64":
        return ""
    media_type = source.get("media_type") or "image/png"
    data = source.get("data") or ""
    return f"data:{media_type};base64,{data}"


def anthropic_message_from_openai(response: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    content = openai_message_to_anthropic_content(message, ctx)
    usage = response.get("usage") or {}

    return {
        "id": str(response.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": ctx.requested_model or ctx.upstream_model,
        "content": content,
        "stop_reason": map_finish_reason(finish_reason, bool(message.get("tool_calls"))),
        "stop_sequence": None,
        "usage": convert_usage(usage),
    }


def openai_message_to_anthropic_content(message: dict[str, Any], ctx: TransformContext) -> list[dict[str, Any]]:
    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        if text:
            content_blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        upstream_name = str(function.get("name") or "tool")
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                "name": ctx.from_upstream_tool_name(upstream_name),
                "input": parse_tool_arguments(function.get("arguments")),
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return content_blocks


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    if not isinstance(value, str):
        return {"_value": value}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def convert_usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    }


def map_finish_reason(finish_reason: Any, has_tool_calls: bool = False) -> str:
    if has_tool_calls or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop" or finish_reason is None:
        return "end_turn"
    return "stop_sequence"


def new_message_start(ctx: TransformContext, message_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id or f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": ctx.requested_model or ctx.upstream_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def now_ms() -> int:
    return int(time.time() * 1000)
