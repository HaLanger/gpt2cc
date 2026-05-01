from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, BinaryIO, Callable

from .config import Config
from .sse import encode_sse, iter_sse_data
from .transform import TransformContext, map_finish_reason, new_message_start, normalize_blocks, parse_tool_arguments
from .upstream import build_headers, open_stream_url, post_json_url


Writer = Callable[[bytes], None]


def transform_anthropic_to_gemini(request: dict[str, Any], config: Config) -> tuple[dict[str, Any], TransformContext]:
    requested_model = str(request.get("model") or "")
    upstream_model = config.resolve_model(requested_model)
    ctx = TransformContext(requested_model=requested_model, upstream_model=upstream_model)
    contents: list[dict[str, Any]] = []

    for message in request.get("messages", []):
        converted = convert_message(message, ctx)
        if converted:
            contents.extend(converted)

    payload: dict[str, Any] = {"contents": contents or [{"role": "user", "parts": [{"text": ""}]}]}
    system_instruction = system_to_text(request.get("system"))
    if system_instruction:
        payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

    generation_config: dict[str, Any] = {}
    if "max_tokens" in request:
        generation_config["maxOutputTokens"] = request["max_tokens"]
    if "temperature" in request:
        generation_config["temperature"] = request["temperature"]
    if "top_p" in request:
        generation_config["topP"] = request["top_p"]
    if "stop_sequences" in request and request["stop_sequences"]:
        generation_config["stopSequences"] = request["stop_sequences"]
    if generation_config:
        payload["generationConfig"] = generation_config

    tools = request.get("tools") or []
    if tools:
        payload["tools"] = [{"function_declarations": [convert_tool(tool, ctx) for tool in tools]}]
        tool_config = convert_tool_choice(request.get("tool_choice"), ctx)
        if tool_config:
            payload["toolConfig"] = tool_config

    return payload, ctx


def convert_message(message: dict[str, Any], ctx: TransformContext) -> list[dict[str, Any]]:
    role = "model" if message.get("role") == "assistant" else "user"
    parts: list[dict[str, Any]] = []
    for block in normalize_blocks(message.get("content", "")):
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"text": str(block.get("text") or "")})
        elif block_type == "image":
            image_part = image_block_to_gemini_part(block)
            if image_part:
                parts.append(image_part)
        elif block_type == "tool_use":
            parts.append(
                {
                    "functionCall": {
                        "name": ctx.to_upstream_tool_name(str(block.get("name") or "tool")),
                        "args": block.get("input") or {},
                    }
                }
            )
        elif block_type == "tool_result":
            parts.append(
                {
                    "functionResponse": {
                        "name": str(block.get("name") or "tool"),
                        "response": {"result": tool_result_content(block)},
                    }
                }
            )
        elif block_type not in {"thinking", "redacted_thinking"}:
            parts.append({"text": json.dumps(block, ensure_ascii=False)})
    return [{"role": role, "parts": parts or [{"text": ""}]}]


def image_block_to_gemini_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") or {}
    if source.get("type") != "base64":
        return None
    return {"inline_data": {"mime_type": source.get("media_type") or "image/png", "data": source.get("data") or ""}}


def tool_result_content(block: dict[str, Any]) -> Any:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    texts = []
    for item in normalize_blocks(content):
        if item.get("type") == "text":
            texts.append(str(item.get("text") or ""))
        else:
            texts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(text for text in texts if text)


def system_to_text(system: Any) -> str:
    if not system:
        return ""
    texts = []
    for block in normalize_blocks(system):
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif "text" in block:
            texts.append(str(block.get("text") or ""))
    return "\n".join(text for text in texts if text)


def convert_tool(tool: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
    declaration: dict[str, Any] = {
        "name": ctx.to_upstream_tool_name(str(tool.get("name") or "tool")),
        "description": str(tool.get("description") or ""),
    }
    schema = tool.get("input_schema") or tool.get("parameters")
    if isinstance(schema, dict):
        declaration["parameters"] = schema
    return declaration


def convert_tool_choice(tool_choice: Any, ctx: TransformContext) -> dict[str, Any] | None:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"any", "required"}:
            return {"functionCallingConfig": {"mode": "ANY"}}
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        return None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [ctx.to_upstream_tool_name(str(tool_choice.get("name") or ""))],
            }
        }
    return None


def gemini_headers(config: Config, stream: bool) -> dict[str, str]:
    headers = build_headers(config, stream=stream)
    if config.upstream_api_key and config.upstream_auth_header == "Authorization":
        headers.pop("Authorization", None)
        headers.setdefault("x-goog-api-key", config.upstream_api_key)
    return headers


def post_gemini(config: Config, payload: dict[str, Any]):
    return post_json_url(config, config.upstream_gemini_generate_url, payload, gemini_headers(config, stream=False))


def open_gemini_stream(config: Config, payload: dict[str, Any]) -> BinaryIO:
    return open_stream_url(config, config.upstream_gemini_stream_url, payload, gemini_headers(config, stream=True))


def anthropic_message_from_gemini(response: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
    candidate = (response.get("candidates") or [{}])[0]
    content = candidate.get("content") or {}
    finish_reason = candidate.get("finishReason")
    blocks = gemini_parts_to_anthropic(content.get("parts") or [], ctx)
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": ctx.requested_model or ctx.upstream_model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": map_gemini_finish_reason(finish_reason, any(block.get("type") == "tool_use" for block in blocks)),
        "stop_sequence": None,
        "usage": convert_gemini_usage(response.get("usageMetadata") or {}),
    }


def gemini_parts_to_anthropic(parts: list[dict[str, Any]], ctx: TransformContext) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if part.get("text"):
            blocks.append({"type": "text", "text": str(part["text"])})
        function_call = part.get("functionCall")
        if isinstance(function_call, dict):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": ctx.from_upstream_tool_name(str(function_call.get("name") or "tool")),
                    "input": parse_tool_arguments(function_call.get("args") or {}),
                }
            )
    return blocks


def map_gemini_finish_reason(reason: Any, has_tool_call: bool) -> str:
    if has_tool_call:
        return "tool_use"
    if reason == "MAX_TOKENS":
        return "max_tokens"
    if reason in {"STOP", None}:
        return "end_turn"
    return "stop_sequence"


def convert_gemini_usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("promptTokenCount") or 0),
        "output_tokens": int(usage.get("candidatesTokenCount") or 0),
    }


def stream_gemini_to_anthropic(response: BinaryIO, ctx: TransformContext, writer: Writer) -> None:
    message_id = f"msg_{uuid.uuid4().hex}"
    writer(encode_sse("message_start", new_message_start(ctx, message_id)))
    text_index: int | None = None
    next_index = 0
    finish_reason: Any = None
    tool_blocks: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}

    for chunk in iter_gemini_chunks(response):
        usage = chunk.get("usageMetadata") or usage
        candidate = (chunk.get("candidates") or [{}])[0]
        if candidate.get("finishReason") is not None:
            finish_reason = candidate.get("finishReason")
        parts = ((candidate.get("content") or {}).get("parts") or [])
        for part in parts:
            if part.get("text"):
                if text_index is None:
                    text_index = next_index
                    next_index += 1
                    writer(encode_sse("content_block_start", {"type": "content_block_start", "index": text_index, "content_block": {"type": "text", "text": ""}}))
                writer(encode_sse("content_block_delta", {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": str(part["text"])}}))
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                tool_blocks.append(function_call)

    if text_index is not None:
        writer(encode_sse("content_block_stop", {"type": "content_block_stop", "index": text_index}))
    for function_call in tool_blocks:
        index = next_index
        next_index += 1
        args = function_call.get("args") or {}
        writer(encode_sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": f"call_{uuid.uuid4().hex[:24]}", "name": ctx.from_upstream_tool_name(str(function_call.get("name") or "tool")), "input": args}}))
        writer(encode_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
    stop_reason = map_gemini_finish_reason(finish_reason, bool(tool_blocks))
    converted_usage = convert_gemini_usage(usage)
    writer(encode_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": converted_usage["output_tokens"]}}))
    writer(encode_sse("message_stop", {"type": "message_stop"}))


def iter_gemini_chunks(response: BinaryIO):
    yielded = False
    for data in iter_sse_data(response):
        yielded = True
        yield json.loads(data)
    if yielded:
        return
    raw = response.read()
    if not raw:
        return
    value = json.loads(raw.decode("utf-8"))
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        yield value
