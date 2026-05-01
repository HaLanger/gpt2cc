from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, BinaryIO

from .sse import encode_sse, iter_sse_data
from .transform import TransformContext, convert_usage, map_finish_reason, new_message_start


Writer = Callable[[bytes], None]
LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolStreamState:
    content_index: int
    id: str = ""
    name: str = ""
    started: bool = False
    arguments: str = ""


@dataclass(slots=True)
class StreamState:
    ctx: TransformContext
    writer: Writer
    message_started: bool = False
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    text_index: int | None = None
    next_content_index: int = 0
    tool_states: dict[int, ToolStreamState] = field(default_factory=dict)
    finish_reason: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    text_delta_count: int = 0
    tool_delta_count: int = 0

    def send(self, event: str, data: dict[str, Any]) -> None:
        self.writer(encode_sse(event, data))

    def ensure_message_start(self) -> None:
        if not self.message_started:
            self.send("message_start", new_message_start(self.ctx, self.message_id))
            self.message_started = True

    def ensure_text_block(self) -> int:
        self.ensure_message_start()
        if self.text_index is None:
            index = self.next_content_index
            self.next_content_index += 1
            self.text_index = index
            self.send(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            )
        return self.text_index

    def write_text_delta(self, text: str) -> None:
        if not text:
            return
        self.text_delta_count += 1
        if self.text_delta_count == 1:
            LOG.info("stream diagnostics: first text delta received from upstream")
        index = self.ensure_text_block()
        self.send(
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
        )

    def close_text_block(self) -> None:
        if self.text_index is not None:
            self.send("content_block_stop", {"type": "content_block_stop", "index": self.text_index})
            self.text_index = None

    def ensure_tool_block(self, upstream_index: int, tool_delta: dict[str, Any]) -> ToolStreamState:
        self.ensure_message_start()
        state = self.tool_states.get(upstream_index)
        if state is None:
            state = ToolStreamState(content_index=self.next_content_index)
            self.next_content_index += 1
            self.tool_states[upstream_index] = state

        if tool_delta.get("id"):
            state.id = str(tool_delta["id"])
        function = tool_delta.get("function") or {}
        if function.get("name"):
            state.name = str(function["name"])

        if not state.started and state.name:
            original_name = self.ctx.from_upstream_tool_name(state.name)
            self.close_text_block()
            self.send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state.id or f"call_{uuid.uuid4().hex[:24]}",
                        "name": original_name,
                        "input": {},
                    },
                },
            )
            state.started = True
        return state

    def write_tool_delta(self, upstream_index: int, tool_delta: dict[str, Any]) -> None:
        state = self.ensure_tool_block(upstream_index, tool_delta)
        function = tool_delta.get("function") or {}
        args = function.get("arguments") or ""
        if args:
            self.tool_delta_count += 1
            if self.tool_delta_count == 1:
                LOG.info("stream diagnostics: first tool input delta received from upstream")
            state.arguments += str(args)
            if not state.started:
                state.name = state.name or "tool"
                self.ensure_tool_block(upstream_index, {"id": state.id, "function": {"name": state.name}})
            self.send(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state.content_index,
                    "delta": {"type": "input_json_delta", "partial_json": str(args)},
                },
            )

    def close_tool_blocks(self) -> None:
        for upstream_index in sorted(self.tool_states):
            state = self.tool_states[upstream_index]
            if not state.started:
                state.name = state.name or "tool"
                self.ensure_tool_block(upstream_index, {"id": state.id, "function": {"name": state.name}})
            self.send("content_block_stop", {"type": "content_block_stop", "index": state.content_index})
        self.tool_states.clear()

    def stop(self) -> None:
        self.ensure_message_start()
        self.close_text_block()
        self.close_tool_blocks()
        usage = convert_usage(self.usage)
        stop_reason = map_finish_reason(self.finish_reason, False)
        self.send(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
        self.send("message_stop", {"type": "message_stop"})
        LOG.info(
            "stream diagnostics: complete text_deltas=%s tool_deltas=%s finish_reason=%s",
            self.text_delta_count,
            self.tool_delta_count,
            stop_reason,
        )


def stream_openai_to_anthropic(response: BinaryIO, ctx: TransformContext, writer: Writer) -> None:
    state = StreamState(ctx=ctx, writer=writer)
    for data in iter_sse_data(response):
        if data == "[DONE]":
            break

        chunk = json.loads(data)
        if chunk.get("id") and not state.message_started:
            state.message_id = str(chunk["id"])

        if chunk.get("usage"):
            state.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason") is not None:
            state.finish_reason = choice.get("finish_reason")

        if delta.get("content"):
            state.write_text_delta(str(delta["content"]))

        for tool_delta in delta.get("tool_calls") or []:
            upstream_index = int(tool_delta.get("index") or 0)
            state.write_tool_delta(upstream_index, tool_delta)

    state.stop()
