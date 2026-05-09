from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from .config import Config
from .upstream import build_headers, open_stream_url, post_json_url


Writer = Callable[[bytes], None]
KNOWN_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(slots=True)
class AnthropicStreamResult:
    usage: dict[str, int]


def build_anthropic_payload(request: dict[str, Any], config: Config) -> dict[str, Any]:
    payload = dict(request)
    payload["model"] = config.resolve_model(str(request.get("model") or ""))
    if config.force_stream:
        payload["stream"] = True
    return payload


def anthropic_headers(config: Config, stream: bool) -> dict[str, str]:
    headers = build_headers(config, stream=stream)
    headers.setdefault("anthropic-version", "2023-06-01")
    if config.upstream_api_key and config.upstream_auth_header == "Authorization":
        headers.pop("Authorization", None)
        headers.setdefault("x-api-key", config.upstream_api_key)
    return headers


def post_anthropic_message(config: Config, payload: dict[str, Any]):
    return post_json_url(config, config.upstream_messages_url, payload, anthropic_headers(config, stream=False))


def open_anthropic_stream(config: Config, payload: dict[str, Any]) -> BinaryIO:
    return open_stream_url(config, config.upstream_messages_url, payload, anthropic_headers(config, stream=True))


def normalize_anthropic_usage(value: Any) -> dict[str, int]:
    usage = value if isinstance(value, dict) else {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "cache_write_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
    }


def extract_anthropic_usage_from_message(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return normalize_anthropic_usage({})
    if value.get("type") != "message":
        return normalize_anthropic_usage({})
    return normalize_anthropic_usage(value.get("usage"))


def stream_anthropic_passthrough(response: BinaryIO, writer: Writer) -> AnthropicStreamResult:
    usage = normalize_anthropic_usage({})
    event_lines: list[str] = []
    for raw_line in response:
        writer(raw_line)
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line:
            event_lines.append(line)
            continue
        usage = _update_usage_from_sse_lines(event_lines, usage)
        event_lines.clear()
    if event_lines:
        usage = _update_usage_from_sse_lines(event_lines, usage)
    return AnthropicStreamResult(usage=usage)


def _update_usage_from_sse_lines(lines: list[str], usage: dict[str, int]) -> dict[str, int]:
    data_lines = []
    for line in lines:
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return usage
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return usage
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return usage
    if not isinstance(payload, dict):
        return usage
    event_type = str(payload.get("type") or "")
    if event_type == "message_start":
        message = payload.get("message")
        if isinstance(message, dict):
            return normalize_anthropic_usage(message.get("usage"))
    if event_type == "message_delta":
        delta_usage = payload.get("usage")
        if isinstance(delta_usage, dict):
            normalized = normalize_anthropic_usage(delta_usage)
            usage["output_tokens"] = normalized["output_tokens"] or usage["output_tokens"]
    return usage
