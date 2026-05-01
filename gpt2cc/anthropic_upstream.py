from __future__ import annotations

from typing import Any, BinaryIO, Callable

from .config import Config
from .upstream import build_headers, open_stream_url, post_json_url


Writer = Callable[[bytes], None]


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


def stream_anthropic_passthrough(response: BinaryIO, writer: Writer) -> None:
    for chunk in iter(lambda: response.read(65536), b""):
        writer(chunk)
