from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__
from .config import Config, load_config
from .image import (
    anthropic_message_from_image_result,
    build_image_edit_request,
    build_image_generation_payload,
    edit_image,
    generate_image,
    is_image_model,
    request_has_reference_images,
    stream_image_result_to_anthropic,
)
from .streaming import stream_openai_to_anthropic
from .tokens import estimate_tokens
from .transform import anthropic_message_from_openai, transform_anthropic_to_openai
from .upstream import UpstreamError, open_stream_with_retry, post_json


LOG = logging.getLogger(__name__)


def make_handler(config: Config) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"ccproxy/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.info("%s - %s", self.address_string(), fmt % args)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path in {"/", "/health", "/healthz"}:
                    self._send_json(
                        {
                            "ok": True,
                            "name": "ccproxy",
                            "version": __version__,
                            "anthropic_compatible": True,
                            "upstream": config.upstream_base_url,
                        }
                    )
                    return
                if path == "/debug/config":
                    self._require_auth()
                    self._send_json(config.redacted())
                    return
                if path == "/v1/models":
                    models = self._known_anthropic_models()
                    self._send_json(
                        {
                            "data": [
                                {
                                    "type": "model",
                                    "id": model_id,
                                    "display_name": model_id,
                                    "created_at": "2025-01-01T00:00:00Z",
                                }
                                for model_id in models
                            ],
                            "has_more": False,
                            "first_id": models[0],
                            "last_id": models[-1],
                        }
                    )
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not_found_error", f"unknown endpoint: {path}")
            except PermissionError as exc:
                self._send_error(HTTPStatus.UNAUTHORIZED, "authentication_error", str(exc))

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/v1/messages":
                    self._handle_messages()
                    return
                if path == "/v1/messages/count_tokens":
                    self._handle_count_tokens()
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not_found_error", f"unknown endpoint: {path}")
            except UpstreamError as exc:
                LOG.warning("upstream error %s: %s", exc.status, exc)
                self._send_error(status_from_upstream(exc.status), "api_error", str(exc), exc.body)
            except PermissionError as exc:
                self._send_error(HTTPStatus.UNAUTHORIZED, "authentication_error", str(exc))
            except json.JSONDecodeError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", f"invalid JSON: {exc}")
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", str(exc))
            except Exception as exc:
                LOG.exception("request failed")
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "api_error", str(exc))

        def _handle_messages(self) -> None:
            self._require_auth()
            request = self._read_json()
            if config.debug_payloads:
                LOG.debug("anthropic request: %s", json.dumps(request, ensure_ascii=False))

            upstream_payload, ctx = transform_anthropic_to_openai(request, config)
            if is_image_model(ctx.upstream_model, config):
                has_references = request_has_reference_images(request, config)
                endpoint = "images/edits" if has_references else "images/generations"
                LOG.info(
                    "model route: requested=%s upstream=%s endpoint=%s stream=%s tools_ignored=%s references=%s",
                    ctx.requested_model or "<empty>",
                    ctx.upstream_model,
                    endpoint,
                    upstream_payload.get("stream"),
                    len(upstream_payload.get("tools") or []),
                    has_references,
                )

                if has_references:
                    edit_request = build_image_edit_request(request, config, ctx)
                    if config.debug_payloads:
                        safe_payload = dict(edit_request.fields)
                        safe_payload["prompt"] = f"<{len(edit_request.prompt)} chars>"
                        LOG.debug(
                            "upstream image edit request: %s files=%s",
                            json.dumps(safe_payload, ensure_ascii=False),
                            len(edit_request.files),
                        )

                    if upstream_payload.get("stream"):
                        self._send_stream_headers()
                        stream_image_result_to_anthropic(
                            lambda: edit_image(config, edit_request, ctx),
                            ctx,
                            self._write_stream,
                            f"Calling image edit model {ctx.upstream_model} with {edit_request.reference_count} reference image(s)...\n\n",
                        )
                        return

                    result = edit_image(config, edit_request, ctx)
                    self._send_json(anthropic_message_from_image_result(result, ctx))
                    return

                image_payload = build_image_generation_payload(request, config, ctx)
                if config.debug_payloads:
                    safe_payload = dict(image_payload)
                    safe_payload["prompt"] = f"<{len(str(image_payload.get('prompt') or ''))} chars>"
                    LOG.debug("upstream image generation request: %s", json.dumps(safe_payload, ensure_ascii=False))

                if upstream_payload.get("stream"):
                    self._send_stream_headers()
                    stream_image_result_to_anthropic(
                        lambda: generate_image(config, image_payload, ctx),
                        ctx,
                        self._write_stream,
                        f"Calling image generation model {ctx.upstream_model}...\n\n",
                    )
                    return

                result = generate_image(config, image_payload, ctx)
                self._send_json(anthropic_message_from_image_result(result, ctx))
                return

            LOG.info(
                "model route: requested=%s upstream=%s endpoint=chat/completions stream=%s tools=%s",
                ctx.requested_model or "<empty>",
                ctx.upstream_model,
                upstream_payload.get("stream"),
                len(upstream_payload.get("tools") or []),
            )
            if config.debug_payloads:
                safe_payload = dict(upstream_payload)
                LOG.debug("upstream request: %s", json.dumps(safe_payload, ensure_ascii=False))

            if upstream_payload.get("stream"):
                upstream_stream = open_stream_with_retry(config, upstream_payload)
                self._send_stream_headers()
                with upstream_stream as response:
                    stream_openai_to_anthropic(response, ctx, self._write_stream)
                return

            upstream_response = post_json(config, upstream_payload)
            data = upstream_response.json()
            result = anthropic_message_from_openai(data, ctx)
            self._send_json(result)

        def _handle_count_tokens(self) -> None:
            self._require_auth()
            request = self._read_json()
            self._send_json({"input_tokens": estimate_tokens(request)})

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length") or "0")
            if content_length > config.max_body_bytes:
                raise ValueError(f"request body too large: {content_length} bytes")
            raw = self.rfile.read(content_length)
            if not raw:
                return {}
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request JSON must be an object")
            return value

        def _require_auth(self) -> None:
            if not config.proxy_api_key:
                return
            x_api_key = self.headers.get("x-api-key") or self.headers.get("anthropic-api-key")
            auth = self.headers.get("authorization") or ""
            bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
            if config.proxy_api_key not in {x_api_key, bearer}:
                raise PermissionError("invalid proxy API key")

        def _send_stream_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._send_common_headers()
            self.end_headers()
            self.close_connection = True

        def _write_stream(self, data: bytes) -> None:
            self.wfile.write(data)
            self.wfile.flush()

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_common_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_error(
            self,
            status: HTTPStatus,
            error_type: str,
            message: str,
            upstream_body: bytes | None = None,
        ) -> None:
            if upstream_body and config.debug_payloads:
                LOG.debug("upstream error body: %s", upstream_body.decode("utf-8", errors="replace"))
            payload = {"type": "error", "error": {"type": error_type, "message": message}}
            self._send_json(payload, status)

        def _send_common_headers(self) -> None:
            if config.cors:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "content-type,x-api-key,authorization,anthropic-version")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

        @staticmethod
        def _known_anthropic_models() -> list[str]:
            return [
                "claude-sonnet-4-5-20250929",
                "claude-opus-4-1-20250805",
                "claude-3-5-sonnet-20241022",
            ]

    return Handler


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def status_from_upstream(status: int) -> HTTPStatus:
    if status in {400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}:
        return HTTPStatus(status)
    if 400 <= status < 500:
        return HTTPStatus.BAD_REQUEST
    if 500 <= status < 600:
        return HTTPStatus.BAD_GATEWAY
    return HTTPStatus.INTERNAL_SERVER_ERROR


def run(config: Config) -> None:
    server = ReusableThreadingHTTPServer((config.host, config.port), make_handler(config))
    LOG.info("ccproxy %s listening on http://%s:%s", __version__, config.host, config.port)
    LOG.info("upstream chat endpoint: %s", config.upstream_chat_url)
    LOG.info("upstream images endpoint: %s", config.upstream_images_generations_url)
    LOG.info("upstream image edits endpoint: %s", config.upstream_images_edits_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("stopping")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Messages API proxy for Claude Code.")
    parser.add_argument("--host", help="listen host; overrides CCPROXY_HOST")
    parser.add_argument("--port", type=int, help="listen port; overrides CCPROXY_PORT")
    args = parser.parse_args()

    config = load_config()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    run(config)


if __name__ == "__main__":
    main()
