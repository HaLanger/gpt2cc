from __future__ import annotations

import argparse
import html
import json
import logging
import os
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from typing import Any

from . import __version__
from .anthropic_upstream import (
    build_anthropic_payload,
    extract_anthropic_usage_from_message,
    open_anthropic_stream,
    post_anthropic_message,
    stream_anthropic_passthrough,
)
from .config import Config, ConfigStore, ensure_config_file, load_config
from .gemini import (
    anthropic_message_from_gemini,
    open_gemini_stream,
    post_gemini,
    stream_gemini_to_anthropic,
    transform_anthropic_to_gemini,
)
from .image import (
    anthropic_message_from_image_result,
    build_image_edit_request,
    build_image_generation_payload,
    edit_image,
    generate_image,
    image_usage,
    is_image_model,
    request_has_reference_images,
    stream_image_result_to_anthropic,
)
from .streaming import stream_openai_to_anthropic
from .tokens import estimate_tokens
from .transform import anthropic_message_from_openai, convert_usage, transform_anthropic_to_openai
from .upstream import UpstreamError, open_stream_with_retry, post_json
from .usage_stats import UsagePrice, append_usage_record, build_usage_record, load_usage_stats, summarize_usage_records, usage_record_to_dict


LOG = logging.getLogger(__name__)


def make_handler(config: Config) -> type[BaseHTTPRequestHandler]:
    store = ConfigStore(config)

    class Handler(BaseHTTPRequestHandler):
        server_version = f"gpt2cc/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            quiet_paths = {"/admin/state", "/admin/usage/summary", "/admin/usage/history"}
            if self.command == "GET" and self.path.split("?", 1)[0] in quiet_paths:
                return
            LOG.info("%s - %s", self.address_string(), fmt % args)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path in {"/", "/health", "/healthz"}:
                    current = store.snapshot()
                    self._send_json(
                        {
                            "ok": True,
                            "name": "gpt2cc",
                            "version": __version__,
                            "anthropic_compatible": True,
                            "upstream": current.upstream_base_url,
                        }
                    )
                    return
                if path == "/admin":
                    self._require_auth()
                    self._send_html(admin_html(store.state()))
                    return
                if path == "/admin/usage":
                    self._require_auth()
                    self._send_html(usage_html(store.state()))
                    return
                if path == "/admin/state":
                    self._require_auth()
                    self._send_json(store.state())
                    return
                if path == "/admin/usage/summary":
                    self._require_auth()
                    self._send_json(self._usage_summary_payload())
                    return
                if path == "/admin/usage/history":
                    self._require_auth()
                    self._send_json(self._usage_history_payload(self._parse_limit()))
                    return
                if path == "/debug/config":
                    self._require_auth()
                    current = store.snapshot()
                    payload = current.redacted()
                    payload["admin_state"] = store.state()
                    self._send_json(payload)
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
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", str(exc))

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/admin/providers":
                    self._require_auth()
                    self._send_json(store.add_or_update_provider(self._read_json()))
                    return
                if path == "/admin/providers/delete":
                    self._require_auth()
                    payload = self._read_json()
                    self._send_json(store.delete_provider(str(payload.get("id") or "")))
                    return
                if path == "/admin/active":
                    self._require_auth()
                    payload = self._read_json()
                    self._send_json(store.set_active(str(payload.get("provider_id") or ""), str(payload.get("model") or "")))
                    return
                if path == "/admin/model-routes":
                    self._require_auth()
                    payload = self._read_json()
                    self._send_json(
                        store.set_model_route(
                            str(payload.get("requested_model") or ""),
                            str(payload.get("provider_id") or payload.get("provider") or ""),
                            str(payload.get("model") or ""),
                        )
                    )
                    return
                if path == "/admin/model-routes/delete":
                    self._require_auth()
                    payload = self._read_json()
                    self._send_json(store.delete_model_route(str(payload.get("requested_model") or "")))
                    return
                if path == "/admin/primary-route-model":
                    self._require_auth()
                    payload = self._read_json()
                    requested_model = str(payload.get("requested_model") or "")
                    if requested_model:
                        self._send_json(store.bind_primary_route_model(requested_model))
                    else:
                        self._send_json(store.unbind_primary_route_model())
                    return
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
            initial_config = store.snapshot()
            request = self._read_json(initial_config)
            requested_model = str(request.get("model") or "")
            store.record_seen_model(requested_model)
            request_config = store.snapshot_for_model(requested_model)
            if request_config.debug_payloads:
                LOG.debug("anthropic request: %s", json.dumps(request, ensure_ascii=False))

            if request_config.upstream_protocol == "anthropic":
                self._handle_anthropic_messages(request, request_config)
                return
            if request_config.upstream_protocol == "gemini":
                self._handle_gemini_messages(request, request_config)
                return
            self._handle_openai_messages(request, request_config)

        def _provider_name(self, request_config: Config) -> str:
            for provider in request_config.providers:
                if provider.get("id") == request_config.active_provider:
                    return str(provider.get("name") or provider.get("id") or request_config.active_provider)
            return request_config.active_provider or ""

        def _usage_price(self, request_config: Config, upstream_model: str) -> UsagePrice | None:
            provider_id = str(request_config.active_provider or "")
            prices = request_config.provider_pricing.get(provider_id) or {}
            fields = prices.get(upstream_model) or {}
            if not fields:
                return None
            return UsagePrice(
                provider_id=provider_id,
                model=upstream_model,
                input_per_million=fields.get("input_per_million"),
                output_per_million=fields.get("output_per_million"),
                cache_read_per_million=fields.get("cache_read_per_million"),
            )

        def _record_usage(
            self,
            request: dict[str, Any],
            request_config: Config,
            endpoint: str,
            stream: bool,
            usage: dict[str, int],
            upstream_model: str,
            route_source: str | None = None,
        ) -> None:
            try:
                record = build_usage_record(
                    protocol=request_config.upstream_protocol,
                    requested_model=str(request.get("model") or ""),
                    provider_id=str(request_config.active_provider or ""),
                    provider_name=self._provider_name(request_config),
                    upstream_model=upstream_model,
                    route_source=route_source or request_config.resolve_model_route(str(request.get("model") or "")).source,
                    stream=stream,
                    endpoint=endpoint,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
                    cache_write_input_tokens=int(usage.get("cache_write_input_tokens") or 0),
                    price=self._usage_price(request_config, upstream_model),
                )
                append_usage_record(request_config.stats_path, record)
            except Exception as exc:
                LOG.warning("could not persist usage stats: %s", exc)

        def _handle_anthropic_messages(self, request: dict[str, Any], request_config: Config) -> None:
            upstream_payload = build_anthropic_payload(request, request_config)
            LOG.info(
                "model route: requested=%s upstream=%s endpoint=anthropic/messages stream=%s protocol=anthropic route=%s provider=%s",
                request.get("model") or "<empty>",
                upstream_payload.get("model"),
                upstream_payload.get("stream"),
                request_config.resolve_model_route(str(request.get("model") or "")).source,
                request_config.active_provider_label(),
            )
            if upstream_payload.get("stream"):
                upstream_stream = open_anthropic_stream(request_config, upstream_payload)
                self._send_stream_headers()
                with upstream_stream as response:
                    stream_result = stream_anthropic_passthrough(response, self._write_stream)
                self._record_usage(request, request_config, "anthropic/messages", True, stream_result.usage, str(upstream_payload.get("model") or ""))
                return
            upstream_response = post_anthropic_message(request_config, upstream_payload)
            data = upstream_response.json()
            self._send_json(data)
            self._record_usage(request, request_config, "anthropic/messages", False, extract_anthropic_usage_from_message(data), str(upstream_payload.get("model") or ""))

        def _handle_gemini_messages(self, request: dict[str, Any], request_config: Config) -> None:
            upstream_payload, ctx = transform_anthropic_to_gemini(request, request_config)
            stream = bool(request.get("stream") or request_config.force_stream)
            LOG.info(
                "model route: requested=%s upstream=%s endpoint=gemini/generateContent stream=%s protocol=gemini route=%s provider=%s",
                ctx.requested_model or "<empty>",
                ctx.upstream_model,
                stream,
                ctx.route_source,
                request_config.active_provider_label(),
            )
            if request_config.debug_payloads:
                LOG.debug("upstream gemini request: %s", json.dumps(upstream_payload, ensure_ascii=False))
            if stream:
                upstream_stream = open_gemini_stream(request_config, upstream_payload)
                self._send_stream_headers()
                with upstream_stream as response:
                    stream_result = stream_gemini_to_anthropic(response, ctx, self._write_stream)
                self._record_usage(request, request_config, "gemini/generateContent", True, stream_result.usage, ctx.upstream_model)
                return
            upstream_response = post_gemini(request_config, upstream_payload)
            data = upstream_response.json()
            self._send_json(anthropic_message_from_gemini(data, ctx))
            from .gemini import convert_gemini_usage
            self._record_usage(request, request_config, "gemini/generateContent", False, convert_gemini_usage(data.get("usageMetadata") or {}), ctx.upstream_model)

        def _handle_openai_messages(self, request: dict[str, Any], request_config: Config) -> None:
            upstream_payload, ctx = transform_anthropic_to_openai(request, request_config)
            if is_image_model(ctx.upstream_model, request_config):
                has_references = request_has_reference_images(request, request_config)
                endpoint = "images/edits" if has_references else "images/generations"
                LOG.info(
                    "model route: requested=%s upstream=%s endpoint=%s stream=%s tools_ignored=%s references=%s route=%s provider=%s",
                    ctx.requested_model or "<empty>",
                    ctx.upstream_model,
                    endpoint,
                    upstream_payload.get("stream"),
                    len(upstream_payload.get("tools") or []),
                    has_references,
                    ctx.route_source,
                    request_config.active_provider_label(),
                )

                if has_references:
                    edit_request = build_image_edit_request(request, request_config, ctx)
                    if request_config.debug_payloads:
                        safe_payload = dict(edit_request.fields)
                        safe_payload["prompt"] = f"<{len(edit_request.prompt)} chars>"
                        LOG.debug(
                            "upstream image edit request: %s files=%s",
                            json.dumps(safe_payload, ensure_ascii=False),
                            len(edit_request.files),
                        )

                    if upstream_payload.get("stream"):
                        self._send_stream_headers()
                        stream_result = stream_image_result_to_anthropic(
                            lambda: edit_image(request_config, edit_request, ctx),
                            ctx,
                            self._write_stream,
                            f"Calling image edit model {ctx.upstream_model} with {edit_request.reference_count} reference image(s)...\n\n",
                        )
                        if stream_result.succeeded:
                            self._record_usage(request, request_config, endpoint, True, stream_result.usage, ctx.upstream_model)
                        return

                    result = edit_image(request_config, edit_request, ctx)
                    self._send_json(anthropic_message_from_image_result(result, ctx))
                    self._record_usage(request, request_config, endpoint, False, image_usage(result.raw_response), ctx.upstream_model)
                    return

                image_payload = build_image_generation_payload(request, request_config, ctx)
                if request_config.debug_payloads:
                    safe_payload = dict(image_payload)
                    safe_payload["prompt"] = f"<{len(str(image_payload.get('prompt') or ''))} chars>"
                    LOG.debug("upstream image generation request: %s", json.dumps(safe_payload, ensure_ascii=False))

                if upstream_payload.get("stream"):
                    self._send_stream_headers()
                    stream_result = stream_image_result_to_anthropic(
                        lambda: generate_image(request_config, image_payload, ctx),
                        ctx,
                        self._write_stream,
                        f"Calling image generation model {ctx.upstream_model}...\n\n",
                    )
                    if stream_result.succeeded:
                        self._record_usage(request, request_config, endpoint, True, stream_result.usage, ctx.upstream_model)
                    return

                result = generate_image(request_config, image_payload, ctx)
                self._send_json(anthropic_message_from_image_result(result, ctx))
                self._record_usage(request, request_config, endpoint, False, image_usage(result.raw_response), ctx.upstream_model)
                return

            LOG.info(
                "model route: requested=%s upstream=%s endpoint=chat/completions stream=%s tools=%s route=%s provider=%s",
                ctx.requested_model or "<empty>",
                ctx.upstream_model,
                upstream_payload.get("stream"),
                len(upstream_payload.get("tools") or []),
                ctx.route_source,
                request_config.active_provider_label(),
            )
            if request_config.debug_payloads:
                safe_payload = dict(upstream_payload)
                LOG.debug("upstream request: %s", json.dumps(safe_payload, ensure_ascii=False))

            if upstream_payload.get("stream"):
                upstream_stream = open_stream_with_retry(request_config, upstream_payload)
                self._send_stream_headers()
                with upstream_stream as response:
                    stream_result = stream_openai_to_anthropic(response, ctx, self._write_stream)
                self._record_usage(request, request_config, "chat/completions", True, stream_result.usage, ctx.upstream_model)
                return

            upstream_response = post_json(request_config, upstream_payload)
            data = upstream_response.json()
            result = anthropic_message_from_openai(data, ctx)
            self._send_json(result)
            self._record_usage(request, request_config, "chat/completions", False, convert_usage(data.get("usage") or {}, request_config.upstream_base_url), ctx.upstream_model)

        def _handle_count_tokens(self) -> None:
            self._require_auth()
            request = self._read_json(store.snapshot())
            self._send_json({"input_tokens": estimate_tokens(request)})

        def _parse_limit(self, default: int = 50, maximum: int = 500) -> int:
            query = parse_qs(urlsplit(self.path).query)
            raw = (query.get("limit") or [str(default)])[0]
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("limit must be an integer") from exc
            return max(1, min(maximum, value))

        def _parse_usage_datetime(self, name: str, raw: str) -> datetime:
            value = str(raw or "").strip()
            if not value:
                raise ValueError(f"{name} must not be blank")
            normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError(f"invalid {name} datetime: {value}") from exc
            if parsed.tzinfo is None:
                raise ValueError(f"invalid {name} datetime: timezone is required")
            return parsed.astimezone(timezone.utc)

        def _usage_filters(self) -> tuple[datetime | None, datetime | None]:
            query = parse_qs(urlsplit(self.path).query)
            start_raw = (query.get("start") or [""])[0].strip()
            end_raw = (query.get("end") or [""])[0].strip()
            start = self._parse_usage_datetime("start", start_raw) if start_raw else None
            end = self._parse_usage_datetime("end", end_raw) if end_raw else None
            return start, end

        def _filter_usage_records(self, records: list[Any]) -> list[Any]:
            start, end = self._usage_filters()
            if start is None and end is None:
                return records
            filtered = []
            for record in records:
                record_ts = self._parse_usage_datetime("record ts", str(record.ts or ""))
                if start is not None and record_ts < start:
                    continue
                if end is not None and record_ts > end:
                    continue
                filtered.append(record)
            return filtered

        def _usage_summary_payload(self) -> dict[str, Any]:
            stats_path = store.snapshot().stats_path
            records = self._filter_usage_records(load_usage_stats(stats_path).records or [])
            summary = summarize_usage_records(records)
            total_cost = 0.0
            priced_records = 0
            merged_models: dict[str, dict[str, Any]] = {}
            provider_models: dict[tuple[str, str], dict[str, Any]] = {}
            for record in records:
                model_key = record.upstream_model or "<unknown>"
                merged = merged_models.setdefault(
                    model_key,
                    {
                        "upstream_model": model_key,
                        "records": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "providers": set(),
                    },
                )
                merged["records"] += 1
                merged["input_tokens"] += record.input_tokens
                merged["output_tokens"] += record.output_tokens
                merged["cache_read_input_tokens"] += record.cache_read_input_tokens
                merged["cache_write_input_tokens"] += record.cache_write_input_tokens
                merged["providers"].add(record.provider_id or record.provider_name or "")

                provider_key = (record.provider_id or "", model_key)
                item = provider_models.setdefault(
                    provider_key,
                    {
                        "provider_id": record.provider_id,
                        "provider_name": record.provider_name,
                        "upstream_model": model_key,
                        "records": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "cache_hit_rate": None,
                        "total_cost": 0.0,
                        "has_pricing": False,
                        "pricing": None,
                    },
                )
                item["records"] += 1
                item["input_tokens"] += record.input_tokens
                item["output_tokens"] += record.output_tokens
                item["cache_read_input_tokens"] += record.cache_read_input_tokens
                item["cache_write_input_tokens"] += record.cache_write_input_tokens
                if record.price is not None:
                    item["has_pricing"] = True
                    item["pricing"] = {
                        "input_per_million": record.price.input_per_million,
                        "output_per_million": record.price.output_per_million,
                        "cache_read_per_million": record.price.cache_read_per_million,
                    }
                if record.cost is not None:
                    item["total_cost"] += record.cost.total
                    total_cost += record.cost.total
                    priced_records += 1

            merged_list = []
            for item in merged_models.values():
                providers = sorted(value for value in item.pop("providers") if value)
                item["provider_count"] = len(providers)
                item["providers"] = providers
                item["cache_hit_rate"] = _cache_hit_rate(item["input_tokens"], item["cache_read_input_tokens"])
                merged_list.append(item)
            merged_list.sort(key=lambda item: (-int(item["input_tokens"] + item["output_tokens"]), str(item["upstream_model"])))

            provider_list = []
            for item in provider_models.values():
                item["cache_hit_rate"] = _cache_hit_rate(item["input_tokens"], item["cache_read_input_tokens"])
                provider_list.append(item)
            provider_list.sort(key=lambda item: (-float(item["total_cost"]), str(item["provider_id"]), str(item["upstream_model"])))

            return {
                "stats_path": stats_path,
                "records": len(records),
                "totals": {
                    "records": summary.records,
                    "input_tokens": summary.input_tokens,
                    "output_tokens": summary.output_tokens,
                    "cache_read_input_tokens": summary.cache_read_input_tokens,
                    "cache_write_input_tokens": summary.cache_write_input_tokens,
                    "cache_hit_rate": summary.cache_hit_rate,
                    "total_cost": total_cost if priced_records else None,
                    "priced_records": priced_records,
                    "has_pricing": bool(priced_records),
                },
                "merged_by_model": merged_list,
                "provider_model_breakdown": provider_list,
            }

        def _usage_history_payload(self, limit: int) -> dict[str, Any]:
            stats_path = store.snapshot().stats_path
            records = self._filter_usage_records(load_usage_stats(stats_path).records or [])
            recent = list(reversed(records[-limit:]))
            return {
                "stats_path": stats_path,
                "records": [usage_record_to_dict(record) for record in recent],
                "returned": len(recent),
                "total": len(records),
                "limit": limit,
            }

        def _read_json(self, request_config: Config | None = None) -> dict[str, Any]:
            limit = (request_config or store.snapshot()).max_body_bytes
            content_length = int(self.headers.get("Content-Length") or "0")
            if content_length > limit:
                raise ValueError(f"request body too large: {content_length} bytes")
            raw = self.rfile.read(content_length)
            if not raw:
                return {}
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request JSON must be an object")
            return value

        def _require_auth(self) -> None:
            proxy_api_key = store.snapshot().proxy_api_key
            if not proxy_api_key:
                return
            x_api_key = self.headers.get("x-api-key") or self.headers.get("anthropic-api-key")
            auth = self.headers.get("authorization") or ""
            bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
            if proxy_api_key not in {x_api_key, bearer}:
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

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self._send_common_headers()
            self.end_headers()
            self.wfile.write(raw)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
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
            if upstream_body and store.snapshot().debug_payloads:
                LOG.debug("upstream error body: %s", upstream_body.decode("utf-8", errors="replace"))
            payload = {"type": "error", "error": {"type": error_type, "message": message}}
            self._send_json(payload, status)

        def _send_common_headers(self) -> None:
            if store.snapshot().cors:
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

    Handler.config_store = store  # type: ignore[attr-defined]
    return Handler


def _cache_hit_rate(input_tokens: int, cache_read_input_tokens: int) -> float | None:
    denominator = int(input_tokens) + int(cache_read_input_tokens)
    if denominator <= 0:
        return None
    return int(cache_read_input_tokens) / denominator


def admin_html(state: dict[str, Any]) -> str:
    auth_hint = "true" if state.get("auth_required") else "false"
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>gpt2cc relay console</title>
<style>
:root {{ color-scheme: light; --bg:#f6f7fb; --card:#ffffff; --text:#172033; --muted:#657086; --line:#e5e9f2; --brand:#2563eb; --brand2:#0f172a; --bad:#dc2626; --ok:#059669; --warn:#b45309; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:linear-gradient(135deg,#f8fbff,#f3f4f8); color:var(--text); }}
.shell {{ max-width:1180px; margin:0 auto; padding:32px 20px 48px; }}
.hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:24px; }}
h1 {{ margin:0; font-size:34px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); margin-top:8px; }}
.status {{ min-width:280px; background:var(--brand2); color:white; padding:18px 20px; border-radius:22px; box-shadow:0 20px 60px rgba(15,23,42,.16); }}
.status small {{ color:#b9c4d8; display:block; margin-bottom:6px; }} .status strong {{ display:block; font-size:18px; word-break:break-all; }}
.toolbar {{ display:grid; grid-template-columns:minmax(220px,1fr) auto auto; gap:10px; margin:16px 0; }}
.panel,.drawer {{ background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:22px; box-shadow:0 12px 38px rgba(30,41,59,.08); }} .panel {{ padding:16px; }}
input,textarea,select {{ width:100%; border:1px solid var(--line); border-radius:13px; padding:10px 12px; font:inherit; background:white; color:var(--text); }} textarea {{ min-height:126px; resize:vertical; }}
button {{ border:0; border-radius:12px; padding:10px 13px; font-weight:750; cursor:pointer; background:#e8eefc; color:#1e3a8a; white-space:nowrap; }} button.primary {{ background:var(--brand); color:white; }} button.danger {{ background:#fee2e2; color:#991b1b; }} button.ghost {{ background:#f8fafc; color:#334155; border:1px solid var(--line); }} button:hover {{ filter:brightness(.97); }}
.banner {{ display:none; margin-bottom:14px; padding:11px 13px; border-radius:14px; font-weight:700; }} .banner.ok {{ display:block; background:#dcfce7; color:#166534; }} .banner.err {{ display:block; background:#fee2e2; color:#991b1b; }} .auth {{ display:none; margin-bottom:14px; }} .auth.show {{ display:block; }}
.list {{ display:grid; gap:8px; }} .row {{ display:grid; grid-template-columns:1.35fr 1.2fr .9fr auto; gap:12px; align-items:center; padding:12px; border:1px solid var(--line); background:white; border-radius:16px; }} .row.active {{ border-color:#93c5fd; box-shadow:0 10px 30px rgba(37,99,235,.10); }}
.route-row {{ display:grid; grid-template-columns:1.3fr .8fr .8fr auto; gap:10px; align-items:center; padding:10px; border:1px solid var(--line); border-radius:14px; background:white; margin-top:8px; }} .route-form {{ display:grid; grid-template-columns:1.2fr .8fr .8fr auto; gap:10px; align-items:end; margin-top:14px; }} .route-summary {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; }} .summary-card {{ background:white; border:1px solid var(--line); border-radius:16px; padding:13px; }} .summary-card strong {{ display:block; margin-bottom:5px; }} .route-drawer {{ width:min(860px,calc(100vw - 36px)); }} .hidden {{ display:none; }} .muted {{ color:var(--muted); }}
.name {{ font-weight:800; }} .sub {{ color:var(--muted); font-size:12px; margin-top:3px; word-break:break-all; }} .models {{ display:flex; flex-wrap:wrap; gap:5px; max-height:160px; overflow:auto; }} .model {{ border:1px solid var(--line); background:#f8fafc; border-radius:999px; padding:4px 8px; font-size:12px; }} button.model {{ color:#334155; font-weight:750; }} button.model.active-model {{ background:#dbeafe; border-color:#93c5fd; color:#1d4ed8; }} .pill {{ border-radius:999px; padding:4px 9px; background:#eef2ff; color:#1d4ed8; font-size:12px; font-weight:800; display:inline-block; margin-left:6px; }} .protocol {{ border-radius:999px; padding:3px 8px; background:#ecfdf5; color:#047857; font-size:11px; font-weight:800; display:inline-block; margin-left:6px; }} .actions {{ display:flex; gap:7px; justify-content:flex-end; flex-wrap:wrap; }} .empty {{ text-align:center; color:var(--muted); padding:32px; }}
.drawer-backdrop {{ position:fixed; inset:0; background:rgba(15,23,42,.28); opacity:0; pointer-events:none; transition:.18s; }} .drawer-backdrop.show {{ opacity:1; pointer-events:auto; }} .drawer {{ position:fixed; top:18px; right:18px; bottom:18px; width:min(460px,calc(100vw - 36px)); padding:20px; overflow:auto; transform:translateX(calc(100% + 28px)); transition:.2s; }} .drawer.show {{ transform:translateX(0); }} .drawer-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }} .drawer h2 {{ margin:0; }} label {{ display:block; color:#334155; font-size:13px; font-weight:750; margin:13px 0 6px; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin:12px 0 16px; }} .stat-card {{ background:white; border:1px solid var(--line); border-radius:16px; padding:14px; }} .stat-card b {{ display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }} .stat-card strong {{ font-size:20px; display:block; }}
.stats-layout {{ display:grid; grid-template-columns:1.4fr 1fr; gap:16px; }} .chart-list {{ display:grid; gap:10px; margin-top:12px; }} .chart-row {{ background:white; border:1px solid var(--line); border-radius:16px; padding:12px; }} .chart-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-end; margin-bottom:8px; }} .chart-bar {{ display:flex; height:12px; border-radius:999px; overflow:hidden; background:#eef2ff; }} .seg-input {{ background:#2563eb; }} .seg-output {{ background:#8b5cf6; }} .seg-cache-read {{ background:#10b981; }} .seg-cache-write {{ background:#f59e0b; }}
.table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:#475569; font-weight:800; }} .inline-note {{ margin-top:10px; padding:10px 12px; border:1px dashed #cbd5e1; border-radius:14px; color:#475569; background:#f8fafc; }} .inline-note.warn {{ color:#92400e; background:#fffbeb; border-color:#fcd34d; }} .history-list {{ display:grid; gap:8px; margin-top:12px; }} .history-item {{ border:1px solid var(--line); border-radius:14px; padding:10px 12px; background:white; }} .tiny {{ font-size:12px; }}
@media (max-width:1050px) {{ .stats-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .stats-layout {{ grid-template-columns:1fr; }} }}
@media (max-width:900px) {{ .hero,.toolbar,.row {{ grid-template-columns:1fr; }} .actions {{ justify-content:flex-start; }} }}
</style>
</head>
<body>
<div class=\"shell\">
  <div class=\"hero\"><div><h1>gpt2cc relay console</h1><div class=\"subtitle\">管理中转站、API key 和模型，新请求会立即使用当前选择。</div></div><div class=\"status\"><small>当前主路由</small><strong id=\"activeTitle\">Loading...</strong><small id=\"configPath\"></small></div></div>
  <div class=\"actions\" style=\"justify-content:flex-start;margin-bottom:16px\"><button class=\"primary\" onclick=\"openRoutes()\">模型路由</button><button class=\"ghost\" onclick=\"location.href='/admin/usage'\">用量统计</button><button class=\"ghost\" onclick=\"load()\">刷新状态</button></div>
  <div id=\"banner\" class=\"banner\"></div>
  <div id=\"authBox\" class=\"panel auth\"><b>代理密钥</b><div class=\"muted\">此服务启用了 GPT2CC_PROXY_API_KEY。密钥只保存在本次浏览器会话。</div><div class=\"toolbar\"><input id=\"proxyKey\" type=\"password\" autocomplete=\"off\" placeholder=\"Proxy API key\"><button class=\"primary\" onclick=\"saveProxyKey()\">保存并连接</button></div></div>
  <section class=\"panel\" style=\"margin-bottom:16px\"><h2 style=\"margin:0 0 10px\">当前路由概览</h2><div id=\"routeOverview\" class=\"route-summary\"></div></section>
  <section class=\"panel\"><div class=\"toolbar\"><input id=\"searchBox\" placeholder=\"搜索中转站名称、ID、域名或模型...\" oninput=\"render()\"><button class=\"ghost\" onclick=\"clearSearch()\">清除搜索</button><button class=\"primary\" onclick=\"openForm()\">添加中转站</button></div><div id=\"providers\" class=\"list\"></div></section>
</div>
<div id=\"drawerBackdrop\" class=\"drawer-backdrop\" onclick=\"closeForm()\"></div>
<aside id=\"drawer\" class=\"drawer\"><div class=\"drawer-head\"><h2 id=\"formTitle\">添加中转站</h2><button class=\"ghost\" onclick=\"closeForm()\">关闭</button></div><label>ID</label><input id=\"providerId\" placeholder=\"my-relay\"><label>名称</label><input id=\"providerName\" placeholder=\"My Relay\"><label>协议</label><select id=\"protocol\"><option value=\"openai\">OpenAI-compatible</option><option value=\"anthropic\">Anthropic Messages</option><option value=\"gemini\">Gemini native</option></select><label>Base URL</label><input id=\"baseUrl\" placeholder=\"https://relay.example.com/v1\"><label>API key</label><input id=\"apiKey\" type=\"password\" placeholder=\"编辑时留空表示保留原 key\"><label>模型（每行一个）</label><textarea id=\"models\" oninput=\"refreshPricingModels(el('pricingModel').value)\" placeholder=\"gpt-4.1&#10;gpt-image-2\"></textarea><label>价格（可选，按模型填写；单位：美元 / 1M tokens）</label><label>模型</label><select id=\"pricingModel\" onchange=\"loadPricingFields()\"></select><label>输入价格</label><input id=\"pricingInput\" type=\"number\" step=\"any\" placeholder=\"input_per_million\"><label>输出价格</label><input id=\"pricingOutput\" type=\"number\" step=\"any\" placeholder=\"output_per_million\"><label>缓存读取价格</label><input id=\"pricingCacheRead\" type=\"number\" step=\"any\" placeholder=\"cache_read_per_million\"><div class=\"sub\">同名模型在不同中转站可分别设置价格；三个价格都留空则不计算费用。</div><div class=\"actions\"><button onclick=\"applyPricingFields()\">应用当前模型价格</button><button class=\"ghost\" onclick=\"clearPricingFields()\">清空当前模型价格</button></div><div id=\"pricingSummary\" class=\"sub\"></div><div class=\"actions\"><button class=\"primary\" onclick=\"saveProvider()\">保存中转站</button><button onclick=\"resetForm()\">清空</button></div><p class=\"muted\">关闭面板不会清空未保存内容；再次点“添加中转站”会继续显示。</p></aside>
<aside id=\"routeDrawer\" class=\"drawer route-drawer\"><div class=\"drawer-head\"><h2>模型路由配置</h2><button class=\"ghost\" onclick=\"closeRoutes()\">关闭</button></div><div class=\"muted\">已配置路由会优先命中；未配置模型使用当前主路由。最近发现会自动刷新。</div><div class=\"route-form\"><label>Claude Code 模型<input id=\"routeRequested\" placeholder=\"claude-opus-4-7\"></label><label>中转站<select id=\"routeProvider\" onchange=\"syncRouteModels()\"></select></label><label>上游模型<select id=\"routeModel\"></select></label><button class=\"primary\" onclick=\"saveRoute()\">保存路由</button></div><h3>已配置路由</h3><div id=\"configuredRoutes\"></div><h3><button class=\"ghost\" onclick=\"toggleSeenRoutes()\">最近发现</button></h3><div id=\"seenRoutes\" class=\"hidden\"></div></aside>
<aside id=\"primaryBindDrawer\" class=\"drawer\"><div class=\"drawer-head\"><h2>绑定主模型</h2><button class=\"ghost\" onclick=\"closePrimaryBind()\">关闭</button></div><div class=\"muted\">选择 Claude Code 默认请求的模型。绑定后，切换当前主路由会同步更新这个模型的路由。</div><div id=\"primaryBindList\" style=\"margin-top:14px\"></div></aside>
<script>
const AUTH_REQUIRED = {auth_hint};
let state = null;
function el(id) {{ return document.getElementById(id); }}
function key() {{ return sessionStorage.getItem('gpt2cc_proxy_key') || ''; }}
function headers() {{ const h={{'content-type':'application/json'}}; if(key()) h['x-api-key']=key(); return h; }}
function show(msg, cls='ok') {{ const b=el('banner'); b.className='banner '+cls; b.textContent=msg; setTimeout(()=>{{b.className='banner';}},3500); }}
function esc(s) {{ return String(s ?? '').replace(/[&<>\"]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}
function saveProxyKey() {{ sessionStorage.setItem('gpt2cc_proxy_key', el('proxyKey').value); load(); }}
async function api(path, opts={{}}) {{ const r=await fetch(path, {{...opts, headers:{{...headers(), ...(opts.headers||{{}})}}}}); if(r.status===401) {{ el('authBox').classList.add('show'); throw new Error('需要代理密钥'); }} const data=await r.json(); if(!r.ok) throw new Error(data.error?.message || '请求失败'); return data; }}
async function load(silent=false) {{ try {{ const selections={{}}; document.querySelectorAll('select[id^="sel-"]').forEach(s=>selections[s.id]=s.value); const routeProviderEl=el('routeProvider'); const routeModelEl=el('routeModel'); const routeSelection={{provider:routeProviderEl?.value||'', model:routeModelEl?.value||''}}; state=await api('/admin/state'); el('authBox').classList.toggle('show', AUTH_REQUIRED && !key()); render(); Object.entries(selections).forEach(([id,value])=>{{ const sel=el(id); if(sel&&[...sel.options].some(o=>o.value===value)) sel.value=value; }}); if(routeSelection.provider && (state.providers||[]).some(p=>p.id===routeSelection.provider)) {{ routeProviderEl.value=routeSelection.provider; syncRouteModels(routeSelection.model); }} }} catch(e) {{ if(!silent) show(e.message,'err'); }} }}
function providerText(p) {{ return [p.id,p.name,p.protocol,p.upstream_base_url,...(p.models||[])].join(' ').toLowerCase(); }}
function filteredProviders() {{ const q=(el('searchBox').value||'').trim().toLowerCase(); return !q ? state.providers : state.providers.filter(p=>providerText(p).includes(q)); }}
function clearSearch() {{ el('searchBox').value=''; render(); }}
function render() {{ const active=state.providers.find(p=>p.id===state.active_provider); el('activeTitle').textContent=(active?.name||state.active_provider)+' / '+state.active_model; el('configPath').textContent='配置文件：'+state.config_path; const items=filteredProviders(); el('providers').innerHTML=items.length ? items.map(p=>rowHtml(p)).join('') : '<div class="empty">没有匹配的中转站</div>'; renderRouteOverview(); renderRoutes(); }}
function protocolName(protocol) {{ return ({{openai:'OpenAI',anthropic:'Anthropic',gemini:'Gemini'}})[protocol||'openai'] || protocol || 'OpenAI'; }}
function rowHtml(p) {{ const active=p.id===state.active_provider; const protocol=p.protocol||'openai'; const models=p.models||[]; const visible=models.map(m=>`<button class="model ${{active&&m===state.active_model?'active-model':''}}" onclick="activateModel('${{esc(p.id)}}',decodeURIComponent('${{encodeURIComponent(m)}}'))">${{esc(m)}}</button>`).join('') || '<span class="muted">未配置模型</span>'; return `<div class="row ${{active?'active':''}}"><div><div class="name">${{esc(p.name)}}<span class="protocol">${{esc(protocolName(protocol))}}</span>${{active?'<span class="pill">Active</span>':''}}</div><div class="sub">${{esc(p.id)}} · ${{p.has_api_key?'key 已保存':'未设置 key'}}</div></div><div class="sub">${{esc(p.upstream_base_url)}}</div><div class="models">${{visible}}</div><div class="actions"><select id="sel-${{esc(p.id)}}">${{models.map(m=>`<option ${{m===state.active_model?'selected':''}}>${{esc(m)}}</option>`).join('')}}</select><button class="primary" onclick="activate('${{esc(p.id)}}')">切换</button><button onclick="editProvider('${{esc(p.id)}}')">编辑</button><button class="danger" onclick="deleteProvider('${{esc(p.id)}}')">删除</button></div></div>`; }}
function routeText(model) {{ const r=(state.model_routes||{{}})[model]; if(!r) return '未配置'; return `${{r.provider}} / ${{r.model}}`; }}
function routeRow(model, info) {{ return `<div class="route-row"><div><b>${{esc(model)}}</b><div class="sub">次数 ${{info?.count||0}} · ${{esc(info?.last_seen||'')}}</div></div><div>${{esc(routeText(model))}}</div><div></div><div class="actions"><button onclick="fillRoute('${{encodeURIComponent(model)}}')">设置路由</button></div></div>`; }}
function configuredRouteRow(model, route) {{ return `<div class="route-row"><div><b>${{esc(model)}}</b></div><div>${{esc(route.provider)}}</div><div>${{esc(route.model)}}</div><div class="actions"><button onclick="fillRoute('${{encodeURIComponent(model)}}')">编辑</button><button class="danger" onclick="deleteRoute('${{encodeURIComponent(model)}}')">删除</button></div></div>`; }}
function renderRoutes() {{ const seen=state.seen_models||{{}}; const routes=state.model_routes||{{}}; const routeProviderEl=el('routeProvider'); const routeModelEl=el('routeModel'); const selectedProvider=routeProviderEl.value; const selectedModel=routeModelEl.value; const seenKeys=Object.keys(seen).sort((a,b)=>String(seen[b].last_seen||'').localeCompare(String(seen[a].last_seen||''))); el('seenRoutes').innerHTML=seenKeys.length ? seenKeys.map(m=>routeRow(m, seen[m])).join('') : '<div class="empty">还没有发现 Claude Code 请求模型；发起一次请求后会自动显示在这里。</div>'; const routeKeys=Object.keys(routes).sort(); el('configuredRoutes').innerHTML=routeKeys.length ? routeKeys.map(m=>configuredRouteRow(m, routes[m])).join('') : '<div class="empty">还没有配置模型路由。</div>'; routeProviderEl.innerHTML=(state.providers||[]).map(p=>`<option value="${{esc(p.id)}}">${{esc(p.name||p.id)}}</option>`).join(''); if(selectedProvider && (state.providers||[]).some(p=>p.id===selectedProvider)) routeProviderEl.value=selectedProvider; syncRouteModels(selectedModel); }}
function syncRouteModels(preferredModel='') {{ const routeProviderEl=el('routeProvider'); const routeModelEl=el('routeModel'); const previous=preferredModel||routeModelEl.value; const p=(state.providers||[]).find(x=>x.id===routeProviderEl.value) || (state.providers||[])[0]; routeModelEl.innerHTML=((p&&p.models)||[]).map(m=>`<option>${{esc(m)}}</option>`).join(''); if(previous && ((p&&p.models)||[]).includes(previous)) routeModelEl.value=previous; }}
function fillRoute(encoded) {{ const model=decodeURIComponent(encoded); const r=(state.model_routes||{{}})[model]||{{}}; el('routeRequested').value=model; el('routeProvider').value=r.provider||state.active_provider; syncRouteModels(); el('routeModel').value=r.model||state.active_model; el('routeRequested').scrollIntoView({{behavior:'smooth',block:'center'}}); }}
async function saveRoute() {{ try {{ await api('/admin/model-routes', {{method:'POST', body:JSON.stringify({{requested_model:el('routeRequested').value,provider_id:el('routeProvider').value,model:el('routeModel').value}})}}); await load(); show('模型路由已保存'); }} catch(e) {{ show(e.message,'err'); }} }}
async function deleteRoute(encoded) {{ if(!confirm('删除这个模型路由？')) return; try {{ await api('/admin/model-routes/delete', {{method:'POST', body:JSON.stringify({{requested_model:decodeURIComponent(encoded)}})}}); await load(); show('模型路由已删除'); }} catch(e) {{ show(e.message,'err'); }} }}
function providerName(id) {{ const p=(state.providers||[]).find(x=>x.id===id); return p ? (p.name||p.id) : id; }}
function renderRouteOverview() {{ const routes=state.model_routes||{{}}; const routeKeys=Object.keys(routes).sort(); const bound=state.primary_route_model||''; const bindText=bound ? `已绑定：${{esc(bound)}}` : '未绑定 Claude Code 默认模型'; const bindAction=bound ? '<button class="danger" onclick="unbindPrimaryRoute()">解绑主模型</button>' : '<button onclick="openPrimaryBind()">绑定主模型</button>'; const defaultCard=`<div class="summary-card"><strong>当前主路由</strong><div>${{esc(providerName(state.active_provider))}} / ${{esc(state.active_model)}}</div><div class="sub">${{bindText}}</div><div class="actions" style="justify-content:flex-start;margin-top:8px">${{bindAction}}</div></div>`; const routeCards=routeKeys.length ? routeKeys.map(m=>`<div class="summary-card"><strong>${{esc(m)}}</strong><div>${{esc(providerName(routes[m].provider))}} / ${{esc(routes[m].model)}}</div></div>`).join('') : '<div class="summary-card"><strong>具体模型路由</strong><div class="muted">尚未配置，点击上方“模型路由”添加。</div></div>'; el('routeOverview').innerHTML=defaultCard+routeCards; }}
function openRoutes() {{ el('routeDrawer').classList.add('show'); el('drawerBackdrop').classList.add('show'); renderRoutes(); }}
function primaryBindRow(model, info) {{ const active=model===state.primary_route_model; return `<div class="route-row"><div><b>${{esc(model)}}</b><div class="sub">次数 ${{info?.count||0}} · ${{esc(info?.last_seen||'')}}</div></div><div>${{esc(routeText(model))}}</div><div>${{active?'<span class="pill">已绑定</span>':''}}</div><div class="actions"><button class="primary" onclick="savePrimaryBind('${{encodeURIComponent(model)}}')">绑定</button></div></div>`; }}
function openPrimaryBind() {{ const seen=state.seen_models||{{}}; const keys=Object.keys(seen).sort((a,b)=>String(seen[b].last_seen||'').localeCompare(String(seen[a].last_seen||''))); el('primaryBindList').innerHTML=keys.length ? keys.map(m=>primaryBindRow(m, seen[m])).join('') : '<div class="empty">还没有发现 Claude Code 请求模型；先发起一次 Claude Code 请求后再绑定。</div>'; el('primaryBindDrawer').classList.add('show'); el('drawerBackdrop').classList.add('show'); }}
function closePrimaryBind() {{ const drawerEl=el('drawer'); el('primaryBindDrawer').classList.remove('show'); if(!drawerEl.classList.contains('show')&&!el('routeDrawer').classList.contains('show')) el('drawerBackdrop').classList.remove('show'); }}
async function savePrimaryBind(encoded) {{ try {{ await api('/admin/primary-route-model', {{method:'POST', body:JSON.stringify({{requested_model:decodeURIComponent(encoded)}})}}); closePrimaryBind(); await load(); show('主模型已绑定，后续切换主路由会同步更新它的模型路由'); }} catch(e) {{ show(e.message,'err'); }} }}
async function unbindPrimaryRoute() {{ if(!confirm('解绑主模型？现有模型路由会保留，不再随主路由切换。')) return; try {{ await api('/admin/primary-route-model', {{method:'POST', body:JSON.stringify({{requested_model:''}})}}); await load(); show('主模型已解绑'); }} catch(e) {{ show(e.message,'err'); }} }}
function createPricingMap() {{ return {{}}; }}
function normalizeModels() {{ return el('models').value.split('\\n').map(x=>x.trim()).filter(Boolean); }}
function pricingState() {{ if(!window.providerPricingDraft) window.providerPricingDraft=createPricingMap(); return window.providerPricingDraft; }}
function refreshPricingModels(preferred='') {{ const pricingModelEl=el('pricingModel'); const current=preferred||pricingModelEl.value||''; const modelList=normalizeModels(); const available=modelList.length?modelList:['']; pricingModelEl.innerHTML=available.map(m=>`<option value="${{esc(m)}}">${{esc(m||'未选择模型')}}</option>`).join(''); if(current && available.includes(current)) pricingModelEl.value=current; loadPricingFields(); renderPricingSummary(); }}
function loadPricingFields() {{ const selected=el('pricingModel').value||''; const entry=(pricingState()[selected]||{{}}); el('pricingInput').value=entry.input_per_million ?? ''; el('pricingOutput').value=entry.output_per_million ?? ''; el('pricingCacheRead').value=entry.cache_read_per_million ?? ''; }}
function parsePriceInput(value) {{ const raw=String(value ?? '').trim(); if(!raw) return null; const parsed=Number(raw); if(!Number.isFinite(parsed)) throw new Error('价格必须是数字'); return parsed; }}
function applyPricingFields() {{ const selected=el('pricingModel').value||''; if(!selected) return; const entry={{ input_per_million:parsePriceInput(el('pricingInput').value), output_per_million:parsePriceInput(el('pricingOutput').value), cache_read_per_million:parsePriceInput(el('pricingCacheRead').value) }}; if(Object.values(entry).every(v=>v==null)) delete pricingState()[selected]; else pricingState()[selected]=entry; renderPricingSummary(); }}
function clearPricingFields() {{ el('pricingInput').value=''; el('pricingOutput').value=''; el('pricingCacheRead').value=''; applyPricingFields(); }}
function collectPricingPayload() {{ applyPricingFields(); const allowed=new Set(normalizeModels()); const payload={{}}; Object.entries(pricingState()).forEach(([model,fields])=>{{ if(!allowed.has(model)) return; if(Object.values(fields||{{}}).every(v=>v==null)) return; payload[model]=fields; }}); return payload; }}
function renderPricingSummary() {{ const items=Object.entries(pricingState()).filter(([model, fields])=>model && !Object.values(fields||{{}}).every(v=>v==null)); el('pricingSummary').textContent=items.length ? items.map(([model, fields])=>`${{model}}: in ${{fields.input_per_million ?? '-'}}, out ${{fields.output_per_million ?? '-'}}, cache ${{fields.cache_read_per_million ?? '-'}}`).join(' | ') : '当前未配置模型价格'; }}
function closeRoutes() {{ const drawerEl=el('drawer'); const primaryBindDrawerEl=el('primaryBindDrawer'); el('routeDrawer').classList.remove('show'); if(!drawerEl.classList.contains('show')&&!primaryBindDrawerEl.classList.contains('show')) el('drawerBackdrop').classList.remove('show'); }}
function toggleSeenRoutes() {{ el('seenRoutes').classList.toggle('hidden'); }}
function openForm() {{ el('drawer').classList.add('show'); el('drawerBackdrop').classList.add('show'); refreshPricingModels(el('pricingModel').value); }}
function closeForm() {{ const routeDrawerEl=el('routeDrawer'); const primaryBindDrawerEl=el('primaryBindDrawer'); el('drawer').classList.remove('show'); if(!routeDrawerEl.classList.contains('show')&&!primaryBindDrawerEl.classList.contains('show')) el('drawerBackdrop').classList.remove('show'); }}
function editProvider(id) {{ const p=state.providers.find(x=>x.id===id); if(!p) return; el('formTitle').textContent='编辑中转站'; el('providerId').value=p.id; el('providerName').value=p.name; el('protocol').value=p.protocol||'openai'; el('baseUrl').value=p.upstream_base_url; el('apiKey').value=''; el('models').value=(p.models||[]).join('\\n'); window.providerPricingDraft=JSON.parse(JSON.stringify((state.provider_pricing||{{}})[p.id]||{{}})); refreshPricingModels((p.models||[])[0]||''); openForm(); }}
function resetForm() {{ el('formTitle').textContent='添加中转站'; el('providerId').value=''; el('providerName').value=''; el('protocol').value='openai'; el('baseUrl').value=''; el('apiKey').value=''; el('models').value=''; window.providerPricingDraft=createPricingMap(); refreshPricingModels(''); }}
async function saveProvider() {{ try {{ await api('/admin/providers', {{method:'POST', body:JSON.stringify({{id:el('providerId').value,name:el('providerName').value,protocol:el('protocol').value,upstream_base_url:el('baseUrl').value,upstream_api_key:el('apiKey').value,models:normalizeModels(),pricing:collectPricingPayload()}})}}); resetForm(); closeForm(); await load(); show('中转站已保存'); }} catch(e) {{ show(e.message,'err'); }} }}
async function activate(id) {{ try {{ const sel=el('sel-'+id); await activateModel(id, sel?.value||''); }} catch(e) {{ show(e.message,'err'); }} }}

async function activateModel(id, model) {{ try {{ await api('/admin/active', {{method:'POST', body:JSON.stringify({{provider_id:id, model}})}}); await load(); show('已切换，新的请求会立即使用该配置'); }} catch(e) {{ show(e.message,'err'); }} }}
async function deleteProvider(id) {{ if(!confirm('删除这个中转站？')) return; try {{ await api('/admin/providers/delete', {{method:'POST', body:JSON.stringify({{id}})}}); await load(); show('中转站已删除'); }} catch(e) {{ show(e.message,'err'); }} }}
load(); setInterval(()=>load(true), 2500);
</script>
</body>
</html>"""


def usage_html(state: dict[str, Any]) -> str:
    auth_hint = "true" if state.get("auth_required") else "false"
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>gpt2cc usage stats</title>
<style>
* {{ box-sizing:border-box; }} body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; background:#fafafa; color:#0f172a; }}
.top {{ height:58px; border-bottom:1px solid #e5e7eb; display:flex; align-items:center; justify-content:space-between; padding:0 22px; background:white; }} h1 {{ font-size:18px; margin:0; }} a,button {{ border:0; border-radius:10px; padding:9px 12px; background:#f1f5f9; color:#1e293b; font-weight:750; text-decoration:none; cursor:pointer; }} button.primary {{ background:#111827; color:white; }}
.shell {{ width:min(920px,calc(100vw - 36px)); margin:20px auto 48px; }} .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:12px 0 20px; }} .card,.panel {{ background:white; border:1px solid #e5e7eb; border-radius:10px; }} .card {{ padding:18px; min-height:106px; }} .label {{ color:#6b7280; font-size:12px; margin-bottom:6px; }} .value {{ font-size:24px; font-weight:850; letter-spacing:-.03em; }} .sub {{ color:#6b7280; font-size:12px; margin-top:5px; }} .panel {{ padding:16px; margin-bottom:20px; }} .panel h2 {{ margin:0 0 14px; font-size:15px; }}
.model-row {{ display:grid; grid-template-columns:140px 1fr 64px; gap:10px; align-items:center; margin:11px 0; font-size:13px; }} .bar-track {{ height:16px; border-radius:4px; background:#eeeeee; overflow:hidden; }} .bar {{ height:100%; border-radius:4px; background:#222; }} .daily-chart {{ display:flex; gap:2px; align-items:end; height:140px; padding:12px 0 4px; }} .day {{ flex:1; min-width:10px; background:#ececec; border-radius:2px 2px 0 0; position:relative; }} .day-fill {{ position:absolute; left:0; right:0; bottom:0; background:#5b6ec1; border-radius:2px 2px 0 0; }} .day-fill.cache {{ background:#10b981; height:2px; bottom:0; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ text-align:left; padding:9px 10px; border-top:1px solid #e5e7eb; }} th {{ color:#6b7280; font-weight:750; }} .muted {{ color:#6b7280; }} .auth {{ display:none; margin:20px 0; padding:14px; }} .auth.show {{ display:block; }} input {{ border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; width:260px; max-width:100%; }} .banner {{ display:none; margin:10px 0; padding:10px 12px; border-radius:10px; background:#fee2e2; color:#991b1b; }} .banner.show {{ display:block; }}
@media (max-width:780px) {{ .cards {{ grid-template-columns:1fr 1fr; }} .model-row {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class=\"top\"><h1>用量统计</h1><div><a href=\"/admin\">返回管理台</a> <button onclick=\"loadUsage()\">刷新</button></div></div>
<div class=\"shell\">
  <div id=\"authBox\" class=\"auth panel\"><b>代理密钥</b><div class=\"sub\">此页面需要 GPT2CC_PROXY_API_KEY。</div><div style=\"margin-top:10px\"><input id=\"proxyKey\" type=\"password\" placeholder=\"Proxy API key\"> <button class=\"primary\" onclick=\"saveProxyKey()\">保存并连接</button></div></div>
  <div id=\"banner\" class=\"banner\"></div>
  <section class=\"panel\"><h2>时间范围</h2><div style=\"display:flex;gap:10px;flex-wrap:wrap;align-items:end\"><div><div class=\"sub\">开始</div><input id=\"startAt\" type=\"datetime-local\"></div><div><div class=\"sub\">结束</div><input id=\"endAt\" type=\"datetime-local\"></div><button onclick=\"applyQuickRange('week')\">最近一周</button><button onclick=\"applyQuickRange('day')\">最近一天</button><button onclick=\"applyQuickRange('hour')\">最近 1 小时</button><button class=\"ghost\" onclick=\"clearUsageRange()\">清空范围</button><button class=\"primary\" onclick=\"loadUsage()\">应用筛选</button></div></section>
  <div id=\"cards\" class=\"cards\"></div>
  <section class=\"panel\"><h2>模型分布</h2><div id=\"modelDistribution\"></div></section>
  <section class=\"panel\"><h2>最近 20 条请求</h2><table><thead><tr><th>时间</th><th>Provider / 模型</th><th>输入</th><th>输出</th><th>缓存命中情况</th><th>费用</th></tr></thead><tbody id=\"requestRows\"></tbody></table></section>
  <section class=\"panel\"><h2>Provider / Model 成本明细</h2><table><thead><tr><th>Provider</th><th>模型</th><th>输入</th><th>输出</th><th>缓存读取</th><th>费用</th></tr></thead><tbody id=\"costRows\"></tbody></table><div id=\"statsPath\" class=\"sub\"></div></section>
</div>
<script>
const AUTH_REQUIRED = {auth_hint};
let summary=null, history=null;
function el(id) {{ return document.getElementById(id); }}
function key() {{ return sessionStorage.getItem('gpt2cc_proxy_key') || ''; }}
function headers() {{ const h={{'content-type':'application/json'}}; if(key()) h['x-api-key']=key(); return h; }}
function esc(s) {{ return String(s ?? '').replace(/[&<>\"]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c])); }}
function fmt(n) {{ const v=Number(n||0); if(v>=1000000) return (v/1000000).toFixed(v>=10000000?1:2)+'M'; if(v>=1000) return (v/1000).toFixed(1)+'K'; return v.toLocaleString('en-US'); }}
function pct(v) {{ return v==null ? '--' : (Number(v)*100).toFixed(1)+'%'; }}
function money(v) {{ return v==null ? '--' : (Number(v)<0.01&&Number(v)>0 ? '<$0.01' : '$'+Number(v).toFixed(2)); }}
function show(msg) {{ const bannerEl=el('banner'); bannerEl.textContent=msg; bannerEl.classList.add('show'); }}
function saveProxyKey() {{ sessionStorage.setItem('gpt2cc_proxy_key', el('proxyKey').value); loadUsage(); }}
function formatLocalDateTime(value) {{ const pad=(n)=>String(n).padStart(2,'0'); return `${{value.getFullYear()}}-${{pad(value.getMonth()+1)}}-${{pad(value.getDate())}}T${{pad(value.getHours())}}:${{pad(value.getMinutes())}}`; }}
function rangeQueryParams(limit=null) {{ const params=new URLSearchParams(); if(limit!=null) params.set('limit', String(limit)); if(el('startAt').value) params.set('start', new Date(el('startAt').value).toISOString()); if(el('endAt').value) params.set('end', new Date(el('endAt').value).toISOString()); const query=params.toString(); return query ? `?${{query}}` : ''; }}
function applyQuickRange(kind) {{ const end=new Date(); const start=new Date(end.getTime()); if(kind==='week') start.setDate(start.getDate()-7); else if(kind==='day') start.setDate(start.getDate()-1); else start.setHours(start.getHours()-1); el('startAt').value=formatLocalDateTime(start); el('endAt').value=formatLocalDateTime(end); loadUsage(); }}
function clearUsageRange() {{ el('startAt').value=''; el('endAt').value=''; loadUsage(); }}
async function api(path) {{ const r=await fetch(path, {{headers:headers()}}); if(r.status===401) {{ el('authBox').classList.add('show'); throw new Error('需要代理密钥'); }} const data=await r.json(); if(!r.ok) throw new Error(data.error?.message||'请求失败'); return data; }}
async function loadUsage() {{ try {{ el('banner').classList.remove('show'); const usageSummaryPath='/admin/usage/summary'+rangeQueryParams(); const usageHistoryPath='/admin/usage/history'+rangeQueryParams(20); [summary, history] = await Promise.all([api(usageSummaryPath), api(usageHistoryPath)]); el('authBox').classList.toggle('show', AUTH_REQUIRED && !key()); render(); }} catch(e) {{ show(e.message); }} }}
function card(label,value,sub='') {{ return `<div class=\"card\"><div class=\"label\">${{esc(label)}}</div><div class=\"value\">${{value}}</div><div class=\"sub\">${{esc(sub)}}</div></div>`; }}
function render() {{ const t=summary?.totals||{{}}; el('cards').innerHTML=[card('总 Token 数',fmt((t.input_tokens||0)+(t.output_tokens||0)),`${{fmt(t.input_tokens)}} 输入 / ${{fmt(t.output_tokens)}} 输出`),card('总会话数',fmt(t.records),`当前筛选结果 ${{fmt(t.records)}}`),card('预估费用',money(t.total_cost),t.has_pricing?'基于 provider_pricing':'未配置价格'),card('缓存命中率',pct(t.cache_hit_rate),`${{fmt(t.cache_read_input_tokens)}} 读取 / ${{fmt(t.cache_write_input_tokens)}} 写入`)].join(''); renderModels(); renderRequests(); renderCosts(); el('statsPath').textContent='统计文件：'+(summary?.stats_path||''); }}
function renderModels() {{ const rows=summary?.merged_by_model||[]; const max=Math.max(1,...rows.map(r=>(r.input_tokens||0)+(r.output_tokens||0)+(r.cache_read_input_tokens||0))); el('modelDistribution').innerHTML=rows.length?rows.map(r=>{{ const total=(r.input_tokens||0)+(r.output_tokens||0)+(r.cache_read_input_tokens||0); return `<div class=\"model-row\"><div>${{esc(r.upstream_model)}}</div><div class=\"bar-track\"><div class=\"bar\" style=\"width:${{Math.max(2,total/max*100).toFixed(1)}}%\"></div></div><div class=\"muted\">${{fmt(total)}}</div></div>`; }}).join(''):'<div class=\"muted\">暂无模型用量</div>'; }}
function renderRequests() {{ const rows=(history?.records||[]).slice(0,20); el('requestRows').innerHTML=rows.length?rows.map(r=>{{ const cacheRead=r.cache_read_input_tokens||0; const cacheWrite=r.cache_write_input_tokens||0; const hit=r.cache_hit_rate==null?'--':pct(r.cache_hit_rate); const cache=cacheRead||cacheWrite?`${{hit}} · 读取 ${{fmt(cacheRead)}} / 写入 ${{fmt(cacheWrite)}}`:'未命中'; const providerModel=`${{r.provider_name||r.provider_id||''}} / ${{r.upstream_model||''}}`; return `<tr><td>${{esc(new Date(r.ts).toLocaleString())}}</td><td>${{esc(providerModel)}}</td><td>${{fmt(r.input_tokens)}}</td><td>${{fmt(r.output_tokens)}}</td><td>${{esc(cache)}}</td><td>${{r.cost?money(r.cost.total):'未配置'}}</td></tr>`; }}).join(''):'<tr><td colspan="6" class="muted">暂无最近请求记录</td></tr>'; }}
function renderCosts() {{ const rows=summary?.provider_model_breakdown||[]; el('costRows').innerHTML=rows.length?rows.map(r=>`<tr><td>${{esc(r.provider_name||r.provider_id)}}</td><td>${{esc(r.upstream_model)}}</td><td>${{fmt(r.input_tokens)}}</td><td>${{fmt(r.output_tokens)}}</td><td>${{fmt(r.cache_read_input_tokens)}}</td><td>${{r.has_pricing?money(r.total_cost):'未配置'}}</td></tr>`).join(''):'<tr><td colspan=\"6\" class=\"muted\">暂无成本明细</td></tr>'; }}
loadUsage(); setInterval(()=>loadUsage(), 5000);
</script>
</body>
</html>"""


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


def admin_url(config: Config) -> str:
    host = config.host.strip() or "localhost"
    if host in {"127.0.0.1", "0.0.0.0", "::", "[::]"}:
        host = "localhost"
    if ":" in host and not host.startswith("[") and host != "localhost":
        host = f"[{host}]"
    return f"http://{host}:{config.port}/admin"


def should_open_admin(cli_no_open: bool) -> bool:
    if cli_no_open:
        return False
    value = os.getenv("GPT2CC_OPEN_ADMIN") or os.getenv("CCPROXY_OPEN_ADMIN")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "n", "off"}


def run(config: Config, open_admin: bool = True) -> None:
    server = ReusableThreadingHTTPServer((config.host, config.port), make_handler(config))
    url = admin_url(config)
    LOG.info("gpt2cc %s listening on http://%s:%s", __version__, config.host, config.port)
    LOG.info("admin console: %s", url)
    LOG.info("config file: %s", config.config_path)
    LOG.info("upstream protocol: %s", config.upstream_protocol)
    LOG.info("upstream chat endpoint: %s", config.upstream_chat_url)
    LOG.info("upstream images endpoint: %s", config.upstream_images_generations_url)
    LOG.info("upstream image edits endpoint: %s", config.upstream_images_edits_url)
    if open_admin:
        try:
            webbrowser.open(url)
        except Exception as exc:
            LOG.warning("could not open admin console automatically: %s", exc)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("stopping")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Messages API proxy for Claude Code.")
    parser.add_argument("--host", help="listen host; overrides GPT2CC_HOST")
    parser.add_argument("--port", type=int, help="listen port; overrides GPT2CC_PORT")
    parser.add_argument("--no-open-admin", action="store_true", help="do not open the admin console in a browser")
    args = parser.parse_args()

    config = load_config()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    created_config = ensure_config_file(config)
    if created_config:
        LOG.info("created config file: %s", config.config_path)

    run(config, open_admin=should_open_admin(args.no_open_admin))


if __name__ == "__main__":
    main()
