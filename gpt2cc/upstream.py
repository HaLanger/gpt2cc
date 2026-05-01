from __future__ import annotations

import http.client
import json
import logging
import ssl
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO

from .config import Config


LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class UpstreamResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        return json.loads(self.body.decode("utf-8"))


class UpstreamError(RuntimeError):
    def __init__(self, status: int, message: str, body: bytes | None = None):
        super().__init__(message)
        self.status = status
        self.body = body or b""


@dataclass(slots=True)
class MultipartFile:
    field_name: str
    filename: str
    content_type: str
    data: bytes


def build_ssl_context(config: Config) -> ssl.SSLContext:
    if not config.upstream_ssl_verify:
        LOG.warning("upstream TLS certificate verification is disabled")
        return ssl._create_unverified_context()
    if config.upstream_ca_bundle:
        return ssl.create_default_context(cafile=config.upstream_ca_bundle)
    return ssl.create_default_context()


def build_headers(config: Config, stream: bool = False, content_type: str = "application/json") -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": "gpt2cc/0.1.0",
    }
    if content_type:
        headers["Content-Type"] = content_type
    if config.upstream_api_key:
        if config.upstream_auth_scheme:
            headers[config.upstream_auth_header] = f"{config.upstream_auth_scheme} {config.upstream_api_key}"
        else:
            headers[config.upstream_auth_header] = config.upstream_api_key
    headers.update(config.extra_headers)
    return headers


def post_json(config: Config, payload: dict[str, Any]) -> UpstreamResponse:
    return post_json_url(config, config.upstream_chat_url, payload)


def post_image_generation(config: Config, payload: dict[str, Any]) -> UpstreamResponse:
    return post_json_url(config, config.upstream_images_generations_url, payload)


def post_image_edit(config: Config, fields: dict[str, str], files: list[MultipartFile]) -> UpstreamResponse:
    return post_multipart_url(config, config.upstream_images_edits_url, fields, files)


def post_json_url(
    config: Config,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> UpstreamResponse:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: UpstreamError | None = None
    ssl_context = build_ssl_context(config)
    request_headers = headers or build_headers(config, stream=False)
    for attempt in range(config.max_retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=body,
                headers=request_headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=config.timeout_seconds, context=ssl_context) as response:
                data = response.read()
                return UpstreamResponse(response.status, dict(response.headers.items()), data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            raise UpstreamError(exc.code, decode_error(data) or exc.reason, data) from exc
        except http.client.IncompleteRead as exc:
            last_error = UpstreamError(
                502,
                format_incomplete_read(exc),
                exc.partial,
            )
        except urllib.error.URLError as exc:
            last_error = UpstreamError(502, format_url_error(exc))
        except ssl.SSLError as exc:
            last_error = UpstreamError(502, format_ssl_error(exc))
        except TimeoutError:
            last_error = UpstreamError(504, "upstream request timed out")

        if attempt < config.max_retries:
            time.sleep(min(2**attempt, 5))

    assert last_error is not None
    raise last_error


def post_multipart_url(
    config: Config,
    url: str,
    fields: dict[str, str],
    files: list[MultipartFile],
) -> UpstreamResponse:
    boundary = f"gpt2cc-{uuid.uuid4().hex}"
    body = encode_multipart(fields, files, boundary)
    last_error: UpstreamError | None = None
    ssl_context = build_ssl_context(config)
    headers = build_headers(config, stream=False, content_type=f"multipart/form-data; boundary={boundary}")
    for attempt in range(config.max_retries + 1):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=config.timeout_seconds, context=ssl_context) as response:
                data = response.read()
                return UpstreamResponse(response.status, dict(response.headers.items()), data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            raise UpstreamError(exc.code, decode_error(data) or exc.reason, data) from exc
        except http.client.IncompleteRead as exc:
            last_error = UpstreamError(
                502,
                format_incomplete_read(exc),
                exc.partial,
            )
        except urllib.error.URLError as exc:
            last_error = UpstreamError(502, format_url_error(exc))
        except ssl.SSLError as exc:
            last_error = UpstreamError(502, format_ssl_error(exc))
        except TimeoutError:
            last_error = UpstreamError(504, "upstream request timed out")

        if attempt < config.max_retries:
            time.sleep(min(2**attempt, 5))

    assert last_error is not None
    raise last_error


def encode_multipart(fields: dict[str, str], files: list[MultipartFile], boundary: str) -> bytes:
    chunks: list[bytes] = []
    boundary_bytes = boundary.encode("ascii")
    for name, value in fields.items():
        chunks.extend(
            [
                b"--" + boundary_bytes + b"\r\n",
                f'Content-Disposition: form-data; name="{escape_multipart_name(name)}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for file in files:
        chunks.extend(
            [
                b"--" + boundary_bytes + b"\r\n",
                (
                    f'Content-Disposition: form-data; name="{escape_multipart_name(file.field_name)}"; '
                    f'filename="{escape_multipart_name(file.filename)}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {file.content_type or 'application/octet-stream'}\r\n\r\n".encode("ascii"),
                file.data,
                b"\r\n",
            ]
        )
    chunks.extend([b"--" + boundary_bytes + b"--\r\n"])
    return b"".join(chunks)


def escape_multipart_name(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def open_stream_url(
    config: Config,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> BinaryIO:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ssl_context = build_ssl_context(config)
    try:
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers or build_headers(config, stream=True),
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=config.timeout_seconds, context=ssl_context)
    except urllib.error.HTTPError as exc:
        data = exc.read()
        raise UpstreamError(exc.code, decode_error(data) or exc.reason, data) from exc
    except urllib.error.URLError as exc:
        raise UpstreamError(502, format_url_error(exc)) from exc
    except ssl.SSLError as exc:
        raise UpstreamError(502, format_ssl_error(exc)) from exc
    except TimeoutError as exc:
        raise UpstreamError(504, "upstream request timed out") from exc


def open_stream(config: Config, payload: dict[str, Any]) -> BinaryIO:
    return open_stream_url(config, config.upstream_chat_url, payload)


def open_stream_with_retry(config: Config, payload: dict[str, Any]) -> BinaryIO:
    try:
        return open_stream(config, payload)
    except UpstreamError as exc:
        if (
            config.retry_without_stream_options
            and exc.status in {400, 422}
            and "stream_options" in payload
        ):
            LOG.info("upstream rejected stream_options; retrying without it")
            retry_payload = dict(payload)
            retry_payload.pop("stream_options", None)
            return open_stream(config, retry_payload)
        raise


def decode_error(data: bytes) -> str:
    if not data:
        return ""
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception:
        return data.decode("utf-8", errors="replace")[:4000]
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return json.dumps(value, ensure_ascii=False)[:4000]


def format_incomplete_read(exc: http.client.IncompleteRead) -> str:
    expected = getattr(exc, "expected", None)
    if expected is None:
        return f"upstream response ended before the full body was received: {len(exc.partial)} bytes read"
    return (
        "upstream response ended before the full body was received: "
        f"{len(exc.partial)} bytes read, {expected} more expected"
    )


def format_url_error(exc: urllib.error.URLError) -> str:
    reason = exc.reason
    if isinstance(reason, ssl.SSLError):
        return format_ssl_error(reason)
    return str(reason)


def format_ssl_error(exc: ssl.SSLError) -> str:
    message = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" not in message:
        return message
    return (
        f"{message}. The upstream HTTPS certificate chain is not trusted by this Python runtime. "
        "Prefer setting GPT2CC_UPSTREAM_CA_BUNDLE to a PEM file containing the missing root/intermediate CA. "
        "For temporary local troubleshooting only, set GPT2CC_UPSTREAM_SSL_VERIFY=false and restart gpt2cc."
    )
