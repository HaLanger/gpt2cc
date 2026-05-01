from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "yes", "y", "on"}


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def parse_jsonish_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("map JSON must be an object")
        return {str(k): str(v) for k, v in value.items()}

    result: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid map item {item!r}; expected from=to")
        source, target = item.split("=", 1)
        result[source.strip()] = target.strip()
    return result


def parse_jsonish_object(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON value must be an object")
    return {str(k): str(v) for k, v in value.items()}


def parse_map_value(value: Any) -> dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return parse_jsonish_map(str(value))


def parse_object_value(value: Any) -> dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return parse_jsonish_object(str(value))


def parse_list_value(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("list JSON must be an array")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_value(env_name: str) -> str | None:
    value = os.getenv(env_name)
    if value is not None:
        return value
    if env_name.startswith("GPT2CC_"):
        return os.getenv("CCPROXY_" + env_name.removeprefix("GPT2CC_"))
    return None


def _load_json_config() -> dict[str, Any]:
    path = _env_value("GPT2CC_CONFIG")
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"GPT2CC_CONFIG/CCPROXY_CONFIG does not exist: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("GPT2CC_CONFIG/CCPROXY_CONFIG must contain a JSON object")
    return data


def _cfg(data: dict[str, Any], key: str, env_name: str, default: Any = None) -> Any:
    env_value = _env_value(env_name)
    if env_value is not None:
        return env_value
    return data.get(key, default)


@dataclass(slots=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 3456
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_chat_path: str = "/chat/completions"
    upstream_images_generations_path: str = "/images/generations"
    upstream_images_edits_path: str = "/images/edits"
    upstream_api_key: str = ""
    upstream_auth_header: str = "Authorization"
    upstream_auth_scheme: str = "Bearer"
    upstream_ssl_verify: bool = True
    upstream_ca_bundle: str = ""
    proxy_api_key: str = ""
    model: str = ""
    model_map: dict[str, str] = field(default_factory=dict)
    pass_through_model: bool = False
    timeout_seconds: float = 600.0
    max_retries: int = 1
    max_body_bytes: int = 25 * 1024 * 1024
    log_level: str = "INFO"
    debug_payloads: bool = False
    cors: bool = True
    stream_include_usage: bool = True
    retry_without_stream_options: bool = True
    max_tokens_field: str = "max_tokens"
    omit_temperature: bool = False
    omit_top_p: bool = False
    force_stream: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    image_models: list[str] = field(default_factory=lambda: ["gpt-image-*", "dall-e-3", "dall-e-2"])
    image_output_dir: str = "generated-images"
    image_size: str = "auto"
    image_quality: str = "auto"
    image_background: str = "auto"
    image_output_format: str = "png"
    image_n: int = 1
    image_moderation: str = ""
    image_input_fidelity: str = ""
    image_max_reference_images: int = 16

    @property
    def upstream_chat_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        path = self.upstream_chat_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    @property
    def upstream_images_generations_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        path = self.upstream_images_generations_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    @property
    def upstream_images_edits_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        path = self.upstream_images_edits_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def resolve_model(self, requested: str | None) -> str:
        requested = requested or ""
        if requested in self.model_map:
            return self.model_map[requested]
        if self.model:
            return self.model
        if self.pass_through_model and requested:
            return requested
        return requested or "gpt-4.1"

    def redacted(self) -> dict[str, Any]:
        safe_extra_headers: dict[str, str] = {}
        for key, value in self.extra_headers.items():
            key_lower = key.lower()
            if "key" in key_lower or "auth" in key_lower or "token" in key_lower:
                safe_extra_headers[key] = "***"
            else:
                safe_extra_headers[key] = value

        return {
            "host": self.host,
            "port": self.port,
            "upstream_base_url": self.upstream_base_url,
            "upstream_chat_path": self.upstream_chat_path,
            "upstream_images_generations_path": self.upstream_images_generations_path,
            "upstream_images_edits_path": self.upstream_images_edits_path,
            "upstream_api_key": "***" if self.upstream_api_key else "",
            "upstream_auth_header": self.upstream_auth_header,
            "upstream_ssl_verify": self.upstream_ssl_verify,
            "upstream_ca_bundle": self.upstream_ca_bundle,
            "proxy_api_key": "***" if self.proxy_api_key else "",
            "model": self.model,
            "model_map": self.model_map,
            "pass_through_model": self.pass_through_model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_body_bytes": self.max_body_bytes,
            "log_level": self.log_level,
            "debug_payloads": self.debug_payloads,
            "cors": self.cors,
            "stream_include_usage": self.stream_include_usage,
            "retry_without_stream_options": self.retry_without_stream_options,
            "max_tokens_field": self.max_tokens_field,
            "omit_temperature": self.omit_temperature,
            "omit_top_p": self.omit_top_p,
            "force_stream": self.force_stream,
            "extra_headers": safe_extra_headers,
            "image_models": self.image_models,
            "image_output_dir": self.image_output_dir,
            "image_size": self.image_size,
            "image_quality": self.image_quality,
            "image_background": self.image_background,
            "image_output_format": self.image_output_format,
            "image_n": self.image_n,
            "image_moderation": self.image_moderation,
            "image_input_fidelity": self.image_input_fidelity,
            "image_max_reference_images": self.image_max_reference_images,
        }


def load_config() -> Config:
    load_dotenv()
    file_config = _load_json_config()

    config = Config(
        host=str(_cfg(file_config, "host", "GPT2CC_HOST", "127.0.0.1")),
        port=int(_cfg(file_config, "port", "GPT2CC_PORT", 3456)),
        upstream_base_url=str(
            _cfg(
                file_config,
                "upstream_base_url",
                "GPT2CC_UPSTREAM_BASE_URL",
                "https://api.openai.com/v1",
            )
        ),
        upstream_chat_path=str(
            _cfg(file_config, "upstream_chat_path", "GPT2CC_UPSTREAM_CHAT_PATH", "/chat/completions")
        ),
        upstream_images_generations_path=str(
            _cfg(
                file_config,
                "upstream_images_generations_path",
                "GPT2CC_UPSTREAM_IMAGES_GENERATIONS_PATH",
                "/images/generations",
            )
        ),
        upstream_images_edits_path=str(
            _cfg(file_config, "upstream_images_edits_path", "GPT2CC_UPSTREAM_IMAGES_EDITS_PATH", "/images/edits")
        ),
        upstream_api_key=str(_cfg(file_config, "upstream_api_key", "GPT2CC_UPSTREAM_API_KEY", "")),
        upstream_auth_header=str(
            _cfg(file_config, "upstream_auth_header", "GPT2CC_UPSTREAM_AUTH_HEADER", "Authorization")
        ),
        upstream_auth_scheme=str(_cfg(file_config, "upstream_auth_scheme", "GPT2CC_UPSTREAM_AUTH_SCHEME", "Bearer")),
        upstream_ssl_verify=str(
            _cfg(file_config, "upstream_ssl_verify", "GPT2CC_UPSTREAM_SSL_VERIFY", "true")
        ).lower()
        in TRUTHY,
        upstream_ca_bundle=str(_cfg(file_config, "upstream_ca_bundle", "GPT2CC_UPSTREAM_CA_BUNDLE", "")),
        proxy_api_key=str(_cfg(file_config, "proxy_api_key", "GPT2CC_PROXY_API_KEY", "")),
        model=str(_cfg(file_config, "model", "GPT2CC_UPSTREAM_MODEL", "")),
        model_map=parse_map_value(_cfg(file_config, "model_map", "GPT2CC_MODEL_MAP", "")),
        pass_through_model=str(
            _cfg(file_config, "pass_through_model", "GPT2CC_PASS_THROUGH_MODEL", "false")
        ).lower()
        in TRUTHY,
        timeout_seconds=float(_cfg(file_config, "timeout_seconds", "GPT2CC_TIMEOUT_SECONDS", 600.0)),
        max_retries=int(_cfg(file_config, "max_retries", "GPT2CC_MAX_RETRIES", 1)),
        max_body_bytes=int(_cfg(file_config, "max_body_bytes", "GPT2CC_MAX_BODY_BYTES", 25 * 1024 * 1024)),
        log_level=str(_cfg(file_config, "log_level", "GPT2CC_LOG_LEVEL", "INFO")),
        debug_payloads=str(_cfg(file_config, "debug_payloads", "GPT2CC_DEBUG_PAYLOADS", "false")).lower()
        in TRUTHY,
        cors=str(_cfg(file_config, "cors", "GPT2CC_CORS", "true")).lower() in TRUTHY,
        stream_include_usage=str(
            _cfg(file_config, "stream_include_usage", "GPT2CC_STREAM_INCLUDE_USAGE", "true")
        ).lower()
        in TRUTHY,
        retry_without_stream_options=str(
            _cfg(file_config, "retry_without_stream_options", "GPT2CC_RETRY_WITHOUT_STREAM_OPTIONS", "true")
        ).lower()
        in TRUTHY,
        max_tokens_field=str(_cfg(file_config, "max_tokens_field", "GPT2CC_MAX_TOKENS_FIELD", "max_tokens")),
        omit_temperature=str(_cfg(file_config, "omit_temperature", "GPT2CC_OMIT_TEMPERATURE", "false")).lower()
        in TRUTHY,
        omit_top_p=str(_cfg(file_config, "omit_top_p", "GPT2CC_OMIT_TOP_P", "false")).lower() in TRUTHY,
        force_stream=str(_cfg(file_config, "force_stream", "GPT2CC_FORCE_STREAM", "false")).lower() in TRUTHY,
        extra_headers=parse_object_value(_cfg(file_config, "extra_headers", "GPT2CC_UPSTREAM_EXTRA_HEADERS", "")),
        image_models=parse_list_value(
            _cfg(file_config, "image_models", "GPT2CC_IMAGE_MODELS", "gpt-image-*,dall-e-3,dall-e-2")
        ),
        image_output_dir=str(_cfg(file_config, "image_output_dir", "GPT2CC_IMAGE_OUTPUT_DIR", "generated-images")),
        image_size=str(_cfg(file_config, "image_size", "GPT2CC_IMAGE_SIZE", "auto")),
        image_quality=str(_cfg(file_config, "image_quality", "GPT2CC_IMAGE_QUALITY", "auto")),
        image_background=str(_cfg(file_config, "image_background", "GPT2CC_IMAGE_BACKGROUND", "auto")),
        image_output_format=str(_cfg(file_config, "image_output_format", "GPT2CC_IMAGE_OUTPUT_FORMAT", "png")),
        image_n=int(_cfg(file_config, "image_n", "GPT2CC_IMAGE_N", 1)),
        image_moderation=str(_cfg(file_config, "image_moderation", "GPT2CC_IMAGE_MODERATION", "")),
        image_input_fidelity=str(_cfg(file_config, "image_input_fidelity", "GPT2CC_IMAGE_INPUT_FIDELITY", "")),
        image_max_reference_images=int(
            _cfg(file_config, "image_max_reference_images", "GPT2CC_IMAGE_MAX_REFERENCE_IMAGES", 16)
        ),
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return config
