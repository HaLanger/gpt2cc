from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "yes", "y", "on"}
DEFAULT_CONFIG_PATH = "gpt2cc.config.json"
PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SUPPORTED_PROTOCOLS = {"openai", "anthropic", "gemini"}


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


def config_path_from_env() -> str:
    return _env_value("GPT2CC_CONFIG") or DEFAULT_CONFIG_PATH


def _load_json_config() -> dict[str, Any]:
    config_path = Path(config_path_from_env())
    if not config_path.exists():
        return {}
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
    upstream_protocol: str = "openai"
    upstream_chat_path: str = "/chat/completions"
    upstream_messages_path: str = "/messages"
    upstream_gemini_generate_path: str = "/models/{model}:generateContent"
    upstream_gemini_stream_path: str = "/models/{model}:streamGenerateContent"
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
    providers: list[dict[str, Any]] = field(default_factory=list)
    active_provider: str = "default"
    active_model: str = ""
    config_path: str = DEFAULT_CONFIG_PATH

    def active_provider_label(self) -> str:
        provider = next((p for p in self.providers if p.get("id") == self.active_provider), None)
        if not provider:
            return self.active_provider or "<none>"
        name = str(provider.get("name") or provider.get("id") or "").strip()
        provider_id = str(provider.get("id") or "").strip()
        if name and provider_id and name != provider_id:
            return f"{name} ({provider_id})"
        return name or provider_id or "<none>"

    @property
    def upstream_chat_url(self) -> str:
        return self._join_upstream_path(self.upstream_chat_path)

    @property
    def upstream_images_generations_url(self) -> str:
        return self._join_upstream_path(self.upstream_images_generations_path)

    @property
    def upstream_images_edits_url(self) -> str:
        return self._join_upstream_path(self.upstream_images_edits_path)

    @property
    def upstream_messages_url(self) -> str:
        return self._join_upstream_path(self.upstream_messages_path)

    @property
    def upstream_gemini_generate_url(self) -> str:
        return self._join_upstream_path(self.upstream_gemini_generate_path.format(model=self.model))

    @property
    def upstream_gemini_stream_url(self) -> str:
        return self._join_upstream_path(self.upstream_gemini_stream_path.format(model=self.model))

    def _join_upstream_path(self, path: str) -> str:
        base = self.upstream_base_url.rstrip("/")
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
            "upstream_protocol": self.upstream_protocol,
            "upstream_chat_path": self.upstream_chat_path,
            "upstream_messages_path": self.upstream_messages_path,
            "upstream_gemini_generate_path": self.upstream_gemini_generate_path,
            "upstream_gemini_stream_path": self.upstream_gemini_stream_path,
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
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "providers": redacted_providers(self.providers),
            "config_path": self.config_path,
        }


def normalize_provider(value: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    provider_id = str(value.get("id") or "").strip()
    if not provider_id or not PROVIDER_ID_RE.match(provider_id):
        raise ValueError("provider id may only contain letters, numbers, dots, underscores, and dashes")

    base_url = str(value.get("upstream_base_url") or value.get("base_url") or "").strip().rstrip("/")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise ValueError("provider upstream_base_url must start with http:// or https://")

    api_key = str(value.get("upstream_api_key") or value.get("api_key") or "")
    if not api_key and existing:
        api_key = str(existing.get("upstream_api_key") or "")

    protocol = str(value.get("protocol") or value.get("upstream_protocol") or (existing or {}).get("protocol") or "openai").strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError("provider protocol must be one of: anthropic, gemini, openai")

    models = parse_list_value(value.get("models"))
    provider = {
        "id": provider_id,
        "name": str(value.get("name") or provider_id).strip() or provider_id,
        "protocol": protocol,
        "upstream_base_url": base_url,
        "upstream_api_key": api_key,
        "models": models,
        "upstream_chat_path": str(value.get("upstream_chat_path") or (existing or {}).get("upstream_chat_path") or "/chat/completions"),
        "upstream_messages_path": str(value.get("upstream_messages_path") or (existing or {}).get("upstream_messages_path") or "/messages"),
        "upstream_gemini_generate_path": str(
            value.get("upstream_gemini_generate_path")
            or (existing or {}).get("upstream_gemini_generate_path")
            or "/models/{model}:generateContent"
        ),
        "upstream_gemini_stream_path": str(
            value.get("upstream_gemini_stream_path")
            or (existing or {}).get("upstream_gemini_stream_path")
            or "/models/{model}:streamGenerateContent"
        ),
        "upstream_images_generations_path": str(
            value.get("upstream_images_generations_path")
            or (existing or {}).get("upstream_images_generations_path")
            or "/images/generations"
        ),
        "upstream_images_edits_path": str(
            value.get("upstream_images_edits_path") or (existing or {}).get("upstream_images_edits_path") or "/images/edits"
        ),
        "upstream_auth_header": str(value.get("upstream_auth_header") or (existing or {}).get("upstream_auth_header") or "Authorization"),
        "upstream_auth_scheme": str(value.get("upstream_auth_scheme") if value.get("upstream_auth_scheme") is not None else (existing or {}).get("upstream_auth_scheme", "Bearer")),
    }
    return provider


def provider_from_config(config: Config) -> dict[str, Any]:
    model = config.model or "gpt-4.1"
    return {
        "id": "default",
        "name": "Default relay",
        "protocol": config.upstream_protocol,
        "upstream_base_url": config.upstream_base_url,
        "upstream_api_key": config.upstream_api_key,
        "models": [model] if model else [],
        "upstream_chat_path": config.upstream_chat_path,
        "upstream_messages_path": config.upstream_messages_path,
        "upstream_gemini_generate_path": config.upstream_gemini_generate_path,
        "upstream_gemini_stream_path": config.upstream_gemini_stream_path,
        "upstream_images_generations_path": config.upstream_images_generations_path,
        "upstream_images_edits_path": config.upstream_images_edits_path,
        "upstream_auth_header": config.upstream_auth_header,
        "upstream_auth_scheme": config.upstream_auth_scheme,
    }


def redacted_providers(providers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for provider in providers:
        safe = dict(provider)
        has_key = bool(safe.get("upstream_api_key"))
        safe["upstream_api_key"] = "***" if has_key else ""
        safe["has_api_key"] = has_key
        result.append(safe)
    return result


def apply_provider(config: Config, provider: dict[str, Any], model: str) -> Config:
    return replace(
        config,
        upstream_protocol=str(provider.get("protocol") or "openai"),
        upstream_base_url=str(provider.get("upstream_base_url") or config.upstream_base_url),
        upstream_api_key=str(provider.get("upstream_api_key") or ""),
        upstream_chat_path=str(provider.get("upstream_chat_path") or config.upstream_chat_path),
        upstream_messages_path=str(provider.get("upstream_messages_path") or config.upstream_messages_path),
        upstream_gemini_generate_path=str(
            provider.get("upstream_gemini_generate_path") or config.upstream_gemini_generate_path
        ),
        upstream_gemini_stream_path=str(provider.get("upstream_gemini_stream_path") or config.upstream_gemini_stream_path),
        upstream_images_generations_path=str(
            provider.get("upstream_images_generations_path") or config.upstream_images_generations_path
        ),
        upstream_images_edits_path=str(provider.get("upstream_images_edits_path") or config.upstream_images_edits_path),
        upstream_auth_header=str(provider.get("upstream_auth_header") or config.upstream_auth_header),
        upstream_auth_scheme=str(provider.get("upstream_auth_scheme") if provider.get("upstream_auth_scheme") is not None else config.upstream_auth_scheme),
        model=model,
    )


def copy_config(config: Config) -> Config:
    return replace(
        config,
        model_map=dict(config.model_map),
        extra_headers=dict(config.extra_headers),
        image_models=list(config.image_models),
        providers=copy.deepcopy(config.providers),
    )


class ConfigStore:
    def __init__(self, config: Config):
        self._lock = threading.RLock()
        self._config = copy_config(config)

    def snapshot(self) -> Config:
        with self._lock:
            return copy_config(self._config)

    def state(self) -> dict[str, Any]:
        with self._lock:
            providers = redacted_providers(self._config.providers)
            providers.sort(key=lambda provider: provider.get("id") != self._config.active_provider)
            return {
                "active_provider": self._config.active_provider,
                "active_model": self._config.active_model or self._config.model,
                "active_protocol": self._config.upstream_protocol,
                "providers": providers,
                "config_path": self._config.config_path,
                "auth_required": bool(self._config.proxy_api_key),
            }

    def add_or_update_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            existing = next((p for p in self._config.providers if p.get("id") == payload.get("id")), None)
            provider = normalize_provider(payload, existing)
            providers = [p for p in self._config.providers if p.get("id") != provider["id"]]
            providers.append(provider)
            active_provider = self._config.active_provider
            active_model = self._config.active_model
            config = replace(self._config, providers=providers)
            if not active_provider or active_provider == provider["id"]:
                active_provider = provider["id"]
                active_model = active_model or (provider["models"][0] if provider["models"] else self._config.model)
                config = self._activate(config, active_provider, active_model)
            self._config = config
            self.save()
            return self.state()

    def delete_provider(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            providers = [p for p in self._config.providers if p.get("id") != provider_id]
            if len(providers) == len(self._config.providers):
                raise ValueError("provider not found")
            if not providers:
                raise ValueError("at least one provider is required")
            config = replace(self._config, providers=providers)
            if self._config.active_provider == provider_id:
                provider = providers[0]
                model = provider["models"][0] if provider.get("models") else self._config.model
                config = self._activate(config, str(provider["id"]), model)
            self._config = config
            self.save()
            return self.state()

    def set_active(self, provider_id: str, model: str) -> dict[str, Any]:
        with self._lock:
            self._config = self._activate(self._config, provider_id, model)
            self.save()
            return self.state()

    def _activate(self, config: Config, provider_id: str, model: str) -> Config:
        provider = next((p for p in config.providers if p.get("id") == provider_id), None)
        if not provider:
            raise ValueError("provider not found")
        models = [str(item) for item in provider.get("models") or []]
        model = str(model or (models[0] if models else config.model or "gpt-4.1")).strip()
        if models and model not in models:
            raise ValueError("model is not listed for this provider")
        config = replace(config, active_provider=provider_id, active_model=model)
        return apply_provider(config, provider, model)

    def save(self) -> None:
        data = config_to_json(self._config)
        path = Path(self._config.config_path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config_to_json(config: Config) -> dict[str, Any]:
    return {
        "host": config.host,
        "port": config.port,
        "upstream_base_url": config.upstream_base_url,
        "upstream_protocol": config.upstream_protocol,
        "upstream_chat_path": config.upstream_chat_path,
        "upstream_messages_path": config.upstream_messages_path,
        "upstream_gemini_generate_path": config.upstream_gemini_generate_path,
        "upstream_gemini_stream_path": config.upstream_gemini_stream_path,
        "upstream_images_generations_path": config.upstream_images_generations_path,
        "upstream_images_edits_path": config.upstream_images_edits_path,
        "upstream_api_key": config.upstream_api_key,
        "upstream_auth_header": config.upstream_auth_header,
        "upstream_auth_scheme": config.upstream_auth_scheme,
        "upstream_ssl_verify": config.upstream_ssl_verify,
        "upstream_ca_bundle": config.upstream_ca_bundle,
        "proxy_api_key": config.proxy_api_key,
        "model": config.model,
        "model_map": config.model_map,
        "pass_through_model": config.pass_through_model,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "max_body_bytes": config.max_body_bytes,
        "log_level": config.log_level,
        "debug_payloads": config.debug_payloads,
        "cors": config.cors,
        "stream_include_usage": config.stream_include_usage,
        "retry_without_stream_options": config.retry_without_stream_options,
        "max_tokens_field": config.max_tokens_field,
        "omit_temperature": config.omit_temperature,
        "omit_top_p": config.omit_top_p,
        "force_stream": config.force_stream,
        "extra_headers": config.extra_headers,
        "image_models": config.image_models,
        "image_output_dir": config.image_output_dir,
        "image_size": config.image_size,
        "image_quality": config.image_quality,
        "image_background": config.image_background,
        "image_output_format": config.image_output_format,
        "image_n": config.image_n,
        "image_moderation": config.image_moderation,
        "image_input_fidelity": config.image_input_fidelity,
        "image_max_reference_images": config.image_max_reference_images,
        "active_provider": config.active_provider,
        "active_model": config.active_model,
        "providers": config.providers,
    }


def ensure_config_file(config: Config) -> bool:
    path = Path(config.config_path)
    if path.exists():
        return False
    path.write_text(json.dumps(config_to_json(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def load_config() -> Config:
    load_dotenv()
    config_path = config_path_from_env()
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
        upstream_protocol=str(_cfg(file_config, "upstream_protocol", "GPT2CC_UPSTREAM_PROTOCOL", "openai")).lower(),
        upstream_chat_path=str(
            _cfg(file_config, "upstream_chat_path", "GPT2CC_UPSTREAM_CHAT_PATH", "/chat/completions")
        ),
        upstream_messages_path=str(_cfg(file_config, "upstream_messages_path", "GPT2CC_UPSTREAM_MESSAGES_PATH", "/messages")),
        upstream_gemini_generate_path=str(
            _cfg(
                file_config,
                "upstream_gemini_generate_path",
                "GPT2CC_UPSTREAM_GEMINI_GENERATE_PATH",
                "/models/{model}:generateContent",
            )
        ),
        upstream_gemini_stream_path=str(
            _cfg(
                file_config,
                "upstream_gemini_stream_path",
                "GPT2CC_UPSTREAM_GEMINI_STREAM_PATH",
                "/models/{model}:streamGenerateContent",
            )
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
        config_path=config_path,
    )

    if config.upstream_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError("upstream protocol must be one of: anthropic, gemini, openai")

    raw_providers = file_config.get("providers")
    if isinstance(raw_providers, list) and raw_providers:
        providers = [normalize_provider(item) for item in raw_providers if isinstance(item, dict)]
    else:
        providers = [provider_from_config(config)]
    active_provider = str(_cfg(file_config, "active_provider", "GPT2CC_ACTIVE_PROVIDER", providers[0]["id"]))
    env_model = _env_value("GPT2CC_UPSTREAM_MODEL")
    active_model = str(env_model or _cfg(file_config, "active_model", "GPT2CC_ACTIVE_MODEL", config.model or ""))
    if not active_model:
        provider = next((p for p in providers if p["id"] == active_provider), providers[0])
        active_model = (provider.get("models") or [config.model or "gpt-4.1"])[0]
    config = replace(config, providers=providers, active_provider=active_provider, active_model=active_model)
    active = next((p for p in providers if p["id"] == active_provider), providers[0])
    config = apply_provider(config, active, active_model)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return config
