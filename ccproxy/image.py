from __future__ import annotations

import base64
import binascii
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .sse import encode_sse
from .transform import TransformContext, block_to_text, new_message_start, normalize_blocks
from .upstream import MultipartFile, UpstreamError, post_image_edit, post_image_generation


LOG = logging.getLogger(__name__)
Writer = Callable[[bytes], None]
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class GeneratedImage:
    path: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    mime_type: str | None = None


@dataclass(slots=True)
class ImageGenerationResult:
    text: str
    images: list[GeneratedImage] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReferenceImage:
    filename: str
    media_type: str
    data: bytes


@dataclass(slots=True)
class ImageEditRequest:
    fields: dict[str, str]
    files: list[MultipartFile]
    prompt: str
    reference_count: int


def is_image_model(model: str, config: Config) -> bool:
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    for pattern in config.image_models:
        pattern = pattern.strip().lower()
        if not pattern:
            continue
        if pattern.endswith("*") and normalized.startswith(pattern[:-1]):
            return True
        if normalized == pattern:
            return True
    return False


def request_has_reference_images(request: dict[str, Any], config: Config) -> bool:
    return bool(extract_reference_images(request, config.image_max_reference_images))


def build_image_generation_payload(request: dict[str, Any], config: Config, ctx: TransformContext) -> dict[str, Any]:
    prompt = extract_image_prompt(request)
    if not prompt:
        raise ValueError("image generation requires a non-empty user prompt")

    payload = common_image_options(config, ctx, prompt, string_values=False)
    return payload


def build_image_edit_request(request: dict[str, Any], config: Config, ctx: TransformContext) -> ImageEditRequest:
    prompt = extract_image_prompt(request)
    if not prompt:
        raise ValueError("image edit requires a non-empty user prompt")

    references = extract_reference_images(request, config.image_max_reference_images)
    if not references:
        raise ValueError("image edit requires at least one reference image")

    fields = common_image_options(config, ctx, prompt, string_values=True)
    if config.image_input_fidelity:
        fields["input_fidelity"] = config.image_input_fidelity

    files = [
        MultipartFile(
            field_name="image[]",
            filename=reference.filename,
            content_type=reference.media_type,
            data=reference.data,
        )
        for reference in references
    ]
    return ImageEditRequest(fields=fields, files=files, prompt=prompt, reference_count=len(references))


def common_image_options(
    config: Config,
    ctx: TransformContext,
    prompt: str,
    string_values: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": ctx.upstream_model,
        "prompt": prompt,
    }

    if config.image_n > 0:
        payload["n"] = str(config.image_n) if string_values else config.image_n
    if config.image_size and config.image_size.lower() != "omit":
        payload["size"] = config.image_size
    if config.image_quality and config.image_quality.lower() != "omit":
        payload["quality"] = config.image_quality
    if config.image_output_format and config.image_output_format.lower() != "omit":
        payload["output_format"] = normalize_output_format(config.image_output_format)
    if config.image_moderation:
        payload["moderation"] = config.image_moderation

    background = (config.image_background or "").strip()
    if background and background.lower() not in {"omit", "auto"}:
        if ctx.upstream_model.lower() == "gpt-image-2" and background.lower() == "transparent":
            LOG.warning("gpt-image-2 does not support transparent background; omitting background")
        else:
            payload["background"] = background

    if string_values:
        return {key: str(value) for key, value in payload.items()}
    return payload


def generate_image(config: Config, payload: dict[str, Any], ctx: TransformContext) -> ImageGenerationResult:
    response = post_image_generation(config, payload).json()
    result = image_result_from_response(response, config, ctx, str(payload.get("prompt") or ""), reference_count=0)
    LOG.info("image generation complete: model=%s images=%s", ctx.upstream_model, len(result.images))
    return result


def edit_image(config: Config, edit_request: ImageEditRequest, ctx: TransformContext) -> ImageGenerationResult:
    response = post_image_edit(config, edit_request.fields, edit_request.files).json()
    result = image_result_from_response(
        response,
        config,
        ctx,
        edit_request.prompt,
        reference_count=edit_request.reference_count,
    )
    LOG.info(
        "image edit complete: model=%s references=%s images=%s",
        ctx.upstream_model,
        edit_request.reference_count,
        len(result.images),
    )
    return result


def anthropic_message_from_image_result(
    result: ImageGenerationResult,
    ctx: TransformContext,
    response_id: str | None = None,
) -> dict[str, Any]:
    usage = image_usage(result.raw_response)
    return {
        "id": response_id or str(result.raw_response.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": ctx.requested_model or ctx.upstream_model,
        "content": [{"type": "text", "text": result.text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def stream_image_result_to_anthropic(
    operation: Callable[[], ImageGenerationResult],
    ctx: TransformContext,
    writer: Writer,
    start_text: str,
) -> None:
    message_id = f"msg_{uuid.uuid4().hex}"
    content_index = 0
    writer(encode_sse("message_start", new_message_start(ctx, message_id)))
    writer(
        encode_sse(
            "content_block_start",
            {"type": "content_block_start", "index": content_index, "content_block": {"type": "text", "text": ""}},
        )
    )

    def write_delta(text: str) -> None:
        writer(
            encode_sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": content_index, "delta": {"type": "text_delta", "text": text}},
            )
        )

    write_delta(start_text)
    output_tokens = 0
    try:
        result = operation()
        write_delta(result.text)
        usage = image_usage(result.raw_response)
        output_tokens = usage.get("output_tokens", 0)
    except UpstreamError as exc:
        LOG.warning("image upstream error %s: %s", exc.status, exc)
        write_delta(f"Image request failed: {exc}\n")
    except Exception as exc:
        LOG.exception("image request failed")
        write_delta(f"Image request failed: {exc}\n")

    writer(encode_sse("content_block_stop", {"type": "content_block_stop", "index": content_index}))
    writer(
        encode_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
    )
    writer(encode_sse("message_stop", {"type": "message_stop"}))


def extract_image_prompt(request: dict[str, Any]) -> str:
    content = last_user_content(request)
    prompt = user_content_to_prompt(content)
    if prompt:
        return prompt
    return user_content_to_prompt(request.get("system"))


def last_user_content(request: dict[str, Any]) -> Any:
    for message in reversed(request.get("messages") or []):
        if message.get("role") == "user":
            return message.get("content")
    return None


def user_content_to_prompt(content: Any) -> str:
    parts: list[str] = []
    for block in normalize_blocks(content):
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text") or "").strip()
        elif block_type in {"tool_result", "image", "thinking", "redacted_thinking"}:
            text = ""
        else:
            text = block_to_text(block).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def extract_reference_images(request: dict[str, Any], max_images: int) -> list[ReferenceImage]:
    references: list[ReferenceImage] = []
    content = last_user_content(request)
    for block in normalize_blocks(content):
        if block.get("type") != "image":
            continue
        reference = reference_image_from_block(block, len(references) + 1)
        if reference:
            references.append(reference)
        if max_images > 0 and len(references) >= max_images:
            break
    return references


def reference_image_from_block(block: dict[str, Any], index: int) -> ReferenceImage | None:
    source = block.get("source") or {}
    if not isinstance(source, dict):
        return None

    source_type = str(source.get("type") or "").lower()
    data = source.get("data") or ""
    if source_type != "base64" and not str(data).startswith("data:image/"):
        LOG.warning("ignoring unsupported reference image source type: %s", source_type or "<empty>")
        return None

    media_type = str(source.get("media_type") or "").strip()
    b64_data = str(data)
    if b64_data.startswith("data:image/"):
        header, _, payload = b64_data.partition(",")
        b64_data = payload
        if not media_type:
            media_type = header.removeprefix("data:").split(";", 1)[0]
    if not media_type:
        media_type = "image/png"

    try:
        image_bytes = base64.b64decode(b64_data, validate=True)
    except binascii.Error as exc:
        LOG.warning("ignoring invalid base64 reference image: %s", exc)
        return None

    ext = extension_from_media_type(media_type)
    return ReferenceImage(filename=f"reference-{index}.{ext}", media_type=media_type, data=image_bytes)


def image_result_from_response(
    response: dict[str, Any],
    config: Config,
    ctx: TransformContext,
    prompt: str,
    reference_count: int = 0,
) -> ImageGenerationResult:
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("image response did not contain data")

    output_dir = Path(config.image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images: list[GeneratedImage] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        generated = GeneratedImage(
            url=str(item.get("url")) if item.get("url") else None,
            revised_prompt=str(item.get("revised_prompt")) if item.get("revised_prompt") else None,
            mime_type=str(item.get("mime_type")) if item.get("mime_type") else None,
        )
        b64_json = item.get("b64_json")
        if isinstance(b64_json, str) and b64_json:
            generated.path = str(save_b64_image(b64_json, output_dir, config, ctx, index))
        images.append(generated)

    if not images:
        raise ValueError("image response contained no usable image entries")

    return ImageGenerationResult(
        text=format_image_result_text(images, ctx, prompt, config, reference_count),
        images=images,
        raw_response=response,
    )


def save_b64_image(
    b64_json: str,
    output_dir: Path,
    config: Config,
    ctx: TransformContext,
    index: int,
) -> Path:
    try:
        image_bytes = base64.b64decode(b64_json, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"invalid base64 image data: {exc}") from exc

    ext = extension_from_format(config.image_output_format)
    model_part = SAFE_FILENAME_RE.sub("-", ctx.upstream_model.strip() or "image").strip("-")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{model_part}-{index}-{uuid.uuid4().hex[:8]}.{ext}"
    path = output_dir / filename
    path.write_bytes(image_bytes)
    return path.resolve()


def format_image_result_text(
    images: list[GeneratedImage],
    ctx: TransformContext,
    prompt: str,
    config: Config,
    reference_count: int,
) -> str:
    lines = [
        "Image request complete.",
        "",
        f"Model: {ctx.upstream_model}",
        f"Endpoint: {'images/edits' if reference_count else 'images/generations'}",
        f"Reference images: {reference_count}",
        f"Size: {config.image_size}",
        f"Quality: {config.image_quality}",
        f"Format: {normalize_output_format(config.image_output_format)}",
        "",
        "Results:",
    ]

    for index, image in enumerate(images, start=1):
        if image.path:
            lines.append(f"{index}. {image.path}")
        elif image.url:
            lines.append(f"{index}. {image.url}")
        else:
            lines.append(f"{index}. Image generated, but the response did not include base64 data or a URL.")
        if image.revised_prompt:
            lines.append(f"   revised_prompt: {image.revised_prompt}")

    lines.extend(["", "Prompt:", prompt])
    lines.append("")
    lines.append("Claude Code usually does not render images inline. Open the local file path above to view it.")
    return "\n".join(lines)


def image_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
    }


def normalize_output_format(value: str) -> str:
    value = (value or "png").strip().lower()
    if value == "jpg":
        return "jpeg"
    return value or "png"


def extension_from_format(value: str) -> str:
    value = normalize_output_format(value)
    if value == "jpeg":
        return "jpg"
    if value in {"png", "webp"}:
        return value
    return "png"


def extension_from_media_type(media_type: str) -> str:
    normalized = media_type.lower().split(";", 1)[0].strip()
    if normalized in {"image/jpeg", "image/jpg"}:
        return "jpg"
    if normalized == "image/webp":
        return "webp"
    return "png"
