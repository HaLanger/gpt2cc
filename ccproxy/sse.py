from __future__ import annotations

import json
from typing import Any, BinaryIO, Iterable


def encode_sse(event: str, data: dict[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def encode_done() -> bytes:
    return b"data: [DONE]\n\n"


def iter_sse_data(response: BinaryIO) -> Iterable[str]:
    event_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            data = _event_data(event_lines)
            event_lines.clear()
            if data is not None:
                yield data
            continue
        event_lines.append(line)

    data = _event_data(event_lines)
    if data is not None:
        yield data


def _event_data(lines: list[str]) -> str | None:
    data_lines: list[str] = []
    for line in lines:
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    return "\n".join(data_lines)
