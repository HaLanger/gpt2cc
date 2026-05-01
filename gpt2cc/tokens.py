from __future__ import annotations

import json
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Cheap token estimate for Anthropic count_tokens compatibility.

    Claude Code uses this endpoint for budgeting, but an OpenAI-compatible
    upstream normally cannot count Anthropic-formatted content directly.
    """

    if value is None:
        return 0
    if isinstance(value, str):
        # A conservative mixed CJK/ASCII approximation.
        cjk = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
        asciiish = max(len(value) - cjk, 0)
        return max(1, cjk + asciiish // 4)
    if isinstance(value, (int, float, bool)):
        return 1
    if isinstance(value, list):
        return sum(estimate_tokens(item) for item in value) + len(value)
    if isinstance(value, dict):
        return sum(estimate_tokens(k) + estimate_tokens(v) for k, v in value.items()) + len(value)
    return estimate_tokens(json.dumps(value, ensure_ascii=False))
