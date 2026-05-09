from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

STATS_FILE_VERSION = 1
COST_PRECISION = Decimal("0.0000000001")
PRICE_FIELDS = ("input_per_million", "output_per_million", "cache_read_per_million")
TRUTHY = {"1", "true", "yes", "y", "on"}
_STATS_LOCK = threading.RLock()


@dataclass(slots=True)
class UsagePrice:
    provider_id: str
    model: str
    input_per_million: float | None = None
    output_per_million: float | None = None
    cache_read_per_million: float | None = None


@dataclass(slots=True)
class UsageCost:
    input: float
    output: float
    cache_read: float
    total: float


@dataclass(slots=True)
class UsageRecord:
    ts: str
    protocol: str
    requested_model: str
    provider_id: str
    provider_name: str
    upstream_model: str
    route_source: str
    stream: bool
    endpoint: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_hit_rate: float | None = None
    price: UsagePrice | None = None
    cost: UsageCost | None = None


@dataclass(slots=True)
class UsageStatsDocument:
    version: int = STATS_FILE_VERSION
    records: list[UsageRecord] | None = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []


@dataclass(slots=True)
class UsageSummary:
    records: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int
    cache_hit_rate: float | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def calculate_cache_hit_rate(input_tokens: int, cache_read_input_tokens: int) -> float | None:
    denominator = int(input_tokens) + int(cache_read_input_tokens)
    if denominator <= 0:
        return None
    return int(cache_read_input_tokens) / denominator


def _coerce_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _coerce_str(value: Any) -> str:
    return str(value or "")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY
    return bool(value)


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantize_cost(value: Decimal) -> float:
    return float(value.quantize(COST_PRECISION, rounding=ROUND_HALF_UP))


def _decimal_cost(token_count: int, price_per_million: float | None) -> Decimal:
    if token_count <= 0 or price_per_million is None:
        return Decimal("0")
    return (Decimal(token_count) * Decimal(str(price_per_million))) / Decimal("1000000")


def normalize_usage_price(value: Any) -> UsagePrice | None:
    if not isinstance(value, dict):
        return None
    provider_id = _coerce_str(value.get("provider_id")).strip()
    model = _coerce_str(value.get("model")).strip()
    if not provider_id or not model:
        return None
    prices = {
        "input_per_million": _coerce_float(value.get("input_per_million")),
        "output_per_million": _coerce_float(value.get("output_per_million")),
        "cache_read_per_million": _coerce_float(value.get("cache_read_per_million")),
    }
    if any(raw is not None and prices[field] is None for field, raw in ((name, value.get(name)) for name in PRICE_FIELDS)):
        return None
    return UsagePrice(
        provider_id=provider_id,
        model=model,
        input_per_million=prices["input_per_million"],
        output_per_million=prices["output_per_million"],
        cache_read_per_million=prices["cache_read_per_million"],
    )


def compute_usage_cost(token_counts: dict[str, int], price: UsagePrice | None) -> UsageCost | None:
    if price is None:
        return None
    if all(getattr(price, field) is None for field in PRICE_FIELDS):
        return None
    input_cost = _decimal_cost(_coerce_int(token_counts.get("input_tokens")), price.input_per_million)
    output_cost = _decimal_cost(_coerce_int(token_counts.get("output_tokens")), price.output_per_million)
    cache_read_cost = _decimal_cost(_coerce_int(token_counts.get("cache_read_input_tokens")), price.cache_read_per_million)
    total = input_cost + output_cost + cache_read_cost
    return UsageCost(
        input=_quantize_cost(input_cost),
        output=_quantize_cost(output_cost),
        cache_read=_quantize_cost(cache_read_cost),
        total=_quantize_cost(total),
    )


def build_usage_record(
    *,
    protocol: str,
    requested_model: str,
    provider_id: str,
    provider_name: str,
    upstream_model: str,
    route_source: str,
    stream: bool,
    endpoint: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    ts: str | None = None,
    price: UsagePrice | None = None,
) -> UsageRecord:
    token_counts = {
        "input_tokens": _coerce_int(input_tokens),
        "output_tokens": _coerce_int(output_tokens),
        "cache_read_input_tokens": _coerce_int(cache_read_input_tokens),
        "cache_write_input_tokens": _coerce_int(cache_write_input_tokens),
    }
    return UsageRecord(
        ts=ts or utc_now_iso(),
        protocol=_coerce_str(protocol),
        requested_model=_coerce_str(requested_model),
        provider_id=_coerce_str(provider_id),
        provider_name=_coerce_str(provider_name),
        upstream_model=_coerce_str(upstream_model),
        route_source=_coerce_str(route_source),
        stream=_coerce_bool(stream),
        endpoint=_coerce_str(endpoint),
        input_tokens=token_counts["input_tokens"],
        output_tokens=token_counts["output_tokens"],
        cache_read_input_tokens=token_counts["cache_read_input_tokens"],
        cache_write_input_tokens=token_counts["cache_write_input_tokens"],
        cache_hit_rate=calculate_cache_hit_rate(
            token_counts["input_tokens"], token_counts["cache_read_input_tokens"]
        ),
        price=price,
        cost=compute_usage_cost(token_counts, price),
    )


def usage_record_to_dict(record: UsageRecord) -> dict[str, Any]:
    data = asdict(record)
    if data.get("price") is None:
        data.pop("price", None)
    if data.get("cost") is None:
        data.pop("cost", None)
    if data.get("cache_hit_rate") is None:
        data["cache_hit_rate"] = None
    return data


def normalize_usage_record(value: Any) -> UsageRecord | None:
    if not isinstance(value, dict):
        return None
    input_tokens = _coerce_int(value.get("input_tokens"))
    cache_read_input_tokens = _coerce_int(value.get("cache_read_input_tokens"))
    price = normalize_usage_price(value.get("price"))
    token_counts = {
        "input_tokens": input_tokens,
        "output_tokens": _coerce_int(value.get("output_tokens")),
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_write_input_tokens": _coerce_int(value.get("cache_write_input_tokens")),
    }
    cost_value = value.get("cost")
    cost: UsageCost | None
    if isinstance(cost_value, dict):
        total = _coerce_float(cost_value.get("total"))
        if total is None:
            total = 0.0
        cost = UsageCost(
            input=_coerce_float(cost_value.get("input")) or 0.0,
            output=_coerce_float(cost_value.get("output")) or 0.0,
            cache_read=_coerce_float(cost_value.get("cache_read")) or 0.0,
            total=total,
        )
    else:
        cost = compute_usage_cost(token_counts, price)
    cache_hit_rate = _coerce_float(value.get("cache_hit_rate"))
    if cache_hit_rate is None:
        cache_hit_rate = calculate_cache_hit_rate(input_tokens, cache_read_input_tokens)
    return UsageRecord(
        ts=_coerce_str(value.get("ts")),
        protocol=_coerce_str(value.get("protocol")),
        requested_model=_coerce_str(value.get("requested_model")),
        provider_id=_coerce_str(value.get("provider_id")),
        provider_name=_coerce_str(value.get("provider_name")),
        upstream_model=_coerce_str(value.get("upstream_model")),
        route_source=_coerce_str(value.get("route_source")),
        stream=_coerce_bool(value.get("stream")),
        endpoint=_coerce_str(value.get("endpoint")),
        input_tokens=input_tokens,
        output_tokens=token_counts["output_tokens"],
        cache_read_input_tokens=cache_read_input_tokens,
        cache_write_input_tokens=token_counts["cache_write_input_tokens"],
        cache_hit_rate=cache_hit_rate,
        price=price,
        cost=cost,
    )


def load_usage_stats(path: str | Path) -> UsageStatsDocument:
    stats_path = Path(path)
    if not stats_path.exists():
        return UsageStatsDocument(records=[])
    try:
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return UsageStatsDocument(records=[])
    if not isinstance(raw, dict):
        return UsageStatsDocument(records=[])
    version = raw.get("version")
    try:
        version_number = int(version)
    except (TypeError, ValueError):
        version_number = STATS_FILE_VERSION
    raw_records = raw.get("records")
    records: list[UsageRecord] = []
    if isinstance(raw_records, list):
        for item in raw_records:
            record = normalize_usage_record(item)
            if record is not None:
                records.append(record)
    return UsageStatsDocument(version=version_number, records=records)


def save_usage_stats(path: str | Path, document: UsageStatsDocument) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(document.version),
        "records": [usage_record_to_dict(record) for record in document.records or []],
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=stats_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    tmp_path.replace(stats_path)


def append_usage_record(path: str | Path, record: UsageRecord) -> UsageStatsDocument:
    with _STATS_LOCK:
        document = load_usage_stats(path)
        document.records.append(record)
        save_usage_stats(path, document)
        return document


def summarize_usage_records(records: list[UsageRecord]) -> UsageSummary:
    total_input = sum(record.input_tokens for record in records)
    total_output = sum(record.output_tokens for record in records)
    total_cache_read = sum(record.cache_read_input_tokens for record in records)
    total_cache_write = sum(record.cache_write_input_tokens for record in records)
    return UsageSummary(
        records=len(records),
        input_tokens=total_input,
        output_tokens=total_output,
        cache_read_input_tokens=total_cache_read,
        cache_write_input_tokens=total_cache_write,
        cache_hit_rate=calculate_cache_hit_rate(total_input, total_cache_read),
    )
