import json
import tempfile
import threading
import unittest
from pathlib import Path

from gpt2cc.usage_stats import (
    STATS_FILE_VERSION,
    UsagePrice,
    append_usage_record,
    build_usage_record,
    calculate_cache_hit_rate,
    load_usage_stats,
    summarize_usage_records,
    usage_record_to_dict,
)


class UsageStatsTests(unittest.TestCase):
    def test_load_missing_stats_file_returns_empty_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.stats.json"
            document = load_usage_stats(path)
        self.assertEqual(document.version, STATS_FILE_VERSION)
        self.assertEqual(document.records, [])

    def test_append_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "usage.stats.json"
            price = UsagePrice(
                provider_id="xlab",
                model="gpt-5.4",
                input_per_million=2.5,
                output_per_million=10.0,
                cache_read_per_million=0.3,
            )
            record = build_usage_record(
                ts="2026-05-09T02:34:56Z",
                protocol="openai",
                requested_model="claude-sonnet-4-6",
                provider_id="xlab",
                provider_name="xlab",
                upstream_model="gpt-5.4",
                route_source="model_routes",
                stream=True,
                endpoint="chat/completions",
                input_tokens=123,
                output_tokens=45,
                cache_read_input_tokens=67,
                cache_write_input_tokens=0,
                price=price,
            )
            append_usage_record(path, record)
            reloaded = load_usage_stats(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(reloaded.version, STATS_FILE_VERSION)
        self.assertEqual(len(reloaded.records), 1)
        self.assertEqual(usage_record_to_dict(reloaded.records[0]), usage_record_to_dict(record))
        self.assertEqual(payload["records"][0]["price"]["provider_id"], "xlab")
        self.assertAlmostEqual(payload["records"][0]["cache_hit_rate"], 67 / 123)
        self.assertAlmostEqual(payload["records"][0]["cost"]["total"], 0.0006101, places=7)

    def test_cache_hit_rate_helper_behavior(self):
        self.assertIsNone(calculate_cache_hit_rate(0, 0))
        self.assertEqual(calculate_cache_hit_rate(0, 50), 1.0)
        self.assertAlmostEqual(calculate_cache_hit_rate(123, 67), 67 / 123)

    def test_summary_aggregates_cache_hit_rate_once(self):
        records = [
            build_usage_record(
                ts="2026-05-09T00:00:00Z",
                protocol="openai",
                requested_model="a",
                provider_id="p1",
                provider_name="Provider 1",
                upstream_model="model-a",
                route_source="active",
                stream=False,
                endpoint="chat/completions",
                input_tokens=100,
                output_tokens=10,
                cache_read_input_tokens=50,
                cache_write_input_tokens=5,
            ),
            build_usage_record(
                ts="2026-05-09T00:00:01Z",
                protocol="openai",
                requested_model="b",
                provider_id="p2",
                provider_name="Provider 2",
                upstream_model="model-b",
                route_source="active",
                stream=True,
                endpoint="chat/completions",
                input_tokens=300,
                output_tokens=20,
                cache_read_input_tokens=150,
                cache_write_input_tokens=15,
            ),
        ]
        summary = summarize_usage_records(records)
        self.assertEqual(summary.records, 2)
        self.assertEqual(summary.input_tokens, 400)
        self.assertEqual(summary.output_tokens, 30)
        self.assertEqual(summary.cache_read_input_tokens, 200)
        self.assertEqual(summary.cache_write_input_tokens, 20)
        self.assertAlmostEqual(summary.cache_hit_rate, 200 / 400)

    def test_load_tolerates_malformed_optional_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "usage.stats.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "records": [
                            {
                                "ts": "2026-05-09T02:34:56Z",
                                "protocol": "openai",
                                "requested_model": "claude-sonnet-4-6",
                                "provider_id": "xlab",
                                "provider_name": "xlab",
                                "upstream_model": "gpt-5.4",
                                "route_source": "model_routes",
                                "stream": "true",
                                "endpoint": "chat/completions",
                                "input_tokens": "123",
                                "output_tokens": "45",
                                "cache_read_input_tokens": "67",
                                "cache_write_input_tokens": "",
                                "price": {"provider_id": "xlab", "model": "gpt-5.4", "input_per_million": "bad"},
                                "cost": {"input": "0.1", "output": "", "cache_read": None, "total": "0.1"}
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            document = load_usage_stats(path)
        self.assertEqual(len(document.records), 1)
        record = document.records[0]
        self.assertEqual(record.input_tokens, 123)
        self.assertEqual(record.cache_write_input_tokens, 0)
        self.assertIsNone(record.price)
        self.assertIsNotNone(record.cost)
        self.assertAlmostEqual(record.cache_hit_rate, 67 / 123)
    def test_concurrent_appends_keep_all_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "usage.stats.json"

            def append(index: int) -> None:
                append_usage_record(
                    path,
                    build_usage_record(
                        ts=f"2026-05-09T00:00:{index:02d}Z",
                        protocol="openai",
                        requested_model=f"claude-{index}",
                        provider_id="p1",
                        provider_name="Provider 1",
                        upstream_model="model-a",
                        route_source="active",
                        stream=False,
                        endpoint="chat/completions",
                        input_tokens=index,
                    ),
                )

            threads = [threading.Thread(target=append, args=(index,)) for index in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            document = load_usage_stats(path)
        self.assertEqual(len(document.records), 20)
        self.assertEqual(sum(record.input_tokens for record in document.records), sum(range(20)))

    def test_cost_uses_uncached_input_only_and_clamps_when_cache_exceeds_input(self):
        price = UsagePrice(
            provider_id="relay",
            model="deepseek",
            input_per_million=2.0,
            output_per_million=8.0,
            cache_read_per_million=0.5,
        )
        normal = build_usage_record(
            ts="2026-05-09T00:00:00Z",
            protocol="anthropic",
            requested_model="claude-sonnet-4-6",
            provider_id="relay",
            provider_name="Relay",
            upstream_model="deepseek",
            route_source="active",
            stream=False,
            endpoint="anthropic/messages",
            input_tokens=100,
            output_tokens=25,
            cache_read_input_tokens=40,
            price=price,
        )
        self.assertIsNotNone(normal.cost)
        self.assertAlmostEqual(normal.cost.input, 0.00012)
        self.assertAlmostEqual(normal.cost.output, 0.0002)
        self.assertAlmostEqual(normal.cost.cache_read, 0.00002)
        self.assertAlmostEqual(normal.cost.total, 0.00034)

        cache_heavy = build_usage_record(
            ts="2026-05-09T00:00:01Z",
            protocol="anthropic",
            requested_model="claude-sonnet-4-6",
            provider_id="relay",
            provider_name="Relay",
            upstream_model="deepseek",
            route_source="active",
            stream=False,
            endpoint="anthropic/messages",
            input_tokens=50,
            output_tokens=10,
            cache_read_input_tokens=80,
            price=price,
        )
        self.assertIsNotNone(cache_heavy.cost)
        self.assertAlmostEqual(cache_heavy.cost.input, 0.0)
        self.assertAlmostEqual(cache_heavy.cost.cache_read, 0.00004)
        self.assertAlmostEqual(cache_heavy.cost.total, 0.00012)


if __name__ == "__main__":
    unittest.main()
