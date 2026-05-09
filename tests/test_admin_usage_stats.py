from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from gpt2cc.config import Config
from gpt2cc.server import ReusableThreadingHTTPServer, make_handler
from gpt2cc.usage_stats import UsagePrice, append_usage_record, build_usage_record


class AdminUsageStatsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        self.stats_path = str(base / "usage.stats.json")
        self.config = Config(
            host="127.0.0.1",
            port=0,
            proxy_api_key="local-key",
            model="gpt-4.1",
            config_path=str(base / "config.json"),
            stats_path=self.stats_path,
            providers=[
                {
                    "id": "relay-a",
                    "name": "Relay A",
                    "protocol": "openai",
                    "upstream_base_url": "https://a.example/v1",
                    "upstream_api_key": "sk-a",
                    "models": ["gpt-4.1", "gpt-4.1-mini"],
                },
                {
                    "id": "relay-b",
                    "name": "Relay B",
                    "protocol": "openai",
                    "upstream_base_url": "https://b.example/v1",
                    "upstream_api_key": "sk-b",
                    "models": ["gpt-4.1"],
                },
            ],
            active_provider="relay-a",
            active_model="gpt-4.1",
            provider_pricing={
                "relay-a": {"gpt-4.1": {"input_per_million": 2.5, "output_per_million": 10.0, "cache_read_per_million": 0.3}},
                "relay-b": {"gpt-4.1": {"input_per_million": 4.0, "output_per_million": 12.0, "cache_read_per_million": 0.6}},
            },
        )
        append_usage_record(
            self.stats_path,
            build_usage_record(
                ts="2026-05-09T10:00:00Z",
                protocol="openai",
                requested_model="claude-sonnet-4-6",
                provider_id="relay-a",
                provider_name="Relay A",
                upstream_model="gpt-4.1",
                route_source="active",
                stream=True,
                endpoint="chat/completions",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=25,
                cache_write_input_tokens=10,
                price=UsagePrice(provider_id="relay-a", model="gpt-4.1", input_per_million=2.5, output_per_million=10.0, cache_read_per_million=0.3),
            ),
        )
        append_usage_record(
            self.stats_path,
            build_usage_record(
                ts="2026-05-09T11:00:00Z",
                protocol="openai",
                requested_model="claude-opus-4-1",
                provider_id="relay-b",
                provider_name="Relay B",
                upstream_model="gpt-4.1",
                route_source="active",
                stream=False,
                endpoint="chat/completions",
                input_tokens=40,
                output_tokens=20,
                cache_read_input_tokens=10,
                cache_write_input_tokens=0,
                price=UsagePrice(provider_id="relay-b", model="gpt-4.1", input_per_million=4.0, output_per_million=12.0, cache_read_per_million=0.6),
            ),
        )
        append_usage_record(
            self.stats_path,
            build_usage_record(
                ts="2026-05-09T12:00:00Z",
                protocol="openai",
                requested_model="claude-3-5-sonnet",
                provider_id="relay-a",
                provider_name="Relay A",
                upstream_model="gpt-4.1-mini",
                route_source="active",
                stream=False,
                endpoint="chat/completions",
                input_tokens=20,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_write_input_tokens=0,
            ),
        )
        self.server = ReusableThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmpdir.cleanup()

    def request(self, method, path, key="local-key"):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            headers = {"x-api-key": key} if key is not None else {}
            conn.request(method, path, headers=headers)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_usage_summary_requires_auth(self):
        status, _ = self.request("GET", "/admin/usage/summary", key=None)
        self.assertEqual(status, 401)

    def test_usage_summary_merges_models_but_keeps_provider_cost_breakdown(self):
        status, data = self.request("GET", "/admin/usage/summary")
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload["records"], 3)
        self.assertEqual(payload["totals"]["input_tokens"], 160)
        self.assertEqual(payload["totals"]["output_tokens"], 75)
        self.assertEqual(payload["totals"]["cache_read_input_tokens"], 35)
        self.assertEqual(payload["totals"]["cache_write_input_tokens"], 10)
        self.assertTrue(payload["totals"]["has_pricing"])
        merged = {item["upstream_model"]: item for item in payload["merged_by_model"]}
        self.assertEqual(merged["gpt-4.1"]["input_tokens"], 140)
        self.assertEqual(merged["gpt-4.1"]["provider_count"], 2)
        self.assertAlmostEqual(merged["gpt-4.1"]["cache_hit_rate"], 35 / 175)
        breakdown = {(item["provider_id"], item["upstream_model"]): item for item in payload["provider_model_breakdown"]}
        self.assertIn(("relay-a", "gpt-4.1"), breakdown)
        self.assertIn(("relay-b", "gpt-4.1"), breakdown)
        self.assertGreater(breakdown[("relay-a", "gpt-4.1")]["total_cost"], 0)
        self.assertGreater(breakdown[("relay-b", "gpt-4.1")]["total_cost"], 0)
        self.assertFalse(breakdown[("relay-a", "gpt-4.1-mini")]["has_pricing"])

        status, data = self.request(
            "GET",
            "/admin/usage/summary?start=2026-05-09T10:30:00Z&end=2026-05-09T11:30:00Z",
        )
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload["records"], 1)
        self.assertEqual(payload["totals"]["records"], 1)
        self.assertEqual(payload["totals"]["input_tokens"], 40)
        self.assertEqual(payload["totals"]["output_tokens"], 20)
        self.assertEqual(payload["totals"]["cache_read_input_tokens"], 10)
        self.assertEqual(len(payload["provider_model_breakdown"]), 1)
        self.assertEqual(payload["provider_model_breakdown"][0]["provider_id"], "relay-b")
        self.assertEqual(payload["provider_model_breakdown"][0]["upstream_model"], "gpt-4.1")

    def test_usage_history_returns_recent_records_with_limit(self):
        status, data = self.request("GET", "/admin/usage/history?limit=2")
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload["returned"], 2)
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["records"][0]["ts"], "2026-05-09T12:00:00Z")
        self.assertEqual(payload["records"][1]["ts"], "2026-05-09T11:00:00Z")

        status, data = self.request(
            "GET",
            "/admin/usage/history?limit=20&start=2026-05-09T10:30:00Z&end=2026-05-09T11:30:00Z",
        )
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload["returned"], 1)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["records"][0]["ts"], "2026-05-09T11:00:00Z")

    def test_usage_invalid_date_returns_400(self):
        status, data = self.request("GET", "/admin/usage/summary?start=not-a-date")
        self.assertEqual(status, 400)
        self.assertIn("invalid start datetime", data.decode("utf-8"))

    def test_usage_page_requires_auth(self):
        status, _ = self.request("GET", "/admin/usage", key=None)
        self.assertEqual(status, 401)

    def test_usage_page_contains_dedicated_metering_layout(self):
        status, data = self.request("GET", "/admin/usage")
        self.assertEqual(status, 200)
        text = data.decode("utf-8")
        self.assertIn("用量统计", text)
        self.assertIn("总 Token 数", text)
        self.assertIn("总会话数", text)
        self.assertIn("预估费用", text)
        self.assertIn("缓存命中率", text)
        self.assertIn("datetime-local", text)
        self.assertIn("最近一周", text)
        self.assertIn("最近一天", text)
        self.assertIn("最近 1 小时", text)
        self.assertIn("rangeQueryParams", text)
        self.assertIn("toLocaleString", text)
        self.assertIn("/admin/usage/summary", text)
        self.assertIn("/admin/usage/history", text)
        self.assertIn("limit', String(limit)", text)
        self.assertIn("模型分布", text)
        self.assertIn("最近 20 条请求", text)
        self.assertIn("Provider / 模型", text)
        self.assertIn("缓存命中情况", text)
        self.assertIn("Provider / Model 成本明细", text)
        self.assertIn("返回管理台", text)
        self.assertIn("function el(id)", text)
        self.assertIn("function formatLocalDateTime(value)", text)
        self.assertIn("padStart(2,'0')", text)
        self.assertIn("el('banner')", text)
        self.assertIn("el('proxyKey').value", text)
        self.assertIn("el('startAt').value", text)
        self.assertIn("el('endAt').value", text)
        self.assertIn("el('authBox').classList", text)
        self.assertIn("el('cards').innerHTML", text)
        self.assertIn("el('modelDistribution').innerHTML", text)
        self.assertIn("el('requestRows').innerHTML", text)
        self.assertIn("el('costRows').innerHTML", text)
        self.assertIn("el('statsPath').textContent", text)
        self.assertIn("new Date(el('startAt').value).toISOString()", text)
        self.assertIn("new Date(el('endAt').value).toISOString()", text)
        self.assertIn("el('startAt').value=formatLocalDateTime(start)", text)
        self.assertIn("el('endAt').value=formatLocalDateTime(end)", text)
        self.assertNotIn("toISOString().slice(0,16)", text)

    def test_admin_home_links_to_usage_page_without_embedding_dashboard(self):
        status, data = self.request("GET", "/admin")
        self.assertEqual(status, 200)
        text = data.decode("utf-8")
        self.assertIn("用量统计", text)
        self.assertIn("/admin/usage", text)
        self.assertNotIn("Usage Dashboard", text)
        self.assertNotIn("Provider/model cost breakdown", text)


if __name__ == "__main__":
    unittest.main()
