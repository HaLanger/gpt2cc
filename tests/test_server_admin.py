import logging
import http.client
import json
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from gpt2cc.config import Config
from gpt2cc.server import ReusableThreadingHTTPServer, admin_url, make_handler, should_open_admin


class AdminServerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_path = str(Path(self.tmpdir.name) / "config.json")
        config = Config(
            host="127.0.0.1",
            port=0,
            proxy_api_key="local-key",
            upstream_api_key="sk-initial",
            model="gpt-4.1",
            config_path=self.config_path,
        )
        self.server = ReusableThreadingHTTPServer(("127.0.0.1", 0), make_handler(config))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmpdir.cleanup()

    def request(self, method, path, body=None, key="local-key"):
        headers = {}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["content-type"] = "application/json"
        if key is not None:
            headers["x-api-key"] = key
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            conn.request(method, path, body=payload, headers=headers)
            response = conn.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), data
        finally:
            conn.close()

    def extract_admin_script(self):
        status, _, data = self.request("GET", "/admin")
        self.assertEqual(status, 200)
        html = data.decode("utf-8")
        match = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        self.assertIsNotNone(match)
        return match.group(1)

    def test_admin_state_requires_auth(self):
        status, _, _ = self.request("GET", "/admin/state", key=None)
        self.assertEqual(status, 401)

    def test_admin_html_does_not_include_upstream_api_key(self):
        status, _, data = self.request("GET", "/admin")
        self.assertEqual(status, 200)
        text = data.decode("utf-8")
        self.assertIn("gpt2cc relay console", text)
        self.assertIn("Gemini native", text)
        self.assertIn("模型路由", text)
        self.assertIn("当前路由概览", text)
        self.assertIn("最近发现", text)
        self.assertIn("绑定主模型", text)
        self.assertIn("primaryBindDrawer", text)
        self.assertNotIn("prompt(", text)
        self.assertIn("routeSelection", text)
        self.assertIn("function openForm()", text)
        self.assertIn('onclick="openForm()"', text)
        self.assertIn("价格（可选", text)
        self.assertIn('id="pricingModel"', text)
        self.assertIn("输入价格", text)
        self.assertIn("输出价格", text)
        self.assertIn("缓存读取价格", text)
        self.assertIn("applyPricingFields()", text)
        self.assertNotIn("parsePricing()", text)
        self.assertIn("provider_pricing", text)
        self.assertIn("setInterval", text)
        self.assertIn("function el(id)", text)
        self.assertIn("const routeProviderEl=el('routeProvider')", text)
        self.assertIn("const routeModelEl=el('routeModel')", text)
        self.assertIn("el('searchBox').value", text)
        self.assertIn("el('activeTitle').textContent", text)
        self.assertIn("el('configPath').textContent", text)
        self.assertIn("el('providers').innerHTML", text)
        self.assertIn("el('drawer').classList.add('show')", text)
        self.assertIn("el('drawerBackdrop').classList.add('show')", text)
        self.assertIn("el('providerId').value", text)
        self.assertIn("el('baseUrl').value", text)
        self.assertIn("el('models').value", text)
        self.assertIn("const pricingModelEl=el('pricingModel')", text)
        self.assertIn("oninput=\"refreshPricingModels(el('pricingModel').value)\"", text)
        self.assertIn("el('pricingInput').value", text)
        self.assertNotIn("routeProvider?.value", text)
        self.assertNotIn("searchBox.value", text)
        self.assertNotIn("activeTitle.textContent", text)
        self.assertNotIn("providers.innerHTML", text)
        self.assertNotIn("providerId.value", text)
        self.assertNotIn("models.value", text)
        self.assertNotIn("pricingModel.value", text)
        self.assertNotIn("sk-initial", text)

    def test_admin_normalize_models_script_is_valid_and_node_checkable(self):
        script = self.extract_admin_script()
        self.assertIn("function normalizeModels()", script)
        self.assertNotIn("split(/\n+/)", script)
        self.assertNotIn("split(/\\n+/)", script)
        self.assertIn("split('\\n')", script)
        self.assertIn(".map(x=>x.trim())", script)
        self.assertIn(".filter(Boolean)", script)

        node_path = shutil.which("node")
        if not node_path:
            return

        script_path = Path(self.tmpdir.name) / "admin-inline-script.js"
        script_path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [node_path, "--check", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"node --check failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_add_provider_and_switch_active_model(self):
        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "relay2",
                "name": "Relay 2",
                "protocol": "gemini",
                "upstream_base_url": "https://relay2.example/v1",
                "upstream_api_key": "sk-relay2",
                "models": ["gpt-4.1", "gpt-image-2"],
            },
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        relay = next(provider for provider in state["providers"] if provider["id"] == "relay2")
        self.assertEqual(relay["upstream_api_key"], "***")
        self.assertEqual(relay["protocol"], "gemini")
        self.assertNotIn("sk-relay2", data.decode("utf-8"))

        status, _, data = self.request("POST", "/admin/active", {"provider_id": "relay2", "model": "gpt-image-2"})
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(state["active_provider"], "relay2")
        self.assertEqual(state["active_model"], "gpt-image-2")

        status, _, data = self.request("GET", "/debug/config")
        self.assertEqual(status, 200)
        config = json.loads(data.decode("utf-8"))
        self.assertEqual(config["upstream_base_url"], "https://relay2.example/v1")
        self.assertEqual(config["upstream_protocol"], "gemini")
        self.assertEqual(config["model"], "gpt-image-2")
        self.assertEqual(config["upstream_api_key"], "***")

    def test_provider_pricing_api_saves_per_provider_model_prices(self):
        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "relay2",
                "name": "Relay 2",
                "protocol": "openai",
                "upstream_base_url": "https://relay2.example/v1",
                "upstream_api_key": "sk-relay2",
                "models": ["gpt-4.1"],
                "pricing": {"gpt-4.1": {"input_per_million": 2.5, "output_per_million": 10.0, "cache_read_per_million": 0.3}},
            },
        )
        self.assertEqual(status, 200)
        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "relay3",
                "name": "Relay 3",
                "protocol": "openai",
                "upstream_base_url": "https://relay3.example/v1",
                "upstream_api_key": "sk-relay3",
                "models": ["gpt-4.1"],
                "pricing": {"gpt-4.1": {"input_per_million": 4.0, "output_per_million": 12.0, "cache_read_per_million": 0.6}},
            },
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(state["provider_pricing"]["relay2"]["gpt-4.1"]["input_per_million"], 2.5)
        self.assertEqual(state["provider_pricing"]["relay3"]["gpt-4.1"]["input_per_million"], 4.0)
        self.assertNotEqual(state["provider_pricing"]["relay2"]["gpt-4.1"], state["provider_pricing"]["relay3"]["gpt-4.1"])
        self.assertNotIn("sk-relay2", data.decode("utf-8"))

        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "relay2",
                "name": "Relay 2",
                "protocol": "openai",
                "upstream_base_url": "https://relay2.example/v1",
                "models": ["gpt-4.1"],
                "pricing": {},
            },
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertNotIn("relay2", state["provider_pricing"])

    def test_invalid_provider_payload_returns_400(self):
        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {"id": "bad id", "upstream_base_url": "relay.example", "models": ["gpt-4.1"]},
        )
        self.assertEqual(status, 400)
        self.assertIn("provider id", data.decode("utf-8"))

    def test_model_route_api_saves_and_deletes_route(self):
        status, _, _ = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "strong",
                "name": "Strong",
                "protocol": "openai",
                "upstream_base_url": "https://strong.example/v1",
                "upstream_api_key": "sk-strong",
                "models": ["deepseek-v4"],
            },
        )
        self.assertEqual(status, 200)
        status, _, data = self.request(
            "POST",
            "/admin/model-routes",
            {"requested_model": "claude-opus-4-7", "provider_id": "strong", "model": "deepseek-v4"},
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(state["model_routes"]["claude-opus-4-7"], {"provider": "strong", "model": "deepseek-v4"})

        status, _, data = self.request(
            "POST",
            "/admin/model-routes/delete",
            {"requested_model": "claude-opus-4-7"},
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertNotIn("claude-opus-4-7", state["model_routes"])

    def test_primary_route_binding_api_updates_state(self):
        status, _, _ = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "main",
                "name": "Main",
                "protocol": "openai",
                "upstream_base_url": "https://main.example/v1",
                "upstream_api_key": "sk-main",
                "models": ["gpt-4.1", "gpt-4.1-mini"],
            },
        )
        self.assertEqual(status, 200)
        status, _, _ = self.request("POST", "/admin/active", {"provider_id": "main", "model": "gpt-4.1"})
        self.assertEqual(status, 200)
        status, _, data = self.request(
            "POST",
            "/admin/primary-route-model",
            {"requested_model": "claude-sonnet-4-6"},
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(state["primary_route_model"], "claude-sonnet-4-6")
        self.assertEqual(state["model_routes"]["claude-sonnet-4-6"], {"provider": "main", "model": "gpt-4.1"})
        status, _, data = self.request("POST", "/admin/active", {"provider_id": "main", "model": "gpt-4.1-mini"})
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(state["model_routes"]["claude-sonnet-4-6"], {"provider": "main", "model": "gpt-4.1-mini"})
        status, _, data = self.request("POST", "/admin/primary-route-model", {"requested_model": ""})
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(state["primary_route_model"], "")

    def test_admin_state_polling_is_not_logged(self):
        logger = logging.getLogger("gpt2cc.server")
        with self.assertLogs(logger, level="INFO") as captured:
            status, _, _ = self.request("GET", "/admin")
        self.assertEqual(status, 200)
        self.assertTrue(any("GET /admin HTTP" in message for message in captured.output))

        with self.assertNoLogs(logger, level="INFO"):
            status, _, _ = self.request("GET", "/admin/state")
        self.assertEqual(status, 200)

        with self.assertNoLogs(logger, level="INFO"):
            status, _, _ = self.request(
                "GET",
                "/admin/usage/summary?start=2026-05-09T10:30:00Z&end=2026-05-09T11:30:00Z",
            )
        self.assertEqual(status, 200)

        with self.assertNoLogs(logger, level="INFO"):
            status, _, _ = self.request(
                "GET",
                "/admin/usage/history?limit=20&start=2026-05-09T10:30:00Z&end=2026-05-09T11:30:00Z",
            )
        self.assertEqual(status, 200)

    def test_admin_url_uses_localhost_for_loopback(self):
        self.assertEqual(admin_url(Config(host="127.0.0.1", port=3456)), "http://localhost:3456/admin")
        self.assertEqual(admin_url(Config(host="0.0.0.0", port=3456)), "http://localhost:3456/admin")

    def test_should_open_admin_respects_opt_out(self):
        self.assertFalse(should_open_admin(True))


if __name__ == "__main__":
    unittest.main()
