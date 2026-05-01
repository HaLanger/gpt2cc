import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from gpt2cc.config import Config
from gpt2cc.server import ReusableThreadingHTTPServer, make_handler


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

    def test_admin_state_requires_auth(self):
        status, _, _ = self.request("GET", "/admin/state", key=None)
        self.assertEqual(status, 401)

    def test_admin_html_does_not_include_upstream_api_key(self):
        status, _, data = self.request("GET", "/admin")
        self.assertEqual(status, 200)
        text = data.decode("utf-8")
        self.assertIn("gpt2cc relay console", text)
        self.assertNotIn("sk-initial", text)

    def test_add_provider_and_switch_active_model(self):
        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {
                "id": "relay2",
                "name": "Relay 2",
                "upstream_base_url": "https://relay2.example/v1",
                "upstream_api_key": "sk-relay2",
                "models": ["gpt-4.1", "gpt-image-2"],
            },
        )
        self.assertEqual(status, 200)
        state = json.loads(data.decode("utf-8"))
        relay = next(provider for provider in state["providers"] if provider["id"] == "relay2")
        self.assertEqual(relay["upstream_api_key"], "***")
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
        self.assertEqual(config["model"], "gpt-image-2")
        self.assertEqual(config["upstream_api_key"], "***")

    def test_invalid_provider_payload_returns_400(self):
        status, _, data = self.request(
            "POST",
            "/admin/providers",
            {"id": "bad id", "upstream_base_url": "relay.example", "models": ["gpt-4.1"]},
        )
        self.assertEqual(status, 400)
        self.assertIn("provider id", data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
