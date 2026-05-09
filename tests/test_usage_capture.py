import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from gpt2cc.config import Config
from gpt2cc.image import ImageGenerationResult
from gpt2cc.server import ReusableThreadingHTTPServer, make_handler
from gpt2cc.usage_stats import UsageRecord


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class UsageCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        self.config = Config(
            host="127.0.0.1",
            port=0,
            proxy_api_key="local-key",
            model="gpt-test",
            config_path=str(base / "config.json"),
            stats_path=str(base / "usage.stats.json"),
            providers=[{"id": "default", "name": "Default relay", "protocol": "openai", "upstream_base_url": "https://example.invalid", "upstream_api_key": "", "models": ["gpt-test"]}],
            active_provider="default",
            active_model="gpt-test",
            provider_pricing={"default": {"gpt-test": {"input_per_million": 1.0, "output_per_million": 2.0, "cache_read_per_million": 0.5}}},
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

    def request(self, body):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            payload = json.dumps(body).encode("utf-8")
            conn.request("POST", "/v1/messages", body=payload, headers={"content-type": "application/json", "x-api-key": "local-key"})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_openai_non_stream_persists_usage_record_with_cost(self):
        upstream_payload = {
            "id": "chatcmpl_1",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3, "cache_creation_tokens": 2}},
        }
        captured: list[UsageRecord] = []
        with patch("gpt2cc.server.post_json", return_value=_FakeResponse(upstream_payload)), patch("gpt2cc.server.append_usage_record", side_effect=lambda path, record: captured.append(record)):
            status, _ = self.request({"model": "claude-test", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        record = captured[0]
        self.assertEqual(record.protocol, "openai")
        self.assertEqual(record.endpoint, "chat/completions")
        self.assertEqual(record.input_tokens, 10)
        self.assertEqual(record.output_tokens, 4)
        self.assertEqual(record.cache_read_input_tokens, 3)
        self.assertEqual(record.cache_write_input_tokens, 2)
        self.assertIsNotNone(record.cost)

    def test_anthropic_non_stream_persists_normalized_usage(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.config.upstream_protocol = "anthropic"
        self.server = ReusableThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

        upstream_payload = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 5, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1},
        }
        captured: list[UsageRecord] = []
        with patch("gpt2cc.server.post_anthropic_message", return_value=_FakeResponse(upstream_payload)), patch("gpt2cc.server.append_usage_record", side_effect=lambda path, record: captured.append(record)):
            status, _ = self.request({"model": "claude-test", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        record = captured[0]
        self.assertEqual(record.protocol, "anthropic")
        self.assertEqual(record.input_tokens, 8)
        self.assertEqual(record.output_tokens, 5)
        self.assertEqual(record.cache_read_input_tokens, 2)
        self.assertEqual(record.cache_write_input_tokens, 1)

    def test_image_generation_non_stream_persists_normalized_usage(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.config.model = "gpt-image-2"
        self.config.active_model = "gpt-image-2"
        self.config.image_models = ["gpt-image-*"]
        self.config.providers[0]["models"] = ["gpt-image-2"]
        self.config.provider_pricing = {}
        self.server = ReusableThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        result = ImageGenerationResult(
            text="saved image",
            raw_response={
                "id": "img_1",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "cached_tokens": 4,
                    "cache_creation_tokens": 2,
                },
            },
        )
        captured: list[UsageRecord] = []
        with patch("gpt2cc.server.generate_image", return_value=result), patch("gpt2cc.server.append_usage_record", side_effect=lambda path, record: captured.append(record)):
            status, _ = self.request({"model": "claude-test", "stream": False, "messages": [{"role": "user", "content": "draw"}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        record = captured[0]
        self.assertEqual(record.endpoint, "images/generations")
        self.assertEqual(record.input_tokens, 12)
        self.assertEqual(record.output_tokens, 3)
        self.assertEqual(record.cache_read_input_tokens, 4)
        self.assertEqual(record.cache_write_input_tokens, 2)


if __name__ == "__main__":
    unittest.main()
