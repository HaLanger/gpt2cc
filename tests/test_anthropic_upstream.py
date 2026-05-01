from __future__ import annotations

import io
import json
import unittest

from gpt2cc.anthropic_upstream import build_anthropic_payload, stream_anthropic_passthrough
from gpt2cc.config import Config


class AnthropicUpstreamTests(unittest.TestCase):
    def test_build_anthropic_payload_replaces_model(self):
        config = Config(model="claude-3-5-sonnet-latest")
        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
        payload = build_anthropic_payload(request, config)
        self.assertEqual(payload["model"], "claude-3-5-sonnet-latest")
        self.assertEqual(payload["messages"], request["messages"])

    def test_stream_anthropic_passthrough_writes_bytes(self):
        output = []
        stream_anthropic_passthrough(io.BytesIO(b"event: message_stop\n\n"), output.append)
        self.assertEqual(output, [b"event: message_stop\n\n"])


if __name__ == "__main__":
    unittest.main()
