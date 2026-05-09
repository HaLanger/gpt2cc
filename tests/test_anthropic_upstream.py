from __future__ import annotations

import io
import unittest

from gpt2cc.anthropic_upstream import build_anthropic_payload, extract_anthropic_usage_from_message, stream_anthropic_passthrough
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

    def test_stream_anthropic_passthrough_writes_bytes_and_extracts_usage(self):
        raw = (
            b'event: message_start\n'
            b'data: {"type":"message_start","message":{"id":"msg_1","type":"message","usage":{"input_tokens":11,"cache_read_input_tokens":5}}}\n\n'
            b'event: message_delta\n'
            b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
            b'event: message_stop\n'
            b'data: {"type":"message_stop"}\n\n'
        )
        output = []
        result = stream_anthropic_passthrough(io.BytesIO(raw), output.append)
        self.assertEqual(b"".join(output), raw)
        self.assertEqual(result.usage["input_tokens"], 11)
        self.assertEqual(result.usage["cache_read_input_tokens"], 5)
        self.assertEqual(result.usage["output_tokens"], 7)

    def test_extract_anthropic_usage_ignores_non_message_shapes(self):
        self.assertEqual(
            extract_anthropic_usage_from_message({"type": "other", "usage": {"input_tokens": 99}}),
            {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_write_input_tokens": 0},
        )
    def test_stream_anthropic_passthrough_writes_incrementally(self):
        class IncrementalStream:
            def __init__(self):
                self.lines = iter([
                    b"event: message_start\n",
                    b'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n',
                    b"\n",
                ])
                self.output: list[bytes] = []

            def __iter__(self):
                return self

            def __next__(self):
                line = next(self.lines)
                if line == b"\n":
                    self.output.append(b"saw-prior-output:" + b"".join(self.output))
                return line

        stream = IncrementalStream()
        result = stream_anthropic_passthrough(stream, stream.output.append)
        self.assertEqual(result.usage["input_tokens"], 3)
        markers = [item for item in stream.output if item.startswith(b"saw-prior-output:")]
        self.assertTrue(markers)
        self.assertIn(b"event: message_start\n", markers[0])


if __name__ == "__main__":
    unittest.main()
