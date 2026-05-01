from __future__ import annotations

import io
import json
import unittest

from gpt2cc.config import Config
from gpt2cc.gemini import anthropic_message_from_gemini, stream_gemini_to_anthropic, transform_anthropic_to_gemini


class GeminiTests(unittest.TestCase):
    def test_transforms_text_request_to_gemini_payload(self):
        request = {
            "model": "claude-sonnet-4-6",
            "system": "Be concise",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        payload, ctx = transform_anthropic_to_gemini(request, Config(model="gemini-2.5-pro"))
        self.assertEqual(ctx.upstream_model, "gemini-2.5-pro")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "hello")
        self.assertEqual(payload["system_instruction"]["parts"][0]["text"], "Be concise")
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 20)

    def test_transforms_tool_definition_and_call(self):
        request = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"name": "Read", "description": "read file", "input_schema": {"type": "object"}}],
        }
        payload, ctx = transform_anthropic_to_gemini(request, Config(model="gemini-2.5-pro"))
        declaration = payload["tools"][0]["function_declarations"][0]
        self.assertEqual(declaration["name"], ctx.to_upstream_tool_name("Read"))
        self.assertEqual(declaration["parameters"], {"type": "object"})

    def test_gemini_response_to_anthropic_message(self):
        _, ctx = transform_anthropic_to_gemini({"model": "claude", "messages": []}, Config(model="gemini"))
        response = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3},
        }
        message = anthropic_message_from_gemini(response, ctx)
        self.assertEqual(message["content"], [{"type": "text", "text": "hi"}])
        self.assertEqual(message["stop_reason"], "end_turn")
        self.assertEqual(message["usage"], {"input_tokens": 2, "output_tokens": 3})

    def test_stream_gemini_to_anthropic(self):
        _, ctx = transform_anthropic_to_gemini({"model": "claude", "messages": []}, Config(model="gemini"))
        chunk = {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}
        output = []
        raw = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        stream_gemini_to_anthropic(io.BytesIO(raw), ctx, output.append)
        body = b"".join(output).decode("utf-8")
        self.assertIn("message_start", body)
        self.assertIn("hi", body)
        self.assertIn("message_stop", body)


if __name__ == "__main__":
    unittest.main()
