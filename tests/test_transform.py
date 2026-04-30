import json
import unittest

from ccproxy.config import Config
from ccproxy.transform import anthropic_message_from_openai, transform_anthropic_to_openai


class TransformTests(unittest.TestCase):
    def test_text_request(self):
        payload, ctx = transform_anthropic_to_openai(
            {
                "model": "claude-test",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
            },
            Config(model="gpt-test"),
        )
        self.assertEqual(ctx.upstream_model, "gpt-test")
        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(payload["max_tokens"], 100)

    def test_tool_roundtrip(self):
        payload, ctx = transform_anthropic_to_openai(
            {
                "model": "claude-test",
                "messages": [{"role": "user", "content": "list files"}],
                "tools": [
                    {
                        "name": "Bash",
                        "description": "Run a shell command",
                        "input_schema": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    }
                ],
            },
            Config(model="gpt-test"),
        )
        self.assertEqual(payload["tools"][0]["function"]["name"], "Bash")

        anthropic = anthropic_message_from_openai(
            {
                "id": "chatcmpl_1",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": json.dumps({"command": "ls"})},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            ctx,
        )
        self.assertEqual(anthropic["stop_reason"], "tool_use")
        self.assertEqual(anthropic["content"][0]["name"], "Bash")
        self.assertEqual(anthropic["content"][0]["input"], {"command": "ls"})
        self.assertEqual(anthropic["usage"], {"input_tokens": 10, "output_tokens": 5})

    def test_tool_result_maps_to_openai_tool_message(self):
        payload, _ = transform_anthropic_to_openai(
            {
                "model": "claude-test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "pwd"}}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "/tmp"}],
                    },
                ],
            },
            Config(model="gpt-test"),
        )
        self.assertEqual(payload["messages"][0]["role"], "assistant")
        self.assertEqual(payload["messages"][0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(payload["messages"][1], {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"})


if __name__ == "__main__":
    unittest.main()
