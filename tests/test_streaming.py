import io
import json
import unittest

from gpt2cc.streaming import stream_openai_to_anthropic
from gpt2cc.transform import TransformContext


def sse(data):
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


class StreamingTests(unittest.TestCase):
    def test_text_stream(self):
        upstream = io.BytesIO(
            b"".join(
                [
                    sse({"id": "chatcmpl_1", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
                    sse({"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]}),
                    sse({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    sse({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}),
                    b"data: [DONE]\n\n",
                ]
            )
        )
        chunks = []
        stream_openai_to_anthropic(
            upstream,
            TransformContext(requested_model="claude-test", upstream_model="gpt-test"),
            chunks.append,
        )
        text = b"".join(chunks).decode("utf-8")
        self.assertIn("event: message_start", text)
        self.assertIn("event: content_block_start", text)
        self.assertIn('"type":"text_delta","text":"hel"', text)
        self.assertIn('"type":"text_delta","text":"lo"', text)
        self.assertIn('"stop_reason":"end_turn"', text)
        self.assertIn("event: message_stop", text)
        self.assertNotIn("[DONE]", text)

    def test_tool_stream(self):
        upstream = io.BytesIO(
            b"".join(
                [
                    sse(
                        {
                            "id": "chatcmpl_2",
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {"name": "Bash", "arguments": ""},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ),
                    sse(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {"index": 0, "function": {"arguments": "{\"command\":\"pwd\"}"}}
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    ),
                    sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
                    b"data: [DONE]\n\n",
                ]
            )
        )
        ctx = TransformContext(requested_model="claude-test", upstream_model="gpt-test")
        ctx.tool_name_from_upstream["Bash"] = "Bash"
        chunks = []
        stream_openai_to_anthropic(upstream, ctx, chunks.append)
        text = b"".join(chunks).decode("utf-8")
        self.assertIn('"type":"tool_use"', text)
        self.assertIn('"name":"Bash"', text)
        self.assertIn('"type":"input_json_delta","partial_json":"{\\"command\\":\\"pwd\\"}"', text)
        self.assertIn('"stop_reason":"tool_use"', text)


if __name__ == "__main__":
    unittest.main()
