import base64
import tempfile
import unittest
from pathlib import Path

from gpt2cc.config import Config
from gpt2cc.image import (
    build_image_edit_request,
    build_image_generation_payload,
    extract_reference_images,
    image_result_from_response,
    is_image_model,
    request_has_reference_images,
)
from gpt2cc.transform import TransformContext


class ImageTests(unittest.TestCase):
    def test_detects_image_model_patterns(self):
        config = Config()
        self.assertTrue(is_image_model("gpt-image-2", config))
        self.assertTrue(is_image_model("dall-e-3", config))
        self.assertFalse(is_image_model("gpt-5.4", config))

    def test_build_image_generation_payload_uses_last_user_text(self):
        config = Config(
            model="gpt-image-2",
            image_size="1024x1024",
            image_quality="high",
            image_output_format="png",
            image_n=1,
        )
        ctx = TransformContext(requested_model="claude-test", upstream_model="gpt-image-2")
        payload = build_image_generation_payload(
            {
                "system": "You are Claude Code.",
                "messages": [
                    {"role": "user", "content": "first prompt"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": [{"type": "text", "text": "make a cyberpunk poster"}]},
                ],
            },
            config,
            ctx,
        )
        self.assertEqual(payload["model"], "gpt-image-2")
        self.assertEqual(payload["prompt"], "make a cyberpunk poster")
        self.assertEqual(payload["size"], "1024x1024")
        self.assertEqual(payload["quality"], "high")

    def test_extracts_reference_images_from_last_user_message(self):
        image_bytes = b"fake-reference-image"
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "use this style"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                    ],
                }
            ]
        }
        references = extract_reference_images(request, 16)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].media_type, "image/png")
        self.assertEqual(references[0].data, image_bytes)
        self.assertTrue(request_has_reference_images(request, Config()))

    def test_build_image_edit_request_uses_multipart_files(self):
        image_bytes = b"fake-reference-image"
        config = Config(image_input_fidelity="high")
        ctx = TransformContext(requested_model="claude-test", upstream_model="gpt-image-2")
        edit_request = build_image_edit_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "make a new image based on this reference"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64.b64encode(image_bytes).decode("ascii"),
                                },
                            },
                        ],
                    }
                ]
            },
            config,
            ctx,
        )
        self.assertEqual(edit_request.fields["model"], "gpt-image-2")
        self.assertEqual(edit_request.fields["prompt"], "make a new image based on this reference")
        self.assertEqual(edit_request.fields["input_fidelity"], "high")
        self.assertEqual(len(edit_request.files), 1)
        self.assertEqual(edit_request.files[0].field_name, "image[]")
        self.assertEqual(edit_request.files[0].filename, "reference-1.jpg")
        self.assertEqual(edit_request.files[0].content_type, "image/jpeg")
        self.assertEqual(edit_request.files[0].data, image_bytes)

    def test_saves_base64_image_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(image_output_dir=tmpdir, image_output_format="png")
            ctx = TransformContext(requested_model="claude-test", upstream_model="gpt-image-2")
            result = image_result_from_response(
                {
                    "data": [
                        {
                            "b64_json": base64.b64encode(b"fake-png-data").decode("ascii"),
                            "revised_prompt": "revised",
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 7},
                },
                config,
                ctx,
                "a test image",
                reference_count=1,
            )
            self.assertEqual(len(result.images), 1)
            self.assertIsNotNone(result.images[0].path)
            path = Path(result.images[0].path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"fake-png-data")
            self.assertIn("Image request complete", result.text)
            self.assertIn("Endpoint: images/edits", result.text)
            self.assertIn("a test image", result.text)


if __name__ == "__main__":
    unittest.main()
