import http.client
import unittest
from unittest.mock import patch

from gpt2cc.config import Config
from gpt2cc.upstream import MultipartFile, UpstreamError, post_json_url, post_multipart_url


class FailingResponse:
    status = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        raise http.client.IncompleteRead(b'{"data":[', 1024)


class SuccessfulResponse:
    status = 200
    headers = {"content-type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b'{"ok":true}'


class UpstreamTests(unittest.TestCase):
    def test_json_retries_incomplete_read(self):
        config = Config(max_retries=1, timeout_seconds=1)
        with patch("gpt2cc.upstream.time.sleep"), patch(
            "gpt2cc.upstream.urllib.request.urlopen", side_effect=[FailingResponse(), SuccessfulResponse()]
        ) as urlopen:
            response = post_json_url(config, "https://relay.example/v1/chat/completions", {"model": "gpt-test"})
        self.assertEqual(response.body, b'{"ok":true}')
        self.assertEqual(urlopen.call_count, 2)

    def test_json_raises_clear_error_after_incomplete_read_retries(self):
        config = Config(max_retries=1, timeout_seconds=1)
        with patch("gpt2cc.upstream.time.sleep"), patch(
            "gpt2cc.upstream.urllib.request.urlopen", side_effect=[FailingResponse(), FailingResponse()]
        ):
            with self.assertRaises(UpstreamError) as caught:
                post_json_url(config, "https://relay.example/v1/chat/completions", {"model": "gpt-test"})
        self.assertEqual(caught.exception.status, 502)
        self.assertIn("ended before the full body", str(caught.exception))
        self.assertEqual(caught.exception.body, b'{"data":[')

    def test_multipart_retries_incomplete_read(self):
        config = Config(max_retries=1, timeout_seconds=1)
        files = [MultipartFile("image[]", "reference.png", "image/png", b"image-bytes")]
        with patch("gpt2cc.upstream.time.sleep"), patch(
            "gpt2cc.upstream.urllib.request.urlopen", side_effect=[FailingResponse(), SuccessfulResponse()]
        ) as urlopen:
            response = post_multipart_url(
                config,
                "https://relay.example/v1/images/edits",
                {"model": "gpt-image-2", "prompt": "test"},
                files,
            )
        self.assertEqual(response.body, b'{"ok":true}')
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
