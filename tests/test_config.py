import os
import unittest
from unittest.mock import patch

from gpt2cc.config import load_config


class ConfigTests(unittest.TestCase):
    def test_tls_env_options(self):
        env = {
            "GPT2CC_UPSTREAM_SSL_VERIFY": "false",
            "GPT2CC_UPSTREAM_CA_BUNDLE": r"C:\certs\relay-ca.pem",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
        self.assertFalse(config.upstream_ssl_verify)
        self.assertEqual(config.upstream_ca_bundle, r"C:\certs\relay-ca.pem")

    def test_legacy_env_options_still_work(self):
        env = {"CCPROXY_UPSTREAM_MODEL": "gpt-image-2"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
        self.assertEqual(config.model, "gpt-image-2")

    def test_new_env_prefix_wins_over_legacy_prefix(self):
        env = {
            "GPT2CC_UPSTREAM_MODEL": "gpt-image-2",
            "CCPROXY_UPSTREAM_MODEL": "gpt-4.1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
        self.assertEqual(config.model, "gpt-image-2")


if __name__ == "__main__":
    unittest.main()
