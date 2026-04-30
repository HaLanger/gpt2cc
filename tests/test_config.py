import os
import unittest
from unittest.mock import patch

from ccproxy.config import load_config


class ConfigTests(unittest.TestCase):
    def test_tls_env_options(self):
        env = {
            "CCPROXY_UPSTREAM_SSL_VERIFY": "false",
            "CCPROXY_UPSTREAM_CA_BUNDLE": r"C:\certs\relay-ca.pem",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config()
        self.assertFalse(config.upstream_ssl_verify)
        self.assertEqual(config.upstream_ca_bundle, r"C:\certs\relay-ca.pem")


if __name__ == "__main__":
    unittest.main()
