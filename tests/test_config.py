import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpt2cc.config import ConfigStore, load_config


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

    def test_loads_provider_config_and_applies_active_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "active_provider": "relay2",
                        "active_model": "gpt-image-2",
                        "providers": [
                            {
                                "id": "relay1",
                                "name": "Relay 1",
                                "upstream_base_url": "https://relay1.example/v1",
                                "upstream_api_key": "sk-relay1",
                                "models": ["gpt-4.1"],
                            },
                            {
                                "id": "relay2",
                                "name": "Relay 2",
                                "upstream_base_url": "https://relay2.example/v1",
                                "upstream_api_key": "sk-relay2",
                                "models": ["gpt-image-2"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                config = load_config()
        self.assertEqual(config.upstream_base_url, "https://relay2.example/v1")
        self.assertEqual(config.upstream_api_key, "sk-relay2")
        self.assertEqual(config.model, "gpt-image-2")

    def test_config_store_redacts_and_preserves_existing_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                store = ConfigStore(load_config())
                store.add_or_update_provider(
                    {
                        "id": "relay",
                        "name": "Relay",
                        "upstream_base_url": "https://relay.example/v1",
                        "upstream_api_key": "sk-secret",
                        "models": ["gpt-4.1"],
                    }
                )
                state = store.add_or_update_provider(
                    {
                        "id": "relay",
                        "name": "Relay Updated",
                        "upstream_base_url": "https://relay.example/v1",
                        "models": ["gpt-4.1", "gpt-image-2"],
                    }
                )
        relay = next(provider for provider in state["providers"] if provider["id"] == "relay")
        self.assertEqual(relay["upstream_api_key"], "***")
        self.assertTrue(relay["has_api_key"])
        self.assertEqual(store.snapshot().providers[-1]["upstream_api_key"], "sk-secret")
    def test_config_store_state_places_active_provider_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "active_provider": "relay1",
                        "active_model": "gpt-4.1",
                        "providers": [
                            {
                                "id": "relay1",
                                "name": "Relay 1",
                                "upstream_base_url": "https://relay1.example/v1",
                                "upstream_api_key": "sk-relay1",
                                "models": ["gpt-4.1"],
                            },
                            {
                                "id": "relay2",
                                "name": "Relay 2",
                                "upstream_base_url": "https://relay2.example/v1",
                                "upstream_api_key": "sk-relay2",
                                "models": ["gpt-image-2"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                store = ConfigStore(load_config())
                state = store.set_active("relay2", "gpt-image-2")
        self.assertEqual(state["providers"][0]["id"], "relay2")
        self.assertEqual(state["active_model"], "gpt-image-2")

    def test_active_provider_label_uses_name_and_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "active_provider": "relay1",
                        "active_model": "gpt-4.1",
                        "providers": [
                            {
                                "id": "relay1",
                                "name": "Main Relay",
                                "upstream_base_url": "https://relay1.example/v1",
                                "upstream_api_key": "sk-relay1",
                                "models": ["gpt-4.1"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                config = load_config()
        self.assertEqual(config.active_provider_label(), "Main Relay (relay1)")


if __name__ == "__main__":
    unittest.main()
