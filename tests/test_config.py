import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpt2cc.config import (
    Config,
    ConfigStore,
    ensure_config_file,
    load_config,
    normalize_provider,
    parse_provider_pricing_value,
    stats_path_from_config_path,
)


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
    def test_config_store_saves_provider_pricing_by_provider_and_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                store = ConfigStore(load_config())
                state = store.add_or_update_provider(
                    {
                        "id": "relay-a",
                        "name": "Relay A",
                        "upstream_base_url": "https://a.example/v1",
                        "upstream_api_key": "sk-a",
                        "models": ["gpt-4.1"],
                        "pricing": {"gpt-4.1": {"input_per_million": 2.5, "output_per_million": 10.0}},
                    }
                )
                state = store.add_or_update_provider(
                    {
                        "id": "relay-b",
                        "name": "Relay B",
                        "upstream_base_url": "https://b.example/v1",
                        "upstream_api_key": "sk-b",
                        "models": ["gpt-4.1"],
                        "pricing": {"gpt-4.1": {"input_per_million": 4.0, "output_per_million": 12.0}},
                    }
                )
        self.assertEqual(state["provider_pricing"]["relay-a"]["gpt-4.1"]["input_per_million"], 2.5)
        self.assertEqual(state["provider_pricing"]["relay-b"]["gpt-4.1"]["input_per_million"], 4.0)

    def test_config_store_empty_provider_pricing_clears_provider_prices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                config_path=str(Path(tmpdir) / "config.json"),
                provider_pricing={"relay": {"gpt-4.1": {"input_per_million": 2.5}}},
            )
            store = ConfigStore(config)
            state = store.add_or_update_provider(
                {
                    "id": "relay",
                    "name": "Relay",
                    "upstream_base_url": "https://relay.example/v1",
                    "models": ["gpt-4.1"],
                    "pricing": {},
                }
            )
        self.assertNotIn("relay", state["provider_pricing"])

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
    def test_legacy_provider_defaults_to_openai_protocol(self):
        provider = normalize_provider(
            {
                "id": "relay",
                "upstream_base_url": "https://relay.example/v1",
                "upstream_api_key": "sk-relay",
                "models": ["gpt-4.1"],
            }
        )
        self.assertEqual(provider["protocol"], "openai")

    def test_invalid_provider_protocol_fails(self):
        with self.assertRaisesRegex(ValueError, "provider protocol"):
            normalize_provider(
                {
                    "id": "relay",
                    "protocol": "unknown",
                    "upstream_base_url": "https://relay.example/v1",
                    "models": ["gpt-4.1"],
                }
            )

    def test_ensure_config_file_creates_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                config = load_config()
                self.assertTrue(ensure_config_file(config))
                data = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(data["providers"][0]["protocol"], "openai")
                config_path.write_text('{"sentinel":true}\n', encoding="utf-8")
                self.assertFalse(ensure_config_file(config))
                self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), {"sentinel": True})
    def test_model_routes_snapshot_selects_provider_without_changing_active(self):
        config = Config(
            model="gpt-4.1",
            active_provider="main",
            active_model="gpt-4.1",
            providers=[
                {
                    "id": "main",
                    "name": "Main",
                    "protocol": "openai",
                    "upstream_base_url": "https://main.example/v1",
                    "upstream_api_key": "sk-main",
                    "models": ["gpt-4.1"],
                    "upstream_chat_path": "/chat/completions",
                    "upstream_messages_path": "/messages",
                    "upstream_gemini_generate_path": "/models/{model}:generateContent",
                    "upstream_gemini_stream_path": "/models/{model}:streamGenerateContent",
                    "upstream_images_generations_path": "/images/generations",
                    "upstream_images_edits_path": "/images/edits",
                    "upstream_auth_header": "Authorization",
                    "upstream_auth_scheme": "Bearer",
                },
                {
                    "id": "strong",
                    "name": "Strong",
                    "protocol": "gemini",
                    "upstream_base_url": "https://gemini.example/v1beta",
                    "upstream_api_key": "sk-strong",
                    "models": ["gemini-2.5-pro"],
                    "upstream_chat_path": "/chat/completions",
                    "upstream_messages_path": "/messages",
                    "upstream_gemini_generate_path": "/models/{model}:generateContent",
                    "upstream_gemini_stream_path": "/models/{model}:streamGenerateContent",
                    "upstream_images_generations_path": "/images/generations",
                    "upstream_images_edits_path": "/images/edits",
                    "upstream_auth_header": "Authorization",
                    "upstream_auth_scheme": "Bearer",
                },
            ],
            model_routes={"claude-opus-4-7": {"provider": "strong", "model": "gemini-2.5-pro"}},
        )
        store = ConfigStore(config)
        routed = store.snapshot_for_model("claude-opus-4-7")
        route = routed.resolve_model_route("claude-opus-4-7")
        self.assertEqual(routed.active_provider, "strong")
        self.assertEqual(routed.model, "gemini-2.5-pro")
        self.assertEqual(routed.upstream_protocol, "gemini")
        self.assertEqual(routed.route_source, "model_routes")
        self.assertEqual(route.upstream, "gemini-2.5-pro")
        self.assertEqual(route.source, "model_routes")
        self.assertEqual(store.state()["active_provider"], "main")

    def test_primary_route_binding_tracks_active_route(self):
        config = Config(
            model="gpt-4.1",
            active_provider="main",
            active_model="gpt-4.1",
            providers=[
                {
                    "id": "main",
                    "name": "Main",
                    "protocol": "openai",
                    "upstream_base_url": "https://main.example/v1",
                    "upstream_api_key": "sk-main",
                    "models": ["gpt-4.1", "gpt-4.1-mini"],
                    "upstream_chat_path": "/chat/completions",
                    "upstream_messages_path": "/messages",
                    "upstream_gemini_generate_path": "/models/{model}:generateContent",
                    "upstream_gemini_stream_path": "/models/{model}:streamGenerateContent",
                    "upstream_images_generations_path": "/images/generations",
                    "upstream_images_edits_path": "/images/edits",
                    "upstream_auth_header": "Authorization",
                    "upstream_auth_scheme": "Bearer",
                }
            ],
        )
        store = ConfigStore(config)
        state = store.bind_primary_route_model("claude-sonnet-4-6")
        self.assertEqual(state["primary_route_model"], "claude-sonnet-4-6")
        self.assertEqual(state["model_routes"]["claude-sonnet-4-6"], {"provider": "main", "model": "gpt-4.1"})
        state = store.set_active("main", "gpt-4.1-mini")
        self.assertEqual(state["model_routes"]["claude-sonnet-4-6"], {"provider": "main", "model": "gpt-4.1-mini"})
        state = store.unbind_primary_route_model()
        self.assertEqual(state["primary_route_model"], "")

    def test_seen_models_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(model="gpt-4.1", config_path=str(Path(tmpdir) / "config.json"))
            store = ConfigStore(config)
            store.record_seen_model("claude-sonnet-4-6")
            store.record_seen_model("claude-sonnet-4-6")
            seen = store.state()["seen_models"]["claude-sonnet-4-6"]
        self.assertEqual(seen["count"], 2)
        self.assertTrue(seen["last_seen"])


    def test_stats_path_default_and_custom_derivation(self):
        self.assertEqual(stats_path_from_config_path("gpt2cc.config.json"), "gpt2cc.stats.json")
        self.assertEqual(
            stats_path_from_config_path(str(Path("/tmp") / "custom-config.json")),
            str(Path("/tmp") / "custom-config.stats.json"),
        )

    def test_provider_pricing_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "relay.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider_pricing": {
                            "xlab": {
                                "gpt-5.4": {
                                    "input_per_million": 2.5,
                                    "output_per_million": 10.0,
                                    "cache_read_per_million": 0.3,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GPT2CC_CONFIG": str(config_path)}, clear=True):
                config = load_config()
                self.assertEqual(config.stats_path, str(Path(tmpdir) / "relay.stats.json"))
                self.assertEqual(config.provider_pricing["xlab"]["gpt-5.4"]["input_per_million"], 2.5)
                store = ConfigStore(config)
                store.save()
            saved = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["provider_pricing"]["xlab"]["gpt-5.4"]["output_per_million"], 10.0)
        self.assertNotIn("stats_path", saved)

    def test_parse_provider_pricing_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "pricing field input_per_million"):
            parse_provider_pricing_value({"xlab": {"gpt-5.4": {"input_per_million": "abc"}}})


if __name__ == "__main__":
    unittest.main()
