import csv
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from tts_trainer.text_generation import (_openai_compatible_request, generate_texts,
                                         text_corpus_path,
                                         validate_text_generation_config)
from tts_trainer.experiments import resolve_experiment


class TextGenerationTests(unittest.TestCase):
    def _config(self, root: Path, text_generation: dict, languages=("en", "de"),
                *, name="text-test", filename="config.json") -> Path:
        config = {
            "experiment": {
                "name": name,
                "languages": list(languages),
                "dataset_root": str(root / "datasets"),
                "run_root": str(root / "runs"),
                "artifact_root": str(root / "artifacts"),
            },
            "text_generation": {"enabled": True, **text_generation},
            "logging": {"level": "WARNING"},
        }
        path = root / filename
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _rows(self, path: Path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_builtin_is_deterministic_and_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, {
                "provider": "builtin", "sentences_per_language": 12, "seed": 7,
                "filters": {"deduplicate": True, "reject_mixed_language": True},
            })
            first = generate_texts(config)
            first_text = first.read_text(encoding="utf-8")
            second = generate_texts(config)
            self.assertEqual(second.read_text(encoding="utf-8"), first_text)
            rows = self._rows(first)
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(row["language"] == "de" for row in rows), 12)
            report = json.loads(first.with_suffix(".report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["accepted"], {"de": 12, "en": 12})
            self.assertIn("fingerprint", report)

    def test_same_corpus_is_reused_across_model_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {
                "provider": "builtin", "sentences_per_language": 4, "seed": 99,
            }
            first_config = self._config(
                root, settings, languages=("en",), name="reader-a", filename="a.json",
            )
            second_config = self._config(
                root, settings, languages=("en",), name="reader-b", filename="b.json",
            )
            first = generate_texts(first_config)
            initial_mtime = first.stat().st_mtime_ns
            second = generate_texts(second_config)
            self.assertEqual(first, second)
            self.assertEqual(second.stat().st_mtime_ns, initial_mtime)
            self.assertIn("text_corpora", second.parts)
            self.assertNotIn("reader-a", second.parts)
            self.assertNotIn("reader-b", second.parts)

    def test_request_batch_size_does_not_change_corpus_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {
                "provider": "openai_compatible",
                "endpoint": "http://local/v1",
                "model": "text-model",
                "sentences_per_language": 20,
                "batch_size": 20,
            }
            first_config = self._config(root, settings, languages=("en",))
            first_raw, first_layout = resolve_experiment(first_config)
            first_path = text_corpus_path(first_raw["text_generation"], first_layout)

            settings["request_batch_size"] = 50
            second_config = self._config(
                root, settings, languages=("en",), filename="larger-requests.json",
            )
            second_raw, second_layout = resolve_experiment(second_config)
            second_path = text_corpus_path(second_raw["text_generation"], second_layout)
            self.assertEqual(first_path, second_path)

    def test_file_provider_filters_languages_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "language", "category"])
                writer.writeheader()
                writer.writerow({"text": "Hello there.", "language": "en", "category": "daily"})
                writer.writerow({"text": "Hello there.", "language": "en", "category": "daily"})
                writer.writerow({"text": "Guten Morgen.", "language": "de", "category": "daily"})
                writer.writerow({"text": "Bonjour.", "language": "fr", "category": "daily"})
            config = self._config(root, {
                "provider": "file", "input": str(source), "sentences_per_language": 1,
                "filters": {"deduplicate": True, "reject_mixed_language": True},
            })
            rows = self._rows(generate_texts(config))
            self.assertEqual([(row["language"], row["text"]) for row in rows],
                             [("en", "Hello there."), ("de", "Guten Morgen.")])

    def test_openai_compatible_provider_uses_json_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, {
                "provider": "openai_compatible", "endpoint": "http://local/v1",
                "model": "text-model", "sentences_per_language": 2,
                "batch_size": 2,
            }, languages=("en",))
            calls = []

            def requester(raw, prompt):
                calls.append((raw["model"], prompt))
                return json.dumps([
                    {"text": "Please review notification one.", "category": "daily"},
                    {"text": "Does the meeting start at nine?", "category": "question"},
                ])

            rows = self._rows(generate_texts(config, requester=requester))
            self.assertEqual(len(rows), 2)
            self.assertEqual(calls[0][0], "text-model")
            self.assertEqual({row["source"] for row in rows}, {"openai_compatible"})

    def test_openai_provider_refills_rows_rejected_by_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, {
                "provider": "openai_compatible", "endpoint": "http://local/v1",
                "model": "text-model", "sentences_per_language": 2,
                "batch_size": 2,
            }, languages=("en",))
            calls = []

            def requester(_raw, _prompt):
                calls.append(1)
                if len(calls) == 1:
                    return json.dumps([
                        {"text": "A", "category": "short"},
                        {"text": "This sentence is valid.", "category": "daily"},
                    ])
                return json.dumps([
                    {"text": f"Refill sentence {index} is valid.", "category": "daily"}
                    for index in range(4)
                ])

            output = generate_texts(config, requester=requester)
            self.assertEqual(len(self._rows(output)), 2)
            self.assertEqual(len(calls), 2)
            report = json.loads(output.with_suffix(".report.json").read_text())
            self.assertEqual(report["accepted"], {"en": 2})
            self.assertEqual(report["rejected"]["length"], 1)

    def test_openai_provider_resumes_partial_shared_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, {
                "provider": "openai_compatible", "endpoint": "http://local/v1",
                "model": "text-model", "sentences_per_language": 2,
                "batch_size": 2,
            }, languages=("en",))

            def incomplete_requester(_raw, _prompt):
                return json.dumps([
                    {"text": "This sentence survives filtering.", "category": "daily"},
                    {"text": "A", "category": "short"},
                ])

            with self.assertRaisesRegex(RuntimeError, "en missing 1"):
                generate_texts(config, requester=incomplete_requester)

            calls = []

            def refill_requester(_raw, _prompt):
                calls.append(1)
                return json.dumps([
                    {"text": f"Recovered sentence {index} is valid.", "category": "daily"}
                    for index in range(4)
                ])

            output = generate_texts(config, requester=refill_requester)
            self.assertEqual(len(self._rows(output)), 2)
            self.assertEqual(len(calls), 1)

    def test_openai_provider_checkpoints_each_successful_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, {
                "provider": "openai_compatible", "endpoint": "http://local/v1",
                "model": "text-model", "sentences_per_language": 4,
                "batch_size": 2,
            }, languages=("en",))
            calls = 0

            def interrupted_requester(_raw, _prompt):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("temporary network failure")
                return json.dumps([
                    {"text": f"Persisted sentence {index} is valid.", "category": "daily"}
                    for index in range(2)
                ])

            with self.assertRaisesRegex(RuntimeError, "temporary network failure"):
                generate_texts(config, requester=interrupted_requester)

            partial = next((root / "datasets" / "text_corpora").rglob("texts.partial.jsonl"))
            self.assertTrue(partial.is_file())
            resumed_calls = 0

            def resumed_requester(_raw, _prompt):
                nonlocal resumed_calls
                resumed_calls += 1
                return json.dumps([
                    {"text": f"Recovered sentence {index} is also valid.", "category": "daily"}
                    for index in range(2)
                ])

            output = generate_texts(config, requester=resumed_requester)
            self.assertEqual(resumed_calls, 1)
            self.assertEqual(len(self._rows(output)), 4)
            self.assertFalse(partial.exists())

    def test_nested_openai_compatible_config_accepts_base_url_alias(self):
        config = {
            "provider": "openai_compatible",
            "openai_compatible": {
                "base_url": "https://llm.example/v1",
                "model": "text-model",
                "api_key_env": "TEXT_LLM_API_KEY",
            },
        }
        validate_text_generation_config(config)

    def test_api_key_value_is_rejected_without_echoing_it(self):
        secret = "secret-value.with-punctuation"
        config = {
            "provider": "openai_compatible",
            "endpoint": "https://llm.example/v1",
            "model": "text-model",
            "api_key_env": secret,
        }
        with self.assertRaisesRegex(ValueError, "environment variable name") as raised:
            validate_text_generation_config(config)
        self.assertNotIn(secret, str(raised.exception))

    def test_http_auth_error_is_actionable_and_does_not_echo_key(self):
        config = {
            "provider": "openai_compatible",
            "endpoint": "https://llm.example/v1",
            "model": "text-model",
            "api_key_env": "TEXT_LLM_API_KEY",
        }
        error = urllib.error.HTTPError(
            config["endpoint"], 401, "Unauthorized", {},
            io.BytesIO(b'{"error":"invalid token"}'),
        )
        with patch.dict(os.environ, {"TEXT_LLM_API_KEY": "private-key"}), \
                patch("tts_trainer.text_generation.urllib.request.urlopen",
                      side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as raised:
                _openai_compatible_request(config, "prompt")
        self.assertIn("endpoint and plan", str(raised.exception))
        self.assertNotIn("private-key", str(raised.exception))

    def test_tls_url_error_has_proxy_hint(self):
        config = {
            "provider": "openai_compatible",
            "endpoint": "https://llm.example/v1",
            "model": "text-model",
            "api_key_env": None,
        }
        with patch("tts_trainer.text_generation.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("TLS failed")):
            with self.assertRaisesRegex(RuntimeError, "HTTPS_PROXY/NO_PROXY"):
                _openai_compatible_request(config, "prompt")


if __name__ == "__main__":
    unittest.main()
