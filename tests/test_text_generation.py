import csv
import io
import json
import os
import re
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from tts_trainer.text_generation import (_corpus_paths, _legacy_corpus_identity,
                                         _openai_compatible_request, generate_texts,
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

    def test_smaller_language_selection_reuses_compatible_larger_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {
                "provider": "openai_compatible",
                "endpoint": "http://local/v1",
                "model": "text-model",
                "sentences_per_language": 4,
                "request_batch_size": 4,
            }
            larger = self._config(
                root, settings, languages=("en", "fr"), filename="larger.json",
            )

            def requester(_config, prompt):
                language = "fr" if "language code fr" in prompt else "en"
                prefix = "Phrase française" if language == "fr" else "English sentence"
                return json.dumps([
                    {"text": f"{prefix} number {index} is valid.", "category": "daily"}
                    for index in range(1, 5)
                ])

            larger_output = generate_texts(larger, requester=requester)
            larger_english = [
                row["text"] for row in self._rows(larger_output)
                if row["language"] == "en"
            ]
            larger_report = larger_output.with_suffix(".report.json")
            legacy_report = json.loads(larger_report.read_text(encoding="utf-8"))
            legacy_report.pop("family_fingerprint")
            larger_report.write_text(json.dumps(legacy_report), encoding="utf-8")

            reduced_settings = {**settings, "sentences_per_language": 2}
            reduced = self._config(
                root, reduced_settings, languages=("en",), filename="reduced.json",
            )

            def must_not_request(*_args, **_kwargs):
                self.fail("compatible larger corpus should satisfy reduced selection")

            reduced_output = generate_texts(reduced, requester=must_not_request)
            reduced_rows = self._rows(reduced_output)
            self.assertNotEqual(reduced_output, larger_output)
            self.assertEqual(len(reduced_rows), 2)
            self.assertEqual(
                [row["text"] for row in reduced_rows], larger_english[:2],
            )

            incompatible_settings = {
                **reduced_settings, "model": "different-text-model",
            }
            incompatible = self._config(
                root, incompatible_settings, languages=("en",),
                filename="incompatible.json",
            )
            incompatible_calls = 0

            def incompatible_requester(_config, _prompt):
                nonlocal incompatible_calls
                incompatible_calls += 1
                return json.dumps([
                    {"text": f"Different model sentence {index} is valid.",
                     "category": "daily"}
                    for index in range(1, 3)
                ])

            generate_texts(incompatible, requester=incompatible_requester)
            self.assertEqual(incompatible_calls, 1)

    def test_larger_selection_requests_only_missing_compatible_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_settings = {
                "provider": "openai_compatible",
                "endpoint": "http://local/v1",
                "model": "text-model",
                "sentences_per_language": 2,
                "request_batch_size": 10,
            }
            base = self._config(root, base_settings, languages=("en",))
            generate_texts(base, requester=lambda *_args: json.dumps([
                {"text": "Existing sentence number one is valid.", "category": "daily"},
                {"text": "Existing sentence number two is valid.", "category": "daily"},
            ]))

            expanded = self._config(
                root, {**base_settings, "sentences_per_language": 4},
                languages=("en",), filename="expanded.json",
            )
            requested_counts = []

            def requester(_config, prompt):
                match = re.search(r"Create (\d+) unique", prompt)
                requested_counts.append(int(match.group(1)))
                return json.dumps([
                    {"text": "New sentence number three is valid.", "category": "daily"},
                    {"text": "New sentence number four is valid.", "category": "daily"},
                ])

            output = generate_texts(expanded, requester=requester)
            self.assertEqual(requested_counts, [2])
            self.assertEqual(len(self._rows(output)), 4)

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

            settings["batch_size"] = 100
            settings["request_batch_size"] = 50
            settings["timeout_seconds"] = 300
            settings["max_retries"] = 8
            settings["retry_backoff_seconds"] = 0.1
            second_config = self._config(
                root, settings, languages=("en",), filename="larger-requests.json",
            )
            second_raw, second_layout = resolve_experiment(second_config)
            second_path = text_corpus_path(second_raw["text_generation"], second_layout)
            self.assertEqual(first_path, second_path)

    def test_timeout_is_retried_and_then_succeeds(self):
        config = {
            "provider": "openai_compatible",
            "endpoint": "https://llm.example/v1",
            "model": "text-model",
            "api_key_env": None,
            "max_retries": 1,
            "retry_backoff_seconds": 0,
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "[]"}}],
                }).encode("utf-8")

        with patch("tts_trainer.text_generation.urllib.request.urlopen",
                   side_effect=[TimeoutError("read timed out"), Response()]) as urlopen:
            self.assertEqual(_openai_compatible_request(config, "prompt"), "[]")
        self.assertEqual(urlopen.call_count, 2)

    def test_legacy_batch_hashed_checkpoint_is_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, {
                "provider": "openai_compatible",
                "endpoint": "http://local/v1",
                "model": "text-model",
                "sentences_per_language": 2,
                "batch_size": 20,
            }, languages=("en",))
            raw, layout = resolve_experiment(config)
            text_config = raw["text_generation"]
            legacy_id, legacy_fingerprint = _legacy_corpus_identity(text_config, layout)
            legacy_output, _ = _corpus_paths(text_config, layout, legacy_id)
            legacy_partial = legacy_output.with_suffix(".partial.jsonl")
            legacy_partial.parent.mkdir(parents=True)
            legacy_partial.write_text(
                json.dumps({
                    "_checkpoint": {"format": 1, "fingerprint": legacy_fingerprint},
                }) + "\n" + json.dumps({
                    "text": "The first persisted sentence is valid.",
                    "language": "en", "category": "daily",
                    "source": "openai_compatible",
                }) + "\n",
                encoding="utf-8",
            )
            updated = json.loads(config.read_text(encoding="utf-8"))
            updated["text_generation"].update({
                "request_batch_size": 100,
                "max_retries": 8,
                "retry_backoff_seconds": 0.1,
            })
            config.write_text(json.dumps(updated), encoding="utf-8")
            calls = 0

            def requester(_raw, _prompt):
                nonlocal calls
                calls += 1
                return json.dumps([{
                    "text": "The recovered second sentence is valid.",
                    "category": "daily",
                }])

            output = generate_texts(config, requester=requester)
            self.assertEqual(calls, 1)
            self.assertEqual(len(self._rows(output)), 2)
            self.assertNotEqual(output, legacy_output)
            self.assertFalse(legacy_partial.exists())

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
