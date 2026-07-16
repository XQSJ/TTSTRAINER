import csv
import json
import tempfile
import unittest
from pathlib import Path

from tts_trainer.text_generation import generate_texts


class TextGenerationTests(unittest.TestCase):
    def _config(self, root: Path, text_generation: dict, languages=("en", "de")) -> Path:
        config = {
            "experiment": {
                "name": "text-test",
                "languages": list(languages),
                "dataset_root": str(root / "datasets"),
                "run_root": str(root / "runs"),
                "artifact_root": str(root / "artifacts"),
            },
            "text_generation": {"enabled": True, **text_generation},
            "logging": {"level": "WARNING"},
        }
        path = root / "config.json"
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
            report = json.loads((first.parent / "text-generation-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["accepted"], {"de": 12, "en": 12})

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


if __name__ == "__main__":
    unittest.main()
