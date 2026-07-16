import json
import tempfile
import unittest
from pathlib import Path

from tts_trainer.experiments import prepare_experiment, resolve_experiment, validate_model_name


class ExperimentTests(unittest.TestCase):
    def test_name_creates_isolated_run_and_artifact_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "model.json"
            raw = {
                "experiment": {"name": "voice-a-v2", "metadata": "data.csv",
                               "dataset_root": str(root / "datasets"),
                               "run_root": str(root / "runs"), "artifact_root": str(root / "artifacts"),
                               "initialization": {"mode": "scratch", "checkpoint": None}}
            }
            config.write_text(json.dumps(raw), encoding="utf-8")
            resolved, layout = resolve_experiment(config)
            prepare_experiment(layout, resolved, config)
            self.assertEqual(layout.run_dir, root / "runs" / "voice-a-v2")
            self.assertEqual(layout.languages, ("zh", "en", "ja", "ko", "fr", "es", "pt"))
            self.assertEqual(layout.artifacts_dir, root / "artifacts" / "voice-a-v2")
            self.assertTrue((layout.run_dir / "resolved-config.json").is_file())
            self.assertTrue(layout.checkpoints_dir.is_dir())
            self.assertTrue(layout.logs_dir.is_dir())
            self.assertTrue(layout.artifacts_dir.is_dir())

    def test_name_provides_default_namespaced_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps({"experiment": {"name": "reader_a"}}), encoding="utf-8")
            _, layout = resolve_experiment(config)
            self.assertEqual(layout.dataset_dir, Path("datasets/reader_a"))
            self.assertEqual(layout.metadata, Path("datasets/reader_a/metadata.phonemes.csv"))

    def test_expand_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps({"experiment": {"name": "v2", "metadata": "data.csv",
                                                          "initialization": {"mode": "expand_speakers"}}}))
            with self.assertRaisesRegex(ValueError, "requires a checkpoint"):
                resolve_experiment(config)

    def test_rejects_path_like_model_name(self):
        with self.assertRaises(ValueError):
            validate_model_name("../overwrite")

    def test_validates_configured_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps({"experiment": {"name": "english", "languages": ["en"]}}))
            _, layout = resolve_experiment(config)
            self.assertEqual(layout.languages, ("en",))
            config.write_text(json.dumps({"experiment": {"name": "bad", "languages": ["en", "xx"]}}))
            with self.assertRaisesRegex(ValueError, "unregistered configured"):
                resolve_experiment(config)

    def test_builtin_registry_supports_all_qwen_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps({
                "experiment": {
                    "name": "qwen-ten",
                    "languages": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                }
            }))
            _, layout = resolve_experiment(config)
            self.assertEqual(layout.language_specs["de"].frontend_voice, "de")
            self.assertEqual(layout.language_specs["ru"].teacher_language, "Russian")

    def test_custom_registered_language_can_use_external_data(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps({
                "language_registry": {
                    "pl": {
                        "name": "Polish",
                        "teacher": None,
                        "frontend": {"provider": "espeak-ng", "voice": "pl"},
                        "smoke_text": "Dzień dobry.",
                    }
                },
                "experiment": {"name": "polish", "languages": ["pl"]},
                "generation": {"enabled": False},
            }))
            _, layout = resolve_experiment(config)
            self.assertEqual(layout.languages, ("pl",))
            self.assertIsNone(layout.language_specs["pl"].teacher_provider)

    def test_resolved_config_records_dynamic_language_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "model.json"
            config.write_text(json.dumps({
                "experiment": {
                    "name": "latin", "languages": ["en", "fr"],
                    "dataset_root": str(root / "datasets"),
                    "run_root": str(root / "runs"), "artifact_root": str(root / "artifacts"),
                },
                "model": {"num_languages": 7},
            }))
            raw, layout = resolve_experiment(config)
            prepare_experiment(layout, raw, config)
            recorded = json.loads((layout.run_dir / "resolved-config.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["experiment"]["languages"], ["en", "fr"])
            self.assertEqual(recorded["model"]["num_languages"], 2)


if __name__ == "__main__":
    unittest.main()
