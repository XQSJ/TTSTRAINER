import json
import tempfile
import unittest
from pathlib import Path

from tts_trainer.project_config import load_project_config
from tts_trainer.vits.config import load_vits_config


class ProjectConfigTests(unittest.TestCase):
    def test_relative_extends_and_deep_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "system" / "base.json"; base.parent.mkdir()
            base.write_text(json.dumps({"model": {"a": 1, "b": 2}, "training": {"epochs": 10}}))
            user = root / "user" / "train.json"; user.parent.mkdir()
            user.write_text(json.dumps({"extends": "../system/base.json", "model": {"b": 3},
                                        "training": {"batch_size": 4}}))
            result = load_project_config(user)
            self.assertEqual(result["model"], {"a": 1, "b": 3})
            self.assertEqual(result["training"], {"epochs": 10, "batch_size": 4})

    def test_public_train1_config_resolves(self):
        config = load_project_config("training_configs/train1.json")
        self.assertEqual(config["model"]["hidden_channels"], 128)
        self.assertEqual(config["training"]["batch_size"], 8)
        self.assertEqual(config["experiment"]["name"], "model_1")
        self.assertEqual(config["experiment"]["languages"], ["zh", "en", "ja", "ko", "fr", "es", "pt"])
        self.assertEqual(config["language_registry"]["de"]["teacher"]["language"], "German")

    def test_training_config_keeps_expert_defaults_internal(self):
        config = load_project_config("training_configs/train2.json")
        self.assertEqual(config["experiment"]["name"], "model_2")
        self.assertEqual(config["experiment"]["languages"], ["en", "fr", "es", "pt"])
        self.assertEqual(config["generation"]["voice"]["mode"], "design")
        self.assertEqual(config["model"]["hidden_channels"], 128)
        self.assertEqual(config["generation"]["generation_kwargs"]["max_new_tokens"], 2048)

    def test_public_workflow_examples_resolve(self):
        clone = load_project_config("training_configs/clone.example.json")
        resume = load_project_config("training_configs/resume.example.json")
        expand = load_project_config("training_configs/add-speaker.example.json")
        self.assertEqual(clone["generation"]["voice"]["mode"], "clone")
        self.assertEqual(resume["experiment"]["initialization"]["mode"], "resume")
        self.assertEqual(expand["experiment"]["initialization"]["mode"], "expand_speakers")
        self.assertEqual(expand["generation"]["include_metadata"], ["datasets/model_1/metadata.csv"])
        self.assertEqual(load_vits_config("training_configs/train1.json").hop_length, 256)
        european = load_project_config("training_configs/european.example.json")
        self.assertEqual(european["experiment"]["languages"], ["en", "de", "fr", "ru", "es", "pt", "it"])

    def test_public_configs_keep_valid_bilingual_json_comments(self):
        config_paths = sorted(Path("training_configs").glob("*.json"))
        self.assertTrue(config_paths)
        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertIn("_comment", raw)
                self.assertIn(" / ", raw["_comment"])
                self.assertIn("experiment", load_project_config(config_path))

    def test_rejects_circular_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text('{"extends":"b.json"}')
            (root / "b.json").write_text('{"extends":"a.json"}')
            with self.assertRaisesRegex(ValueError, "circular"):
                load_project_config(root / "a.json")


if __name__ == "__main__":
    unittest.main()
