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

    def test_public_preset_hides_internal_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "quality.json"
            config.write_text(json.dumps({
                "preset": "quality",
                "experiment": {"name": "clear_config", "languages": ["en"]},
                "training": {"batch_size": 2},
            }), encoding="utf-8")
            resolved = load_project_config(config)
            self.assertEqual(resolved["model"]["hidden_channels"], 256)
            self.assertEqual(resolved["training"]["batch_size"], 2)
            self.assertEqual(resolved["training"]["log_every_steps"], 50)
            self.assertEqual(resolved["training"]["checkpoint_every_steps"], 10000)
            self.assertEqual(resolved["training"]["checkpoint_every_epochs"], 5)
            # These options caused the 2026-07-30 regression and must not
            # silently return through a preset. / 这些选项曾导致 2026-07-30
            # 回归，不能通过 preset 悄悄重新进入训练目标。
            for key in (
                "aligned_prior_mel_weight",
                "aligned_prior_mel_start_steps",
                "aligned_prior_mel_warmup_steps",
            ):
                self.assertNotIn(key, resolved["training"])
            self.assertEqual(resolved["validation"]["every_epochs"], 5)

    def test_mobile_preset_is_isolated_from_quality_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality_path = root / "quality.json"
            mobile_path = root / "mobile.json"
            common = {
                "experiment": {"name": "frontend-check", "languages": ["zh", "en"]},
            }
            quality_path.write_text(json.dumps({
                **common, "preset": "quality",
            }), encoding="utf-8")
            mobile_path.write_text(json.dumps({
                **common, "preset": "mobile",
            }), encoding="utf-8")
            quality = load_project_config(quality_path)
            mobile = load_project_config(mobile_path)
            self.assertEqual(quality["frontend"]["provider"], "language-router")
            self.assertFalse(quality["frontend"].get("piper_compatible", False))
            self.assertEqual(mobile["frontend"]["provider"], "espeak-ng")
            self.assertTrue(mobile["frontend"]["mobile_direct"])
            self.assertFalse(mobile["frontend"].get("piper_compatible", False))

    def test_mobile_routed_keeps_mobile_model_and_quality_frontends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobile-routed.json"
            path.write_text(json.dumps({
                "preset": "mobile_routed",
                "experiment": {
                    "name": "mobile-routed",
                    "languages": ["zh", "en", "ja", "ko"],
                },
            }), encoding="utf-8")
            resolved = load_project_config(path)
            self.assertEqual(
                resolved["model"]["duration_predictor_type"],
                "stochastic_mobile",
            )
            self.assertEqual(
                resolved["model"]["duration_predictor_flow_layers"], 2,
            )
            self.assertEqual(
                resolved["frontend"]["provider"], "language-router",
            )
            self.assertFalse(resolved["frontend"]["mobile_direct"])

    def test_rejects_unknown_public_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.json"
            config.write_text('{"preset":"mystery"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config preset"):
                load_project_config(config)

    def test_public_train1_uses_quality_defaults(self):
        config = load_project_config("training_configs/train1.json")
        self.assertEqual(config["model"]["hidden_channels"], 256)
        self.assertEqual(config["training"]["batch_size"], 4)
        self.assertEqual(config["training"]["log_every_steps"], 50)
        self.assertEqual(config["training"]["checkpoint_every_epochs"], 5)
        self.assertEqual(config["experiment"]["name"], "model_1")
        self.assertEqual(config["experiment"]["languages"], ["zh", "en", "ja", "ko", "fr", "es", "pt"])
        self.assertEqual(config["language_registry"]["de"]["teacher"]["language"], "German")
        self.assertTrue(config["validation"]["enabled"])
        self.assertEqual(config["validation"]["metric"], "combined_mel")
        self.assertEqual(config["validation"]["export_checkpoint"], "last")
        self.assertTrue(config["quality"]["enabled"])
        self.assertTrue(config["training"]["text_prior_refinement"]["enabled"])

    def test_refinement_example_keeps_advanced_defaults_internal(self):
        config = load_project_config(
            "training_configs/refine-text-prior.example.json",
        )
        self.assertEqual(
            config["experiment"]["initialization"]["mode"],
            "refine_text_prior",
        )
        self.assertEqual(
            config["validation"]["metric"], "combined_mel",
        )
        self.assertTrue(
            config["training"]["text_prior_refinement"]["enabled"],
        )

    def test_training_config_keeps_expert_defaults_internal(self):
        config = load_project_config("training_configs/train2.json")
        self.assertEqual(config["experiment"]["name"], "model_2")
        self.assertEqual(config["experiment"]["languages"], ["en", "fr", "es", "pt"])
        self.assertEqual(config["generation"]["voices"]["voice_02"]["mode"], "design")
        self.assertEqual(config["model"]["hidden_channels"], 256)
        self.assertEqual(config["generation"]["generation_kwargs"]["max_new_tokens"], 2048)
        self.assertEqual(config["generation"]["voices"]["voice_02"]["id"], "voice_02")
        self.assertEqual(config["text_generation"]["sentences_per_language"], 2000)

    def test_public_dataset_block_expands_to_internal_pipeline_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simple.json"
            path.write_text(json.dumps({
                "experiment": {"name": "simple", "languages": ["en"]},
                "dataset": {
                    "sentences_per_language": 321,
                    "text": {"provider": "builtin"},
                    "voice": {"id": "voice_a", "mode": "design"},
                    "speakers": {"reader": "voice_a"},
                    "include": ["old.csv"],
                },
            }), encoding="utf-8")
            config = load_project_config(path)
            self.assertTrue(config["text_generation"]["enabled"])
            self.assertEqual(config["text_generation"]["sentences_per_language"], 321)
            self.assertEqual(config["generation"]["voice"]["id"], "voice_a")
            self.assertEqual(
                config["generation"]["speaker_assignments"],
                {"reader": "voice_a"},
            )
            self.assertEqual(config["generation"]["include_metadata"], ["old.csv"])

    def test_multiple_public_voices_have_explicit_task_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voices.json"
            path.write_text(json.dumps({
                "task": "prepare",
                "experiment": {"name": "voices", "languages": ["en"]},
                "dataset": {
                    "sentences_per_language": 12,
                    "voices": {
                        "warm": {
                            "mode": "design",
                            "prompt": "Warm",
                            "reference_strategy": "cascade",
                            "regenerate": {
                                "audio": True,
                                "references": True,
                                "languages": ["en"],
                            },
                        },
                        "clone": {"mode": "clone", "reference_audio": "voice.wav"},
                    },
                },
            }), encoding="utf-8")
            config = load_project_config(path)
            self.assertEqual(config["task"], "prepare")
            self.assertEqual(set(config["generation"]["voices"]), {"warm", "clone"})
            self.assertEqual(config["generation"]["voices"]["warm"]["id"], "warm")
            self.assertEqual(
                config["generation"]["voices"]["warm"]["regenerate"],
                {
                    "audio": True,
                    "references": True,
                    "languages": ["en"],
                },
            )
            self.assertEqual(
                config["generation"]["voices"]["warm"]["reference_strategy"],
                "cascade",
            )
            self.assertNotIn("speaker_assignments", config["generation"])

    def test_task_validation_prevents_ambiguous_workflows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare = root / "prepare.json"
            prepare.write_text(json.dumps({
                "task": "prepare",
                "dataset": {
                    "voices": {"a": {"mode": "design"}},
                    "speakers": {"reader": "a"},
                },
            }))
            with self.assertRaisesRegex(ValueError, "does not use dataset.speakers"):
                load_project_config(prepare)

            train = root / "train.json"
            train.write_text(json.dumps({
                "task": "train",
                "dataset": {"voices": {"a": {"mode": "design"}}},
            }))
            with self.assertRaisesRegex(ValueError, "requires dataset.speakers"):
                load_project_config(train)

            invalid = root / "invalid.json"
            invalid.write_text('{"task":"guess"}')
            with self.assertRaisesRegex(ValueError, "task must be prepare or train"):
                load_project_config(invalid)

            invalid_voice = root / "invalid-voice.json"
            invalid_voice.write_text(json.dumps({
                "task": "prepare",
                "dataset": {
                    "voices": {
                        "a": {
                            "mode": "design",
                            "regenerate_audio": "yes",
                        },
                    },
                },
            }))
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                load_project_config(invalid_voice)

    def test_public_workflow_examples_resolve(self):
        clone = load_project_config("training_configs/clone.example.json")
        prepare = load_project_config("training_configs/prepare-voices.example.json")
        resume = load_project_config("training_configs/resume.example.json")
        warm_start = load_project_config("training_configs/sdp-warm-start.example.json")
        expand = load_project_config("training_configs/add-speaker.example.json")
        multi = load_project_config("training_configs/multi-speaker.example.json")
        self.assertEqual(clone["generation"]["voices"]["my_voice"]["mode"], "clone")
        self.assertEqual(clone["training"]["mixed_precision"], "fp32")
        self.assertEqual(clone["training"]["stage"], "auto")
        self.assertEqual(prepare["task"], "prepare")
        self.assertEqual(
            set(prepare["generation"]["voices"]),
            {"voice_xiaoling_a", "voice_xiaoling_b"},
        )
        self.assertEqual(resume["experiment"]["initialization"]["mode"], "resume")
        self.assertEqual(
            warm_start["experiment"]["initialization"]["mode"], "warm_start",
        )
        self.assertEqual(
            warm_start["experiment"]["initialization"]["exclude"],
            ["duration_predictor"],
        )
        self.assertEqual(expand["experiment"]["initialization"]["mode"], "expand_speakers")
        self.assertEqual(expand["generation"]["include_metadata"], [])
        self.assertFalse(multi["text_generation"]["enabled"])
        self.assertEqual(multi["generation"]["speaker_assignments"], {
            "xiaoling_a": "voice_xiaoling_a",
            "xiaoling_b": "voice_xiaoling_b",
        })
        self.assertEqual(load_vits_config("training_configs/train1.json").hop_length, 256)
        self.assertEqual(
            load_project_config("training_configs/train1.json")["model"]
            ["duration_predictor_type"],
            "stochastic_quality",
        )
        with tempfile.TemporaryDirectory() as directory:
            mobile_source = Path(directory) / "mobile-duration-test.json"
            mobile_source.write_text(json.dumps({
                "_comment": "测试 / Test", "preset": "mobile",
                "experiment": {"name": "mobile", "languages": ["zh"]},
            }), encoding="utf-8")
            mobile = load_project_config(mobile_source)
        self.assertEqual(
            mobile["model"]["duration_predictor_type"],
            "stochastic_mobile",
        )

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
