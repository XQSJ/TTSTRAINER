import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tts_trainer.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_configured_stages_run_in_order_and_write_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "automatic"
            config = {
                "experiment": {
                    "name": "automatic",
                    "languages": ["en"],
                    "dataset_root": str(root / "datasets"),
                    "metadata": str(dataset / "metadata.phonemes.csv"),
                    "run_root": str(root / "runs"),
                    "artifact_root": str(root / "artifacts"),
                },
                "audio": {"sample_rate": 22050},
                "frontend": {"require_phonemes": True},
                "text_generation": {"enabled": True},
                "generation": {"enabled": True},
                "pipeline": {"generate_texts": True, "generate_samples": True,
                             "phonemize": True, "validate": True,
                             "train": True, "export": True, "validate_onnx": True},
            }
            config_path = root / "train.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            raw_metadata = dataset / "metadata.csv"
            text_manifest = dataset / "texts.generated.csv"
            checkpoint = root / "runs/automatic/checkpoints/last"
            onnx = root / "artifacts/automatic/model.onnx"
            calls = []

            def fake_generate_texts(_):
                calls.append("generate_texts"); return text_manifest

            def fake_generate(_, **kwargs):
                self.assertEqual(kwargs["text_manifest_path"], text_manifest)
                calls.append("generate"); return raw_metadata

            def fake_phonemize(source, destination, frontend):
                calls.append("phonemize"); return destination

            def fake_validate(*args, **kwargs):
                calls.append("validate")
                return SimpleNamespace(items=(SimpleNamespace(language="en"), SimpleNamespace(language="en")),
                                       language_counts={"en": 2})

            def fake_train(*args, **kwargs):
                calls.append("train"); return checkpoint

            def fake_export(*args, **kwargs):
                calls.append("export")
                onnx.parent.mkdir(parents=True, exist_ok=True); onnx.write_bytes(b"onnx")
                return onnx

            with patch("tts_trainer.pipeline.generate_texts", fake_generate_texts), \
                    patch("tts_trainer.pipeline.generate_samples", fake_generate), \
                    patch("tts_trainer.pipeline.check_language_support", return_value=[]), \
                    patch("tts_trainer.pipeline.frontend_from_config", return_value=object()), \
                    patch("tts_trainer.pipeline.phonemize_manifest", fake_phonemize), \
                    patch("tts_trainer.pipeline.validate_manifest", fake_validate), \
                    patch("tts_trainer.pipeline.train_vits", fake_train), \
                    patch("tts_trainer.pipeline.export_vits_onnx", fake_export), \
                    patch("tts_trainer.pipeline.validate_onnx_runtime", return_value=(1, 1, 100)):
                report_path = run_pipeline(config_path, max_steps=3)

            self.assertEqual(calls, ["generate_texts", "generate", "phonemize", "validate", "train", "export"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["name"], "automatic")
            self.assertEqual(report["stages"]["validate_onnx"], [1, 1, 100])


if __name__ == "__main__":
    unittest.main()
