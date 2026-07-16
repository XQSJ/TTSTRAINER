import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_trainer.manifest import Item, validate_manifest
from tts_trainer.quality import inspect_audio_item
from tts_trainer.sample_generation import (_postprocess_training_wav,
                                           generate_samples)


class FakeDesignModel:
    def __init__(self, calls):
        self.calls = calls

    def generate_voice_design(self, **kwargs):
        self.calls.append(("design", kwargs))
        return [np.linspace(-0.1, 0.1, 160, dtype=np.float32)], 16000


class FakeCloneModel:
    def __init__(self, calls):
        self.calls = calls

    def create_voice_clone_prompt(self, **kwargs):
        self.calls.append(("prompt", kwargs))
        return ["reusable-prompt"]

    def generate_voice_clone(self, **kwargs):
        self.calls.append(("clone", kwargs))
        return [np.linspace(-0.2, 0.2, 160, dtype=np.float32) for _ in kwargs["text"]], 16000

    def get_supported_languages(self):
        return ["Chinese", "English", "French"]


class SampleGenerationTests(unittest.TestCase):
    def test_edge_silence_postprocess_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rate = 8000
            active_time = np.arange(rate, dtype=np.float32) / rate
            active = 0.2 * np.sin(2 * np.pi * 220 * active_time)
            samples = np.concatenate([
                np.zeros(rate, dtype=np.float32), active,
                np.zeros(rate, dtype=np.float32),
            ])
            path = root / "padded.wav"
            sf.write(path, samples, rate, subtype="PCM_16")
            settings = {
                "enabled": True, "trim_edge_silence": True,
                "silence_threshold_dbfs": -45.0,
                "keep_edge_silence_seconds": 0.1,
            }
            result = _postprocess_training_wav(path, settings)
            self.assertIsNotNone(result)
            self.assertLess(result["after_seconds"], 1.25)
            self.assertIsNone(_postprocess_training_wav(path, settings))
            quality = inspect_audio_item(
                Item(path, "hello world", "en", "voice_a", tuple("hello world")),
                {"maximum_edge_silence_seconds": 0.2},
            )
            self.assertTrue(quality["passed"])
            self.assertLessEqual(quality["metrics"]["leading_silence_seconds"], 0.101)
            self.assertLessEqual(quality["metrics"]["trailing_silence_seconds"], 0.101)

    def _base(self, root: Path, mode: str) -> tuple[Path, list]:
        texts = root / "texts.csv"
        with texts.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["text", "language"])
            writer.writeheader()
            writer.writerow({"text": "Hello world", "language": "en"})
            writer.writerow({"text": "Bonjour", "language": "fr"})
            writer.writerow({"text": "你好", "language": "zh"})
        voice = {"mode": mode, "speaker": "voice_a", "reference_text": "Exact reference text."}
        if mode == "design":
            voice.update({"prompt": "Warm and calm adult voice.", "reference_language": "en"})
        else:
            reference = root / "uploaded.wav"
            with wave.open(str(reference), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 100)
            voice["reference_audio"] = str(reference)
        config = {
            "experiment": {
                "name": f"sample-{mode}",
                "languages": ["en", "fr"],
                "dataset_root": str(root / "datasets"),
                "metadata": str(root / "datasets" / f"sample-{mode}" / "metadata.phonemes.csv"),
                "run_root": str(root / "runs"),
                "artifact_root": str(root / "artifacts"),
            },
            "audio": {"sample_rate": 8000},
            "generation": {
                "enabled": True,
                "qwen_runtime": "installed",
                "auto_download_models": False,
                "text_manifest": str(texts),
                "batch_size": 2,
                "models": {"voice_design": "voice-design-1.7b", "voice_clone": "base-1.7b"},
                "runtime": {"device": "cpu", "dtype": "float32", "attention": "sdpa"},
                "voice": voice,
            },
        }
        path = root / f"{mode}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        calls = []
        return path, calls

    def test_voice_design_then_clone_generates_pcm_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            config, calls = self._base(Path(directory), "design")

            def loader(key, **kwargs):
                calls.append(("load", key, kwargs))
                return FakeDesignModel(calls) if key == "voice-design-1.7b" else FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            report = validate_manifest(metadata, 8000)
            self.assertEqual(len(report.items), 2)
            self.assertEqual([call[0] for call in calls], ["load", "design", "load", "prompt", "clone"])
            self.assertEqual(calls[0][2]["runtime_mode"], "installed")
            self.assertIsNone(calls[0][2]["source_path"])
            self.assertTrue((metadata.parent / "references/voice_a.designed.wav").is_file())

            report.items[0].audio.unlink()
            resumed_calls = []

            def resumed_loader(key, **kwargs):
                resumed_calls.append(("load", key))
                self.assertEqual(key, "base-1.7b")
                return FakeCloneModel(resumed_calls)

            generate_samples(config, model_loader=resumed_loader)
            self.assertEqual([call[0] for call in resumed_calls], ["load", "prompt", "clone"])

    def test_uploaded_reference_uses_base_model_only(self):
        with tempfile.TemporaryDirectory() as directory:
            config, calls = self._base(Path(directory), "clone")

            def loader(key, **kwargs):
                calls.append(("load", key, kwargs))
                return FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            self.assertEqual([call[0] for call in calls], ["load", "prompt", "clone"])
            self.assertTrue((metadata.parent / "references/voice_a.uploaded.wav").is_file())
            with wave.open(str(validate_manifest(metadata, 8000).items[0].audio), "rb") as wav:
                self.assertEqual(wav.getframerate(), 8000)
                self.assertEqual(wav.getsampwidth(), 2)

    def test_new_speaker_can_merge_old_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "clone")
            old_audio = root / "old.wav"
            with wave.open(str(old_audio), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
                wav.writeframes(b"\0\0" * 100)
            old_metadata = root / "old.csv"
            with old_metadata.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
                writer.writeheader()
                writer.writerow({"audio": old_audio.name, "text": "Old voice", "language": "en",
                                 "speaker": "voice_old"})
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["generation"]["include_metadata"] = [str(old_metadata)]
            config.write_text(json.dumps(raw), encoding="utf-8")

            metadata = generate_samples(config, model_loader=lambda key, **kwargs: FakeCloneModel(calls))
            report = validate_manifest(metadata, 8000, require_single_speaker=False)
            self.assertEqual(len(report.items), 3)
            self.assertEqual({item.speaker for item in report.items}, {"voice_old", "voice_a"})


if __name__ == "__main__":
    unittest.main()
