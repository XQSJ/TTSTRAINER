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
from tts_trainer.sample_generation import (_identity_digest,
                                           _postprocess_training_wav,
                                           _voice_identity,
                                           generate_samples)
from tts_trainer.experiments import resolve_experiment


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
        if isinstance(kwargs["ref_audio"], Path):
            raise TypeError("Qwen does not accept pathlib.Path reference inputs")
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
        voice = {
            "id": "shared_voice_a", "mode": mode, "speaker": "voice_a",
            "reference_text": "Exact reference text.",
        }
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
            self.assertEqual(
                [call[0] for call in calls],
                ["load", "design", "load", "prompt", "clone", "clone"],
            )
            self.assertEqual(calls[0][2]["runtime_mode"], "installed")
            self.assertIsNone(calls[0][2]["source_path"])
            voice_root = metadata.parent.parent / "voices/shared_voice_a"
            self.assertTrue((voice_root / "voice.json").is_file())
            with (voice_root / "manifest.csv").open(
                newline="", encoding="utf-8",
            ) as stream:
                self.assertNotIn("speaker", next(csv.DictReader(stream)).keys())
            self.assertTrue((voice_root / "references/designed.wav").is_file())
            dataset = json.loads((metadata.parent / "dataset.json").read_text())
            self.assertEqual(dataset["voice_id"], "shared_voice_a")
            self.assertNotIn("voice_revision", dataset)
            self.assertEqual(Path(dataset["voice_dataset"]), voice_root.resolve())

            report.items[0].audio.unlink()
            resumed_calls = []

            def resumed_loader(key, **kwargs):
                resumed_calls.append(("load", key))
                self.assertEqual(key, "base-1.7b")
                return FakeCloneModel(resumed_calls)

            generate_samples(config, model_loader=resumed_loader)
            self.assertEqual([call[0] for call in resumed_calls], ["load", "prompt", "clone"])
            self.assertIsInstance(resumed_calls[1][1]["ref_audio"], str)

    def test_uploaded_reference_uses_base_model_only(self):
        with tempfile.TemporaryDirectory() as directory:
            config, calls = self._base(Path(directory), "clone")

            def loader(key, **kwargs):
                calls.append(("load", key, kwargs))
                return FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            self.assertEqual(
                [call[0] for call in calls],
                ["load", "prompt", "clone", "clone"],
            )
            self.assertIsInstance(calls[1][1]["ref_audio"], str)
            voice_root = metadata.parent.parent / "voices/shared_voice_a"
            self.assertTrue((voice_root / "references/uploaded.wav").is_file())
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

    def test_expand_speakers_automatically_reuses_checkpoint_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, old_calls = self._base(root, "clone")
            old_metadata = generate_samples(
                config, model_loader=lambda key, **kwargs: FakeCloneModel(old_calls),
            )
            old_layout = root / "runs/old-model"
            checkpoint = old_layout / "checkpoints/last"
            checkpoint.mkdir(parents=True)
            (old_layout / "run-layout.json").write_text(json.dumps({
                "dataset_dir": str(old_metadata.parent),
                "metadata": str(old_metadata.parent / "metadata.phonemes.csv"),
            }), encoding="utf-8")

            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["experiment"]["name"] = "expanded-model"
            raw["experiment"]["initialization"] = {
                "mode": "expand_speakers",
                "checkpoint": str(checkpoint),
            }
            raw["generation"]["voice"]["id"] = "shared_voice_b"
            raw["generation"]["voice"]["speaker"] = "voice_b"
            expanded_config = root / "expanded.json"
            expanded_config.write_text(json.dumps(raw), encoding="utf-8")
            new_calls = []

            metadata = generate_samples(
                expanded_config,
                model_loader=lambda key, **kwargs: FakeCloneModel(new_calls),
            )
            report = validate_manifest(metadata, 8000, require_single_speaker=False)
            self.assertEqual(len(report.items), 4)
            self.assertEqual({item.speaker for item in report.items}, {"voice_a", "voice_b"})

    def test_same_voice_dataset_is_reused_by_different_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_config, calls = self._base(root, "design")

            def loader(key, **kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" else FakeCloneModel(calls)

            first_metadata = generate_samples(first_config, model_loader=loader)
            first_audio = tuple(item.audio for item in validate_manifest(first_metadata, 8000).items)

            second_raw = json.loads(first_config.read_text(encoding="utf-8"))
            second_raw["experiment"]["name"] = "another-model"
            second_raw["experiment"]["metadata"] = str(
                root / "datasets/another-model/metadata.phonemes.csv"
            )
            second_config = root / "another-model.json"
            second_config.write_text(json.dumps(second_raw), encoding="utf-8")

            def must_not_load(*_args, **_kwargs):
                self.fail("the shared voice WAV cache should avoid loading Qwen")

            second_metadata = generate_samples(second_config, model_loader=must_not_load)
            second_audio = tuple(item.audio for item in validate_manifest(second_metadata, 8000).items)
            self.assertEqual(first_audio, second_audio)
            self.assertNotEqual(first_metadata.parent, second_metadata.parent)

    def test_model_assigns_speakers_without_copying_shared_voice_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            generate_samples(config, model_loader=loader)
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["experiment"]["name"] = "voice-b-data"
            raw["generation"]["voice"]["id"] = "shared_voice_b"
            raw["generation"]["voice"].pop("speaker")
            voice_b_config = root / "voice-b.json"
            voice_b_config.write_text(json.dumps(raw), encoding="utf-8")
            generate_samples(voice_b_config, model_loader=loader)

            raw["experiment"]["name"] = "two-speaker-model"
            raw["generation"].pop("voice")
            raw["generation"].pop("text_manifest")
            raw["generation"]["speaker_assignments"] = {
                "gentle": "shared_voice_a",
                "bright": "shared_voice_b",
            }
            model_config = root / "two-speaker.json"
            model_config.write_text(json.dumps(raw), encoding="utf-8")

            metadata = generate_samples(
                model_config,
                model_loader=lambda *_args, **_kwargs: self.fail(
                    "both shared voice datasets should be cached"
                ),
            )
            report = validate_manifest(
                metadata, 8000, require_single_speaker=False,
            )
            self.assertEqual(
                {item.speaker for item in report.items},
                {"gentle", "bright"},
            )
            self.assertTrue(all("/datasets/voices/" in str(item.audio)
                                for item in report.items))

    def test_reducing_languages_reuses_audio_and_rewrites_metadata_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            original_audio = tuple(
                item.audio for item in validate_manifest(metadata, 8000).items
            )

            reduced_texts = root / "reduced-texts.csv"
            with reduced_texts.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "language"])
                writer.writeheader()
                writer.writerow({"text": "Hello world", "language": "en"})
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["experiment"]["languages"] = ["en"]
            raw["generation"]["text_manifest"] = str(reduced_texts)
            config.write_text(json.dumps(raw), encoding="utf-8")

            def must_not_load(*_args, **_kwargs):
                self.fail("reduced selection should reuse existing voice WAV")

            reduced_metadata = generate_samples(config, model_loader=must_not_load)
            reduced_report = validate_manifest(reduced_metadata, 8000)
            self.assertEqual(len(reduced_report.items), 1)
            self.assertEqual(reduced_report.items[0].language, "en")
            self.assertTrue(all(path.is_file() for path in original_audio))

    def test_increasing_languages_and_count_appends_only_missing_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, first_calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["experiment"]["languages"] = ["en"]
            first_texts = root / "first.csv"
            with first_texts.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "language"])
                writer.writeheader()
                writer.writerow({"text": "Hello world", "language": "en"})
            raw["generation"]["text_manifest"] = str(first_texts)
            config.write_text(json.dumps(raw), encoding="utf-8")

            def first_loader(key, **_kwargs):
                return FakeDesignModel(first_calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(first_calls)

            first_metadata = generate_samples(config, model_loader=first_loader)
            original = validate_manifest(first_metadata, 8000).items[0].audio
            original_hash = original.read_bytes()

            expanded_texts = root / "expanded.csv"
            with expanded_texts.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "language"])
                writer.writeheader()
                writer.writerow({"text": "Hello world", "language": "en"})
                writer.writerow({"text": "A second sentence", "language": "en"})
                writer.writerow({"text": "Bonjour", "language": "fr"})
            raw["experiment"]["languages"] = ["en", "fr"]
            raw["generation"]["text_manifest"] = str(expanded_texts)
            config.write_text(json.dumps(raw), encoding="utf-8")
            expanded_calls = []

            def expanded_loader(key, **_kwargs):
                self.assertEqual(key, "base-1.7b")
                return FakeCloneModel(expanded_calls)

            expanded_metadata = generate_samples(config, model_loader=expanded_loader)
            expanded = validate_manifest(expanded_metadata, 8000)
            self.assertEqual(len(expanded.items), 3)
            self.assertEqual(
                [call[0] for call in expanded_calls],
                ["prompt", "clone", "clone"],
            )
            self.assertEqual(
                [call[1]["language"] for call in expanded_calls[1:]],
                [["English"], ["French"]],
            )
            self.assertEqual(original.read_bytes(), original_hash)
            self.assertEqual(
                Path(json.loads((expanded_metadata.parent / "dataset.json").read_text())[
                    "voice_dataset"
                ]),
                (root / "datasets/voices/shared_voice_a").resolve(),
            )

    def test_voice_text_pool_is_limited_to_requested_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            texts = root / "datasets/voices/shared_voice_a/texts.csv"
            texts.parent.mkdir(parents=True)
            with texts.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "language"])
                writer.writeheader()
                for index in range(3):
                    writer.writerow({
                        "text": f"English sentence number {index}.", "language": "en",
                    })
            raw["experiment"]["languages"] = ["en"]
            raw["text_generation"] = {
                "enabled": True, "provider": "builtin",
                "sentences_per_language": 1,
            }
            raw["generation"]["text_manifest"] = None
            config.write_text(json.dumps(raw), encoding="utf-8")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            self.assertEqual(len(validate_manifest(metadata, 8000).items), 1)

    def test_generate_samples_automatically_prepares_configured_texts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["experiment"]["languages"] = ["en"]
            raw["generation"].pop("text_manifest")
            raw["text_generation"] = {
                "enabled": True,
                "provider": "builtin",
                "sentences_per_language": 2,
            }
            config.write_text(json.dumps(raw), encoding="utf-8")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            report = validate_manifest(metadata, 8000)
            self.assertEqual(len(report.items), 2)
            self.assertTrue(
                (root / "datasets/voices/shared_voice_a/texts.csv").is_file()
            )

    def test_one_config_prepares_multiple_new_voices_and_assembles_speakers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["task"] = "train"
            raw["experiment"]["languages"] = ["en"]
            raw["generation"].pop("text_manifest")
            raw["generation"].pop("voice")
            raw["generation"]["voices"] = {
                "warm_voice": {
                    "id": "warm_voice",
                    "mode": "design",
                    "prompt": "Warm and calm adult voice.",
                    "reference_text": "Exact reference text.",
                    "reference_language": "en",
                },
                "bright_voice": {
                    "id": "bright_voice",
                    "mode": "design",
                    "prompt": "Bright and friendly adult voice.",
                    "reference_text": "Exact reference text.",
                    "reference_language": "en",
                },
            }
            raw["generation"]["speaker_assignments"] = {
                "gentle": "warm_voice",
                "bright": "bright_voice",
            }
            raw["text_generation"] = {
                "enabled": True,
                "provider": "builtin",
                "sentences_per_language": 1,
            }
            config.write_text(json.dumps(raw), encoding="utf-8")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            metadata = generate_samples(config, model_loader=loader)
            report = validate_manifest(
                metadata, 8000, require_single_speaker=False,
            )
            self.assertEqual(len(report.items), 2)
            self.assertEqual(
                {item.speaker for item in report.items}, {"gentle", "bright"},
            )
            self.assertTrue(
                (root / "datasets/voices/warm_voice/manifest.csv").is_file()
            )
            self.assertTrue(
                (root / "datasets/voices/bright_voice/manifest.csv").is_file()
            )
            raw["task"] = "prepare"
            raw["generation"]["speaker_assignments"] = {}
            config.write_text(json.dumps(raw), encoding="utf-8")

            def must_not_load(*_args, **_kwargs):
                self.fail("prepared voices must be reused without loading Qwen")

            summary = generate_samples(config, model_loader=must_not_load)
            prepared = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(summary.name, "prepared-voices.json")
            self.assertEqual(
                set(prepared["voices"]), {"warm_voice", "bright_voice"},
            )

    def test_per_language_design_references_and_language_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["generation"]["voice"]["reference_strategy"] = "per_language"
            config.write_text(json.dumps(raw), encoding="utf-8")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            generate_samples(config, model_loader=loader)
            design_languages = [
                call[1]["language"] for call in calls if call[0] == "design"
            ]
            clone_languages = [
                call[1]["language"] for call in calls if call[0] == "clone"
            ]
            self.assertEqual(design_languages, ["English", "French"])
            self.assertEqual(clone_languages, [["English"], ["French"]])
            references = root / "datasets/voices/shared_voice_a/references"
            self.assertTrue((references / "designed-en.wav").is_file())
            self.assertTrue((references / "designed-fr.wav").is_file())
            self.assertEqual(
                (references / "designed-fr.txt").read_text(encoding="utf-8"),
                "Bonjour",
            )

    def test_cascade_design_localizes_master_reference_before_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["generation"]["voice"]["reference_strategy"] = "cascade"
            config.write_text(json.dumps(raw), encoding="utf-8")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            generate_samples(config, model_loader=loader)
            references = root / "datasets/voices/shared_voice_a/references"
            self.assertTrue((references / "designed.wav").is_file())
            self.assertTrue((references / "localized-en.wav").is_file())
            self.assertTrue((references / "localized-fr.wav").is_file())
            self.assertEqual(
                (references / "localized-en.txt").read_text(encoding="utf-8"),
                "Exact reference text.",
            )
            self.assertEqual(
                (references / "localized-fr.txt").read_text(encoding="utf-8"),
                "Bonjour",
            )
            clone_calls = [
                call[1] for call in calls if call[0] == "clone"
            ]
            self.assertEqual(
                [call["language"] for call in clone_calls],
                [["French"], ["English"], ["French"]],
            )
            self.assertEqual(clone_calls[0]["text"], ["Bonjour"])
            prompt_calls = [
                call[1] for call in calls if call[0] == "prompt"
            ]
            self.assertEqual(len(prompt_calls), 3)

            next((root / "datasets/voices/shared_voice_a/wavs/fr").glob("*.wav")).unlink()
            resumed_calls = []

            def resumed_loader(key, **_kwargs):
                self.assertEqual(key, "base-1.7b")
                return FakeCloneModel(resumed_calls)

            generate_samples(config, model_loader=resumed_loader)
            self.assertEqual(
                [call[0] for call in resumed_calls],
                ["prompt", "clone"],
            )
            self.assertEqual(
                resumed_calls[-1][1]["language"], ["French"],
            )

    def test_prepare_voice_does_not_write_model_dataset_or_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["task"] = "prepare"
            config.write_text(json.dumps(raw), encoding="utf-8")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            output = generate_samples(config, model_loader=loader)
            self.assertEqual(output.name, "manifest.csv")
            self.assertTrue(output.is_file())
            self.assertTrue(
                (output.parent / "audio-postprocess-report.json").is_file()
            )
            self.assertFalse((root / "datasets/sample-design").exists())
            self.assertFalse((root / "artifacts/sample-design").exists())
            self.assertFalse((root / "runs/sample-design/checkpoints").exists())
            self.assertFalse((root / "runs/sample-design/logs").exists())

    def test_voice_regenerate_audio_overrides_cache_per_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            generate_samples(config, model_loader=loader)
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["generation"]["voice"]["regenerate_audio"] = True
            config.write_text(json.dumps(raw), encoding="utf-8")
            regenerated_calls = []

            def regenerated_loader(key, **_kwargs):
                return FakeCloneModel(regenerated_calls)

            generate_samples(config, model_loader=regenerated_loader)
            self.assertEqual(
                [call[0] for call in regenerated_calls],
                ["prompt", "clone", "clone"],
            )

    def test_qwen_device_inherits_experiment_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "clone")
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["experiment"]["device"] = "cuda:1"
            raw["generation"]["runtime"].pop("device")
            config.write_text(json.dumps(raw), encoding="utf-8")
            load_kwargs = []

            def loader(key, **kwargs):
                load_kwargs.append(kwargs)
                return FakeCloneModel(calls)

            generate_samples(config, model_loader=loader)
            self.assertEqual(load_kwargs[0]["device_map"], "cuda:1")

    def test_same_voice_id_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, calls = self._base(root, "design")

            def loader(key, **_kwargs):
                return FakeDesignModel(calls) if key == "voice-design-1.7b" \
                    else FakeCloneModel(calls)

            generate_samples(config, model_loader=loader)
            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["generation"]["voice"]["prompt"] = "A completely different voice."
            config.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already locked"):
                generate_samples(
                    config,
                    model_loader=lambda *_args, **_kwargs: self.fail("must not load Qwen"),
                )

    def test_legacy_revision_cache_moves_to_voice_root_with_compatibility_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._base(root, "clone")
            raw, layout = resolve_experiment(config)
            generation = raw["generation"]
            identity = _voice_identity(raw, generation, generation["voice"])
            revision = _identity_digest(identity)[:12]
            legacy = root / "datasets/voices/shared_voice_a" / revision
            (legacy / "references").mkdir(parents=True)
            (legacy / "wavs/en").mkdir(parents=True)
            (legacy / "voice.json").write_text(json.dumps({
                "format": 1,
                "voice_id": "shared_voice_a",
                "identity": identity,
            }), encoding="utf-8")
            sf.write(
                legacy / "wavs/en/existing.wav",
                np.linspace(-0.1, 0.1, 80, dtype=np.float32),
                8000,
                subtype="PCM_16",
            )

            from tts_trainer.sample_generation import _voice_dataset
            voice_id, voice_root, returned_identity = _voice_dataset(
                raw, layout, generation, generation["voice"],
            )
            self.assertEqual(voice_id, "shared_voice_a")
            self.assertEqual(returned_identity, identity)
            self.assertTrue((voice_root / "voice.json").is_file())
            self.assertTrue((voice_root / "wavs/en/existing.wav").is_file())
            self.assertTrue((legacy / "wavs/en/existing.wav").is_file())

    def test_legacy_model_wavs_are_migrated_to_shared_voice_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._base(root, "clone")
            legacy = root / "datasets/sample-clone/wavs/voice_a"
            legacy.mkdir(parents=True)
            for index, language in enumerate(("en", "fr"), 1):
                sf.write(
                    legacy / f"{language}_{index:06d}_c01.wav",
                    np.linspace(-0.1, 0.1, 80, dtype=np.float32), 8000,
                    subtype="PCM_16",
                )

            def must_not_load(*_args, **_kwargs):
                self.fail("legacy WAVs should migrate without loading Qwen")

            metadata = generate_samples(config, model_loader=must_not_load)
            report = validate_manifest(metadata, 8000)
            self.assertEqual(len(report.items), 2)
            self.assertTrue(all("/voices/shared_voice_a/" in str(item.audio) for item in report.items))


if __name__ == "__main__":
    unittest.main()
