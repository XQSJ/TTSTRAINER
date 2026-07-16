import csv
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import torch

from tts_trainer.checkpoints import load_training_checkpoint, save_training_checkpoint
from tts_trainer.vits import MultilingualVITS, VitsConfig, VitsDiscriminator
from tts_trainer.vits.data import slice_waveforms
from tts_trainer.vits.losses import discriminator_loss, generator_adversarial_loss
from tts_trainer.vits.trainer import train_vits
from tts_trainer.vits.trainer import (_load_expanded_generator,
                                      _resolve_frontend_contract)
from tts_trainer.vits.exporter import (PiperInferenceWrapper, export_vits_onnx,
                                       validate_onnx_runtime, voice_profiles)
from tts_trainer.vits.runtime import OnnxTTS
from tts_trainer.vits.validation import split_train_validation
from tts_trainer.frontend import frontend_contract_from_config
from tts_trainer.manifest import Item


def tiny_config():
    return VitsConfig(
        vocab_size=32, num_languages=7, num_speakers=3, spec_channels=9,
        hidden_channels=16, latent_channels=8, conditioning_channels=16,
        language_embedding_channels=4, speaker_embedding_channels=4,
        text_encoder_layers=1, text_encoder_heads=2, flow_layers=2,
        decoder_initial_channels=32, upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4), segment_frames=6,
    )


class VitsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = tiny_config()
        self.model = MultilingualVITS(self.config)

    def test_training_forward_and_backward(self):
        tokens = torch.tensor([[2, 4, 5, 3], [2, 7, 3, 0]])
        text_lengths = torch.tensor([4, 3])
        spectrogram = torch.randn(2, 9, 8)
        spec_lengths = torch.tensor([8, 7])
        output = self.model(tokens, text_lengths, spectrogram, spec_lengths,
                            torch.tensor([0, 1]), torch.tensor([0, 2]))
        self.assertEqual(output.audio.shape, (2, 1, 24))
        self.assertEqual(output.attention.shape, (2, 8, 4))
        (output.audio.abs().mean() + output.duration_loss).backward()
        self.assertIsNotNone(self.model.conditioning.speaker_embedding.weight.grad)

    def test_inference_uses_language_and_speaker_inputs(self):
        audio, lengths, attention = self.model.infer(
            torch.tensor([[2, 4, 3]]), torch.tensor([3]),
            torch.tensor([2]), torch.tensor([1]), max_frames=20,
        )
        self.assertEqual(audio.ndim, 3)
        self.assertEqual(audio.shape[-1], int(lengths.max()) * self.config.hop_length)
        self.assertEqual(attention.shape[2], 3)

    def test_piper_sid_splits_into_language_and_speaker(self):
        wrapper = PiperInferenceWrapper(self.model.eval())
        captured = {}
        original = self.model.infer_deploy
        def capture(tokens, lengths, language_ids, speaker_ids, scales):
            captured["language"] = language_ids.clone(); captured["speaker"] = speaker_ids.clone()
            return original(tokens, lengths, language_ids, speaker_ids, scales, max_frames=20)
        self.model.infer_deploy = capture
        output = wrapper(torch.tensor([[2, 3]]), torch.tensor([2]),
                         torch.tensor([0.0, 1.0, 1.0]), torch.tensor([9]))
        self.assertEqual(captured["language"].item(), 2)
        self.assertEqual(captured["speaker"].item(), 1)
        self.assertEqual(output.ndim, 3)

    def test_voice_profile_mapping(self):
        profiles = voice_profiles({"a": 0, "b": 1}, {"zh": 0, "en": 1})
        self.assertEqual([(p["sid"], p["speaker"], p["language"]) for p in profiles],
                         [(0, "a", "zh"), (1, "a", "en"),
                          (2, "b", "zh"), (3, "b", "en")])

    def test_discriminator(self):
        discriminator = VitsDiscriminator(periods=(2, 3))
        outputs = discriminator(torch.randn(2, 1, 64))
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(score.ndim == 2 and features for score, features in outputs))

    def test_losses_and_aligned_waveform_slice(self):
        discriminator = VitsDiscriminator(periods=(2,))
        real = slice_waveforms(torch.arange(40.0).view(1, 1, 40), torch.tensor([2]), 3, 4)
        self.assertEqual(real.flatten().tolist(), list(map(float, range(8, 20))))
        real_outputs = discriminator(real)
        fake_outputs = discriminator(torch.zeros_like(real))
        self.assertGreater(discriminator_loss(real_outputs, fake_outputs).item(), 0)
        self.assertGreater(generator_adversarial_loss(fake_outputs).item(), 0)

    def test_checkpoint_round_trip(self):
        discriminator = VitsDiscriminator(periods=(2,))
        optimizer_g = torch.optim.AdamW(self.model.parameters())
        optimizer_d = torch.optim.AdamW(discriminator.parameters())
        with tempfile.TemporaryDirectory() as directory:
            save_training_checkpoint(
                directory, generator=self.model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=2, global_step=12, config=self.config,
                language_map={"zh": 0}, speaker_map={"voice_01": 0},
                tokens=["_", "^", "$", " ", "<unk>"], metrics={"loss": 1.0},
            )
            restored = MultilingualVITS(self.config)
            result = load_training_checkpoint(directory, generator=restored)
            self.assertEqual(result["global_step"], 12)
            self.assertTrue(torch.equal(restored.conditioning.language_embedding.weight,
                                        self.model.conditioning.language_embedding.weight))

    def test_expand_speakers_and_vocabulary_preserves_old_embeddings(self):
        discriminator = VitsDiscriminator(periods=(2,))
        optimizer_g = torch.optim.AdamW(self.model.parameters())
        optimizer_d = torch.optim.AdamW(discriminator.parameters())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old"
            save_training_checkpoint(
                checkpoint, generator=self.model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=1, global_step=2, config=self.config,
                language_map={"en": 1}, speaker_map={"voice_01": 0, "voice_02": 1, "voice_03": 2},
                tokens=["_", "^", "$", " ", "<unk>"],
            )
            expanded_config = VitsConfig(**{
                **self.config.to_dict(), "vocab_size": self.config.vocab_size + 3,
                "num_speakers": self.config.num_speakers + 2,
            })
            expanded = MultilingualVITS(expanded_config)
            _load_expanded_generator(expanded, checkpoint)
            self.assertTrue(torch.equal(
                expanded.conditioning.speaker_embedding.weight[:self.config.num_speakers],
                self.model.conditioning.speaker_embedding.weight,
            ))
            self.assertTrue(torch.equal(
                expanded.text_encoder.embedding.weight[:self.config.vocab_size],
                self.model.text_encoder.embedding.weight,
            ))

    def test_resume_uses_checkpoint_frontend_when_lock_was_not_copied(self):
        previous_contract = frontend_contract_from_config({}, ("en",)).to_dict()
        previous_contract["languages"]["en"]["engine_version"] = "eSpeak NG frozen"
        with tempfile.TemporaryDirectory() as directory:
            result = _resolve_frontend_contract(
                {}, Path(directory) / "metadata.phonemes.csv", ("en",),
                {"frontend": previous_contract},
            )
        self.assertEqual(
            result["languages"]["en"]["engine_version"], "eSpeak NG frozen",
        )

    def test_validation_split_is_deterministic_and_stratified(self):
        items = [
            Item(Path(f"{language}-{index}.wav"), f"text {index}", language,
                 "voice_01", ("a",))
            for language in ("en", "fr") for index in range(4)
        ]
        first = split_train_validation(
            items, fraction=0.25, seed=7, minimum_per_profile=1,
        )
        second = split_train_validation(
            list(reversed(items)), fraction=0.25, seed=7, minimum_per_profile=1,
        )
        self.assertEqual(first[2]["validation_fingerprint"], second[2]["validation_fingerprint"])
        self.assertEqual(len(first[0]), 6)
        self.assertEqual(len(first[1]), 2)

    def test_onnx_export_and_runtime(self):
        discriminator = VitsDiscriminator(periods=(2,))
        optimizer_g = torch.optim.AdamW(self.model.parameters())
        optimizer_d = torch.optim.AdamW(discriminator.parameters())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            save_training_checkpoint(
                checkpoint, generator=self.model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=1, global_step=1, config=self.config,
                language_map={"zh": 0, "en": 1, "ja": 2, "ko": 3,
                              "fr": 4, "es": 5, "pt": 6},
                speaker_map={"voice_01": 0, "voice_02": 1, "voice_03": 2},
                tokens=["_", "^", "$", " ", "<unk>", "a"],
                frontend=frontend_contract_from_config(
                    {}, ("zh", "en", "ja", "ko", "fr", "es", "pt"),
                    engine_version="eSpeak NG test",
                ).to_dict(),
                frontend_conformance={
                    "format": 1,
                    "cases_per_language": 1,
                    "languages": ["en"],
                    "cases": [{
                        "language": "en", "language_id": 1, "text": "a",
                        "phonemes": ["a"], "token_ids": [1, 5, 2],
                    }],
                },
            )
            target = export_vits_onnx(checkpoint, Path(directory) / "export", sample_rate=8000)
            self.assertTrue(target.is_file())
            frontend = json.loads((target.parent / "frontend.json").read_text(encoding="utf-8"))
            self.assertEqual(frontend["engine_version"], "eSpeak NG test")
            self.assertEqual(frontend["provider"], "language-router")
            self.assertEqual(frontend["languages"]["ja"]["provider"], "openjtalk")
            self.assertEqual(frontend["languages"]["en"]["provider"], "espeak-ng")
            self.assertEqual(list(target.parent.glob("*.onnx")), [target])
            self.assertTrue((target.parent / "frontend.conformance.json").is_file())
            shape = validate_onnx_runtime(target)
            self.assertEqual(shape[0:2], (1, 1))
            runtime = OnnxTTS(target.parent)
            audio = runtime.synthesize_units(("a",), language="en", speaker="voice_02", noise_scale=0.0)
            self.assertGreater(audio.shape[0], 0)

    def test_one_step_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            samples = [int(math.sin(i / 8) * 8000) for i in range(192)]
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
                stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
                writer.writeheader(); writer.writerow({"audio": audio.name, "text": "hello", "language": "en", "speaker": "voice_01"})
            config = {
                "experiment": {"name": "tiny-en", "languages": ["en"]},
                "model": tiny_config().to_dict(),
                "audio": {"sample_rate": 8000, "n_fft": 16, "hop_length": 4, "win_length": 16, "n_mels": 4},
                "frontend": {"require_phonemes": False},
                "training": {"batch_size": 1, "learning_rate_generator": 0.0002,
                             "learning_rate_discriminator": 0.0002, "epochs": 1,
                             "checkpoint_every_steps": 50, "seed": 7},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint = train_vits(str(config_path), str(metadata), str(root / "run"),
                                    device_name="cpu", max_steps=1)
            self.assertTrue((checkpoint / "training-state.pt").is_file())
            saved = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["language_map"], {"en": 0})
            self.assertEqual(saved["config"]["num_languages"], 1)

    def test_validation_creates_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(2):
                audio = root / f"sample-{index}.wav"
                samples = [int(math.sin((i + index) / 8) * 8000) for i in range(192)]
                with wave.open(str(audio), "wb") as stream:
                    stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
                    stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
                rows.append({"audio": audio.name, "text": f"hello {index}",
                             "language": "en", "speaker": "voice_01"})
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["audio", "text", "language", "speaker"],
                )
                writer.writeheader(); writer.writerows(rows)
            config = {
                "experiment": {"name": "tiny-best", "languages": ["en"]},
                "model": tiny_config().to_dict(),
                "audio": {"sample_rate": 8000, "n_fft": 16, "hop_length": 4,
                          "win_length": 16, "n_mels": 4},
                "frontend": {"require_phonemes": False},
                "validation": {"enabled": True, "fraction": 0.5,
                               "minimum_per_profile": 1, "batch_size": 1,
                               "metric": "mel", "seed": 7},
                "training": {"batch_size": 1, "learning_rate_generator": 0.0002,
                             "learning_rate_discriminator": 0.0002, "epochs": 1,
                             "checkpoint_every_steps": 50, "seed": 7},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            last = train_vits(
                str(config_path), str(metadata), str(root / "run"),
                device_name="cpu", max_steps=1,
            )
            best = last.parent / "best"
            self.assertTrue((best / "training-state.pt").is_file())
            saved = json.loads((best / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selection"]["best_epoch"], 1)
            self.assertIn("validation", saved["metrics"])
            self.assertTrue((root / "run" / "splits" / "validation.csv").is_file())


if __name__ == "__main__":
    unittest.main()
