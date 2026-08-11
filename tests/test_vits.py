import csv
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tts_trainer.checkpoints import (inherit_resume_best_checkpoint,
                                     load_training_checkpoint,
                                     require_checkpoint_format,
                                     require_warm_start_checkpoint_format,
                                     save_training_checkpoint)
from tts_trainer.vits import MultilingualVITS, VitsConfig, VitsDiscriminator
from tts_trainer.vits.model import integer_durations
from tts_trainer.vits.data import (AudioConfig, LengthBucketBatchSampler,
                                   inspect_alignment_item, slice_waveforms)
from tts_trainer.vits.losses import (discriminator_loss,
                                     generator_adversarial_loss, kl_loss)
from tts_trainer.vits.modules import maximum_path, sinusoidal_position_encoding
from tts_trainer.vits.trainer import (_semantic_reference_root,
                                      acoustic_refinement_gate,
                                      configure_training_stage,
                                      duration_predictor_settings,
                                      preserve_checkpoint_duration_architecture,
                                      profile_balancing_weights,
                                      refinement_mel_weight,
                                      reject_removed_prior_options,
                                      resolve_mixed_precision,
                                      resolve_refinement_config, train_vits)
from tts_trainer.vits.trainer import (_load_expanded_generator,
                                      _load_warm_start_generator,
                                      _resolve_frontend_contract)
from tts_trainer.vits.exporter import (PiperInferenceWrapper, export_vits_onnx,
                                       require_mobile_blank_semantics,
                                       validate_onnx_runtime, voice_profiles,
                                       _export_sherpa_android_text_package)
from tts_trainer.vits.runtime import OnnxTTS
from tts_trainer.vits.validation import split_train_validation
from tts_trainer.frontend import FrontendContract, frontend_contract_from_config
from tts_trainer.frontend.contract import (LEGACY_PIPER_TOKEN_ENCODING,
                                           MOBILE_DIRECT_TOKEN_ENCODING,
                                           PIPER_TOKEN_ENCODING)
from tts_trainer.manifest import Item
from tts_trainer.text import Vocabulary


def tiny_config():
    return VitsConfig(
        vocab_size=32, num_languages=7, num_speakers=3, spec_channels=9,
        hidden_channels=16, latent_channels=8, conditioning_channels=16,
        language_embedding_channels=4, speaker_embedding_channels=4,
        text_encoder_layers=1, text_encoder_heads=2, flow_layers=2,
        decoder_initial_channels=32, upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4), segment_frames=6,
    )


class RuntimeFrontend:
    def phonemize(self, text, language):
        del language
        return tuple(character for character in text if not character.isspace())


class RuntimeChunkingTests(unittest.TestCase):
    def test_text_runtime_synthesizes_chunks_and_inserts_clause_pause(self):
        runtime = OnnxTTS.__new__(OnnxTTS)
        runtime.frontend_contract = None
        runtime.sample_rate = 1000
        with patch.object(
            runtime, "synthesize_units",
            side_effect=lambda units, **kwargs: np.ones(len(units), dtype=np.float32),
        ) as synthesize:
            audio = runtime.synthesize_text(
                "abcd, efgh.", language="en", speaker="voice_01",
                frontend=RuntimeFrontend(), max_phoneme_tokens=8,
                clause_pause_ms=100, sentence_pause_ms=180,
            )
        self.assertEqual(synthesize.call_count, 2)
        self.assertEqual(audio.shape, (110,))
        np.testing.assert_array_equal(audio[5:105], np.zeros(100, dtype=np.float32))

    def test_text_runtime_can_disable_automatic_chunking(self):
        runtime = OnnxTTS.__new__(OnnxTTS)
        runtime.frontend_contract = None
        with patch.object(
            runtime, "synthesize_units", return_value=np.ones(3, dtype=np.float32),
        ) as synthesize:
            runtime.synthesize_text(
                "abcd, efgh.", language="en", speaker="voice_01",
                frontend=RuntimeFrontend(), auto_chunk=False,
            )
        self.assertEqual(synthesize.call_count, 1)


class VitsTests(unittest.TestCase):
    def test_acoustic_refinement_gate_requires_every_profile(self):
        config = {
            "posterior_mel_threshold": 0.45,
            "full_posterior_mel_threshold": 0.45,
            "require_all_profiles": True,
        }
        metrics = {
            "mel": 0.40,
            "profiles": {
                "en/voice": {
                    "mel": 0.42,
                    "posterior_mean_full_mel": 0.41,
                    "posterior_sampled_full_mel": 0.43,
                },
                "zh/voice": {
                    "mel": 0.39,
                    "posterior_mean_full_mel": 0.40,
                    "posterior_sampled_full_mel": 0.48,
                },
            },
        }
        ready, failures = acoustic_refinement_gate(metrics, config)
        self.assertFalse(ready)
        self.assertTrue(any("zh/voice" in failure for failure in failures))
        metrics["profiles"]["zh/voice"]["posterior_sampled_full_mel"] = 0.44
        self.assertEqual(acoustic_refinement_gate(metrics, config), (True, []))

    def test_standard_stage_disables_automatic_refinement(self):
        config = resolve_refinement_config({
            "stage": "standard",
            "text_prior_refinement": {"enabled": True},
        })
        self.assertEqual(config["stage"], "standard")
        self.assertFalse(config["enabled"])

    def test_mixed_precision_is_disabled_off_cuda(self):
        self.assertEqual(
            resolve_mixed_precision({"mixed_precision": "auto"}, torch.device("cpu")),
            ("fp32", None),
        )

    def test_mixed_precision_auto_prefers_bf16_and_falls_back_to_fp32(self):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(
                resolve_mixed_precision({}, torch.device("cuda")),
                ("bf16", torch.bfloat16),
            )
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(
                resolve_mixed_precision({}, torch.device("cuda")),
                ("fp32", None),
            )
            self.assertEqual(
                resolve_mixed_precision(
                    {"mixed_precision": "fp16"}, torch.device("cuda"),
                ),
                ("fp16", torch.float16),
            )

    def test_explicit_unsupported_bf16_is_rejected(self):
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "BF16 support"):
                resolve_mixed_precision(
                    {"mixed_precision": "bf16"}, torch.device("cuda"),
                )

    def test_resume_inherits_historical_best_into_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "source" / "checkpoints"
            best = checkpoints / "best"
            last = checkpoints / "last"
            best.mkdir(parents=True)
            last.mkdir(parents=True)
            selection = {
                "metric": "combined_mel",
                "best_epoch": 20,
                "best_value": 0.9,
                "mode": "min",
            }
            (best / "metadata.json").write_text(
                json.dumps({"epoch": 20, "selection": selection}),
                encoding="utf-8",
            )
            (best / "training-state.pt").write_bytes(b"best-state")
            (last / "metadata.json").write_text(
                json.dumps({"epoch": 50, "selection": selection}),
                encoding="utf-8",
            )
            (last / "training-state.pt").write_bytes(b"last-state")

            inherited = inherit_resume_best_checkpoint(
                last, root / "resumed" / "checkpoints" / "best", selection,
            )

            self.assertIsNotNone(inherited)
            self.assertEqual(
                (inherited / "training-state.pt").read_bytes(), b"best-state",
            )
            metadata = json.loads(
                (inherited / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["epoch"], 20)

    def test_refinement_config_and_mel_warmup(self):
        config = resolve_refinement_config({
            "lr_decay": 0.9,
            "text_prior_refinement": {
                "enabled": True,
                "mel_weight": 2.0,
                "mel_warmup_steps": 10,
            },
        })
        self.assertTrue(config["enabled"])
        self.assertEqual(config["lr_decay"], 0.9)
        self.assertEqual(refinement_mel_weight(config, 0), 0.0)
        self.assertEqual(refinement_mel_weight(config, 5), 1.0)
        self.assertEqual(refinement_mel_weight(config, 10), 2.0)

    def test_removed_prior_loss_options_fail_instead_of_being_ignored(self):
        for key in (
            "aligned_prior_mel_weight",
            "aligned_prior_mel_start_steps",
            "aligned_prior_mel_warmup_steps",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "已移除|removed",
            ):
                reject_removed_prior_options({key: 1})

    def test_profile_sampler_balances_observed_language_speaker_pairs(self):
        items = [
            *[
                Item(Path(f"en-a-{index}.wav"), "a", "en", "a", ("a",))
                for index in range(4)
            ],
            Item(Path("en-b.wav"), "b", "en", "b", ("b",)),
            Item(Path("fr-a.wav"), "c", "fr", "a", ("c",)),
        ]
        weights = profile_balancing_weights(items)
        masses = {}
        for item, weight in zip(items, weights):
            key = (item.language, item.speaker)
            masses[key] = masses.get(key, 0.0) + weight
        self.assertEqual(masses, {
            ("en", "a"): 1.0,
            ("en", "b"): 1.0,
            ("fr", "a"): 1.0,
        })

    def test_length_bucket_sampler_reduces_batch_spread(self):
        torch.manual_seed(7)
        lengths = list(range(1, 65))
        batches = list(LengthBucketBatchSampler(
            [1.0] * len(lengths), lengths, batch_size=4, pool_batches=16,
        ))
        self.assertEqual(len(batches), 16)
        self.assertEqual(sum(map(len, batches)), len(lengths))
        spreads = [
            max(lengths[index] for index in batch)
            - min(lengths[index] for index in batch)
            for batch in batches
        ]
        self.assertLess(sum(spreads) / len(spreads), 10)

    def test_length_bucket_sampler_retains_weighted_replacement(self):
        torch.manual_seed(11)
        sampler = LengthBucketBatchSampler(
            [0.0, 0.0, 1.0], [100, 200, 300], batch_size=2,
        )
        self.assertEqual(
            [index for batch in sampler for index in batch], [2, 2, 2],
        )

    def test_alignment_gate_detects_piper_token_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "short.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(8000)
                stream.writeframes(struct.pack("<32h", *([0] * 32)))
            item = Item(
                audio, "abc", "en", "voice_01", ("a", "b", "c"),
            )
            result = inspect_alignment_item(
                item, Vocabulary.build([item]),
                AudioConfig(
                    sample_rate=8000, n_fft=16, hop_length=4,
                    win_length=16, n_mels=4,
                ),
                piper_compatible=True,
            )
            self.assertEqual(result["audio_frames"], 5)
            self.assertEqual(result["text_tokens"], 9)
            self.assertEqual(result["frame_deficit"], 4)
            self.assertFalse(result["passed"])

    def test_mobile_export_writes_per_language_sherpa_models(self):
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "source-espeak"
            data.mkdir()
            (data / "phontab").write_bytes(b"test")
            model = onnx.helper.make_model(
                onnx.helper.make_graph([], "empty", [], []),
            )
            frontend = frontend_contract_from_config(
                {
                    "provider": "espeak-ng",
                    "mobile_direct": True,
                    "voices": {"en": "en-us", "fr": "fr-fr"},
                },
                ("en", "fr"),
            ).to_dict()
            profiles = voice_profiles(
                {"voice": 0}, {"en": 0, "fr": 1},
            )
            with patch(
                "tts_trainer.vits.exporter._find_espeak_data_dir",
                return_value=data,
            ):
                result = _export_sherpa_android_text_package(
                    onnx, model, root / "export", frontend, profiles,
                    sample_rate=22050,
                    tokens=["_", "^", "$", " ", "<unk>", "a"],
                )
            self.assertTrue(result["supported"])
            self.assertTrue(
                (root / "export/android_text/model-en.onnx").is_file(),
            )
            self.assertTrue(
                (root / "export/android_text/model-fr.onnx").is_file(),
            )
            metadata = {
                item.key: item.value for item in onnx.load(
                    root / "export/android_text/model-fr.onnx",
                ).metadata_props
            }
            self.assertEqual(metadata["voice"], "fr-fr")
            self.assertEqual(metadata["add_blank"], "0")
            self.assertTrue(
                (root / "export/android_text/espeak-ng-data/phontab").is_file(),
            )
            self.assertEqual(
                (root / "export/android_text/tokens.txt").read_text(
                    encoding="utf-8",
                ),
                "_ 0\n^ 1\n$ 2\n3\na 5\n",
            )

    def test_legacy_noise_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "untrained text prior"):
            require_checkpoint_format(1)
        with self.assertRaisesRegex(ValueError, "position-free text encoder"):
            require_checkpoint_format(2)
        with self.assertRaisesRegex(ValueError, "deterministic duration predictor"):
            require_checkpoint_format(3)
        require_warm_start_checkpoint_format(3)

    def test_semantic_quality_uses_shared_voice_reference_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "model_a"
            shared_voice = root / "datasets" / "voices" / "warm_girl" / "revision_a"
            dataset.mkdir(parents=True)
            (dataset / "dataset.json").write_text(json.dumps({
                "voice_dataset": str(shared_voice),
            }), encoding="utf-8")

            self.assertEqual(
                _semantic_reference_root(dataset),
                shared_voice.resolve() / "references",
            )

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

    def test_flow_duration_training_forward_and_backward(self):
        config = VitsConfig(**{
            **self.config.to_dict(),
            "duration_predictor_type": "stochastic_mobile",
            "duration_predictor_channels": 16,
            "duration_predictor_flow_layers": 2,
        })
        model = MultilingualVITS(config)
        output = model(
            torch.tensor([[2, 4, 5, 3], [2, 7, 3, 0]]),
            torch.tensor([4, 3]), torch.randn(2, 9, 8),
            torch.tensor([8, 7]), torch.tensor([0, 1]),
            torch.tensor([0, 2]),
        )
        (output.audio.abs().mean() + output.duration_loss).backward()
        self.assertTrue(torch.isfinite(output.duration_loss))
        gradient = model.duration_predictor.flows[0].output.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_text_embedding_scale_and_positions_are_well_conditioned(self):
        embedding = self.model.text_encoder.embedding.weight.detach()
        expected_std = self.config.hidden_channels ** -0.5
        self.assertLess(abs(float(embedding.std()) - expected_std), expected_std * 0.35)
        self.assertGreater(float(embedding[0].abs().sum()), 0.0)

        positions = sinusoidal_position_encoding(
            4, self.config.hidden_channels, device=embedding.device, dtype=embedding.dtype,
        )
        self.assertEqual(positions.shape, (4, self.config.hidden_channels))
        self.assertFalse(torch.equal(positions[0], positions[1]))

    def test_residual_flow_is_volume_preserving_and_invertible(self):
        mask = torch.ones(2, 1, 7)
        latent = torch.randn(2, self.config.latent_channels, 7)
        condition = self.model.conditioning(torch.tensor([0, 1]), torch.tensor([0, 2]))
        transformed, logdet = self.model.flow(latent, mask, condition)
        restored, reverse_logdet = self.model.flow(
            transformed, mask, condition, reverse=True,
        )
        self.assertTrue(torch.allclose(restored, latent, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(logdet, torch.zeros_like(logdet)))
        self.assertIsNone(reverse_logdet)

    def test_kl_loss_trains_text_prior(self):
        """The inference-only text prior must receive training gradients."""
        tokens = torch.tensor([[2, 4, 5, 3], [2, 7, 3, 0]])
        text_lengths = torch.tensor([4, 3])
        spectrogram = torch.randn(2, 9, 8)
        spec_lengths = torch.tensor([8, 7])
        output = self.model(
            tokens, text_lengths, spectrogram, spec_lengths,
            torch.tensor([0, 1]), torch.tensor([0, 2]),
        )

        from tts_trainer.vits.losses import kl_loss
        loss = kl_loss(
            output.latent_prior, output.posterior_log_scale,
            output.prior_mean, output.prior_log_scale, output.audio_mask,
        )
        loss.backward()

        gradient = self.model.text_encoder.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_standard_vits_objective_reaches_every_generator_subsystem(self):
        tokens = torch.tensor([[2, 4, 5, 3], [2, 7, 3, 0]])
        text_lengths = torch.tensor([4, 3])
        spectrogram = torch.randn(2, 9, 8)
        spec_lengths = torch.tensor([8, 7])
        output = self.model(
            tokens, text_lengths, spectrogram, spec_lengths,
            torch.tensor([0, 1]), torch.tensor([0, 2]),
        )
        loss = (
            output.audio.square().mean()
            + output.duration_loss
            + kl_loss(
                output.latent_prior, output.posterior_log_scale,
                output.prior_mean, output.prior_log_scale, output.audio_mask,
            )
        )
        loss.backward()
        for name in (
            "conditioning", "text_encoder", "posterior_encoder",
            "duration_predictor", "flow", "decoder",
        ):
            module = getattr(self.model, name)
            gradient_sum = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient_sum, 0.0, name)

    def test_mobile_token_sequence_does_not_change_posterior_forward_path(self):
        direct_tokens = torch.tensor([[2, 4, 5, 3]])
        piper_tokens = torch.tensor([[2, 0, 4, 0, 5, 0, 3]])
        spectrogram = torch.randn(1, 9, 8)
        spec_lengths = torch.tensor([8])
        language_ids = torch.tensor([0])
        speaker_ids = torch.tensor([0])

        torch.manual_seed(123)
        direct = self.model(
            direct_tokens, torch.tensor([4]), spectrogram, spec_lengths,
            language_ids, speaker_ids,
        )
        torch.manual_seed(123)
        piper = self.model(
            piper_tokens, torch.tensor([7]), spectrogram, spec_lengths,
            language_ids, speaker_ids,
        )

        self.assertTrue(torch.equal(direct.posterior_mean, piper.posterior_mean))
        self.assertTrue(torch.equal(direct.posterior_log_scale, piper.posterior_log_scale))
        self.assertTrue(torch.equal(direct.latent, piper.latent))
        self.assertTrue(torch.equal(direct.slice_starts, piper.slice_starts))
        self.assertTrue(torch.equal(direct.audio, piper.audio))

    def test_aligned_prior_decoder_is_validation_only(self):
        tokens = torch.tensor([[2, 4, 5, 3]])
        text_lengths = torch.tensor([4])
        spectrogram = torch.randn(1, 9, 8)
        spec_lengths = torch.tensor([8])
        output = self.model(
            tokens, text_lengths, spectrogram, spec_lengths,
            torch.tensor([0]), torch.tensor([0]),
        )
        prior_audio = self.model.decode_aligned_prior(
            output.prior_mean, output.audio_mask,
            torch.tensor([0]), torch.tensor([0]), output.slice_starts,
        )
        self.assertFalse(prior_audio.requires_grad)

    def test_text_prior_refinement_freezes_acoustics_and_trains_prior(self):
        discriminator = VitsDiscriminator(periods=(2,))
        configure_training_stage(
            self.model, discriminator, "text_prior_refinement",
        )
        output = self.model.forward_text_refinement(
            torch.tensor([[2, 4, 5, 3]]), torch.tensor([4]),
            torch.randn(1, 9, 8), torch.tensor([8]),
            torch.tensor([0]), torch.tensor([0]),
        )
        loss = (
            output.audio.square().mean()
            + output.duration_loss
            + output.duration_mean_loss
            + kl_loss(
                output.latent_prior, output.posterior_log_scale,
                output.prior_mean, output.prior_log_scale,
                output.audio_mask,
            )
        )
        loss.backward()

        for name in ("text_encoder", "flow", "duration_predictor"):
            module = getattr(self.model, name)
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, name)
        for name in ("conditioning", "posterior_encoder", "decoder"):
            module = getattr(self.model, name)
            self.assertTrue(all(
                parameter.grad is None for parameter in module.parameters()
            ), name)
        self.assertTrue(all(
            parameter.grad is None for parameter in discriminator.parameters()
        ))

    def test_vectorized_maximum_path_matches_reference_alignment(self):
        value = torch.randn(2, 9, 5)
        text_lengths = torch.tensor([5, 3])
        spec_lengths = torch.tensor([9, 7])

        def reference(scores, text_length, spec_length):
            negative = scores.new_tensor(torch.finfo(scores.dtype).min)
            dynamic = scores.new_full((spec_length, text_length), negative)
            dynamic[0, 0] = scores[0, 0]
            for audio_index in range(1, spec_length):
                start = max(0, text_length - (spec_length - audio_index))
                end = min(text_length, audio_index + 1)
                for text_index in range(start, end):
                    stay = dynamic[audio_index - 1, text_index]
                    move = dynamic[audio_index - 1, text_index - 1] \
                        if text_index else negative
                    dynamic[audio_index, text_index] = (
                        scores[audio_index, text_index] + torch.maximum(stay, move)
                    )
            result = torch.zeros_like(scores)
            text_index = text_length - 1
            for audio_index in range(spec_length - 1, -1, -1):
                result[audio_index, text_index] = 1
                if text_index and audio_index and (
                    dynamic[audio_index - 1, text_index - 1]
                    >= dynamic[audio_index - 1, text_index]
                ):
                    text_index -= 1
            return result

        actual = maximum_path(value, text_lengths, spec_lengths)
        expected = torch.zeros_like(value)
        for batch in range(2):
            expected[batch] = reference(
                value[batch], int(text_lengths[batch]), int(spec_lengths[batch]),
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_inference_uses_language_and_speaker_inputs(self):
        audio, lengths, attention = self.model.infer(
            torch.tensor([[2, 4, 3]]), torch.tensor([3]),
            torch.tensor([2]), torch.tensor([1]), max_frames=20,
        )
        self.assertEqual(audio.ndim, 3)
        self.assertEqual(audio.shape[-1], int(lengths.max()) * self.config.hop_length)
        self.assertEqual(attention.shape[2], 3)

    def test_valid_piper_blank_embedding_receives_gradient(self):
        tokens = torch.tensor([[2, 4, 0, 5, 0, 3]])
        lengths = torch.tensor([6])
        condition = self.model.conditioning(torch.tensor([0]), torch.tensor([0]))
        hidden, _, _, _ = self.model.text_encoder(tokens, lengths, condition)
        hidden[0, 0, 2].backward()
        gradient = self.model.text_encoder.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient[0].abs().sum()), 0.0)

    def test_legacy_mobile_checkpoint_requires_blank_retraining(self):
        metadata = {
            "frontend": {
                "token_encoding": LEGACY_PIPER_TOKEN_ENCODING,
            },
        }
        with self.assertRaisesRegex(ValueError, "omitted the blank immediately after BOS"):
            require_mobile_blank_semantics(metadata)
        with self.assertRaisesRegex(ValueError, "valid blank token"):
            require_mobile_blank_semantics({
                "frontend": {"token_encoding": PIPER_TOKEN_ENCODING},
            })
        require_mobile_blank_semantics({
            "frontend": {"token_encoding": PIPER_TOKEN_ENCODING},
            "learned_blank_token": True,
        })
        require_mobile_blank_semantics({
            "frontend": {"token_encoding": "bos-phonemes-eos-v1"},
        })

    def test_stochastic_duration_predictor_has_deterministic_zero_noise_mode(self):
        tokens = torch.tensor([[2, 4, 3]])
        lengths = torch.tensor([3])
        language_ids = torch.tensor([2])
        speaker_ids = torch.tensor([1])
        condition = self.model.conditioning(language_ids, speaker_ids)
        hidden, _, _, mask = self.model.text_encoder(tokens, lengths, condition)

        deterministic_a = self.model.duration_predictor.sample(
            hidden, mask, condition, 0.0,
        )
        deterministic_b = self.model.duration_predictor.sample(
            hidden, mask, condition, 0.0,
        )
        stochastic = self.model.duration_predictor.sample(
            hidden, mask, condition, 0.6,
        )
        self.assertTrue(torch.equal(deterministic_a, deterministic_b))
        self.assertFalse(torch.equal(deterministic_a, stochastic))

    def test_external_language_and_speaker_vectors_match_id_conditioning(self):
        language_ids = torch.tensor([2])
        speaker_ids = torch.tensor([1])
        by_id = self.model.conditioning(language_ids, speaker_ids)
        by_pack = self.model.conditioning.from_embeddings(
            self.model.conditioning.language_embedding(language_ids),
            self.model.conditioning.speaker_embedding(speaker_ids),
        )
        self.assertTrue(torch.equal(by_id, by_pack))

    def test_stochastic_duration_likelihood_trains_mean_and_scale(self):
        tokens = torch.tensor([[2, 4, 5, 3]])
        lengths = torch.tensor([4])
        condition = self.model.conditioning(torch.tensor([0]), torch.tensor([0]))
        hidden, _, _, mask = self.model.text_encoder(tokens, lengths, condition)
        loss = self.model.duration_predictor.loss(
            hidden, mask, condition, torch.tensor([[[1.0, 2.0, 4.0, 1.0]]]),
        )
        loss.backward()
        gradient = self.model.duration_predictor.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient[0].abs().sum()), 0.0)
        self.assertGreater(float(gradient[1].abs().sum()), 0.0)

    def test_flow_duration_predictor_is_invertible_and_stochastic(self):
        config = VitsConfig(**{
            **self.config.to_dict(),
            "duration_predictor_type": "stochastic_mobile",
            "duration_predictor_channels": 16,
            "duration_predictor_flow_layers": 2,
        })
        model = MultilingualVITS(config)
        tokens = torch.tensor([[2, 4, 5, 3]])
        lengths = torch.tensor([4])
        condition = model.conditioning(torch.tensor([0]), torch.tensor([0]))
        hidden, _, _, mask = model.text_encoder(tokens, lengths, condition)
        predictor = model.duration_predictor
        context = predictor._context(hidden, mask, condition)
        original = torch.randn(1, 2, 4) * mask
        base, logdet = predictor._flow(
            original, context, mask, reverse=False,
        )
        restored, _ = predictor._flow(base, context, mask, reverse=True)
        self.assertTrue(torch.allclose(original, restored, atol=1e-5))
        self.assertEqual(logdet.shape, (1,))
        deterministic_a = predictor.sample(hidden, mask, condition, 0.0)
        deterministic_b = predictor.sample(hidden, mask, condition, 0.0)
        stochastic = predictor.sample(hidden, mask, condition, 0.6)
        self.assertTrue(torch.equal(deterministic_a, deterministic_b))
        self.assertFalse(torch.equal(deterministic_a, stochastic))

    def test_flow_duration_likelihood_trains_couplings(self):
        config = VitsConfig(**{
            **self.config.to_dict(),
            "duration_predictor_type": "stochastic_quality",
            "duration_predictor_channels": 16,
            "duration_predictor_flow_layers": 4,
        })
        model = MultilingualVITS(config)
        tokens = torch.tensor([[2, 4, 5, 3]])
        lengths = torch.tensor([4])
        condition = model.conditioning(torch.tensor([0]), torch.tensor([0]))
        hidden, _, _, mask = model.text_encoder(tokens, lengths, condition)
        loss = model.duration_predictor.loss(
            hidden, mask, condition,
            torch.tensor([[[1.0, 2.0, 4.0, 1.0]]]),
        )
        loss.backward()
        gradients = [
            flow.output.weight.grad
            for flow in model.duration_predictor.flows
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(float(gradient.abs().sum()) > 0.0
                            for gradient in gradients))

    def test_legacy_duration_config_defaults_and_resume_preservation(self):
        legacy = VitsConfig(vocab_size=10)
        self.assertEqual(
            duration_predictor_settings({})["duration_predictor_type"],
            "stochastic_lognormal",
        )
        configured = VitsConfig(**{
            **legacy.to_dict(),
            "duration_predictor_type": "stochastic_quality",
            "duration_predictor_channels": 128,
            "duration_predictor_flow_layers": 6,
        })
        restored = preserve_checkpoint_duration_architecture(
            configured, {"config": {"vocab_size": 10}}, "resume",
        )
        self.assertEqual(restored.duration_predictor_type, "stochastic_lognormal")
        self.assertEqual(restored.duration_predictor_channels, 64)
        self.assertEqual(restored.duration_predictor_flow_layers, 2)

    def test_integer_durations_do_not_double_one_frame_tokens(self):
        log_duration = torch.log(torch.tensor([[[1.01, 1.49, 1.51, 2.49]]]))
        mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
        actual = integer_durations(log_duration, mask, 1.0)
        self.assertTrue(torch.equal(actual, torch.tensor([[[1, 1, 2, 0]]])))

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

    def test_mobile_export_adapter_inserts_canonical_blank_after_bos(self):
        wrapper = PiperInferenceWrapper(
            self.model.eval(), insert_pad_after_bos=True,
        )
        captured = {}
        original = self.model.infer_deploy

        def capture(tokens, lengths, language_ids, speaker_ids, scales):
            captured["tokens"] = tokens.clone()
            captured["lengths"] = lengths.clone()
            return original(
                tokens, lengths, language_ids, speaker_ids, scales,
                max_frames=20,
            )

        self.model.infer_deploy = capture
        wrapper(
            torch.tensor([[1, 5, 0, 2]]), torch.tensor([4]),
            torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0]),
        )
        self.assertEqual(
            captured["tokens"].tolist(), [[1, 0, 5, 0, 2]],
        )
        self.assertEqual(captured["lengths"].tolist(), [5])

    def test_mobile_export_adapter_strips_legacy_and_canonical_piper_pads(self):
        wrapper = PiperInferenceWrapper(
            self.model.eval(), strip_piper_pads=True,
        )
        captured = []
        original = self.model.infer_deploy

        def capture(tokens, lengths, language_ids, speaker_ids, scales):
            captured.append((tokens.clone(), lengths.clone()))
            return original(
                tokens, lengths, language_ids, speaker_ids, scales,
                max_frames=20,
            )

        self.model.infer_deploy = capture
        wrapper(
            torch.tensor([[1, 5, 0, 6, 0, 2]]), torch.tensor([6]),
            torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0]),
        )
        wrapper(
            torch.tensor([[1, 0, 5, 0, 6, 0, 2]]), torch.tensor([7]),
            torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0]),
        )
        self.assertEqual(captured[0][0].tolist(), [[1, 5, 6, 2]])
        self.assertEqual(captured[0][1].tolist(), [4])
        self.assertEqual(captured[1][0].tolist(), [[1, 5, 6, 2]])
        self.assertEqual(captured[1][1].tolist(), [4])

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
        optimizer_refinement = torch.optim.AdamW(
            self.model.text_encoder.parameters(), lr=1e-4,
        )
        scheduler_refinement = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_refinement, gamma=0.9,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_training_checkpoint(
                directory, generator=self.model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=2, global_step=12, config=self.config,
                language_map={"zh": 0}, speaker_map={"voice_01": 0},
                tokens=["_", "^", "$", " ", "<unk>"], metrics={"loss": 1.0},
                optimizer_refinement=optimizer_refinement,
                scheduler_refinement=scheduler_refinement,
                training_phase={
                    "stage": "text_prior_refinement",
                    "refinement_steps": 7,
                },
            )
            restored = MultilingualVITS(self.config)
            restored_optimizer = torch.optim.AdamW(
                restored.text_encoder.parameters(), lr=1e-4,
            )
            restored_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                restored_optimizer, gamma=0.9,
            )
            result = load_training_checkpoint(
                directory, generator=restored,
                optimizer_refinement=restored_optimizer,
                scheduler_refinement=restored_scheduler,
            )
            self.assertEqual(result["global_step"], 12)
            self.assertEqual(
                result["training_objective"],
                "standard-vits-mel-kl-duration-gan-feature-v1",
            )
            self.assertTrue(torch.equal(restored.conditioning.language_embedding.weight,
                                        self.model.conditioning.language_embedding.weight))
            self.assertEqual(
                result["training_phase"]["stage"], "text_prior_refinement",
            )

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

    def test_format_three_checkpoint_warm_starts_backbone_but_resets_duration(self):
        discriminator = VitsDiscriminator(periods=(2,))
        optimizer_g = torch.optim.AdamW(self.model.parameters())
        optimizer_d = torch.optim.AdamW(discriminator.parameters())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old"
            save_training_checkpoint(
                checkpoint, generator=self.model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=200, global_step=95000, config=self.config,
                language_map={"zh": 0}, speaker_map={"voice_01": 0},
                tokens=["_", "^", "$", " ", "<unk>"],
            )
            metadata_path = checkpoint / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["format"] = 3
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            state_path = checkpoint / "training-state.pt"
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            state["format"] = 3
            state["generator"]["duration_predictor.projection.weight"].fill_(9.0)
            torch.save(state, state_path)

            target = MultilingualVITS(self.config)
            duration_before = target.duration_predictor.projection.weight.detach().clone()
            report = _load_warm_start_generator(target, checkpoint, ())
            self.assertEqual(report["checkpoint_format"], 3)
            self.assertIn("duration_predictor", report["excluded_modules"])
            self.assertTrue(torch.equal(
                target.conditioning.language_embedding.weight,
                self.model.conditioning.language_embedding.weight,
            ))
            self.assertTrue(torch.equal(
                target.duration_predictor.projection.weight, duration_before,
            ))

    def test_warm_start_automatically_replaces_different_duration_architecture(self):
        discriminator = VitsDiscriminator(periods=(2,))
        optimizer_g = torch.optim.AdamW(self.model.parameters())
        optimizer_d = torch.optim.AdamW(discriminator.parameters())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy"
            save_training_checkpoint(
                checkpoint, generator=self.model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=10, global_step=100, config=self.config,
                language_map={"zh": 0}, speaker_map={"voice_01": 0},
                tokens=["_", "^", "$", " ", "<unk>"],
            )
            flow_config = VitsConfig(**{
                **self.config.to_dict(),
                "duration_predictor_type": "stochastic_mobile",
                "duration_predictor_channels": 16,
                "duration_predictor_flow_layers": 2,
            })
            target = MultilingualVITS(flow_config)
            report = _load_warm_start_generator(target, checkpoint, ())
            self.assertIn("duration_predictor", report["excluded_modules"])
            self.assertEqual(
                report["duration_predictor_from"], "stochastic_lognormal",
            )
            self.assertEqual(
                report["duration_predictor_to"], "stochastic_mobile",
            )
            self.assertTrue(torch.equal(
                target.text_encoder.embedding.weight,
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
            deployment = json.loads(
                (target.parent / "model.onnx.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(deployment["format"], 2)
            self.assertEqual(len(deployment["model_sha256"]), 64)
            self.assertEqual(
                deployment["scales"],
                ["noise_scale", "length_scale", "duration_noise_scale"],
            )
            self.assertEqual(deployment["scales_default"], [0.667, 1.0, 0.35])
            self.assertTrue(
                deployment["export_validation"]["pytorch_onnx_parity"]
            )
            self.assertEqual(list(target.parent.glob("*.onnx")), [target])
            self.assertTrue((target.parent / "frontend.conformance.json").is_file())
            pack_index = json.loads(
                (target.parent / "frontend-packs/manifest.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(
                pack_index["languages"]["ja"]["provider"], "openjtalk",
            )
            self.assertEqual(
                deployment["frontend_packs"]["manifest"],
                "frontend-packs/manifest.json",
            )
            composable = target.parent / "composable"
            catalog = json.loads(
                (composable / "catalog.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(catalog["layout"], "composable-vits-v1")
            self.assertEqual(len(catalog["languages"]), 7)
            self.assertEqual(len(catalog["voices"]), 3)
            self.assertTrue((composable / "core/model.onnx").is_file())
            self.assertTrue(
                (composable / "packages/language-ja.zip").is_file()
            )
            self.assertTrue(
                (composable / "packages/voice-voice_02.zip").is_file()
            )
            core_manifest = json.loads(
                (composable / "core/manifest.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(core_manifest["speaker_embedding_dim"], 4)
            self.assertEqual(core_manifest["language_embedding_dim"], 4)
            self.assertEqual(
                core_manifest["inputs"][-2:],
                ["language_embedding", "speaker_embedding"],
            )
            language_manifest = json.loads(
                (composable / "languages/ja/manifest.json").read_text(
                    encoding="utf-8",
                ),
            )
            voice_manifest = json.loads(
                (composable / "voices/voice_02/manifest.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(
                language_manifest["compatible_core_sha256"],
                core_manifest["core_id"],
            )
            self.assertEqual(
                voice_manifest["compatible_core_sha256"],
                core_manifest["core_id"],
            )
            shape = validate_onnx_runtime(target)
            self.assertEqual(shape[0:2], (1, 1))
            runtime = OnnxTTS(target.parent)
            self.assertEqual(runtime.encode(("a",)).tolist(), [[1, 5, 2]])
            runtime.frontend_contract = FrontendContract(
                provider="espeak-ng",
                languages={"en": {"provider": "espeak-ng", "voice": "en-us"}},
                token_encoding=PIPER_TOKEN_ENCODING,
            )
            self.assertEqual(runtime.encode(("a",)).tolist(), [[1, 0, 5, 0, 2]])
            audio = runtime.synthesize_units(
                ("a",), language="en", speaker="voice_02",
                noise_scale=0.0, duration_noise_scale=0.0,
            )
            self.assertGreater(audio.shape[0], 0)

    def test_mobile_onnx_strips_sherpa_wire_pads_before_direct_model(self):
        mobile_config = VitsConfig(**{
            **self.config.to_dict(), "num_languages": 1, "num_speakers": 1,
            "duration_predictor_type": "stochastic_mobile",
            "duration_predictor_channels": 16,
            "duration_predictor_flow_layers": 2,
        })
        mobile_model = MultilingualVITS(mobile_config)
        discriminator = VitsDiscriminator(periods=(2,))
        optimizer_g = torch.optim.AdamW(mobile_model.parameters())
        optimizer_d = torch.optim.AdamW(discriminator.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            espeak_data = root / "espeak-ng-data"
            espeak_data.mkdir()
            (espeak_data / "phontab").write_bytes(b"test")
            mobile_frontend = frontend_contract_from_config(
                {
                    "provider": "espeak-ng",
                    "mobile_direct": True,
                    "voices": {"en": "en-us"},
                },
                ("en",),
                engine_version="eSpeak NG mobile test",
            )
            save_training_checkpoint(
                checkpoint, generator=mobile_model, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                epoch=1, global_step=1, config=mobile_config,
                language_map={"en": 0},
                speaker_map={"voice_01": 0},
                tokens=["_", "^", "$", " ", "<unk>", "a"],
                frontend=mobile_frontend.to_dict(),
                frontend_conformance={
                    "format": 1,
                    "cases_per_language": 1,
                    "languages": ["en"],
                    "piper_compatible": False,
                    "cases": [{
                        "language": "en", "language_id": 0, "text": "a",
                        "phonemes": ["a"],
                        "token_ids": [1, 5, 2],
                    }],
                },
            )
            with patch(
                "tts_trainer.vits.exporter._find_espeak_data_dir",
                return_value=espeak_data,
            ):
                target = export_vits_onnx(
                    checkpoint, root / "export", sample_rate=8000,
                )
            deployment = json.loads(
                (target.parent / "model.onnx.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                deployment["model_token_encoding"],
                MOBILE_DIRECT_TOKEN_ENCODING,
            )
            self.assertEqual(
                deployment["wire_token_encoding"],
                LEGACY_PIPER_TOKEN_ENCODING,
            )
            self.assertEqual(
                deployment["input_adapter"], "strip-piper-pads-v1",
            )
            self.assertEqual(
                deployment["duration_predictor"]["type"],
                "stochastic-mobile",
            )
            self.assertEqual(
                deployment["duration_predictor"]["flow_layers"], 2,
            )
            runtime = OnnxTTS(target.parent)
            # Match sherpa-onnx 1.13.4's historical wire sequence. The ONNX
            # adapter removes transport pads before invoking the VITS core.
            self.assertEqual(runtime.encode(("a",)).tolist(), [[1, 5, 0, 2]])
            audio = runtime.synthesize_units(
                ("a",), language="en", speaker="voice_01",
                noise_scale=0.0, duration_noise_scale=0.0,
            )
            self.assertGreater(audio.shape[0], 0)

    def test_one_step_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            samples = [int(math.sin(i / 8) * 8000) for i in range(192)]
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
                stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            incompatible = root / "incompatible.wav"
            with wave.open(str(incompatible), "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
                stream.writeframes(struct.pack("<32h", *([0] * 32)))
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
                writer.writeheader()
                writer.writerow({
                    "audio": audio.name, "text": "hello",
                    "language": "en", "speaker": "voice_01",
                })
                writer.writerow({
                    "audio": incompatible.name,
                    "text": "this text cannot fit five frames",
                    "language": "en", "speaker": "voice_01",
                })
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
            alignment = json.loads(
                (
                    root / "run" / "quality" / "alignment-quality-report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                (alignment["passed"], alignment["failed"]), (1, 1),
            )
            saved = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["language_map"], {"en": 0})
            self.assertEqual(saved["config"]["num_languages"], 1)
            self.assertNotIn("prior_mel", saved["metrics"]["train"])

            refinement_config = {
                **config,
                "experiment": {
                    "name": "tiny-en-refined",
                    "languages": ["en"],
                    "initialization": {
                        "mode": "refine_text_prior",
                        "checkpoint": str(checkpoint),
                    },
                },
            }
            refinement_path = root / "refinement.json"
            refinement_path.write_text(
                json.dumps(refinement_config), encoding="utf-8",
            )
            refined = train_vits(
                str(refinement_path), str(metadata), str(root / "refined-run"),
                device_name="cpu", max_steps=1,
            )
            source_state = torch.load(
                checkpoint / "training-state.pt", map_location="cpu",
                weights_only=False,
            )["generator"]
            refined_state = torch.load(
                refined / "training-state.pt", map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                refined_state["training_phase"]["stage"],
                "text_prior_refinement",
            )
            self.assertEqual(
                refined_state["training_phase"]["refinement_steps"], 1,
            )
            for prefix in ("conditioning.", "posterior_encoder.", "decoder."):
                for name, value in source_state.items():
                    if name.startswith(prefix):
                        self.assertTrue(
                            torch.equal(value, refined_state["generator"][name]),
                            name,
                        )

            resume_config = {
                **config,
                "experiment": {
                    "name": "tiny-en-refined",
                    "languages": ["en"],
                    "initialization": {
                        "mode": "resume",
                        "checkpoint": str(refined),
                    },
                },
                "training": {**config["training"], "epochs": 2},
            }
            resume_path = root / "resume-refinement.json"
            resume_path.write_text(
                json.dumps(resume_config), encoding="utf-8",
            )
            resumed = train_vits(
                str(resume_path), str(metadata), str(root / "refined-run"),
                device_name="cpu", max_steps=1,
            )
            resumed_state = torch.load(
                resumed / "training-state.pt", map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                resumed_state["training_phase"]["stage"],
                "text_prior_refinement",
            )
            self.assertEqual(
                resumed_state["training_phase"]["refinement_steps"], 2,
            )

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
                             "checkpoint_every_steps": 50, "seed": 7,
                             "text_prior_refinement": {
                                 "enabled": True,
                                 "start_steps": 0,
                                 "posterior_mel_threshold": None,
                             }},
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
            self.assertIn("prior_mel", saved["metrics"]["validation"])
            self.assertIn("combined_mel", saved["metrics"]["validation"])
            state = torch.load(
                best / "training-state.pt", map_location="cpu", weights_only=False,
            )
            self.assertIsNotNone(state["scheduler_g"])
            last_metadata = json.loads(
                (last / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                last_metadata["training_phase"]["stage"],
                "text_prior_refinement",
            )
            preview = root / "run" / "validation-audio" / "epoch-0001"
            self.assertTrue((preview / "target.wav").is_file())
            self.assertTrue((preview / "posterior-reconstruction.wav").is_file())
            self.assertTrue(
                (preview / "posterior-sampled-reconstruction.wav").is_file()
            )
            self.assertTrue((preview / "aligned-text-prior.wav").is_file())
            self.assertTrue((preview / "text-only-inference.wav").is_file())
            diagnostics = json.loads(
                (preview / "diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertIn("duration_ratio", diagnostics)
            self.assertIn("posterior_mean_full_mel", diagnostics)
            self.assertIn("posterior_sampled_full_mel", diagnostics)
            self.assertIn("posterior_scale_mean", diagnostics)
            self.assertEqual(diagnostics["format"], 3)
            self.assertIn("en/voice_01", last_metadata["metrics"]["validation"]["profiles"])
            profile_preview = (
                preview / "profiles" / "en" / "voice_01" / "diagnostics.json"
            )
            self.assertTrue(profile_preview.is_file())
            with wave.open(
                str(preview / "posterior-reconstruction.wav"), "rb",
            ) as posterior:
                self.assertEqual(
                    posterior.getnframes(),
                    diagnostics["target_frames"] * config["audio"]["hop_length"],
                )
            self.assertTrue((root / "run" / "splits" / "validation.csv").is_file())


if __name__ == "__main__":
    unittest.main()
