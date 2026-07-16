import json
import tempfile
import unittest
from pathlib import Path

from tts_trainer.manifest import Item
from tts_trainer.quality_models import (ensure_quality_model,
                                        inspect_quality_model)
from tts_trainer.semantic_quality import (run_semantic_quality_gate,
                                          text_error_rate)


class FakeAsr:
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def transcribe(self, audio, language):
        return self.transcripts[Path(audio).name]


class FakeSpeaker:
    def __init__(self, score):
        self.score = score

    def similarity(self, reference, audio):
        return self.score


class SemanticQualityTests(unittest.TestCase):
    def test_error_rate_uses_cer_and_wer(self):
        self.assertEqual(text_error_rate("你好世界", "你好世", "zh"), ("cer", 0.25))
        self.assertEqual(text_error_rate("hello world", "hello", "en"), ("wer", 0.5))

    def test_mocked_asr_and_speaker_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"; audio.write_bytes(b"test")
            reference = root / "reference.wav"; reference.write_bytes(b"test")
            item = Item(audio, "hello world", "en", "voice_01", ("h",))
            report = run_semantic_quality_gate(
                [item], {
                    "fail_on_error": True,
                    "asr": {"enabled": True, "maximum_error_rate": 0.1},
                    "speaker": {
                        "enabled": True, "minimum_similarity": 0.25,
                        "references": {"voice_01": str(reference)},
                    },
                }, root / "semantic.json",
                asr_evaluator=FakeAsr({"sample.wav": "hello world"}),
                speaker_evaluator=FakeSpeaker(0.8),
            )
            self.assertEqual((report["passed"], report["failed"]), (1, 0))
            saved = json.loads((root / "semantic.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["results"][0]["asr"]["metric"], "wer")

    def test_quality_model_registry_reuses_project_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = inspect_quality_model("asr-small", root)
            self.assertFalse(status.ready)
            for relative in status.spec.required_files:
                target = status.path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"model")
            self.assertEqual(
                ensure_quality_model("asr-small", root, allow_download=False), status.path,
            )


if __name__ == "__main__":
    unittest.main()
