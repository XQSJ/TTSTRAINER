import csv
import tempfile
import unittest
import wave
from pathlib import Path

from tts_trainer.manifest import read_manifest, validate_manifest
from tts_trainer.text import Vocabulary, normalize


class CoreTests(unittest.TestCase):
    def make_dataset(self, root: Path):
        wav = root / "sample.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(22050); stream.writeframes(b"\0\0" * 100)
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
            writer.writeheader(); writer.writerow({"audio": "sample.wav", "text": " 你好　世界 ", "language": "zh", "speaker": "voice_01"})
        return metadata

    def test_manifest_and_vocab(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = self.make_dataset(Path(directory))
            report = validate_manifest(metadata, 22050)
            self.assertEqual(report.language_counts["zh"], 1)
            vocab = Vocabulary.build(read_manifest(metadata))
            encoded = vocab.encode("你好世界", "zh")
            self.assertGreater(len(encoded), 4)
            target = Path(directory) / "vocab.json"; vocab.save(target)
            self.assertEqual(Vocabulary.load(target).tokens, vocab.tokens)

    def test_normalization(self):
        self.assertEqual(normalize(" Ｈｉ   there ", "en"), "Hi there")

    def test_rejects_unknown_language(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = self.make_dataset(Path(directory))
            text = metadata.read_text(encoding="utf-8").replace(",zh,", ",xx,")
            metadata.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported language"):
                validate_manifest(metadata)

    def test_production_manifest_requires_frozen_phonemes(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = self.make_dataset(Path(directory))
            with self.assertRaisesRegex(ValueError, "missing frozen phonemes"):
                validate_manifest(metadata, require_phonemes=True)


if __name__ == "__main__":
    unittest.main()
