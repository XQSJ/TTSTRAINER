import csv
import tempfile
import unittest
from pathlib import Path

from tts_trainer.frontend.espeak import parse_espeak_ipa, phonemize_manifest
from tts_trainer.manifest import read_manifest
from tts_trainer.text import Vocabulary


class FakeFrontend:
    def phonemize(self, text, language):
        return ("h", "ə", " ", "w")


class FrontendTests(unittest.TestCase):
    def test_parses_phone_word_and_break_boundaries(self):
        self.assertEqual(parse_espeak_ipa("h|ə|l|ˈoʊ w|ˈɜː|l|d\nnext"),
                         tuple("həlˈoʊ wˈɜːld next"))

    def test_manifest_freezes_phonemes_and_vocab_uses_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"; destination = root / "processed.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
                writer.writeheader(); writer.writerow({"audio": "x.wav", "text": "Hello world",
                                                        "language": "en", "speaker": "voice_01"})
            phonemize_manifest(source, destination, FakeFrontend())
            item = read_manifest(destination)[0]
            self.assertEqual(item.phonemes, ("h", "ə", " ", "w"))
            vocab = Vocabulary.build([item])
            self.assertEqual(vocab.tokens[:4], ["_", "^", "$", " "])
            self.assertNotIn("H", vocab.tokens)


if __name__ == "__main__":
    unittest.main()
