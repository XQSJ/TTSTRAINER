import csv
import tempfile
import unittest
from pathlib import Path

from tts_trainer.frontend import (FrontendContract, frontend_contract_from_config,
                                  frontend_lock_path, load_frontend_contract,
                                  save_frontend_contract)
from tts_trainer.frontend.espeak import (espeak_frontend_from_config,
                                         parse_espeak_ipa, phonemize_manifest)
from tts_trainer.manifest import read_manifest
from tts_trainer.text import Vocabulary


class FakeFrontend:
    def phonemize(self, text, language):
        return ("h", "ə", " ", "w")

    def contract(self, languages):
        return FrontendContract(
            provider="espeak-ng",
            engine_version="eSpeak NG test",
            languages={language: {"voice": "en-us"} for language in languages},
        )


class FrontendTests(unittest.TestCase):
    def test_parses_phone_word_and_break_boundaries(self):
        self.assertEqual(parse_espeak_ipa("h|ə|l|ˈoʊ w|ˈɜː|l|d\nnext"),
                         tuple("həlˈoʊ wˈɜːld next"))

    def test_language_switch_annotations_are_not_tokens(self):
        self.assertEqual(parse_espeak_ipa("(en)h|ə|(fr)l|o"), tuple("həlo"))

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
            contract = load_frontend_contract(frontend_lock_path(destination))
            self.assertEqual(contract.engine_version, "eSpeak NG test")
            self.assertEqual(contract.languages, {"en": {"voice": "en-us"}})

    def test_configured_voice_override_is_routed_to_espeak(self):
        frontend = espeak_frontend_from_config({
            "executable": "/bin/echo",
            "voices": {"pt": "pt-pt"},
            "strict_language_switches": False,
        })
        self.assertEqual(frontend.voices["pt"], "pt-pt")
        self.assertTrue(frontend.allow_language_switches)

    def test_contract_detects_voice_incompatibility(self):
        default = frontend_contract_from_config({}, ("pt",))
        european = frontend_contract_from_config(
            {"voices": {"pt": "pt-pt"}}, ("pt",)
        )
        self.assertNotEqual(default.compatibility_key(), european.compatibility_key())

    def test_contract_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontend.json"
            expected = frontend_contract_from_config(
                {"voices": {"pt": "pt-pt"}}, ("en", "pt"),
                engine_version="eSpeak NG 1.52",
            )
            save_frontend_contract(expected, path)
            self.assertEqual(load_frontend_contract(path), expected)

    def test_custom_registry_supplies_frontend_voice(self):
        frontend = espeak_frontend_from_config(
            {"executable": "/bin/echo"}, languages=("pl",),
            language_registry={
                "pl": {
                    "name": "Polish", "teacher": None,
                    "frontend": {"provider": "espeak-ng", "voice": "pl"},
                    "smoke_text": "Dzień dobry.",
                }
            },
        )
        self.assertEqual(frontend.voices["pl"], "pl")


if __name__ == "__main__":
    unittest.main()
