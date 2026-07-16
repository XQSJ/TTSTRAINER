import csv
import hashlib
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tts_trainer.frontend import (FrontendContract, frontend_contract_from_config,
                                  frontend_from_config,
                                  build_frontend_conformance,
                                  frontend_lock_path, load_frontend_contract,
                                  save_frontend_contract,
                                  verify_frontend_conformance)
from tts_trainer.frontend.espeak import (espeak_frontend_from_config,
                                         parse_espeak_ipa, phonemize_manifest)
from tts_trainer.frontend.openjtalk import OpenJTalkFrontend
from tts_trainer.frontend.piper_plus import PiperPlusFrontend
from tts_trainer.frontend.resources import (OPENJTALK_REQUIRED_FILES,
                                            ensure_korean_cmudict,
                                            ensure_openjtalk_dictionary,
                                            inspect_korean_cmudict,
                                            inspect_openjtalk_dictionary)
from tts_trainer.manifest import read_manifest
from tts_trainer.manifest import Item
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

    def test_japanese_registry_routes_to_openjtalk(self):
        router = frontend_from_config({}, languages=("en", "ja"))
        self.assertEqual(router.provider_for("en"), "espeak-ng")
        self.assertEqual(router.provider_for("ja"), "openjtalk")
        self.assertIsInstance(router.frontend_for("ja"), OpenJTalkFrontend)
        self.assertEqual(
            router.declared.languages["ja"]["dictionary"],
            "open_jtalk_dic_utf_8-1.11",
        )

    def test_mandarin_and_korean_route_to_piper_plus(self):
        router = frontend_from_config({}, languages=("zh", "ko"))
        self.assertEqual(router.provider_for("zh"), "piper-plus-g2p")
        self.assertEqual(router.provider_for("ko"), "piper-plus-g2p")
        self.assertIsInstance(router.frontend_for("zh"), PiperPlusFrontend)
        self.assertIsInstance(router.frontend_for("ko"), PiperPlusFrontend)
        with patch.object(router.frontend_for("zh"), "version", return_value="zh-test"), \
                patch.object(router.frontend_for("ko"), "version", return_value="ko-test"):
            runtime_contract = router.contract(("zh", "ko"))
        self.assertEqual(
            runtime_contract.declaration_key(), router.declared.declaration_key(),
        )

    def test_piper_plus_keeps_multicharacter_phone_units(self):
        phonemizer = SimpleNamespace(phonemize=lambda text: ["tɕʰ", "i", "tone4"])
        module = SimpleNamespace(get_phonemizer=lambda language: phonemizer)
        frontend = PiperPlusFrontend("zh")
        with patch("tts_trainer.frontend.piper_plus.importlib.util.find_spec", return_value=object()), \
                patch("tts_trainer.frontend.piper_plus.importlib.import_module", return_value=module):
            self.assertEqual(frontend.phonemize("气", "zh"), ("tɕʰ", "i", "tone4"))

    def test_openjtalk_keeps_multi_character_phone_units(self):
        module = SimpleNamespace(g2p=lambda text, kana, join: ["k", "yo", "o", "pau", "N"])
        frontend = OpenJTalkFrontend()
        with patch("tts_trainer.frontend.openjtalk.importlib.util.find_spec", return_value=object()), \
                patch("tts_trainer.frontend.openjtalk.ensure_openjtalk_dictionary", return_value=Path("/tmp/openjtalk-dic")), \
                patch("tts_trainer.frontend.openjtalk.importlib.metadata.version", return_value="0.4.1"), \
                patch("tts_trainer.frontend.openjtalk.importlib.import_module", return_value=module):
            self.assertEqual(frontend.phonemize("今日は晴れです。", "ja"),
                             ("k", "yo", "o", "pau", "N"))
            self.assertEqual(frontend.version(), "pyopenjtalk 0.4.1")

    def test_runtime_versions_are_locked_but_share_the_same_declaration(self):
        first = FrontendContract(provider="language-router", languages={
            "ja": {"provider": "openjtalk", "dictionary": "open_jtalk_dic_utf_8-1.11",
                   "engine_version": "pyopenjtalk 0.4.1"},
        })
        second = FrontendContract(provider="language-router", languages={
            "ja": {"provider": "openjtalk", "dictionary": "open_jtalk_dic_utf_8-1.11",
                   "engine_version": "pyopenjtalk 0.4.2"},
        })
        self.assertNotEqual(first.compatibility_key(), second.compatibility_key())
        self.assertEqual(first.declaration_key(), second.declaration_key())

    def test_project_local_openjtalk_dictionary_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = inspect_openjtalk_dictionary(root)
            self.assertFalse(status.ready)
            with self.assertRaisesRegex(FileNotFoundError, "frontends ensure openjtalk"):
                ensure_openjtalk_dictionary(root, allow_download=False)
            status.path.mkdir(parents=True)
            for name in OPENJTALK_REQUIRED_FILES:
                (status.path / name).write_bytes(b"test")
            ready = inspect_openjtalk_dictionary(root)
            self.assertTrue(ready.ready)
            self.assertEqual(ensure_openjtalk_dictionary(root, allow_download=False), ready.path)

    def test_project_local_korean_dictionary_download_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("cmudict/cmudict", "TEST  T EH1 S T\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            def copy_download(_url, destination):
                shutil.copy2(source, destination)

            with patch("tts_trainer.frontend.resources.KOREAN_CMU_DICT_SHA256", digest), \
                    patch("tts_trainer.frontend.resources.urllib.request.urlretrieve", copy_download):
                data_root = ensure_korean_cmudict(root)
                self.assertEqual(data_root, root / "korean" / "nltk_data")
                self.assertTrue(inspect_korean_cmudict(root).ready)

    def test_conformance_freezes_phonemes_and_token_ids(self):
        item = Item(Path("sample.wav"), "Hello", "en", "voice_01", ("h", "ə", " ", "w"))
        vocabulary = Vocabulary.build([item])
        conformance = build_frontend_conformance([item], vocabulary, {"en": 0})
        self.assertEqual(conformance["cases"][0]["phonemes"], ["h", "ə", " ", "w"])
        self.assertEqual(
            conformance["cases"][0]["token_ids"], vocabulary.encode_item(item),
        )
        self.assertEqual(
            verify_frontend_conformance(conformance, FakeFrontend(), vocabulary), [],
        )


if __name__ == "__main__":
    unittest.main()
