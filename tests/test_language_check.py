import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tts_trainer.experiments import resolve_experiment
from tts_trainer.language_check import check_language_support, format_language_statuses


class FakeFrontend:
    voices = {"de": "de", "pl": "pl"}

    def version(self):
        return "eSpeak NG test"

    def phonemize(self, text, language):
        if language == "pl":
            raise ValueError("missing voice data")
        return tuple("halo")


class LanguageCheckTests(unittest.TestCase):
    def _config(self, root: Path):
        path = root / "model.json"
        path.write_text(json.dumps({
            "experiment": {"name": "german", "languages": ["de"]},
            "generation": {"enabled": True},
        }), encoding="utf-8")
        return path

    def test_reports_ready_teacher_and_g2p(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, layout = resolve_experiment(self._config(Path(directory)))
            with patch("tts_trainer.language_check.espeak_frontend_from_config", return_value=FakeFrontend()):
                statuses = check_language_support(raw, layout)
            self.assertTrue(statuses[0].ready)
            self.assertEqual(statuses[0].teacher, "qwen:German")
            self.assertIn("ready", format_language_statuses(statuses))

    def test_external_data_does_not_require_qwen_but_still_checks_g2p(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.json"
            path.write_text(json.dumps({
                "language_registry": {
                    "pl": {
                        "name": "Polish", "teacher": None,
                        "frontend": {"provider": "espeak-ng", "voice": "pl"},
                        "smoke_text": "Dzień dobry.",
                    }
                },
                "experiment": {"name": "polish", "languages": ["pl"]},
                "generation": {"enabled": False},
            }), encoding="utf-8")
            raw, layout = resolve_experiment(path)
            with patch("tts_trainer.language_check.espeak_frontend_from_config", return_value=FakeFrontend()):
                status = check_language_support(raw, layout)[0]
            self.assertEqual(status.teacher, "external-data")
            self.assertFalse(status.ready)
            self.assertIn("missing voice data", status.error)


if __name__ == "__main__":
    unittest.main()
