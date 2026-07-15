import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tts_trainer.qwen_teacher import configure_qwen_runtime, inspect_qwen_runtime


class QwenRuntimeTests(unittest.TestCase):
    def test_installed_runtime_reports_actionable_install_command(self):
        with patch("tts_trainer.qwen_teacher.importlib.util.find_spec", return_value=None):
            status = inspect_qwen_runtime()
        self.assertFalse(status["ready"])
        self.assertIn("pip install -e '.[qwen]'", status["error"])

    def test_source_runtime_requires_a_qwen_package(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "does not contain qwen_tts"):
                configure_qwen_runtime("source", directory)

    def test_expert_source_runtime_adds_the_checkout_to_sys_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "qwen_tts"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            value = str(root.resolve())
            try:
                status = configure_qwen_runtime("source", root)
                self.assertTrue(status["ready"])
                self.assertEqual(status["source"], value)
                self.assertEqual(sys.path[0], value)
            finally:
                if value in sys.path:
                    sys.path.remove(value)


if __name__ == "__main__":
    unittest.main()
