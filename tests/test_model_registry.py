import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tts_trainer.model_registry import REQUIRED_FILES, ensure_model, inspect_model, require_local_model


def create_fake_model(path: Path):
    for relative in REQUIRED_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model-data")


class ModelRegistryTests(unittest.TestCase):
    def test_detects_missing_model_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(inspect_model("base-1.7b", root).ready)
            with self.assertRaises(FileNotFoundError):
                require_local_model("base-1.7b", root)

    def test_reuses_complete_local_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_fake_model(root / "Qwen3-TTS-12Hz-1.7B-Base")
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                result = ensure_model("base-1.7b", root)
            self.assertTrue(inspect_model("base-1.7b", root).ready)
            self.assertEqual(result.name, "Qwen3-TTS-12Hz-1.7B-Base")

    def test_downloads_into_requested_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_download(repo_id, local_dir):
                create_fake_model(Path(local_dir))
            class Hub:
                snapshot_download = staticmethod(fake_download)
            with patch.dict("sys.modules", {"huggingface_hub": Hub}):
                result = ensure_model("voice-design-1.7b", root)
            marker = json.loads((result / ".download-complete.json").read_text())
            self.assertEqual(marker["repo_id"], "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")


if __name__ == "__main__":
    unittest.main()
