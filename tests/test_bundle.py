import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deployment.build_bundle import DEFAULT_MODELS, copy_sources, write_manifest, write_support_files


class BundleTests(unittest.TestCase):
    def test_builds_metadata_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            output.mkdir()
            copy_sources(output)
            write_support_files(output, DEFAULT_MODELS)
            write_manifest(output, DEFAULT_MODELS)
            self.assertTrue((output / "bundle-manifest.json").is_file())
            self.assertTrue((output / "install_offline.sh").is_file())
            self.assertFalse((output / "source" / "Qwen3-TTS").exists())
            self.assertTrue((output / "source" / "LICENSE").is_file())
            self.assertTrue((output / "source" / "THIRD_PARTY_NOTICES.md").is_file())
            self.assertTrue((output / "source" / "training_configs" / "train1.json").is_file())
            self.assertTrue((output / "source" / "scripts" / "run_pipeline.py").is_file())


if __name__ == "__main__":
    unittest.main()
