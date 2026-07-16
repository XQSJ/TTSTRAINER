import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "deployment" / "build_bundle.py"
SPEC = importlib.util.spec_from_file_location("tts_trainer_build_bundle", MODULE_PATH)
build_bundle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_bundle)


class DeploymentBundleTests(unittest.TestCase):
    def test_wheelhouse_includes_pyopenjtalk_version_build_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = []
            with patch.object(build_bundle, "run",
                              side_effect=lambda *args, **kwargs: commands.append(args)):
                build_bundle.download_wheels(Path(directory))
            command = commands[0]
            self.assertIn("setuptools_scm>=8", command)
            self.assertIn("pyopenjtalk>=0.4.1,<0.5", command)

    def test_downloaded_models_use_runtime_project_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            commands = []
            with patch.object(build_bundle, "ensure_huggingface_client",
                              return_value=output / ".builder"), \
                    patch.object(build_bundle, "run",
                                 side_effect=lambda *args, **kwargs: commands.append(args)):
                build_bundle.download_models(output, ("Org/Test-Model",))
            self.assertIn(
                str(output / "source" / "models" / "qwen" / "Test-Model"),
                commands[0],
            )

    def test_offline_installer_includes_export_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_bundle.write_support_files(output, ("Org/Test-Model",))
            installer = (output / "install_offline.sh").read_text(encoding="utf-8")
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("onnxruntime", installer)
            self.assertIn("piper-plus-g2p", installer)
            self.assertIn("source/models/qwen/", readme)

    def test_optional_quality_dependencies_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_bundle.write_support_files(
                output, ("Org/Test-Model",), include_quality=True,
            )
            installer = (output / "install_offline.sh").read_text(encoding="utf-8")
            self.assertIn("faster-whisper", installer)
            self.assertIn("speechbrain", installer)


if __name__ == "__main__":
    unittest.main()
