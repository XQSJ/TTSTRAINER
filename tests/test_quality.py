import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_trainer.manifest import Item
from tts_trainer.quality import inspect_audio_item, run_audio_quality_gate


class QualityTests(unittest.TestCase):
    def _item(self, root: Path, samples: np.ndarray, *, rate: int = 8000) -> Item:
        path = root / "sample.wav"
        sf.write(path, samples, rate, subtype="PCM_16")
        return Item(path, "hello world", "en", "voice_01", tuple("hello world"))

    def test_clean_audio_passes_signal_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            time = np.arange(8000, dtype=np.float32) / 8000
            item = self._item(root, 0.2 * np.sin(2 * math.pi * 220 * time))
            result = inspect_audio_item(item, {})
            self.assertTrue(result["passed"])
            report = run_audio_quality_gate([item], {}, root / "report.json")
            self.assertEqual((report["passed"], report["failed"]), (1, 0))

    def test_silence_is_rejected_and_report_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self._item(root, np.zeros(8000, dtype=np.float32))
            report_path = root / "report.json"
            with self.assertRaisesRegex(ValueError, "audio quality gate rejected"):
                run_audio_quality_gate([item], {}, report_path)
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
