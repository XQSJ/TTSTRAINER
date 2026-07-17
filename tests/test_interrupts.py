from __future__ import annotations

import os
import signal
import unittest
from unittest.mock import MagicMock, patch

from tts_trainer.interrupts import _exit_code, run_supervised, should_supervise


class InterruptSupervisorTests(unittest.TestCase):
    def test_only_long_running_top_level_commands_are_supervised(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(should_supervise(["run-pipeline", "--config", "train.json"]))
            self.assertTrue(should_supervise(["generate-samples", "--config", "train.json"]))
            self.assertFalse(should_supervise(["languages", "--config", "train.json"]))
            self.assertFalse(should_supervise([]))

    def test_child_does_not_create_another_supervisor(self):
        with patch.dict(os.environ, {"TTS_TRAINER_SUPERVISED": "1"}, clear=True):
            self.assertFalse(should_supervise(["run-pipeline", "--config", "train.json"]))

    def test_signal_return_code_is_normalized_for_the_shell(self):
        self.assertEqual(_exit_code(-2), 130)
        self.assertEqual(_exit_code(0), 0)

    @patch("tts_trainer.interrupts._send_signal")
    @patch("tts_trainer.interrupts.subprocess.Popen")
    def test_ctrl_c_is_forwarded_and_returns_shell_interrupt_code(self, popen, send_signal):
        process = MagicMock()
        process.wait.side_effect = [KeyboardInterrupt(), 130]
        popen.return_value = process

        self.assertEqual(run_supervised(["run-pipeline", "--config", "train.json"]), 130)
        send_signal.assert_called_once_with(process, signal.SIGINT)

