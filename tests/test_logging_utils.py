import logging
import io
import unittest

from tts_trainer.logging_utils import (ConsoleFormatter, TerminalProgress,
                                       format_duration, progress_bar)


class LoggingUtilsTests(unittest.TestCase):
    @staticmethod
    def _record(message="hello %s", args=("world",), *, style=""):
        record = logging.LogRecord(
            "tts_trainer.pipeline", logging.INFO, __file__, 1,
            message, args, None,
        )
        if style:
            record.tts_style = style
        return record

    def test_plain_formatter_has_columns_without_ansi(self):
        rendered = ConsoleFormatter(False).format(self._record())
        self.assertIn("│ INFO", rendered)
        self.assertIn("│ pipeline │ hello world", rendered)
        self.assertNotIn("\033[", rendered)

    def test_color_formatter_uses_ansi(self):
        rendered = ConsoleFormatter(True).format(self._record())
        self.assertIn("\033[", rendered)
        self.assertIn("hello world", rendered)

    def test_section_has_blank_line_and_dividers(self):
        rendered = ConsoleFormatter(False).format(
            self._record("STAGE 1/7\npreflight", (), style="section"),
        )
        self.assertTrue(rendered.startswith("\n"))
        self.assertEqual(rendered.count("━"), 156)
        self.assertIn("STAGE 1/7\npreflight", rendered)

    def test_duration_is_human_readable(self):
        self.assertEqual(format_duration(12.2), "12s")
        self.assertEqual(format_duration(125), "2m 05s")
        self.assertEqual(format_duration(7265), "2h 01m 05s")

    def test_progress_bar_has_fixed_width_and_clamps_values(self):
        self.assertEqual(progress_bar(0, 10, width=4), "[░░░░]")
        self.assertEqual(progress_bar(5, 10, width=4), "[██░░]")
        self.assertEqual(progress_bar(20, 10, width=4), "[████]")

    def test_terminal_progress_updates_and_clears_one_line(self):
        stream = io.StringIO()
        progress = TerminalProgress("TRAIN", 10, enabled=True, stream=stream, width=4)
        progress.update(5, "ETA=10s")
        progress.close()
        rendered = stream.getvalue()
        self.assertIn("TRAIN [██░░]", rendered)
        self.assertIn("50.00% 5/10 | ETA=10s", rendered)
        self.assertGreaterEqual(rendered.count("\r"), 3)


if __name__ == "__main__":
    unittest.main()
