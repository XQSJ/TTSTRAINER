import logging
import unittest

from tts_trainer.logging_utils import ConsoleFormatter, format_duration


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


if __name__ == "__main__":
    unittest.main()
