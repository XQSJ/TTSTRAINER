"""定制词定音：解析、候选、合并与闸门跳过语义。 / Custom words: parsing, candidates, merging, and the skip semantics."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tts_trainer.custom_words import (
    deployment_wordlist,
    has_custom_words,
    load_custom_words,
    merge_into_english_cmudict,
)


class CustomWordsTest(unittest.TestCase):
    def test_missing_config_is_empty_and_skips(self):
        with TemporaryDirectory() as td:
            config = Path(td) / "train.json"
            config.write_text("{}", encoding="utf-8")
            custom = load_custom_words(config)
            self.assertFalse(has_custom_words(custom))

    def test_parse_native_and_shared(self):
        with TemporaryDirectory() as td:
            config = Path(td) / "train.json"
            config.write_text("{}", encoding="utf-8")
            (Path(td) / "custom_words.json").write_text(json.dumps({
                "native_words": {"en": {"fosi": {}}},
                "shared_words": {"kubernetes": {}},
            }), encoding="utf-8")
            custom = load_custom_words(config)
            self.assertTrue(has_custom_words(custom))
            self.assertEqual(custom["native_words"]["en"], {"fosi": {}})
            self.assertEqual(custom["shared_words"], {"kubernetes": {}})

    def test_merge_and_wordlist(self):
        verified = {
            "en": {
                "fosi": {"arpabet": "F OW1 S IY0", "respelling": None},
                "zorp": {"arpabet": "Z AO1 R P", "respelling": "zohrp"},
            },
        }
        entries: dict[str, str] = {}
        merged = merge_into_english_cmudict(verified, entries)
        self.assertEqual(merged, 2)
        self.assertEqual(entries["fosi"], "F OW1 S IY0")
        self.assertEqual(deployment_wordlist(verified), {"zorp": "zohrp"})

    def test_non_english_recorded_as_respelling_only(self):
        verified = {"ja": {"namae": {"arpabet": None, "respelling": "na ma e"}}}
        entries: dict[str, str] = {}
        self.assertEqual(merge_into_english_cmudict(verified, entries), 0)
        self.assertEqual(deployment_wordlist(verified), {"namae": "na ma e"})


if __name__ == "__main__":
    unittest.main()
