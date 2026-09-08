"""语言包资源契约：trainer 产出布局必须与 Android 端物化/前端校验逐项对齐。 / Language-pack resource contract: the exported layout must match the Android materialization and frontend checks item for item."""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPOSABLE = REPO / "src" / "tts_trainer" / "vits" / "composable.py"


class ComposableResourceContract(unittest.TestCase):
    """以源码静态断言钉住两端契约，改动任一侧都会在此失败。 / Pin both sides of the contract with static assertions so either change fails here first."""

    def test_every_language_pack_resource_is_a_directory(self):
        """所有 language-pack 资源最终布局必须是目录；单个文件会被 Android copyDirectory 拒绝。 / Every language-pack payload must end up as a directory; a plain file breaks Android's copyDirectory."""
        source = COMPOSABLE.read_text(encoding="utf-8")
        # en 词典必须落成 runtime/cmudict/cmudict_data.json，而不是 runtime/cmudict 文件
        self.assertIn(
            'destination.mkdir(parents=True, exist_ok=True)',
            source,
            "the en cmudict branch must create the runtime/cmudict directory",
        )
        self.assertIn(
            'destination / "cmudict_data.json"',
            source,
            "the dictionary file must live inside the runtime/cmudict directory",
        )
        # 所有 _tree_identity 的入参都必须是"复制后的目录"，避免再次出现空树哈希
        for match in re.finditer(r"_tree_identity\((\w+)\)", source):
            self.assertEqual(
                match.group(1), "destination",
                f"resource hash must cover the copied directory, got {match.group(1)}",
            )

    def test_resource_ids_cover_every_packaging_branch(self):
        """trainer 产出的资源 id 集合保持封闭：新增分支必须同步 Android 端 switch。 / The produced id set stays closed: a new branch must update the Android switch too."""
        source = COMPOSABLE.read_text(encoding="utf-8")
        ids = set(re.findall(r'"id": "([\w-]+)"', source))
        ids -= {"tts-language-pack", "tts-acoustic-core", "tts-voice-pack"}
        self.assertEqual(
            ids,
            {"cmudict", "espeak-ng-data", "openjtalk-dictionary",
             "piper-plus-g2p", "piper-plus-g2p-data"},
            "if this changes, update ComposablePackStore.materializeFrontendRuntime too",
        )


if __name__ == "__main__":
    unittest.main()
