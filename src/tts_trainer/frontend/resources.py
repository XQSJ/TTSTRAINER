"""前端资源管理：下载并校验 Open JTalk 词典、韩语 CMU 词典等外部资源。 / Frontend resources: download and verify Open JTalk dictionaries, Korean CMU dict and other external resources."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRONTENDS_ROOT = PROJECT_ROOT / "models" / "frontends"  # 资源默认存放根目录 / default resource root
OPENJTALK_DICTIONARY_NAME = "open_jtalk_dic_utf_8-1.11"
# 下载源与 sha256 固定配对，校验失败即删除避免污染缓存。
# Download URLs pair with pinned sha256 digests; failures delete the file to keep the cache clean.
OPENJTALK_DICTIONARY_URL = (
    "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/"
    "open_jtalk_dic_utf_8-1.11.tar.gz"
)
OPENJTALK_DICTIONARY_SHA256 = "fe6ba0e43542cef98339abdffd903e062008ea170b04e7e2a35da805902f382a"
# 词典就绪的最低文件集合，缺任一即视为不完整。
# Minimal file set for a ready dictionary; any gap means incomplete.
OPENJTALK_REQUIRED_FILES = ("char.bin", "matrix.bin", "sys.dic", "unk.dic")
KOREAN_CMU_DICT_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/corpora/cmudict.zip"
)
KOREAN_CMU_DICT_SHA256 = "d07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6"
KOREAN_CMU_DICT_MEMBER = "cmudict/cmudict"


@dataclass(frozen=True)
class FrontendResourceStatus:
    """单个前端资源的就绪状态与缺失项。 / Readiness status and missing items for one frontend resource."""

    key: str
    path: Path
    ready: bool
    missing: tuple[str, ...]
    size_bytes: int


def frontends_root() -> Path:
    """返回资源根目录（可用环境变量覆盖）。 / Return the resource root, overridable by environment."""
    override = os.environ.get("TTS_TRAINER_FRONTENDS_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_FRONTENDS_ROOT


def openjtalk_dictionary_path(root: Path | None = None) -> Path:
    """返回 Open JTalk 词典目录路径。 / Return the Open JTalk dictionary directory path."""
    return (root or frontends_root()) / "openjtalk" / OPENJTALK_DICTIONARY_NAME


def inspect_openjtalk_dictionary(root: Path | None = None) -> FrontendResourceStatus:
    """检查 Open JTalk 词典是否完整就绪。 / Check whether the Open JTalk dictionary is complete and ready."""
    path = openjtalk_dictionary_path(root)
    missing = tuple(name for name in OPENJTALK_REQUIRED_FILES if not (path / name).is_file())
    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.exists() else 0
    return FrontendResourceStatus("openjtalk", path, not missing, missing, size)


def korean_nltk_data_path(root: Path | None = None) -> Path:
    return (root or frontends_root()) / "korean" / "nltk_data"


def korean_cmudict_path(root: Path | None = None) -> Path:
    return korean_nltk_data_path(root) / "corpora" / "cmudict.zip"


def inspect_korean_cmudict(root: Path | None = None) -> FrontendResourceStatus:
    """校验韩语 CMU 词典存在且 sha256 匹配。 / Verify the Korean CMU dict exists with a matching sha256."""
    path = korean_cmudict_path(root)
    valid = path.is_file() and _sha256(path) == KOREAN_CMU_DICT_SHA256
    missing = () if valid else ("corpora/cmudict.zip (missing or checksum mismatch)",)
    size = path.stat().st_size if path.is_file() else 0
    return FrontendResourceStatus("korean", path, not missing, missing, size)


@contextmanager
def _download_lock(root: Path, name: str = "openjtalk-dictionary"):
    """O_EXCL 锁文件保证同一资源同时只有一个下载进程。 / An O_EXCL lock file so only one process downloads a resource at a time."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{name}.download.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another process is downloading frontend resource {name}: {lock}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> Path:
    """防路径穿越地解压词典压缩包并定位词典目录。 / Safely extract the dictionary tarball against path traversal and locate its directory."""
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise RuntimeError(f"unsafe path in Open JTalk dictionary archive: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(f"unsupported entry in Open JTalk dictionary archive: {member.name}")
        source.extractall(destination)
    expected = destination / OPENJTALK_DICTIONARY_NAME
    if expected.is_dir():
        return expected
    matches = [path.parent for path in destination.rglob("char.bin")]
    if len(matches) != 1:
        raise RuntimeError("Open JTalk dictionary archive has an unexpected layout")
    return matches[0]


def ensure_openjtalk_dictionary(root: Path | None = None, *, allow_download: bool = True) -> Path:
    """确保 Open JTalk 词典就绪，必要时带锁下载并校验。 / Ensure the Open JTalk dictionary is ready, downloading under a lock if needed."""
    destination_root = root or frontends_root()
    status = inspect_openjtalk_dictionary(destination_root)
    if status.ready:
        logger.info("frontend resource ready key=openjtalk path=%s size_bytes=%d", status.path, status.size_bytes)
        return status.path
    if not allow_download:
        raise FileNotFoundError(
            f"Open JTalk dictionary is incomplete at {status.path}; missing: {', '.join(status.missing)}. "
            "Run: tts-trainer frontends ensure openjtalk"
        )
    resource_root = destination_root / "openjtalk"
    with _download_lock(resource_root):
        # 双重检查：等锁期间其他进程可能已完成下载。
        # Double-check: another process may have finished while we waited.
        status = inspect_openjtalk_dictionary(destination_root)
        if status.ready:
            return status.path
        resource_root.mkdir(parents=True, exist_ok=True)
        archive = resource_root / f".{OPENJTALK_DICTIONARY_NAME}.tar.gz.part"
        logger.info("frontend resource download starting key=openjtalk path=%s", status.path)
        urllib.request.urlretrieve(OPENJTALK_DICTIONARY_URL, archive)
        digest = _sha256(archive)
        if digest != OPENJTALK_DICTIONARY_SHA256:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"Open JTalk dictionary checksum mismatch: expected {OPENJTALK_DICTIONARY_SHA256}, got {digest}"
            )
        with tempfile.TemporaryDirectory(prefix=".openjtalk-extract-", dir=resource_root) as temporary:
            extracted = _safe_extract(archive, Path(temporary))
            if status.path.exists():
                shutil.rmtree(status.path)
            shutil.move(str(extracted), str(status.path))
        archive.unlink(missing_ok=True)
        # 先解压到临时目录再原子移动，解压中断不会破坏现有词典。
        # Extract into a temp dir then atomically move, so interrupts never corrupt the dictionary.
        completed = inspect_openjtalk_dictionary(destination_root)
        if not completed.ready:
            raise RuntimeError(
                "Open JTalk dictionary extraction is incomplete; missing: " + ", ".join(completed.missing)
            )
        marker = {
            "source": OPENJTALK_DICTIONARY_URL,
            "sha256": OPENJTALK_DICTIONARY_SHA256,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": completed.size_bytes,
        }
        (completed.path / ".download-complete.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8",
        )
        logger.info(
            "frontend resource download completed key=openjtalk path=%s size_bytes=%d",
            completed.path, completed.size_bytes,
        )
        return completed.path


def ensure_korean_cmudict(root: Path | None = None, *, allow_download: bool = True) -> Path:
    """确保 g2pk2 的 CMU 词典存在于项目内而非全局。 / Ensure g2pk2's CMU dictionary is present in the project, never globally."""
    destination_root = root or frontends_root()
    status = inspect_korean_cmudict(destination_root)
    if status.ready:
        return korean_nltk_data_path(destination_root)
    if not allow_download:
        raise FileNotFoundError(
            f"Korean G2P dictionary is missing at {status.path}. "
            "Run: tts-trainer frontends ensure korean"
        )
    resource_root = destination_root / "korean"
    with _download_lock(resource_root, "korean-cmudict"):
        # 等锁后重查，避免重复下载。 / Re-check after acquiring the lock to avoid duplicate downloads.
        status = inspect_korean_cmudict(destination_root)
        if status.ready:
            return korean_nltk_data_path(destination_root)
        status.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status.path.with_suffix(".zip.part")
        logger.info("frontend resource download starting key=korean path=%s", status.path)
        urllib.request.urlretrieve(KOREAN_CMU_DICT_URL, temporary)
        digest = _sha256(temporary)
        if digest != KOREAN_CMU_DICT_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Korean CMU dictionary checksum mismatch: expected "
                f"{KOREAN_CMU_DICT_SHA256}, got {digest}"
            )
        try:
            with zipfile.ZipFile(temporary) as archive:
                if KOREAN_CMU_DICT_MEMBER not in archive.namelist():
                    raise RuntimeError("Korean CMU dictionary archive has an unexpected layout")
                archive.getinfo(KOREAN_CMU_DICT_MEMBER)
        except zipfile.BadZipFile as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Korean CMU dictionary download is not a valid ZIP archive") from exc
        temporary.replace(status.path)
        marker = {
            "source": KOREAN_CMU_DICT_URL,
            "sha256": KOREAN_CMU_DICT_SHA256,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": status.path.stat().st_size,
        }
        (resource_root / ".download-complete.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8",
        )
        logger.info(
            "frontend resource download completed key=korean path=%s size_bytes=%d",
            status.path, status.path.stat().st_size,
        )
        return korean_nltk_data_path(destination_root)


# 英语与韩语共用同一份 nltk cmudict 词表；en 导出用它生成 Android 端的
# cmudict_data.json，使部署词典与 g2p-en 训练词典来自同一数据源。
# English and Korean share the same nltk cmudict corpus; the en export turns
# it into the Android cmudict_data.json so the deployed dictionary comes from
# the same data source g2p-en used during training.
def english_cmudict_json_path(root: Path | None = None) -> Path:
    return (root or frontends_root()) / "english" / "cmudict_data.json"


def _cmudict_first_pronunciations(zip_path: Path) -> dict[str, str]:
    """按 nltk 语义解析 cmudict：key 小写、保留撇号、多发音取首个。 / Parse cmudict with nltk semantics: lowercase keys, apostrophes kept, first pronunciation wins."""
    result: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(KOREAN_CMU_DICT_MEMBER) as stream:
            for raw in stream:
                line = raw.decode("latin-1").strip()
                if not line or line.startswith(";;;"):
                    continue
                pieces = line.split()
                if len(pieces) < 3:
                    continue
                word = pieces[0].lower()
                if word not in result:
                    result[word] = " ".join(pieces[2:])
    return result


def build_english_cmudict_json(root: Path | None = None, *, allow_download: bool = True) -> Path:
    """生成 Android 端 cmudict_data.json；键值与 g2p-en 查询结果逐词一致。 / Build the Android cmudict_data.json whose entries match g2p-en lookups word for word."""
    # nltk 的 read_cmudict_block 丢弃发音序号列（pieces[2:]），key 取小写原词；
    # g2p-en 查询 self.cmu[word][0] 即首个发音。C++ loadCmuDict 期望扁平
    # {word: "HH AH0 L OW1"} 且查词同样使用小写带撇号的形态。
    # nltk's read_cmudict_block drops the index column (pieces[2:]) and
    # lowercases keys; g2p-en reads self.cmu[word][0], i.e. the first
    # pronunciation. C++ loadCmuDict expects a flat {word: "HH AH0 L OW1"}
    # and looks words up in the same lowercase-with-apostrophe form.
    destination_root = root or frontends_root()
    target = english_cmudict_json_path(destination_root)
    if target.is_file():
        return target
    ensure_korean_cmudict(destination_root, allow_download=allow_download)
    source = korean_cmudict_path(destination_root)
    entries = _cmudict_first_pronunciations(source)
    if not entries:
        raise RuntimeError("cmudict corpus produced no entries; refusing to write an empty dictionary")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    logger.info(
        "english cmudict_data.json built entries=%d path=%s", len(entries), target,
    )
    return target
