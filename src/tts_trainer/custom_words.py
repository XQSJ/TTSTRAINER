"""定制词定音：官方/品牌/人名等特殊词的读音候选、确认闸门与分发落点。
/ Custom-word pronunciation: candidate readings, a confirmation gate, and
distribution targets for special words (brands, product names, people).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .frontend.resources import (
    english_cmudict_json_path,
    korean_cmudict_path,
    _cmudict_first_pronunciations,
)


logger = logging.getLogger(__name__)

VERIFIED_FILENAME = "custom_words_verified.json"
CUSTOM_WORDS_FILENAME = "custom_words.json"

# shared_words 的基准语言：跨语言共享词落音到英语词典，并生成部署侧
# custom-wordlist 供各语言 code-switching 混排查读。
# Base language for shared words: readings land in the English dictionary
# and ship as the deployment custom-wordlist for code-switched text.
SHARED_BASE_LANGUAGE = "en"


def load_custom_words(config_path: str | Path) -> dict:
    """读取训练配置内的 custom_words 块（随 preset/extends 合并）；未声明
    返回空结构。 / Read the config's embedded custom_words block (merged
    through preset/extends); return an empty shape when absent."""
    from .project_config import load_project_config
    raw = load_project_config(config_path).get("custom_words") or {}
    native: dict[str, dict[str, dict]] = {}
    for language, words in (raw.get("native_words") or {}).items():
        native[str(language)] = {str(w): dict(v or {}) for w, v in words.items()}
    shared = {str(w): dict(v or {}) for w, v in (raw.get("shared_words") or {}).items()}
    return {"native_words": native, "shared_words": shared}


def has_custom_words(custom: dict) -> bool:
    """配置里是否有任何定制词（决定确认闸门是否进入）。 / Whether any
    customized word exists (decides whether the gate runs at all)."""
    return bool(custom.get("shared_words")) or any(
        words for words in custom.get("native_words", {}).values()
    )


def verified_path(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent / VERIFIED_FILENAME


def load_verified(config_path: str | Path) -> dict:
    """读取已确认定稿：{language: {word: {"arpabet": ..., "respelling": ...}}}。 / Load
    confirmed readings: {language: {word: {"arpabet": ..., "respelling": ...}}}."""
    path = verified_path(config_path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_verified(config_path: str | Path, verified: dict) -> Path:
    path = verified_path(config_path)
    path.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _g2p_en_predictor():
    from g2p_en import G2p
    return G2p()


def candidate_arpabet(language: str, word: str, hint: dict) -> dict:
    """为一个定制词生成候选读音（A 机制）。hint 可带 reference（近似拼写）。

    英语返回 ARPAbet 字符串（cmudict 值格式）；其他语言暂以近似拼写记录
    并在导出时写 custom-wordlist（查读由部署端前端完成）。

    / Produce a candidate reading for one customized word (mechanism A). A
    hint may carry a reference respelling. English returns an ARPAbet string
    (cmudict value format); other languages record the respelling and ship
    via the deployment custom-wordlist.
    """
    reference = str(hint.get("reference") or "").strip()
    if language == SHARED_BASE_LANGUAGE or language == "en":
        if reference:
            # 参考拼写本身先进 cmudict 查，查不到再走预测。 / The reference
            # respelling is looked up in the cmudict first, then predicted.
            entries = _cmudict_first_pronunciations(korean_cmudict_path())
            words = [w for w in reference.lower().split() if w]
            arpabet = [entries.get(w) for w in words]
            if all(arpabet):
                return {"arpabet": " ".join(arpabet), "respelling": reference}
        predictor = _g2p_en_predictor()
        tokens = [t for t in predictor.predict(word.lower()) if t.isascii()]
        return {"arpabet": " ".join(tokens), "respelling": reference or None}
    # 非英语：记录近似拼写（部署端 custom-wordlist 查读）。 / Non-English:
    # record the respelling; the deployment reads it via the custom-wordlist.
    return {"arpabet": None, "respelling": reference or word}


def build_candidates(custom: dict, languages: tuple[str, ...]) -> dict:
    """为全部定制词生成候选。返回 {language: {word: candidate}}。 /
    Build candidates for every customized word: {language: {word: candidate}}."""
    result: dict[str, dict[str, dict]] = {}
    for language, words in custom.get("native_words", {}).items():
        if language not in languages and language != SHARED_BASE_LANGUAGE:
            logger.warning("custom native_words for unselected language %s skipped", language)
            continue
        result.setdefault(language, {})
        for word, hint in words.items():
            result[language][word] = candidate_arpabet(language, word, hint)
    for word, hint in custom.get("shared_words", {}).items():
        result.setdefault(SHARED_BASE_LANGUAGE, {})
        result[SHARED_BASE_LANGUAGE][word] = candidate_arpabet(SHARED_BASE_LANGUAGE, word, hint)
    return result


def synthesize_listen(language: str, word: str, respelling: str | None,
                       output_dir: Path, voice: dict | None = None) -> Path | None:
    """B 机制：用 QwenTTS 按当前音色把候选读音念出来，生成试听 WAV。 /
    Mechanism B: render the candidate with QwenTTS in the current voice for
    listening. Returns the WAV path, or None when the Qwen runtime is absent."""
    try:
        from .qwen_teacher import load_qwen_teacher
    except Exception:
        return None
    text = respelling or word
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"listen-{language}-{word}.wav"
    try:
        model = load_qwen_teacher(voice.get("model_key", "base-1.7b") if voice else "base-1.7b")
        audio = model.tts(text=text)  # 简化调用；音色引用由 voice 上下文提供 / simplified call
        # qwen-tts 返回采样数据的具体字段随版本而异，此处保守写 WAV。 /
        # The qwen-tts payload shape varies; write whatever waveform it returns.
        import soundfile as sf
        sf.write(str(target), audio if not isinstance(audio, tuple) else audio[0],
                 24000)
        return target
    except Exception as error:  # pragma: no cover - depends on optional runtime
        logger.warning("listen synthesis failed for %s: %s", word, error)
        return None


def merge_into_english_cmudict(verified: dict, entries: dict[str, str]) -> int:
    """把已确认的英语/shared 读音合并进 cmudict 词条表，返回合并数。 /
    Merge confirmed English/shared readings into the cmudict entries."""
    merged = 0
    for language, words in verified.items():
        target = SHARED_BASE_LANGUAGE if language == "shared" else language
        if target != "en":
            continue
        for word, reading in words.items():
            arpabet = reading.get("arpabet")
            if arpabet:
                entries[word.lower()] = arpabet
                merged += 1
    return merged


def deployment_wordlist(verified: dict) -> dict[str, str]:
    """生成部署侧 custom-wordlist（词→近似拼写），供 native 注入与应急。 /
    Produce the deployment custom-wordlist (word → respelling)."""
    result: dict[str, str] = {}
    for language, words in verified.items():
        for word, reading in words.items():
            respelling = reading.get("respelling")
            if respelling:
                result[word.lower()] = respelling
    return result


def run_custom_words_gate(config_path: str, *, listen: bool = False,
                          listen_dir: str = "artifacts/custom_words_listen",
                          interactive: bool = True) -> str:
    """阶段② 确认闸门：生成候选 → 逐词过用户 → 产出 custom_words_verified.json。

    未配置 custom_words.json 时直接返回跳过（原流程零影响）；--non-interactive
    全部接受自动候选（CI 模式）；--listen 用 QwenTTS 按当前音色渲染试听。

    / Stage-2 confirmation gate: build candidates, walk the user through every
    word, and emit custom_words_verified.json. With no custom_words.json the
    gate is skipped entirely; --non-interactive accepts every auto candidate
    (CI mode); --listen renders listen-along WAVs with QwenTTS.
    """
    from .project_config import load_project_config
    raw = load_project_config(config_path)
    languages = tuple(raw.get("experiment", {}).get("languages", ()))

    custom = load_custom_words(config_path)
    if not has_custom_words(custom):
        return "custom-words gate skipped (no custom_words.json)"

    candidates = build_candidates(custom, languages)
    verified: dict[str, dict[str, dict]] = {}

    print("=" * 64)
    print("定制词确认闸门 / Special-word confirmation gate")
    print("=" * 64)
    for language in sorted(candidates):
        for word, candidate in sorted(candidates[language].items()):
            arpabet = candidate.get("arpabet")
            respelling = candidate.get("respelling")
            print(f"\n[{language}] {word}")
            print(f"  自动候选 ARPAbet: {arpabet or '(无,近似拼写走部署查读)'}")
            if respelling:
                print(f"  近似拼写: {respelling}")
            if listen:
                audio = synthesize_listen(language, word, respelling,
                                          Path(listen_dir))
                if audio:
                    print(f"  试听音频: {audio}")
                else:
                    print("  试听音频: QwenTTS 不可用,跳过 (可后补)")
            if not interactive:
                print("  -> 自动接受 (non-interactive)")
                verified.setdefault(language, {})[word] = candidate
                continue
            while True:
                answer = input("  确认这个读音? [y]确认 [s]跳过该词 [r]重试试听: ").strip().lower()
                if answer in ("y", ""):
                    verified.setdefault(language, {})[word] = candidate
                    break
                if answer == "s":
                    print("  -> 跳过 (该词回到词典/兜底默认路径)")
                    break
                if answer == "r" and listen:
                    audio = synthesize_listen(language, word, respelling,
                                              Path(listen_dir))
                    if audio:
                        print(f"  试听音频: {audio}")
                    continue
                if answer == "r":
                    print("  未启用 --listen,无法重试试听")
                    continue
                print("  请输入 y/s/r")

    path = save_verified(config_path, verified)
    total = sum(len(words) for words in verified.values())
    return (f"custom-words gate done: confirmed={total} "
            f"skipped={sum(len(w) for w in candidates.values()) - total} "
            f"verified={path}")
