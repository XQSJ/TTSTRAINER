"""语义质量门禁：ASR 回转写与说话人相似度。 / Semantic quality gate: ASR round-trip and speaker similarity."""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

from .manifest import Item
from .logging_utils import TerminalProgress, format_duration, progress_bar
from .quality_models import ensure_quality_model


# 这些语言无词边界，用字符错误率而非词错误率。 / No word boundaries; use character error rate instead of WER.
CHARACTER_ERROR_LANGUAGES = {"zh", "ja"}
logger = logging.getLogger(__name__)


def _normalized_characters(text: str) -> list[str]:
    """NFKC + 小写后的有效字符列表。 / Alphanumeric character list after NFKC + casefold."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [character for character in normalized if character.isalnum()]


def _normalized_words(text: str) -> list[str]:
    """NFKC + 小写后的词列表。 / Word list after NFKC + casefold."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return normalized.split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """滚动一行的 Levenshtein 编辑距离。 / Levenshtein distance with one rolling row."""
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def text_error_rate(reference: str, hypothesis: str, language: str) -> tuple[str, float]:
    """按语言选择 CER/WER 并返回错误率。 / Choose CER or WER by language and return the error rate."""
    if language in CHARACTER_ERROR_LANGUAGES:
        metric = "cer"
        expected = _normalized_characters(reference)
        actual = _normalized_characters(hypothesis)
    else:
        metric = "wer"
        expected = _normalized_words(reference)
        actual = _normalized_words(hypothesis)
    return metric, edit_distance(expected, actual) / max(len(expected), 1)


class FasterWhisperEvaluator:
    """Faster-Whisper ASR 评测器。 / Faster-Whisper ASR evaluator."""

    def __init__(self, model_path: Path, *, device: str = "cpu",
                 compute_type: str = "int8", beam_size: int = 5):
        # 可选依赖缺失时给出安装指引而非裸 ImportError。 / Give an install hint instead of a bare ImportError.
        if importlib.util.find_spec("faster_whisper") is None:
            raise RuntimeError(
                "ASR quality evaluation requires: pip install -e '.[quality]'"
            )
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            str(model_path), device=device, compute_type=compute_type,
            local_files_only=True,
        )
        self.beam_size = beam_size

    def transcribe(self, audio: Path, language: str) -> str:
        """转写音频为文本。 / Transcribe audio to text."""
        # 关闭 VAD 与上文条件，保证短样本结果可复现。 / Disable VAD and text conditioning for reproducibility on short clips.
        segments, _ = self.model.transcribe(
            str(audio), language=language, beam_size=self.beam_size,
            vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class SpeechBrainSpeakerEvaluator:
    """SpeechBrain ECAPA 说话人相似度评测器。 / SpeechBrain ECAPA speaker-similarity evaluator."""

    def __init__(self, model_path: Path, *, device: str = "cpu"):
        if importlib.util.find_spec("speechbrain") is None:
            raise RuntimeError(
                "speaker quality evaluation requires: pip install -e '.[quality]'"
            )
        from speechbrain.inference.speaker import SpeakerRecognition
        self.model = SpeakerRecognition.from_hparams(
            source=str(model_path),
            savedir=str(model_path / ".runtime"),
            run_opts={"device": device},
        )

    def similarity(self, reference: Path, audio: Path) -> float:
        """返回参考音频与待测音频的说话人相似度。 / Speaker similarity between reference and candidate audio."""
        score, _ = self.model.verify_files(str(reference), str(audio))
        return float(score.squeeze().item())


def _speaker_references(items: list[Item], configured: dict,
                        reference_root: Path | None) -> dict[str, Path]:
    """合并显式配置与目录扫描得到说话人参考音频。 / Merge explicit config with directory scanning for speaker references."""
    references = {
        speaker: Path(path).expanduser().resolve()
        for speaker, path in configured.items()
    }
    # 未显式配置的说话人在参考目录下按 speaker.* 唯一匹配。 / Unconfigured speakers match a unique speaker.* file in the root.
    if reference_root and reference_root.is_dir():
        for speaker in {item.speaker for item in items} - set(references):
            matches = sorted(reference_root.glob(f"{speaker}.*"))
            if len(matches) == 1:
                references[speaker] = matches[0].resolve()
    return references


def run_semantic_quality_gate(
    items: list[Item], config: dict, output_path: str | Path,
    *, reference_root: Path | None = None,
    asr_evaluator=None, speaker_evaluator=None,
) -> dict:
    """运行语义质检门禁并输出 JSON 报告。 / Run the semantic quality gate and write a JSON report."""
    asr_config = config.get("asr", {})
    speaker_config = config.get("speaker", {})
    if not asr_config.get("enabled", False) and not speaker_config.get("enabled", False):
        raise ValueError("semantic quality is enabled but both ASR and speaker checks are disabled")

    if asr_config.get("enabled", False) and asr_evaluator is None:
        model = ensure_quality_model(
            asr_config.get("model", "asr-small"),
            allow_download=bool(asr_config.get("auto_download_model", False)),
        )
        asr_evaluator = FasterWhisperEvaluator(
            model, device=asr_config.get("device", "cpu"),
            compute_type=asr_config.get("compute_type", "int8"),
            beam_size=int(asr_config.get("beam_size", 5)),
        )
    if speaker_config.get("enabled", False) and speaker_evaluator is None:
        model = ensure_quality_model(
            speaker_config.get("model", "speaker-ecapa"),
            allow_download=bool(speaker_config.get("auto_download_model", False)),
        )
        speaker_evaluator = SpeechBrainSpeakerEvaluator(
            model, device=speaker_config.get("device", "cpu"),
        )

    references = _speaker_references(
        items, speaker_config.get("references", {}), reference_root,
    )
    if speaker_evaluator is not None:
        missing = sorted({item.speaker for item in items} - set(references))
        if missing:
            raise ValueError(
                "speaker quality references are missing for: " + ", ".join(missing)
            )
        unavailable = sorted(
            speaker for speaker, path in references.items() if not path.is_file()
        )
        if unavailable:
            raise FileNotFoundError(
                "speaker quality reference files are missing for: " + ", ".join(unavailable)
            )

    total = len(items)
    # 日志节流：默认约每 5% 打一条，避免刷屏。 / Throttled logging: roughly one line per 5% by default.
    interval = max(1, int(config.get("progress_every_items", max(1, total // 20))))
    started = time.monotonic()
    results = []
    live_progress = TerminalProgress("SEMANTIC QUALITY", total)
    logger.info(
        "SEMANTIC QUALITY START | total=%d | asr=%s | speaker=%s | progress_every_items=%d",
        total, asr_evaluator is not None, speaker_evaluator is not None, interval,
    )
    for index, item in enumerate(items, 1):
        row = {
            "audio": str(item.audio), "text": item.text,
            "language": item.language, "speaker": item.speaker,
            "failures": [],
        }
        if asr_evaluator is not None:
            transcript = asr_evaluator.transcribe(item.audio, item.language)
            metric, error_rate = text_error_rate(item.text, transcript, item.language)
            row["asr"] = {"transcript": transcript, "metric": metric, "error_rate": error_rate}
            if error_rate > float(asr_config.get("maximum_error_rate", 0.15)):
                row["failures"].append("asr_text_mismatch")
        if speaker_evaluator is not None:
            similarity = speaker_evaluator.similarity(references[item.speaker], item.audio)
            row["speaker"] = {
                "reference": str(references[item.speaker]), "similarity": similarity,
            }
            if similarity < float(speaker_config.get("minimum_similarity", 0.25)):
                row["failures"].append("speaker_mismatch")
        row["passed"] = not row["failures"]
        results.append(row)
        live_progress.update(index, f"language={item.language} speaker={item.speaker}")
        if index % interval == 0 or index == total:
            live_progress.clear()
            elapsed = time.monotonic() - started
            rate = index / max(elapsed, 1e-9)
            logger.info(
                "SEMANTIC QUALITY %s %6.2f%% | checked=%d/%d | failed=%d | speed=%.2f/min | ETA=%s",
                progress_bar(index, total), 100.0 * index / max(total, 1),
                index, total, sum(not result["passed"] for result in results),
                rate * 60.0, format_duration((total - index) / rate),
                extra={"tts_style": "progress"},
            )
            live_progress.update(index, f"language={item.language} speaker={item.speaker}")
    live_progress.close()

    failure_counts = Counter(failure for row in results for failure in row["failures"])
    report = {
        "format": 1,
        "provider": "semantic-quality-v1",
        "items": len(results),
        "passed": sum(row["passed"] for row in results),
        "failed": sum(not row["passed"] for row in results),
        "failure_counts": dict(sorted(failure_counts.items())),
        "results": results,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "SEMANTIC QUALITY DONE | passed=%d | failed=%d | elapsed=%s | report=%s",
        report["passed"], report["failed"],
        format_duration(time.monotonic() - started), target,
        extra={"tts_style": "success" if not report["failed"] else ""},
    )
    if report["failed"] and config.get("fail_on_error", True):
        raise ValueError(
            f"semantic quality gate rejected {report['failed']}/{report['items']} items; see {target}"
        )
    return report
