from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifest import Item


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _edge_silence_seconds(samples: np.ndarray, threshold: float,
                          sample_rate: int) -> tuple[float, float]:
    active = np.flatnonzero(np.abs(samples) > threshold)
    if not active.size:
        duration = len(samples) / sample_rate
        return duration, duration
    return active[0] / sample_rate, (len(samples) - 1 - active[-1]) / sample_rate


def inspect_audio_item(item: Item, config: dict) -> dict:
    samples, sample_rate = sf.read(str(item.audio), dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32).squeeze()
    if samples.ndim != 1:
        raise ValueError(f"{item.audio}: quality inspection expects mono audio")
    duration = len(samples) / sample_rate
    absolute = np.abs(samples)
    peak = float(absolute.max()) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64))) if samples.size else 0.0
    clipping_ratio = float(np.mean(absolute >= float(config.get("clipping_amplitude", 0.999)))) \
        if samples.size else 0.0
    silence_threshold = 10.0 ** (float(config.get("silence_threshold_dbfs", -45.0)) / 20.0)
    leading_silence, trailing_silence = _edge_silence_seconds(
        samples, silence_threshold, sample_rate,
    )
    unit_count = len(item.phonemes) if item.phonemes else len(item.text)
    units_per_second = unit_count / max(duration, 1e-9)
    metrics = {
        "duration_seconds": duration,
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        "dc_offset": float(abs(np.mean(samples, dtype=np.float64))) if samples.size else 0.0,
        "clipping_ratio": clipping_ratio,
        "leading_silence_seconds": leading_silence,
        "trailing_silence_seconds": trailing_silence,
        "units_per_second": units_per_second,
    }
    failures = []
    checks = (
        (duration < float(config.get("minimum_duration_seconds", 0.4)), "audio_too_short"),
        (duration > float(config.get("maximum_duration_seconds", 30.0)), "audio_too_long"),
        (metrics["rms_dbfs"] < float(config.get("minimum_rms_dbfs", -45.0)), "audio_too_quiet"),
        (metrics["rms_dbfs"] > float(config.get("maximum_rms_dbfs", -6.0)), "audio_too_loud"),
        (clipping_ratio > float(config.get("maximum_clipping_ratio", 0.001)), "audio_clipping"),
        (metrics["dc_offset"] > float(config.get("maximum_dc_offset", 0.05)), "dc_offset"),
        (leading_silence > float(config.get("maximum_edge_silence_seconds", 1.5)), "leading_silence"),
        (trailing_silence > float(config.get("maximum_edge_silence_seconds", 1.5)), "trailing_silence"),
        (units_per_second < float(config.get("minimum_units_per_second", 0.5)), "speech_too_slow"),
        (units_per_second > float(config.get("maximum_units_per_second", 40.0)), "speech_too_fast"),
    )
    failures.extend(name for failed, name in checks if failed)
    return {
        "audio": str(item.audio),
        "text": item.text,
        "language": item.language,
        "speaker": item.speaker,
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
    }


def run_audio_quality_gate(items: list[Item], config: dict,
                           output_path: str | Path) -> dict:
    results = [inspect_audio_item(item, config) for item in items]
    failure_counts = Counter(
        failure for result in results for failure in result["failures"]
    )
    report = {
        "format": 1,
        "provider": "signal-quality-v1",
        "items": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "failure_counts": dict(sorted(failure_counts.items())),
        "thresholds": dict(config),
        "results": results,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["failed"] and config.get("fail_on_error", True):
        summary = ", ".join(f"{key}={value}" for key, value in failure_counts.items())
        raise ValueError(
            f"audio quality gate rejected {report['failed']}/{report['items']} items: {summary}; "
            f"see {target}"
        )
    return report
