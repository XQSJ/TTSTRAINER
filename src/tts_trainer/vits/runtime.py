from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..frontend import (FrontendContract, FrontendRouter, chunk_text,
                        frontend_from_contract)
from ..frontend.contract import (LEGACY_PIPER_TOKEN_ENCODING,
                                 PIPER_TOKEN_ENCODING)


LOGGER = logging.getLogger(__name__)


class OnnxTTS:
    """Reference runtime for the single multilingual ONNX core.

    Mobile implementations should reproduce this token/profile mapping before
    calling ONNX Runtime. It intentionally does not rely on a global model cache.
    """
    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort
        self.model_dir = Path(model_dir)
        config = json.loads((self.model_dir / "model.onnx.json").read_text(encoding="utf-8"))
        tokens = json.loads((self.model_dir / "tokens.json").read_text(encoding="utf-8"))["tokens"]
        self.token_ids = {token: index for index, token in enumerate(tokens)}
        self.sample_rate = int(config["sample_rate"])
        self.wire_token_encoding = str(
            config.get("wire_token_encoding", "")
        )
        frontend_raw = config.get("frontend")
        self.frontend_contract = FrontendContract.from_dict(frontend_raw) if isinstance(frontend_raw, dict) else None
        self.profiles = {(row["speaker"], row["language"]): row["sid"] for row in config["voice_profiles"]}
        self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    def encode(self, units: tuple[str, ...]) -> np.ndarray:
        unknown = [unit for unit in units if unit not in self.token_ids]
        if unknown:
            raise ValueError(f"tokens not present in model vocabulary: {sorted(set(unknown))!r}")
        encoded = [self.token_ids[unit] for unit in units]
        if self.wire_token_encoding == LEGACY_PIPER_TOKEN_ENCODING:
            pad = self.token_ids["_"]
            encoded = [
                value
                for token_id in encoded
                for value in (token_id, pad)
            ]
        elif (
            self.frontend_contract
            and self.frontend_contract.token_encoding == PIPER_TOKEN_ENCODING
        ):
            pad = self.token_ids["_"]
            encoded = [
                value
                for token_id in encoded
                for value in (token_id, pad)
            ]
            encoded.insert(0, pad)
        return np.asarray(
            [[self.token_ids["^"], *encoded, self.token_ids["$"]]],
            dtype=np.int64,
        )

    def synthesize_units(self, units: tuple[str, ...], *, language: str, speaker: str,
                         noise_scale: float = 0.667, length_scale: float = 1.0,
                         duration_noise_scale: float = 0.35) -> np.ndarray:
        if noise_scale < 0.0:
            raise ValueError("noise_scale must not be negative")
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive")
        if duration_noise_scale < 0.0:
            raise ValueError("duration_noise_scale must not be negative")
        try:
            sid = self.profiles[(speaker, language)]
        except KeyError as exc:
            raise ValueError(f"unknown voice profile: speaker={speaker!r}, language={language!r}") from exc
        tokens = self.encode(units)
        return self.session.run(None, {
            "input": tokens,
            "input_lengths": np.asarray([tokens.shape[1]], dtype=np.int64),
            "scales": np.asarray(
                [noise_scale, length_scale, duration_noise_scale],
                dtype=np.float32,
            ),
            "sid": np.asarray([sid], dtype=np.int64),
        })[0][0, 0]

    def synthesize_text(self, text: str, *, language: str, speaker: str,
                        frontend: FrontendRouter | None = None,
                        allow_frontend_version_mismatch: bool = False,
                        auto_chunk: bool = True,
                        max_phoneme_tokens: int = 90,
                        sentence_pause_ms: int = 180,
                        clause_pause_ms: int = 100,
                        chunk_pause_ms: int = 60,
                        **scales) -> np.ndarray:
        if frontend is None:
            if not self.frontend_contract:
                raise RuntimeError("model has no frontend contract; supply a FrontendRouter")
            frontend = frontend_from_contract(self.frontend_contract)
        profile = self.frontend_contract.languages.get(language, {}) if self.frontend_contract else {}
        expected = profile.get("engine_version") or (
            self.frontend_contract.engine_version if self.frontend_contract else None
        )
        if expected and not allow_frontend_version_mismatch:
            actual = frontend.version_for(language)
            if actual != expected:
                raise RuntimeError(
                    f"frontend version mismatch: model expects {expected!r}, runtime has {actual!r}; "
                    "use the matching eSpeak-ng build or explicitly allow the mismatch"
                )
        if sentence_pause_ms < 0 or clause_pause_ms < 0 or chunk_pause_ms < 0:
            raise ValueError("pause durations must not be negative")
        if not auto_chunk:
            return self.synthesize_units(
                frontend.phonemize(text, language), language=language,
                speaker=speaker, **scales,
            )

        chunks = chunk_text(
            text, language, frontend.phonemize,
            max_phoneme_tokens=max_phoneme_tokens,
        )
        LOGGER.info(
            "TEXT CHUNKS | language=%s | chunks=%d | max_phoneme_tokens=%d | tokens=%s",
            language, len(chunks), max_phoneme_tokens,
            ",".join(str(len(chunk.units)) for chunk in chunks),
        )
        rendered: list[np.ndarray] = []
        for index, chunk in enumerate(chunks, start=1):
            LOGGER.info(
                "TEXT CHUNK %d/%d | phoneme_tokens=%d | text=%s",
                index, len(chunks), len(chunk.units), chunk.text,
            )
            rendered.append(self.synthesize_units(
                chunk.units, language=language, speaker=speaker, **scales,
            ))
            if index == len(chunks):
                continue
            if chunk.pause_kind == "sentence":
                pause_ms = sentence_pause_ms
            elif chunk.pause_kind == "clause":
                pause_ms = clause_pause_ms
            else:
                pause_ms = chunk_pause_ms
            pause_samples = round(self.sample_rate * pause_ms / 1000.0)
            if pause_samples:
                rendered.append(np.zeros(pause_samples, dtype=np.float32))
        return np.concatenate(rendered).astype(np.float32, copy=False)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), samples, sample_rate, subtype="PCM_16")
    return target
