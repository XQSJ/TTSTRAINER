from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..frontend import FrontendContract, FrontendRouter, frontend_from_contract


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
        frontend_raw = config.get("frontend")
        self.frontend_contract = FrontendContract.from_dict(frontend_raw) if isinstance(frontend_raw, dict) else None
        self.profiles = {(row["speaker"], row["language"]): row["sid"] for row in config["voice_profiles"]}
        self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    def encode(self, units: tuple[str, ...]) -> np.ndarray:
        unknown = [unit for unit in units if unit not in self.token_ids]
        if unknown:
            raise ValueError(f"tokens not present in model vocabulary: {sorted(set(unknown))!r}")
        return np.asarray([[self.token_ids["^"], *(self.token_ids[unit] for unit in units), self.token_ids["$"]]], dtype=np.int64)

    def synthesize_units(self, units: tuple[str, ...], *, language: str, speaker: str,
                         noise_scale: float = 0.667, length_scale: float = 1.0,
                         duration_scale: float = 1.0) -> np.ndarray:
        try:
            sid = self.profiles[(speaker, language)]
        except KeyError as exc:
            raise ValueError(f"unknown voice profile: speaker={speaker!r}, language={language!r}") from exc
        tokens = self.encode(units)
        return self.session.run(None, {
            "input": tokens,
            "input_lengths": np.asarray([tokens.shape[1]], dtype=np.int64),
            "scales": np.asarray([noise_scale, length_scale, duration_scale], dtype=np.float32),
            "sid": np.asarray([sid], dtype=np.int64),
        })[0][0, 0]

    def synthesize_text(self, text: str, *, language: str, speaker: str,
                        frontend: FrontendRouter | None = None,
                        allow_frontend_version_mismatch: bool = False,
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
        return self.synthesize_units(frontend.phonemize(text, language), language=language,
                                     speaker=speaker, **scales)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), samples, sample_rate, subtype="PCM_16")
    return target
