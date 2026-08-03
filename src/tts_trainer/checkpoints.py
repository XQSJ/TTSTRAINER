from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path


# 格式 1/2 的文本先验训练语义有误；格式 3 可迁移主干，但确定性时长预测器
# 与格式 4 的随机时长模型不兼容。
# Formats 1/2 have broken text-prior semantics. Format 3 can warm-start the
# backbone, but its deterministic duration predictor is incompatible with v4.
CHECKPOINT_FORMAT = 4
WARM_START_FORMATS = frozenset((3, CHECKPOINT_FORMAT))
TRAINING_OBJECTIVE = "standard-vits-mel-kl-duration-gan-feature-v1"


def inherit_resume_best_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    selection: dict | None,
) -> Path | None:
    """把续训来源的历史 best 带到新实验。 / Carry resume best into a new run."""
    if not selection or selection.get("best_epoch") is None:
        return None
    source = Path(checkpoint)
    target = Path(destination)
    if target.exists():
        return target
    best_epoch = int(selection["best_epoch"])
    candidates = (source, source.parent / "best")
    source_best = None
    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        state_path = candidate / "training-state.pt"
        if not metadata_path.is_file() or not state_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("epoch", -1)) == best_epoch:
            source_best = candidate
            break
    if source_best is None:
        return None
    if source_best.resolve() == target.resolve():
        return target

    # 先完整复制到临时目录，防止中断后留下半个 best。
    # Copy atomically so an interrupted run cannot leave a partial best.
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.inherit-tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source_best, temporary)
    temporary.replace(target)
    return target


def require_checkpoint_format(value: int) -> None:
    if value == 1:
        raise ValueError(
            "checkpoint format 1 has an untrained text prior and produces noisy "
            "text-only inference; update TTSTRAINER and retrain from scratch with "
            "a new experiment.name"
        )
    if value == 2:
        raise ValueError(
            "checkpoint format 2 used an incorrectly initialized position-free "
            "text encoder and an affine flow incompatible with its KL objective; "
            "update TTSTRAINER and retrain from scratch with a new experiment.name"
        )
    if value == 3:
        raise ValueError(
            "checkpoint format 3 uses the legacy deterministic duration predictor; "
            "start a new experiment with initialization.mode=warm_start to reuse its "
            "text encoder, speaker/language embeddings, flow and decoder"
        )
    if value != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {value}; expected {CHECKPOINT_FORMAT}"
        )


def require_warm_start_checkpoint_format(value: int) -> None:
    """仅接受可安全迁移推理主干的 checkpoint。 / Accept safe backbones only."""
    if value in WARM_START_FORMATS:
        return
    require_checkpoint_format(value)


def save_training_checkpoint(directory: str | Path, *, generator, discriminator,
                             optimizer_g, optimizer_d, epoch: int, global_step: int,
                             config, language_map: dict, speaker_map: dict,
                             tokens: list[str], metrics: dict | None = None,
                             frontend: dict | None = None,
                             frontend_conformance: dict | None = None,
                             selection: dict | None = None,
                             data_split: dict | None = None,
                             quality_summary: dict | None = None,
                             audio=None,
                             scheduler_g=None, scheduler_d=None, scaler=None,
                             optimizer_refinement=None,
                             scheduler_refinement=None,
                             training_phase: dict | None = None) -> Path:
    import torch
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "training_objective": TRAINING_OBJECTIVE,
        "epoch": epoch,
        "global_step": global_step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict() if scheduler_g else None,
        "scheduler_d": scheduler_d.state_dict() if scheduler_d else None,
        "optimizer_refinement": (
            optimizer_refinement.state_dict() if optimizer_refinement else None
        ),
        "scheduler_refinement": (
            scheduler_refinement.state_dict() if scheduler_refinement else None
        ),
        "training_phase": training_phase,
        "scaler": scaler.state_dict() if scaler else None,
    }
    temporary = destination / "training-state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(destination / "training-state.pt")
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "training_objective": TRAINING_OBJECTIVE,
        "epoch": epoch,
        "global_step": global_step,
        "config": asdict(config) if is_dataclass(config) else config,
        "language_map": language_map,
        "speaker_map": speaker_map,
        "tokens": tokens,
        "frontend": frontend,
        "frontend_conformance": frontend_conformance,
        "selection": selection,
        "data_split": data_split,
        "quality_summary": quality_summary,
        "audio": asdict(audio) if is_dataclass(audio) else audio,
        "training_phase": training_phase,
        # Mobile/Piper 把 token 0 同时用作有效音素间 blank；此标记说明该行可训练。
        # Mobile/Piper also uses token 0 as a valid inter-phoneme blank; this
        # marker certifies that the embedding row was trainable.
        "learned_blank_token": True,
        "metrics": metrics or {},
    }
    metadata_tmp = destination / "metadata.json.tmp"
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_tmp.replace(destination / "metadata.json")
    return destination


def load_training_checkpoint(directory: str | Path, *, generator, discriminator=None,
                             optimizer_g=None, optimizer_d=None, scheduler_g=None,
                             scheduler_d=None, scaler=None, map_location="cpu",
                             optimizer_refinement=None,
                             scheduler_refinement=None) -> dict:
    import torch
    source = Path(directory)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    state = torch.load(source / "training-state.pt", map_location=map_location, weights_only=False)
    require_checkpoint_format(int(metadata["format"]))
    require_checkpoint_format(int(state["format"]))
    for label, payload in (("metadata", metadata), ("training state", state)):
        objective = payload.get("training_objective")
        if objective is not None and objective != TRAINING_OBJECTIVE:
            raise ValueError(
                f"unsupported {label} objective {objective!r}; "
                f"expected {TRAINING_OBJECTIVE!r}"
            )
    generator.load_state_dict(state["generator"])
    if discriminator is not None: discriminator.load_state_dict(state["discriminator"])
    if optimizer_g is not None: optimizer_g.load_state_dict(state["optimizer_g"])
    if optimizer_d is not None: optimizer_d.load_state_dict(state["optimizer_d"])
    if scheduler_g is not None and state.get("scheduler_g") is not None:
        scheduler_g.load_state_dict(state["scheduler_g"])
    if scheduler_d is not None and state.get("scheduler_d") is not None:
        scheduler_d.load_state_dict(state["scheduler_d"])
    if (
        optimizer_refinement is not None
        and state.get("optimizer_refinement") is not None
    ):
        optimizer_refinement.load_state_dict(state["optimizer_refinement"])
    if (
        scheduler_refinement is not None
        and state.get("scheduler_refinement") is not None
    ):
        scheduler_refinement.load_state_dict(state["scheduler_refinement"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return {**metadata, "state": state}
