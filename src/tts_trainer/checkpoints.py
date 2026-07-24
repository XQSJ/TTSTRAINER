from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


# Formats 1 and 2 were produced with broken text-prior training semantics.
# Do not silently present those checkpoints as compatible with format 3.
CHECKPOINT_FORMAT = 3


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
    if value != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {value}; expected {CHECKPOINT_FORMAT}"
        )


def save_training_checkpoint(directory: str | Path, *, generator, discriminator,
                             optimizer_g, optimizer_d, epoch: int, global_step: int,
                             config, language_map: dict, speaker_map: dict,
                             tokens: list[str], metrics: dict | None = None,
                             frontend: dict | None = None,
                             frontend_conformance: dict | None = None,
                             selection: dict | None = None,
                             data_split: dict | None = None,
                             quality_summary: dict | None = None,
                             scheduler_g=None, scheduler_d=None, scaler=None) -> Path:
    import torch
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "epoch": epoch,
        "global_step": global_step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict() if scheduler_g else None,
        "scheduler_d": scheduler_d.state_dict() if scheduler_d else None,
        "scaler": scaler.state_dict() if scaler else None,
    }
    temporary = destination / "training-state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(destination / "training-state.pt")
    metadata = {
        "format": CHECKPOINT_FORMAT,
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
        "metrics": metrics or {},
    }
    metadata_tmp = destination / "metadata.json.tmp"
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_tmp.replace(destination / "metadata.json")
    return destination


def load_training_checkpoint(directory: str | Path, *, generator, discriminator=None,
                             optimizer_g=None, optimizer_d=None, scheduler_g=None,
                             scheduler_d=None, scaler=None, map_location="cpu") -> dict:
    import torch
    source = Path(directory)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    state = torch.load(source / "training-state.pt", map_location=map_location, weights_only=False)
    require_checkpoint_format(int(metadata["format"]))
    require_checkpoint_format(int(state["format"]))
    generator.load_state_dict(state["generator"])
    if discriminator is not None: discriminator.load_state_dict(state["discriminator"])
    if optimizer_g is not None: optimizer_g.load_state_dict(state["optimizer_g"])
    if optimizer_d is not None: optimizer_d.load_state_dict(state["optimizer_d"])
    if scheduler_g is not None and state.get("scheduler_g") is not None:
        scheduler_g.load_state_dict(state["scheduler_g"])
    if scheduler_d is not None and state.get("scheduler_d") is not None:
        scheduler_d.load_state_dict(state["scheduler_d"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return {**metadata, "state": state}
