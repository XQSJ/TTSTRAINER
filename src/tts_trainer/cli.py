from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .export import export_onnx
from .manifest import read_manifest, validate_manifest
from .model_registry import MODEL_SPECS, ensure_model, inspect_model, model_path
from .text import Vocabulary
from .train import train
from .vits.trainer import train_vits
from .vits.exporter import export_vits_onnx, validate_onnx_runtime
from .frontend import EspeakFrontend, espeak_frontend_from_config, phonemize_manifest
from .vits.runtime import OnnxTTS, write_wav
from .batch_training import train_many
from .experiments import prepare_experiment, resolve_experiment
from .pipeline import run_pipeline
from .sample_generation import generate_samples
from .qwen_teacher import inspect_qwen_runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tts-trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate metadata and PCM WAV files")
    validate.add_argument("metadata"); validate.add_argument("--sample-rate", type=int)
    validate.add_argument("--multi-speaker", action="store_true")
    validate.add_argument("--require-phonemes", action="store_true")
    vocab = sub.add_parser("vocab"); vocab.add_argument("metadata"); vocab.add_argument("output")
    phonemize = sub.add_parser("phonemize", help="freeze eSpeak phonemes into metadata")
    phonemize.add_argument("metadata"); phonemize.add_argument("output")
    phonemize.add_argument("--config", help="use frontend voices and strictness from a training config")
    frontend_info = sub.add_parser("frontend-info", help="show the resolved eSpeak frontend contract")
    frontend_info.add_argument("--config", default="training_configs/train1.json")
    training = sub.add_parser("train"); training.add_argument("--config", required=True)
    vits_training = sub.add_parser("train-vits", help="train waveform VITS generator and discriminators")
    vits_training.add_argument("--config", default="training_configs/train1.json")
    vits_training.add_argument("--metadata")
    vits_training.add_argument("--output")
    vits_training.add_argument("--device")
    vits_training.add_argument("--max-steps", type=int)
    initialize = sub.add_parser("init-experiment", help="create named dataset, run, log and artifact directories")
    initialize.add_argument("--config", required=True)
    generate = sub.add_parser("generate-samples", help="generate training WAV files with Qwen VoiceDesign or voice clone")
    generate.add_argument("--config", required=True)
    pipeline = sub.add_parser("run-pipeline", help="run all enabled stages from a single config")
    pipeline.add_argument("--config", default="training_configs/train1.json")
    pipeline.add_argument("--max-steps", type=int)
    many = sub.add_parser("train-many", help="train multiple named model configs")
    many.add_argument("configs", nargs="+")
    many.add_argument("--max-parallel", type=int, default=1)
    many.add_argument("--max-steps", type=int)
    export = sub.add_parser("export"); export.add_argument("--config", required=True); export.add_argument("--checkpoint", required=True); export.add_argument("--output", default="artifacts/model.onnx")
    vits_export = sub.add_parser("export-vits", help="export a training checkpoint with Piper-shaped inputs")
    source = vits_export.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint"); source.add_argument("--config")
    vits_export.add_argument("--output")
    vits_export.add_argument("--sample-rate", type=int)
    vits_export.add_argument("--validate-runtime", action="store_true")
    synthesize = sub.add_parser("synthesize-onnx", help="run the project-local ONNX reference runtime")
    synthesize.add_argument("--model-dir", required=True); synthesize.add_argument("--text", required=True)
    synthesize.add_argument("--language", required=True); synthesize.add_argument("--speaker", required=True)
    synthesize.add_argument("--output", default="output.wav")
    models = sub.add_parser("models", help="manage project-local Qwen models")
    model_sub = models.add_subparsers(dest="model_command", required=True)
    status = model_sub.add_parser("status"); status.add_argument("key", nargs="?", choices=MODEL_SPECS)
    ensure = model_sub.add_parser("ensure"); ensure.add_argument("key", choices=MODEL_SPECS)
    path = model_sub.add_parser("path"); path.add_argument("key", choices=MODEL_SPECS)
    qwen_runtime = sub.add_parser("qwen-runtime", help="check the Qwen Python runtime without downloading models")
    qwen_runtime.add_argument("--mode", choices=("installed", "source"), default="installed")
    qwen_runtime.add_argument("--source-path")
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_manifest(
            args.metadata, args.sample_rate,
            require_single_speaker=not args.multi_speaker,
            require_phonemes=args.require_phonemes,
        )
        print(json.dumps({"items": len(report.items), "languages": report.language_counts, "sample_rates": report.sample_rates}, ensure_ascii=False, indent=2))
    elif args.command == "vocab":
        result = Vocabulary.build(read_manifest(args.metadata)); result.save(args.output); print(f"wrote {len(result.tokens)} tokens to {args.output}")
    elif args.command == "phonemize":
        if args.config:
            raw, _ = resolve_experiment(args.config)
            frontend = espeak_frontend_from_config(raw.get("frontend"))
        else:
            frontend = EspeakFrontend()
        print(f"frontend: {frontend.version()}")
        print(phonemize_manifest(args.metadata, args.output, frontend))
    elif args.command == "frontend-info":
        raw, layout = resolve_experiment(args.config)
        frontend = espeak_frontend_from_config(raw.get("frontend"))
        print(json.dumps(frontend.contract(layout.languages).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "train":
        load_config(args.config); print(train(args.config))
    elif args.command == "train-vits":
        print(train_vits(args.config, args.metadata, args.output,
                         device_name=args.device, max_steps=args.max_steps))
    elif args.command == "init-experiment":
        raw, layout = resolve_experiment(args.config)
        prepare_experiment(layout, raw, args.config)
        print(layout.run_dir / "run-layout.json")
    elif args.command == "generate-samples":
        print(generate_samples(args.config))
    elif args.command == "run-pipeline":
        print(run_pipeline(args.config, max_steps=args.max_steps))
    elif args.command == "train-many":
        for config_path in train_many(args.configs, max_parallel=args.max_parallel,
                                      max_steps=args.max_steps):
            print(config_path)
    elif args.command == "export":
        print(export_onnx(args.config, args.checkpoint, args.output))
    elif args.command == "export-vits":
        if args.config:
            raw, layout = resolve_experiment(args.config)
            checkpoint = layout.checkpoints_dir / "last"
            output = Path(args.output) if args.output else layout.artifacts_dir
            sample_rate = args.sample_rate or raw["audio"]["sample_rate"]
        else:
            checkpoint = Path(args.checkpoint)
            output = Path(args.output or "artifacts/vits")
            sample_rate = args.sample_rate or 22050
        result = export_vits_onnx(checkpoint, output, sample_rate=sample_rate)
        print(result)
        if args.validate_runtime: print(f"onnxruntime output shape: {validate_onnx_runtime(result)}")
    elif args.command == "synthesize-onnx":
        runtime = OnnxTTS(args.model_dir)
        samples = runtime.synthesize_text(args.text, language=args.language, speaker=args.speaker)
        print(write_wav(args.output, samples, runtime.sample_rate))
    elif args.command == "qwen-runtime":
        status = inspect_qwen_runtime(args.mode, args.source_path)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["ready"] else 1
    elif args.model_command == "ensure":
        print(ensure_model(args.key))
    elif args.model_command == "path":
        print(model_path(args.key))
    else:
        keys = [args.key] if args.key else list(MODEL_SPECS)
        result = []
        for key in keys:
            status = inspect_model(key)
            result.append({"key": key, "ready": status.ready, "path": str(status.path),
                           "size_bytes": status.size_bytes, "missing": status.missing})
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
