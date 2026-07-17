from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .export import export_onnx
from .manifest import read_manifest, validate_manifest
from .model_registry import MODEL_SPECS, ensure_model, inspect_model, model_path
from .text import Vocabulary
from .train import train
from .vits.trainer import train_vits
from .vits.exporter import export_vits_onnx, validate_onnx_runtime
from .frontend import (EspeakFrontend, ensure_korean_cmudict,
                       ensure_openjtalk_dictionary,
                       frontend_from_config, frontend_from_contract,
                       inspect_korean_cmudict, inspect_openjtalk_dictionary,
                       load_frontend_conformance,
                       load_frontend_contract, phonemize_manifest,
                       verify_frontend_conformance)
from .vits.runtime import OnnxTTS, write_wav
from .batch_training import train_many
from .experiments import prepare_experiment, resolve_experiment
from .pipeline import run_pipeline
from .sample_generation import generate_samples
from .text_generation import generate_texts
from .qwen_teacher import inspect_qwen_runtime
from .language_check import check_language_support, format_language_statuses
from .logging_utils import configure_logging
from .quality import run_audio_quality_gate
from .quality_models import (QUALITY_MODEL_SPECS, ensure_quality_model,
                             inspect_quality_model, quality_model_path)
from .semantic_quality import run_semantic_quality_gate
from .interrupts import run_supervised, should_supervise


def _dispatch(argv=None) -> int:
    configure_logging(os.environ.get("TTS_TRAINER_LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(prog="tts-trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate metadata and PCM WAV files")
    validate.add_argument("metadata"); validate.add_argument("--sample-rate", type=int)
    validate.add_argument("--multi-speaker", action="store_true")
    validate.add_argument("--require-phonemes", action="store_true")
    validate.add_argument("--config", help="validate against languages registered by this config")
    vocab = sub.add_parser("vocab"); vocab.add_argument("metadata"); vocab.add_argument("output")
    phonemize = sub.add_parser("phonemize", help="freeze routed language phonemes into metadata")
    phonemize.add_argument("metadata"); phonemize.add_argument("output")
    phonemize.add_argument("--config", help="use frontend voices and strictness from a training config")
    frontend_info = sub.add_parser("frontend-info", help="show the resolved multilingual frontend contract")
    frontend_info.add_argument("--config", default="training_configs/train1.json")
    languages = sub.add_parser(
        "languages",
        help="check every configured language without downloading Qwen models",
    )
    languages.add_argument("--config", default="training_configs/train1.json")
    languages.add_argument("--selected-only", action="store_true")
    language_check = sub.add_parser("language-check", help="check selected language G2P and teacher mappings")
    language_check.add_argument("codes", nargs="*")
    language_check.add_argument("--config", default="training_configs/train1.json")
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
    generate_text = sub.add_parser("generate-texts", help="generate and validate multilingual training text")
    generate_text.add_argument("--config", required=True)
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
    frontends = sub.add_parser("frontends", help="manage project-local text frontend resources")
    frontend_sub = frontends.add_subparsers(dest="frontend_command", required=True)
    frontend_keys = ("openjtalk", "korean")
    frontend_sub.add_parser("status").add_argument("key", choices=frontend_keys)
    frontend_sub.add_parser("ensure").add_argument("key", choices=frontend_keys)
    verify_frontend = sub.add_parser(
        "verify-frontend",
        help="verify exported text frontend versions, phonemes and token IDs",
    )
    verify_frontend.add_argument("--model-dir", required=True)
    verify_frontend.add_argument("--user-dictionary")
    quality_check = sub.add_parser(
        "quality-check", help="run signal-level audio quality checks without training",
    )
    quality_check.add_argument("--config", default="training_configs/train1.json")
    quality_check.add_argument("--metadata")
    quality_models = sub.add_parser(
        "quality-models", help="manage project-local ASR and speaker quality models",
    )
    quality_model_sub = quality_models.add_subparsers(
        dest="quality_model_command", required=True,
    )
    quality_status = quality_model_sub.add_parser("status")
    quality_status.add_argument("key", nargs="?", choices=QUALITY_MODEL_SPECS)
    quality_ensure = quality_model_sub.add_parser("ensure")
    quality_ensure.add_argument("key", choices=QUALITY_MODEL_SPECS)
    quality_path = quality_model_sub.add_parser("path")
    quality_path.add_argument("key", choices=QUALITY_MODEL_SPECS)
    args = parser.parse_args(argv)
    if args.command == "validate":
        supported = None
        if args.config:
            _, layout = resolve_experiment(args.config)
            supported = layout.language_specs
        report = validate_manifest(
            args.metadata, args.sample_rate,
            require_single_speaker=not args.multi_speaker,
            require_phonemes=args.require_phonemes,
            supported_languages=supported,
        )
        print(json.dumps({"items": len(report.items), "languages": report.language_counts, "sample_rates": report.sample_rates}, ensure_ascii=False, indent=2))
    elif args.command == "vocab":
        result = Vocabulary.build(read_manifest(args.metadata)); result.save(args.output); print(f"wrote {len(result.tokens)} tokens to {args.output}")
    elif args.command == "phonemize":
        if args.config:
            raw, layout = resolve_experiment(args.config)
            frontend = frontend_from_config(
                raw.get("frontend"), languages=layout.languages,
                language_registry=raw.get("language_registry"),
            )
        else:
            frontend = EspeakFrontend()
        print(f"frontend: {frontend.version()}")
        print(phonemize_manifest(args.metadata, args.output, frontend))
    elif args.command == "frontend-info":
        raw, layout = resolve_experiment(args.config)
        frontend = frontend_from_config(
            raw.get("frontend"), languages=layout.languages,
            language_registry=raw.get("language_registry"),
        )
        print(json.dumps(frontend.contract(layout.languages).to_dict(), ensure_ascii=False, indent=2))
    elif args.command in {"languages", "language-check"}:
        raw, layout = resolve_experiment(args.config)
        if args.command == "languages":
            codes = layout.languages if args.selected_only else tuple(layout.language_registry)
        else:
            codes = tuple(args.codes) if args.codes else layout.languages
        statuses = check_language_support(raw, layout, codes)
        print(format_language_statuses(statuses))
        return 0 if all(status.ready for status in statuses) else 1
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
    elif args.command == "generate-texts":
        print(generate_texts(args.config))
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
            checkpoint_name = raw.get("validation", {}).get("export_checkpoint", "best")
            if checkpoint_name not in {"best", "last"}:
                raise ValueError("validation.export_checkpoint must be best or last")
            checkpoint = layout.checkpoints_dir / checkpoint_name
            if checkpoint_name == "best" and not checkpoint.is_dir():
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
    elif args.command == "frontends":
        if args.frontend_command == "ensure":
            ensure_resource = {
                "openjtalk": ensure_openjtalk_dictionary,
                "korean": ensure_korean_cmudict,
            }[args.key]
            print(ensure_resource())
        else:
            inspect_resource = {
                "openjtalk": inspect_openjtalk_dictionary,
                "korean": inspect_korean_cmudict,
            }[args.key]
            status = inspect_resource()
            print(json.dumps({
                "key": status.key,
                "ready": status.ready,
                "path": str(status.path),
                "size_bytes": status.size_bytes,
                "missing": status.missing,
            }, ensure_ascii=False, indent=2))
            return 0 if status.ready else 1
    elif args.command == "verify-frontend":
        model_dir = Path(args.model_dir)
        contract = load_frontend_contract(model_dir / "frontend.json")
        conformance = load_frontend_conformance(model_dir / "frontend.conformance.json")
        frontend_config = {}
        if args.user_dictionary:
            frontend_config["openjtalk"] = {"user_dictionary": args.user_dictionary}
        frontend = frontend_from_contract(contract, frontend_config)
        vocabulary = Vocabulary.load(model_dir / "tokens.json")
        mismatches = verify_frontend_conformance(conformance, frontend, vocabulary)
        version_mismatches = []
        for language, profile in contract.languages.items():
            expected = profile.get("engine_version") or contract.engine_version
            actual = frontend.version_for(language)
            if expected and actual != expected:
                version_mismatches.append({
                    "language": language, "expected": expected, "actual": actual,
                })
        result = {
            "ready": not mismatches and not version_mismatches,
            "cases": len(conformance["cases"]),
            "version_mismatches": version_mismatches,
            "content_mismatches": mismatches,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    elif args.command == "quality-check":
        raw, layout = resolve_experiment(args.config, metadata_override=args.metadata)
        report = validate_manifest(
            layout.metadata, int(raw["audio"]["sample_rate"]),
            require_single_speaker=False,
            require_phonemes=bool(raw.get("frontend", {}).get("require_phonemes", True)),
            supported_languages=layout.language_specs,
        )
        quality_config = raw.get("quality", {})
        destination = layout.run_dir / "quality" / "audio-quality-report.json"
        result = run_audio_quality_gate(list(report.items), quality_config, destination)
        print(json.dumps({
            key: result[key]
            for key in ("provider", "items", "passed", "failed", "failure_counts")
        }, ensure_ascii=False, indent=2))
        semantic_config = quality_config.get("semantic", {})
        if semantic_config.get("enabled", False):
            semantic_result = run_semantic_quality_gate(
                list(report.items), semantic_config,
                layout.run_dir / "quality" / "semantic-quality-report.json",
                reference_root=layout.dataset_dir / "references",
            )
            print(json.dumps({
                key: semantic_result[key]
                for key in ("provider", "items", "passed", "failed", "failure_counts")
            }, ensure_ascii=False, indent=2))
        print(destination)
    elif args.command == "quality-models":
        if args.quality_model_command == "ensure":
            print(ensure_quality_model(args.key))
        elif args.quality_model_command == "path":
            print(quality_model_path(args.key))
        else:
            keys = [args.key] if args.key else list(QUALITY_MODEL_SPECS)
            statuses = [inspect_quality_model(key) for key in keys]
            print(json.dumps([{
                "key": status.spec.key,
                "ready": status.ready,
                "path": str(status.path),
                "size_bytes": status.size_bytes,
                "missing": status.missing,
            } for status in statuses], ensure_ascii=False, indent=2))
            return 0 if all(status.ready for status in statuses) else 1
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


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if should_supervise(arguments):
        return run_supervised(arguments)
    try:
        return _dispatch(arguments)
    except KeyboardInterrupt:
        print(
            "\nINTERRUPT | stopped by user | rerun the same command to reuse completed work",
            file=sys.stderr,
            flush=True,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
