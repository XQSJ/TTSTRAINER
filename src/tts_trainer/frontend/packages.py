from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


FRONTEND_PACK_FORMAT = 1


def _requirements(profile: dict) -> list[dict]:
    provider = str(profile["provider"])
    if provider == "espeak-ng":
        return [
            {
                "kind": "runtime",
                "id": "espeak-ng",
                "shared_group": "espeak-ng",
            },
            {
                "kind": "voice",
                "id": str(profile["voice"]),
                "shared_group": "espeak-ng",
            },
        ]
    if provider == "openjtalk":
        return [
            {"kind": "runtime", "id": "openjtalk"},
            {
                "kind": "dictionary",
                "id": str(profile["dictionary"]),
            },
        ]
    if provider == "piper-plus-g2p":
        return [
            {"kind": "runtime", "id": "piper-plus-g2p"},
            {
                "kind": "rules",
                "id": str(profile.get("resource") or profile["profile"]),
            },
        ]
    raise ValueError(f"unsupported frontend pack provider: {provider}")


def export_frontend_packs(
    output_dir: str | Path,
    frontend: dict,
    conformance: dict | None,
    language_map: dict[str, int],
    *,
    model_sha256: str,
    espeak_data_dir: str | Path | None = None,
) -> dict:
    """Export installable descriptors without bundling third-party runtimes.

    The acoustic model and vocabulary stay in the artifact root. Each language
    directory contains only its frozen frontend contract, required runtime
    resources, and conformance vectors. Platform projects may package or
    download those directories independently.

    声学模型和词表保留在根目录；每个语言目录只保存前端契约、依赖清单和一致性
    向量，平台项目可以独立打包、下载或卸载，且不会复制共享模型权重。
    """
    root = Path(output_dir) / "frontend-packs"
    root.mkdir(parents=True, exist_ok=True)
    shared = {}
    if espeak_data_dir is not None:
        source = Path(espeak_data_dir)
        target = root / "_shared" / "espeak-ng" / "espeak-ng-data"
        shutil.copytree(source, target, dirs_exist_ok=True)
        shared["espeak-ng"] = {
            "provider": "espeak-ng",
            "path": "frontend-packs/_shared/espeak-ng/espeak-ng-data",
            "bundled": True,
        }
    all_cases = list((conformance or {}).get("cases") or [])
    languages = {}
    ordered = sorted(language_map.items(), key=lambda item: item[1])
    for language, language_id in ordered:
        profile = dict(frontend["languages"][language])
        cases = [
            dict(case) for case in all_cases
            if case.get("language") == language
        ]
        pack_dir = root / language
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack_conformance = {
            "format": int((conformance or {}).get("format", 1)),
            "cases_per_language": len(cases),
            "languages": [language],
            "piper_compatible": bool(
                (conformance or {}).get("piper_compatible", False)
            ),
            "cases": cases,
        }
        (pack_dir / "conformance.json").write_text(
            json.dumps(pack_conformance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        bundled = (
            profile["provider"] == "espeak-ng" and "espeak-ng" in shared
        )
        pack = {
            "format": FRONTEND_PACK_FORMAT,
            "language": language,
            "language_id": int(language_id),
            "provider": profile["provider"],
            "profile": profile,
            "normalization": frontend["normalization"],
            "tokens": frontend["tokens"],
            "token_encoding": frontend["token_encoding"],
            "core_model_sha256": model_sha256,
            "delivery": "on-demand",
            "bundled": bundled,
            "requirements": _requirements(profile),
            "conformance": "conformance.json",
        }
        encoded = json.dumps(
            pack, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
        pack["pack_id"] = (
            f"{language}-{profile['provider']}-"
            f"{hashlib.sha256(encoded).hexdigest()[:16]}"
        )
        (pack_dir / "manifest.json").write_text(
            json.dumps(pack, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        languages[language] = {
            "provider": profile["provider"],
            "pack_id": pack["pack_id"],
            "manifest": f"frontend-packs/{language}/manifest.json",
            "bundled": bundled,
        }
    manifest = {
        "format": FRONTEND_PACK_FORMAT,
        "layout": "shared-core-language-frontends-v1",
        "core": {
            "model": "model.onnx",
            "deployment": "model.onnx.json",
            "tokens": "tokens.json",
            "model_sha256": model_sha256,
        },
        "shared": shared,
        "languages": languages,
        "note": (
            "Frontend packs do not duplicate model weights. Install the "
            "provider runtime/resources declared by each pack and verify its "
            "conformance vectors before enabling the language."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "format": FRONTEND_PACK_FORMAT,
        "manifest": "frontend-packs/manifest.json",
        "layout": manifest["layout"],
        "languages": languages,
    }
