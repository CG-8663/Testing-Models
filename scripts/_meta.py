"""Environment snapshot shared across eval scripts.

The goal is mathematical reproducibility: every results JSON records
exactly which weights, tokenizer, dependency versions, hardware, and
seeds produced the numbers, so a third party can reproduce or
falsify them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256_of_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem_bytes = _run(["sysctl", "-n", "hw.memsize"])
    info["memsize_bytes"] = int(mem_bytes) if mem_bytes else None
    iogpu = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    info["iogpu_wired_limit_mb"] = int(iogpu) if iogpu else None
    return info


def _model_fingerprint(model_dir: str | Path) -> dict:
    p = Path(model_dir)
    out: dict = {"model_dir": str(p)}
    if not p.is_dir():
        out["status"] = "not_a_directory"
        return out
    out["config_json_sha256"] = _sha256_of_file(p / "config.json")
    out["tokenizer_json_sha256"] = _sha256_of_file(p / "tokenizer.json")
    out["index_sha256"] = _sha256_of_file(p / "model.safetensors.index.json")
    shards = sorted(p.glob("model-*-of-*.safetensors"))
    out["shard_count"] = len(shards)
    return out


def snapshot(extra: dict | None = None, model_dir: str | Path | None = None) -> dict:
    snap: dict = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "packages": {
            name: _pkg(name) for name in (
                "mlx", "mlx-lm", "lighteval", "datasets", "transformers",
                "requests", "numpy",
            )
        },
        "hardware": _hardware(),
    }
    if model_dir is not None:
        snap["model"] = _model_fingerprint(model_dir)
    if extra:
        snap.update(extra)
    return snap


def write_with_meta(path: str | Path, payload: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"meta": meta, **payload}
    path.write_text(json.dumps(doc, indent=2))
