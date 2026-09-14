#!/usr/bin/env python3
"""Download only models explicitly listed in models/manifests/models.yaml."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifests" / "models.yaml"
LOCK = ROOT / "models" / "manifests" / "resolved.local.json"


def load_models() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["models"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(load_models()), required=True)
    parser.add_argument("--revision", help="Override manifest revision; record it before committing.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = load_models()[args.model_key]
    revision = args.revision or spec["revision"]
    target = ROOT / spec["local_dir"]
    print(f"model: {args.model_key}")
    print(f"repo: {spec['repo_id']}@{revision}")
    print(f"local_dir: {target}")
    if args.dry_run:
        return 0

    target.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    api = HfApi(token=token)
    info = api.model_info(spec["repo_id"], revision=revision)
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=revision,
        local_dir=str(target),
        token=token,
    )
    lock = {}
    if LOCK.exists():
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock[args.model_key] = {
        "repo_id": spec["repo_id"],
        "requested_revision": revision,
        "resolved_commit": info.sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_dir": str(target.relative_to(ROOT)).replace("\\", "/"),
    }
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"resolved commit: {info.sha}")
    print(f"wrote local lock: {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
