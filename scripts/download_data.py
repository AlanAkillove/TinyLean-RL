#!/usr/bin/env python3
"""Materialize a named dataset only after an explicit command-line choice."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "datasets.yaml"
LOCK = ROOT / "data" / "manifests" / "resolved.local.json"


def load_datasets() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["datasets"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-key", choices=sorted(load_datasets()), required=True)
    parser.add_argument("--revision", help="Override manifest revision.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = load_datasets()[args.dataset_key]
    revision = args.revision or spec["revision"]
    target = ROOT / spec["local_dir"]
    print(f"dataset: {args.dataset_key}")
    print(f"source: {spec['repo_id']}@{revision}")
    print(f"local_dir: {target}")
    if args.dry_run:
        return 0

    if spec["status"] == "evaluation_only":
        print("NOTE: this dataset is evaluation-only and must not be used for training.")
    if spec["status"] == "metadata_only":
        print("NOTE: this dataset is metadata-only in P0; materialization is discouraged.")

    target.parent.mkdir(parents=True, exist_ok=True)
    if spec["source_type"] == "git":
        if target.exists() and any(target.iterdir()):
            raise SystemExit(f"Refusing to overwrite non-empty directory: {target}")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", spec["repo_id"], str(target)],
            check=True,
        )
        subprocess.run(["git", "-C", str(target), "checkout", revision], check=True)
        resolved = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    elif spec["source_type"] == "huggingface":
        api = HfApi()
        info = api.dataset_info(spec["repo_id"], revision=revision)
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=revision,
            local_dir=str(target),
        )
        resolved = info.sha
    else:
        raise SystemExit(f"Unsupported source_type: {spec['source_type']}")

    lock = {}
    if LOCK.exists():
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock[args.dataset_key] = {
        "source": spec["repo_id"],
        "requested_revision": revision,
        "resolved_revision": resolved,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_dir": str(target.relative_to(ROOT)).replace("\\", "/"),
    }
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"resolved revision: {resolved}")
    print(f"wrote local lock: {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
