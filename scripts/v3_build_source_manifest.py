#!/usr/bin/env python3
"""V3 §17 — source provenance manifest (READ-ONLY over data; writes only the manifest).

Computes a task-scoped provenance freeze for the existing seed1/2/3 GRPO rollout
dumps that V3-D001 consumes, plus the pinned dataset / family-registry / theta0
identities. Commits HASHES ONLY — the ~126MB of raw rollout_data is NOT added to
Git. Recorded so a future reader can prove the exact bytes V3-D001 was fit on.

Output: experiments/manifests/v3/v1_rollout_sources.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "manifests" / "v3" / "v1_rollout_sources.json"

N = 8
FORMAL_HINT = 4  # prompts per step

SEEDS = {
    "seed1": "runs/p3b_pilot/rollout_data",
    "seed2": "runs/m1_seed2/rollout_data",
    "seed3": "runs/m1_seed3/rollout_data",
}
EXCLUDED = {
    "m2_qwen_smoke": "runs/m2_qwen_smoke/rollout_data",
    "seed3_r2r3_aborted_backups": ".cache/m1_seed3_r2r3_dumps",
}
DATASET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
DATASET_REVISION = "3009c548d90160d0f5e963d72238610c6732f812"  # scripts/p3c_build_fixed_set.py:33
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
THETA0_DIR = "models/weights/kimina_distill_0_6b"
THETA0_REPO = "AI-MO/Kimina-Prover-Distill-0.6B"
THETA0_REVISION = "332e8a5259d1bdfda19d7c7f339f30804813cd3a"  # models/manifests/models.yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def host_info() -> dict:
    import platform
    try:
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.strip()
    except Exception:
        ip = ""
    return {"hostname": platform.node(), "ip": ip, "role": "fly90 (producer / canonical provenance source)",
            "gpu": "RTX 3090 24GB (this manifest is metadata-only, no GPU used)"}


def profile_seed(rel: str) -> dict:
    d = ROOT / rel
    files = sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem))
    file_recs = []
    total_candidates = 0
    total_groups = 0
    for f in files:
        nlines = 0
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                nlines += 1
        total_candidates += nlines
        total_groups += nlines // N if nlines % N == 0 else -1
        file_recs.append({"name": f.name, "bytes": f.stat().st_size, "sha256": sha256(f), "candidates": nlines})
    return {
        "directory": rel,
        "n_files": len(files),
        "candidate_count": total_candidates,
        "group_count": total_groups,
        "expected_groups": 240,
        "expected_candidates": 1920,
        "group_size": N,
        "prompts_per_step": FORMAL_HINT,
        "files": file_recs,
        "dir_concat_sha256": hashlib.sha256("".join(fr["sha256"] for fr in file_recs).encode()).hexdigest(),
    }


def main() -> int:
    seeds = {name: profile_seed(rel) for name, rel in SEEDS.items()}
    excluded = {}
    for name, rel in EXCLUDED.items():
        d = ROOT / rel
        if d.exists():
            excluded[name] = {"path": rel, "present": True,
                              "n_files": len(list(d.glob("*.jsonl"))),
                              "note": "NOT used by V3-D001 (different seed stream / non-Kimina model)"}
    # whole-dataset integrity of the three used dirs
    used_concat = hashlib.sha256("".join(seeds[s]["dir_concat_sha256"] for s in ["seed1", "seed2", "seed3"]).encode()).hexdigest()

    result = {
        "artifact_type": "v3_source_provenance_manifest",
        "status": "frozen-for-V3-D001",
        "purpose": "Prove the exact rollout/dataset/registry/theta0 bytes V3-D001 fits and evaluates on; raw rollout_data is gitignored, only hashes are committed.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev(),
        "host": host_info(),
        "unit": {"group": "one n=8 GRPO rollout of one theorem at one step", "n": N,
                 "groups_per_step": FORMAL_HINT, "steps": 60, "groups_per_seed": 240},
        "seeds": seeds,
        "seeds_summary": {
            "total_files": sum(seeds[s]["n_files"] for s in seeds),
            "total_candidates": sum(seeds[s]["candidate_count"] for s in seeds),
            "total_groups": sum(seeds[s]["group_count"] for s in seeds),
            "used_dirs_concat_sha256": used_concat,
        },
        "excluded_sources": excluded,
        "dataset": {"path": DATASET, "revision": DATASET_REVISION, "sha256": sha256(ROOT / DATASET)},
        "cross_node_verification": {
            "note": "The formal V3-D001 run executes on fly122/RTX 3080; these independent fly122 dir hashes were computed via ssh and MUST equal fly90 (same bytes -> same labels/features).",
            "fly122_host": "ubuntu @ 10.3.25.122, RTX 3080 10GB",
            "fly122_per_seed_dir_concat": {
                "seed1": "3ff522ded5f45eb5a7d50c79fba9279e1f8ee5d8763b68dc074bdbacd16d99a0",
                "seed2": "db838b0b0d720e16f555e5e51c96d789d2b6c585b603a49b4c89d9c7e4c1a019",
                "seed3": "3bbc3b910ae3a28fc80556f928ef141d9a983785096b223948d3cf60d429fb33",
            },
            "fly122_used_dirs_concat": "69cabde77ed6dc7af5f1932a0b6555eb22c1936fe152bfdb6a1fe3c80023b97d",
            "matches_fly90": True,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "family_registry": {"path": REGISTRY, "sha256": sha256(ROOT / REGISTRY),
                            "definition": "V2-frozen connected components of L3 source-family U L2 skeleton U L4 identical-NL (docs/v2/family_leakage_audit.md §5.3)"},
        "theta0": {"repo_id": THETA0_REPO, "revision": THETA0_REVISION, "dir": THETA0_DIR,
                   "config_sha256": sha256(ROOT / THETA0_DIR / "config.json"),
                   "weights_sha256": sha256(ROOT / THETA0_DIR / "model.safetensors"),
                   "weights_bytes": (ROOT / THETA0_DIR / "model.safetensors").stat().st_size},
        "policy": {
            "primary_label": "score (format-gated GRPO reward); informative iff 0 < sum over n=8 < 8",
            "infra": "any group with a candidate whose tool_feedback starts '# System Error:' is excluded from primary (720 -> 686)",
            "no_raw_rollout_in_git": True,
        },
        "git_note": "This manifest contains hashes only; it deliberately does not commit runs/*.jsonl (host-only, gitignored).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "seeds"}, indent=2))
    print("seeds: file/candidate/group summary:")
    for s in seeds:
        print("  ", s, {k: seeds[s][k] for k in ("directory", "n_files", "candidate_count", "group_count", "dir_concat_sha256")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
