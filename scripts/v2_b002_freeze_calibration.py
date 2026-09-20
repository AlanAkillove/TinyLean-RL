#!/usr/bin/env python
"""Freeze the V2-B002 pipeline-calibration replay set (diagnostic-only).

B002's 4096 single-trajectory success rate (11/512 = 2.1%) is far from the
historical theta0 candidate-success regime (E023 sealed holdout: theta0
solved >= 1 sample for 46/128 theorems). This set is a DIAGNOSTIC-ONLY replay
input: a frozen 64-theorem subset of the E023 holdout, to be run through the
exact B002 generation/extraction/verification pipeline to check whether the
runner reproduces the historical success level on historically-known items.

It is NOT a benchmark, NOT a capability claim, and NOT for model comparison.
Selection: first 64 of E023's 128 statement ids when sorted by ascending
sha256(statement_id) (content-addressed ordering; no semantic bias).

Output: experiments/manifests/v2/v2_b002_calibration_set.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v2_family_leakage_audit import sha256_hex

E023_REL = "experiments/manifests/m1_final_holdout.json"
PARQUET_REL = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
REGISTRY_REL = "experiments/manifests/v2/family_component_registry.json"
OUTPUT_REL = "experiments/manifests/v2/v2_b002_calibration_set.json"
TAKE = 64


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the B002 calibration replay set.")
    parser.add_argument("--output", default=OUTPUT_REL)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    e023_path = resolve(E023_REL)
    e023 = json.loads(e023_path.read_text(encoding="utf-8"))
    e023_ids = [t["statement_id"] for t in e023["theorems"]]
    if len(e023_ids) != 128:
        print(f"[ERROR] E023 has {len(e023_ids)} theorems, expected 128", file=sys.stderr)
        return 2

    selected = sorted(e023_ids, key=lambda sid: sha256_hex(sid))[:TAKE]
    if len(selected) != TAKE or len(set(selected)) != TAKE:
        print("[ERROR] calibration selection size/uniqueness check failed", file=sys.stderr)
        return 2

    table = pq.read_table(resolve(PARQUET_REL))
    rows = {}
    for row in table.to_pylist():
        sid = row["statement_id"]
        if sid not in rows:
            rows[sid] = row

    registry = json.loads(resolve(REGISTRY_REL).read_text(encoding="utf-8"))
    comp_of = {}
    for comp in registry["components"]:
        for sid in comp["member_statement_ids"]:
            comp_of[sid] = (comp["component_id"], comp["size"])

    theorems = []
    for rank, sid in enumerate(selected, start=1):
        row = rows.get(sid)
        if row is None:
            print(f"[ERROR] statement {sid} missing from parquet", file=sys.stderr)
            return 2
        if sid not in comp_of:
            print(f"[ERROR] statement {sid} missing from registry", file=sys.stderr)
            return 2
        component_id, component_size = comp_of[sid]
        formal = row["formal_statement"]
        natural = row["informal_problem"] or ""
        theorems.append(
            {
                "rank": rank,
                "statement_id": sid,
                "formal_statement": formal,
                "natural_language": natural,
                "name": row["name"],
                "source": row["data_source"],
                "component_id": component_id,
                "component_size": component_size,
                "statement_sha256": sha256_hex(formal),
                "natural_language_sha256": sha256_hex(natural),
            }
        )

    result = {
        "artifact_type": "v2_b002_calibration_set",
        "status": "frozen",
        "purpose": (
            "diagnostic-only pipeline calibration replay: run the frozen V2-B002 pipeline on a "
            "frozen E023 subset to check the runner reproduces the historical theta0 "
            "candidate-success level on known items"
        ),
        "rules": [
            "diagnostic-only: NOT a benchmark, NOT a capability claim, NOT for model comparison",
            "generation uses the B002 runner with pipeline semantics unchanged (single-path HF, T=1.0, top_p=1.0, 4096 tokens, seed = 20260920 + rank*8); the loader accepts this artifact type as an explicit diagnostic exception",
            "labels via the same 5-prefix truncation + B0 verifier pipeline",
            "the set never enters any B002/B003 analysis pool; results are reported only against the historical E023 regime",
        ],
        "selection_rule": "first 64 of E023's 128 statement ids sorted by ascending sha256(statement_id)",
        "inputs": {
            "e023_holdout": {"path": str(e023_path), "sha256": hashlib.sha256(e023_path.read_bytes()).hexdigest()},
            "registry": {"path": str(resolve(REGISTRY_REL)), "sha256": hashlib.sha256(resolve(REGISTRY_REL).read_bytes()).hexdigest()},
        },
        "counts": {"selected": len(theorems)},
        "theorem_count": len(theorems),
        "theorems": theorems,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/v2_b002_freeze_calibration.py",
    }
    output_path = resolve(args.output)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sources = {}
    for t in theorems:
        sources[t["source"]] = sources.get(t["source"], 0) + 1
    multi = sum(1 for t in theorems if t["component_size"] > 1)
    print(f"[calibration] {len(theorems)} theorems frozen; sources {sources}; multi-variant {multi}/64")
    print(f"artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
