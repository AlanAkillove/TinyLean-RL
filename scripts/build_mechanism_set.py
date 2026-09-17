#!/usr/bin/env python
"""Build the IGR mechanism diagnostic set (Phase 3A of the unattended batch).

Excludes every statement seen by project RL activity - E009/E013/E016/E017
(P3-B steps 1-30), E019 (steps 31-60), the sealed P3-C 64-theorem set, and
any seed-replication training statements that exist - then seals 64 fresh
theorems with seed 20260918 for the temperature x group-size mechanism study.

This set is for rollout-mechanism analysis only; it is NOT the final holdout
(that is constructed separately after all M1 training ends).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from p3c_build_fixed_set import DATASET_REVISION, DEFAULT_DATASET, dump_formal_statements, normalize

from tinylean_rl.evaluation.fixed_set import select_fixed_set

SELECTION_SEED = 20260918
N_SELECTED = 64


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--featured-dumps", default="runs/p3b_pilot/rollout_data")  # E017+E019 steps 1-60
    parser.add_argument("--seed-dump-dirs", default="runs/m1_seed2/rollout_data,runs/m1_seed3/rollout_data")
    parser.add_argument("--p3c-fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--e013", default="experiments/results/p3_0_promptset_temp1.json")
    parser.add_argument("--e016", default="experiments/results/p3_b_n8_calibration.json")
    parser.add_argument("--output", default="experiments/manifests/igr_mechanism_set.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    rows = parquet.read_table(dataset_path).to_pylist()
    unique_rows: dict[str, dict] = {}
    for row in rows:
        unique_rows.setdefault(row["statement_id"], row)
    ids = list(unique_rows.keys())
    formal_to_ids: dict[str, list[str]] = {}
    for statement_id, row in unique_rows.items():
        formal_to_ids.setdefault(normalize(row["formal_statement"]), []).append(statement_id)

    excluded_sources: dict[str, set[str]] = {}
    e013 = json.loads((ROOT / args.e013).read_text(encoding="utf-8"))
    excluded_sources["E013"] = set(e013["sampled_statement_ids"])
    e016 = json.loads((ROOT / args.e016).read_text(encoding="utf-8"))
    excluded_sources["E016"] = set(e016["sampled_statement_ids"])
    # E009 equals the E013 set by construction (documented in p3c_build_fixed_set.py).

    p3c = json.loads((ROOT / args.p3c_fixed_set).read_text(encoding="utf-8"))
    excluded_sources["P3C fixed set (E018)"] = {t["statement_id"] for t in p3c["theorems"]}

    for label, dump_dir in [
        ("E017+E019 p3b_pilot dumps (steps 1-60)", args.featured_dumps),
        *[(f"seed-rep {name}", name) for name in args.seed_dump_dirs.split(",")],
    ]:
        path = ROOT / dump_dir
        if not path.exists() or not any(path.glob("*.jsonl")):
            continue
        texts, unparsed = dump_formal_statements(path)
        if unparsed:
            print(f"[warn] {label}: {len(unparsed)} unparsed lines", file=sys.stderr)
        source_ids: set[str] = set()
        unmatched: list[str] = []
        for text in sorted(texts):
            matches = formal_to_ids.get(text)
            if not matches:
                unmatched.append(text[:120])
                continue
            source_ids.update(matches)
        if unmatched:
            print(f"[ERROR] {label}: {len(unmatched)} statements unmatched to the dataset", file=sys.stderr)
            for sample in unmatched[:5]:
                print(f"  {sample!r}", file=sys.stderr)
            return 2
        excluded_sources[label] = source_ids

    excluded_ids: set[str] = set().union(*excluded_sources.values())
    eligible_ids = [identifier for identifier in ids if identifier not in excluded_ids]
    selected = select_fixed_set(eligible_ids, N_SELECTED, seed=SELECTION_SEED)

    theorems = []
    for index, statement_id in enumerate(selected):
        row = unique_rows[statement_id]
        theorems.append(
            {
                "theorem_index": index,
                "statement_id": statement_id,
                "name": row.get("name", statement_id),
                "formal_statement": row["formal_statement"],
                "natural_language": row.get("natural_language") or "",
            }
        )

    artifact = {
        "artifact_type": "igr_mechanism_set",
        "purpose": "temperature x group-size rollout mechanism diagnostics (Phase 3); NOT a final holdout",
        "selection_seed": SELECTION_SEED,
        "selection_method": "random.Random(seed).sample(eligible_ids in parquet dedup order, 64)",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_revision": DATASET_REVISION,
        "dataset_unique_statements": len(ids),
        "excluded_sources": {source: len(source_ids) for source, source_ids in excluded_sources.items()},
        "excluded_count": len(excluded_ids),
        "eligible_count": len(eligible_ids),
        "selected_count": len(selected),
        "sealed": True,
        "theorems": theorems,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in artifact.items() if key != "theorems"}, indent=2, ensure_ascii=False))
    print(f"Mechanism set: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
