#!/usr/bin/env python
"""Build the M1 FINAL HOLDOUT (Phase 5 of the unattended batch).

Runs ONLY after every M1 training run has ended (seed1/seed2/seed3, M2 smoke).
Excludes every statement the project has ever put through RL training or used
in earlier diagnostic sets - E009/E013/E016/E017 (steps 1-30), E019
(steps 31-60), seed2 dumps, seed3 dumps, the M2 smoke dumps, the P3-C
development set and the IGR mechanism set - then seals 128 fresh theorems
with seed 20260918.

Naming per protocol: "project-unseen, same-source final holdout" (NOT OOD).
After its first evaluation the set is frozen for good: no re-draws, no
hyperparameter changes, no step100 chasing.
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
N_SELECTED = 128

# All training dump directories are DISCOVERED (runs/*/rollout_data) instead of
# being hard-coded, so a seed3 restarted into a fresh directory (or any future
# training run) cannot be silently missed by the exclusions. Every discovered
# source must parse cleanly; otherwise the build refuses to seal.
RUNS_GLOB = "runs/*/rollout_data"


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
    parser.add_argument("--p3c-fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--mechanism-set", default="experiments/manifests/igr_mechanism_set.json")
    parser.add_argument("--e013", default="experiments/results/p3_0_promptset_temp1.json")
    parser.add_argument("--e016", default="experiments/results/p3_b_n8_calibration.json")
    parser.add_argument("--output", default="experiments/manifests/m1_final_holdout.json")
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

    p3c = json.loads((ROOT / args.p3c_fixed_set).read_text(encoding="utf-8"))
    excluded_sources["P3C development set (E018)"] = {t["statement_id"] for t in p3c["theorems"]}
    mechanism = json.loads((ROOT / args.mechanism_set).read_text(encoding="utf-8"))
    excluded_sources["IGR mechanism set (E020-M)"] = {t["statement_id"] for t in mechanism["theorems"]}

    for path in sorted(ROOT.glob(RUNS_GLOB)):
        if not any(path.glob("*.jsonl")):
            continue
        label = f"training dumps {path.parent.name}"
        texts, unparsed = dump_formal_statements(path)
        if unparsed:
            print(f"[ERROR] {label}: {len(unparsed)} unparsed line(s) - refusing to seal", file=sys.stderr)
            for sample in unparsed[:5]:
                print(f"  {sample}", file=sys.stderr)
            return 2
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

    if not any(key.startswith("training dumps ") for key in excluded_sources):
        print("[ERROR] no training dump directories were discovered - refusing to seal", file=sys.stderr)
        return 2

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
        "artifact_type": "m1_final_holdout",
        "naming": "project-unseen, same-source final holdout (NOT OOD; distill training data contact cannot be excluded)",
        "selection_seed": SELECTION_SEED,
        "selection_method": "random.Random(seed).sample(eligible_ids in parquet dedup order, 128)",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_revision": DATASET_REVISION,
        "dataset_unique_statements": len(ids),
        "excluded_sources": {source: len(source_ids) for source, source_ids in excluded_sources.items()},
        "excluded_count": len(excluded_ids),
        "eligible_count": len(eligible_ids),
        "selected_count": len(selected),
        "sealed": True,
        "freeze_rule": "after the first evaluation: no re-draw, no hyperparameter change, no step100 chasing",
        "theorems": theorems,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in artifact.items() if key != "theorems"}, indent=2, ensure_ascii=False))
    print(f"Final holdout: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
