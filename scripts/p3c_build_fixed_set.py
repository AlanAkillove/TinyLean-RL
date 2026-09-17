#!/usr/bin/env python3
"""Build the P3-C (E018) fixed evaluation set.

Collects every Promptset statement used by the project's RL calibration /
P3-B pilot (E009/E013/E016/E017), excludes them, and seals a deterministic
64-theorem fixed set (selection seed 20260917) from the remaining eligible
statements. The selection must never be re-drawn after evaluation results
are observed.

Outputs (committed):
* ``experiments/manifests/p3c_excluded_statement_ids.txt``
* ``experiments/manifests/p3c_fixed_set.json``
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.fixed_set import select_fixed_set

DEFAULT_DATASET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
DATASET_REVISION = "3009c548d90160d0f5e963d72238610c6732f812"
SELECTION_SEED = 20260917
N_SELECTED = 64

FORMAL_BLOCK_RE = re.compile(r"# Formal Statement:\s*\n```lean4\n(.*?)\n```", re.DOTALL)


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def dump_formal_statements(dump_dir: Path) -> tuple[set[str], list[str]]:
    """Extract the (normalized) formal-statement text from every rollout dump."""

    texts: set[str] = set()
    unparsed: list[str] = []
    for path in sorted(dump_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            match = FORMAL_BLOCK_RE.search(record.get("input", ""))
            if match:
                texts.add(normalize(match.group(1)))
            else:
                unparsed.append(f"{path.name}: cannot parse the formal block")
    return texts, unparsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dump-dir", default="runs/p3b_pilot/rollout_data")
    parser.add_argument("--e013", default="experiments/results/p3_0_promptset_temp1.json")
    parser.add_argument("--e016", default="experiments/results/p3_b_n8_calibration.json")
    parser.add_argument("--excluded-out", default="experiments/manifests/p3c_excluded_statement_ids.txt")
    parser.add_argument("--output", default="experiments/manifests/p3c_fixed_set.json")
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
    excluded_sources["E013 p3_0_promptset_temp1.json"] = set(e013["sampled_statement_ids"])

    e016 = json.loads((ROOT / args.e016).read_text(encoding="utf-8"))
    excluded_sources["E016 p3_b_n8_calibration.json"] = set(e016["sampled_statement_ids"])

    dump_texts, unparsed = dump_formal_statements(ROOT / args.dump_dir)
    if unparsed:
        for message in unparsed[:5]:
            print(f"[warn] {message}", file=sys.stderr)
    dump_ids: set[str] = set()
    unmatched: list[str] = []
    ambiguous: list[str] = []
    for text in sorted(dump_texts):
        matches = formal_to_ids.get(text)
        if not matches:
            unmatched.append(text[:120])
            continue
        if len(matches) > 1:
            ambiguous.append(f"{matches} share one formal statement")
        dump_ids.update(matches)
    excluded_sources["E017 p3b_pilot rollout_data (steps 1-30)"] = dump_ids

    if unmatched:
        print(f"[ERROR] {len(unmatched)} dump formal statements did not match the dataset:", file=sys.stderr)
        for sample in unmatched[:5]:
            print(f"  {sample!r}", file=sys.stderr)
        return 2

    excluded_ids: set[str] = set().union(*excluded_sources.values())
    eligible_ids = [identifier for identifier in ids if identifier not in excluded_ids]
    selected = select_fixed_set(eligible_ids, N_SELECTED, seed=SELECTION_SEED)

    excluded_path = ROOT / args.excluded_out
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# P3-C (E018) excluded statement ids - every Promptset statement used by the",
        "# project's RL calibration or the P3-B pilot must not appear in the fixed set.",
        "# sources:",
    ]
    for source, source_ids in excluded_sources.items():
        header.append(f"#   {source}: {len(source_ids)} ids")
    header.append(
        "#   E009 P2.5 W1 profiling: equals the E013 set by construction (same probe,"
    )
    header.append(
        "#     seed 0 / limit 32 / same dataset revision; artifact lives on the Windows host)"
    )
    header.append(f"# unique excluded ids: {len(excluded_ids)}")
    if ambiguous:
        header.append(f"# note: {len(ambiguous)} ambiguous formal-statement collisions kept all ids")
    excluded_path.write_text(
        "\n".join(header) + "\n" + "\n".join(sorted(excluded_ids)) + "\n", encoding="utf-8"
    )

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

    summary = {
        "artifact_type": "p3c_fixed_set",
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
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({key: value for key, value in summary.items() if key != "theorems"}, indent=2, ensure_ascii=False))
    print(f"Excluded ids: {excluded_path}")
    print(f"Fixed set:   {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
