#!/usr/bin/env python
"""E019 fixed-set exposure audit (Phase 1B).

E019 steps 31-60 sampled fresh training prompts from the full Promptset. This
audit extracts the formal statement of every rollout-dump prompt (steps
31-60), maps it to Promptset statement_ids (same regex/normalization as
p3c_build_fixed_set.py), and intersects with the sealed P3-C 64-theorem set.
Output feeds the naming decision: 'held-out-from-pilot' vs
'reused development fixed set with post-selection training exposure'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p3c_build_fixed_set import FORMAL_BLOCK_RE, normalize

DEFAULT_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"


def dump_prompt_texts(dump_dir: Path, start_step: int, end_step: int) -> tuple[dict[str, int], int]:
    """Return {normalized formal statement: prompt count} and the raw dump-line count."""
    prompts: dict[str, int] = {}
    lines = 0
    for step in range(start_step, end_step + 1):
        path = dump_dir / f"{step}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lines += 1
                match = FORMAL_BLOCK_RE.search(record.get("input", ""))
                if match:
                    text = normalize(match.group(1))
                    prompts[text] = prompts.get(text, 0) + 1
    return prompts, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", default="runs/p3b_pilot/rollout_data")
    parser.add_argument("--start-step", type=int, default=31)
    parser.add_argument("--end-step", type=int, default=60)
    parser.add_argument("--fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--output", default="experiments/results/e019_fixed_set_overlap.json")
    args = parser.parse_args()

    import pandas as pd  # local import: only needed here

    fixed_set = json.loads((ROOT / args.fixed_set).read_text(encoding="utf-8"))
    fixed_theorems = fixed_set["theorems"]
    fixed_text_to_index = {normalize(t["formal_statement"]): t["theorem_index"] for t in fixed_theorems}
    fixed_ids = {t["statement_id"] for t in fixed_theorems}

    prompts, dump_lines = dump_prompt_texts(ROOT / args.dump_dir, args.start_step, args.end_step)
    table = pd.read_parquet(ROOT / args.parquet, columns=["statement_id", "formal_statement"])
    text_to_ids: dict[str, set[str]] = {}
    for statement_id, formal in zip(table["statement_id"], table["formal_statement"], strict=True):
        text_to_ids.setdefault(normalize(formal), set()).add(statement_id)

    training_ids: set[str] = set()
    unmatched = 0
    for text in prompts:
        ids = text_to_ids.get(text)
        if ids:
            training_ids |= ids
        else:
            unmatched += 1

    intersection_ids = sorted(training_ids & fixed_ids)
    affected_indices = sorted(
        {fixed_text_to_index[text] for text in prompts if text in fixed_text_to_index}
        | {t["theorem_index"] for t in fixed_theorems if t["statement_id"] in training_ids}
    )

    artifact = {
        "artifact_type": "e019_fixed_set_overlap",
        "steps_range": [args.start_step, args.end_step],
        "dump_lines": dump_lines,
        "unique_prompt_statements": len(prompts),
        "matched_statement_ids": len(training_ids),
        "unmatched_prompt_texts": unmatched,
        "fixed_set_size": len(fixed_theorems),
        "intersection_count": len(intersection_ids),
        "intersection_ids": intersection_ids,
        "affected_fixed_set_indices": affected_indices,
        "interpretation": (
            "intersection > 0 -> the reused P3-C set is a development fixed set with "
            "post-selection training exposure; intersection = 0 -> it remains "
            "pilot-unseen for this training, but adaptive reuse still downgrades it "
            "from confirmatory to development/diagnostic status."
        ),
    }
    output = ROOT / args.output
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[exposure] steps {args.start_step}-{args.end_step}: dump_lines={dump_lines}, "
          f"unique_prompt_statements={len(prompts)}, matched_ids={len(training_ids)}")
    print(f"[exposure] fixed-set intersection: {len(intersection_ids)} ids, "
          f"affected theorem indices: {affected_indices}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
