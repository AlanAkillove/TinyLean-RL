#!/usr/bin/env python3
"""V2-E001 analysis — token-level budget-semantics verdicts.

Reads the raw rollout JSONL (scripts/v2_b1_budget_semantics.py) and checks,
per (theorem_rank, replicate):

- determinism: ``det1 == det2`` (token ids);
- prefix: every direct-budget run equals ``full4096[:len(run)]`` (token ids).

Stage-2 batch records are grouped per (engine, run_tag) with the same checks
plus batch-vs-single reference differences. Writes
``experiments/results/v2_b1_budget_semantics.json`` (gitignored) and prints a
human-readable summary. Comparison is always on token ids — never strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

STAGE1_PREFIX_TAGS = ("det1", "det2", "b1024", "b2048", "b3072")
STAGE2_ENGINES = {
    "hf-batch": ("batch1024a", "batch1024b", "batch4096"),
    "vllm": ("v1024a", "v1024b", "v4096"),
}


def first_diff(left: list[int], right: list[int]) -> int | None:
    for index, (x, y) in enumerate(zip(left, right)):
        if x != y:
            return index
    return None


def prefix_match(short: list[int], full: list[int]) -> dict[str, Any]:
    """True when ``short`` is exactly the head of ``full`` (token ids)."""

    if len(short) > len(full):
        return {"match": False, "n": len(short), "first_diff": None, "reason": "longer_than_full"}
    diff = first_diff(short, full[: len(short)])
    return {"match": diff is None, "n": len(short), "first_diff": diff}


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_stage1(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("stage") != "1":
            continue
        key = (int(record["theorem_rank"]), int(record["replicate"]))
        groups.setdefault(key, {})[record["run_tag"]] = record

    per_candidate: list[dict[str, Any]] = []
    determinism_failures: list[str] = []
    prefix_failures: list[str] = []
    checked_determinism = 0
    checked_prefix = 0
    for key in sorted(groups):
        runs = groups[key]
        label = f"rank={key[0]} rep={key[1]}"
        missing = [
            tag
            for tag in ("det1", "det2", "b1024", "b2048", "b3072", "full4096")
            if tag not in runs
        ]
        if missing:
            per_candidate.append({"label": label, "status": "incomplete", "missing": missing})
            continue
        full = runs["full4096"]["token_ids"]
        entry: dict[str, Any] = {
            "theorem_rank": key[0],
            "replicate": key[1],
            "gen_seed": runs["full4096"]["gen_seed"],
            "full_len": len(full),
            "full_hit_eos": bool(runs["full4096"]["hit_eos"]),
            "checks": {},
        }
        checked_determinism += 1
        det_ok = runs["det1"]["token_ids"] == runs["det2"]["token_ids"]
        entry["checks"]["determinism_det1_eq_det2"] = det_ok
        if not det_ok:
            diff = first_diff(runs["det1"]["token_ids"], runs["det2"]["token_ids"])
            determinism_failures.append(f"{label} first_diff={diff}")
        for tag in STAGE1_PREFIX_TAGS:
            checked_prefix += 1
            verdict = prefix_match(runs[tag]["token_ids"], full)
            entry["checks"][f"prefix_{tag}"] = verdict
            if not verdict["match"]:
                prefix_failures.append(
                    f"{label} {tag} n={verdict['n']} first_diff={verdict['first_diff']}"
                )
        per_candidate.append(entry)

    return {
        "determinism": {
            "checked": checked_determinism,
            "passed": checked_determinism - len(determinism_failures),
            "failures": determinism_failures,
        },
        "prefix": {
            "checked": checked_prefix,
            "passed": checked_prefix - len(prefix_failures),
            "failures": prefix_failures,
        },
        "per_candidate": per_candidate,
    }


def analyze_stage2(records: list[dict[str, Any]], stage1_records: list[dict[str, Any]]) -> dict[str, Any]:
    # stage 1 lookup for batch-vs-single reference
    single: dict[tuple[int, str], list[int]] = {}
    for record in stage1_records:
        if record.get("stage") == "1" and int(record.get("replicate", -1)) == 0:
            single[(int(record["theorem_rank"]), record["run_tag"])] = record["token_ids"]

    by_engine: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("stage") != "2":
            continue
        by_engine.setdefault(record["engine"], {})[record["run_tag"]] = record

    results: dict[str, Any] = {}
    for engine, tags in STAGE2_ENGINES.items():
        runs = by_engine.get(engine)
        if not runs:
            results[engine] = {"status": "skipped"}
            continue
        missing = [tag for tag in tags if tag not in runs]
        if missing:
            results[engine] = {"status": "incomplete", "missing": missing}
            continue
        tag_a, tag_b, tag_full = tags
        entries: list[dict[str, Any]] = []
        det_failures: list[str] = []
        prefix_failures: list[str] = []
        reference_diffs: list[str] = []
        items_a = {item["theorem_rank"]: item for item in runs[tag_a]["items"]}
        items_b = {item["theorem_rank"]: item for item in runs[tag_b]["items"]}
        items_full = {item["theorem_rank"]: item for item in runs[tag_full]["items"]}
        for rank in sorted(items_a):
            ids_a = items_a[rank]["token_ids"]
            ids_b = items_b[rank]["token_ids"]
            ids_full = items_full[rank]["token_ids"]
            label = f"rank={rank}"
            entry: dict[str, Any] = {"theorem_rank": rank, "checks": {}}
            det_ok = ids_a == ids_b
            entry["checks"]["determinism"] = det_ok
            if not det_ok:
                det_failures.append(f"{label} first_diff={first_diff(ids_a, ids_b)}")
            verdict = prefix_match(ids_a, ids_full)
            entry["checks"]["prefix"] = verdict
            if not verdict["match"]:
                prefix_failures.append(
                    f"{label} n={verdict['n']} first_diff={verdict['first_diff']}"
                )
            single_ids = single.get((rank, "b1024"))
            if single_ids is not None:
                same = ids_a == single_ids
                entry["checks"]["batch_vs_single_same_tokens"] = same
                if not same:
                    reference_diffs.append(
                        f"{label} first_diff={first_diff(ids_a, single_ids)}"
                    )
            entries.append(entry)
        results[engine] = {
            "determinism": {"checked": len(entries), "passed": len(entries) - len(det_failures), "failures": det_failures},
            "prefix": {"checked": len(entries), "passed": len(entries) - len(prefix_failures), "failures": prefix_failures},
            "batch_vs_single_reference": {
                "checked": len(entries),
                "identical": len(entries) - len(reference_diffs),
                "differences": reference_diffs,
            },
            "per_candidate": entries,
        }
    return results


def verdict_of(stage1: dict[str, Any]) -> str:
    determinism = stage1["determinism"]
    prefix = stage1["prefix"]
    if determinism["failures"] or prefix["failures"]:
        return "FAIL"
    if determinism["checked"] == 0 or prefix["checked"] == 0:
        return "CONDITIONAL"
    return "PASS"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="experiments/results/v2_b1_rollouts.jsonl")
    parser.add_argument("--output", default="experiments/results/v2_b1_budget_semantics.json")
    args = parser.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.is_absolute():
        rollouts_path = ROOT / rollouts_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    records = load_records(rollouts_path)
    stage1 = analyze_stage1(records)
    stage2 = analyze_stage2(records, records)
    summary = {
        "artifact_type": "v2_e001_budget_semantics",
        "experiment": "V2-E001",
        "track": "B",
        "stage1_single_candidate": {
            "verdict": verdict_of(stage1),
            **stage1,
        },
        "stage2_batch_probes": stage2,
        "raw_artifact": {"path": str(rollouts_path), "sha256": sha256_of(rollouts_path)},
        "record_count": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"records: {len(records)}")
    print(f"stage1 verdict: {summary['stage1_single_candidate']['verdict']}")
    det = summary["stage1_single_candidate"]["determinism"]
    pre = summary["stage1_single_candidate"]["prefix"]
    print(f"  determinism: {det['passed']}/{det['checked']} passed")
    for failure in det["failures"][:10]:
        print(f"    FAIL {failure}")
    print(f"  prefix: {pre['passed']}/{pre['checked']} passed")
    for failure in pre["failures"][:10]:
        print(f"    FAIL {failure}")
    for engine, result in summary["stage2_batch_probes"].items():
        if result.get("status") in {"skipped", "incomplete"}:
            print(f"stage2[{engine}]: {result.get('status')} {result.get('missing', '')}")
            continue
        print(
            f"stage2[{engine}]: determinism {result['determinism']['passed']}/{result['determinism']['checked']}, "
            f"prefix {result['prefix']['passed']}/{result['prefix']['checked']}, "
            f"batch-vs-single identical {result['batch_vs_single_reference']['identical']}/{result['batch_vs_single_reference']['checked']}"
        )
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
