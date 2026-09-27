#!/usr/bin/env python3
"""R001 offline replay: the online credit chain vs the frozen Phase-B artifacts.

No verifier, no GPU, no generation: every candidate is replayed from stored bytes only
(``runs/v5_p001_process/raw/*.json`` for the oracle item, the V1 rollout sources for the response
text, the dataset for the formal statement). Two things are proven per candidate:

1. the online chain (``scripts/v5_r001_online_credit.py``) reproduces the frozen offline
   ``token_credit`` positions exactly - same token index, same label, no shifts;
2. the S3 counter set (owner §14: mapped, ambiguous, outside_response, retokenization_mismatch,
   conflict, infra_censored, no_code) is produced on real data.

The round-trip gate is exercised in its *passing* direction here: the ids fed to the chain are the
re-encoding of the same response text Phase B tokenized, so the comparison isolates the rest of the
chain (frame construction, spans, blame, labels, precedence, masking). The mismatch direction is
covered by fixtures in ``tests/test_v5_r001_online_credit.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_p001_process_run as PR
import v5_process_oracle as O
import v5_r001_online_credit as C

from tinylean_rl.rl.process_credit import LAMBDA_PRIMARY, compute_process_advantage

POOL_LABELS = (
    ("d1_token_positions", "d1"),
    ("d2_token_positions", "d2"),
    ("success_token_positions", "success"),
)


def offline_positions(token_credit: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, label in POOL_LABELS:
        for entry in token_credit[key]:
            out[int(entry[0])] = label
    return out


def load_source_rows(manifest: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    cache: dict[tuple[str, str], list[str]] = {}
    for seed, info in manifest["seeds"].items():
        for name in sorted({entry["name"] for entry in info["files"]}):
            path = ROOT / info["directory"] / name
            if path.exists():
                cache[(seed, name)] = path.read_text(encoding="utf-8").split("\n")
    return cache


def load_formals() -> dict[str, str]:
    return PR.load_formal_by_statement_id()


def replay(limit: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapper = O.TokenMapper()
    formals = load_formals()
    manifest = json.loads(O.S.V1_SOURCES_MANIFEST.read_text())
    rows = load_source_rows(manifest)
    labels_path = ROOT / "runs/v5_p001_process/v5_p001_process_labels.jsonl"
    raw_dir = ROOT / "runs/v5_p001_process/raw"

    results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    skipped = {"no_response_row": 0, "no_item": 0}
    n = 0
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if limit is not None and n >= limit:
                break
            source_rows = rows.get((entry["seed"], entry["file"]))
            if source_rows is None or entry["line"] >= len(source_rows):
                skipped["no_response_row"] += 1
                continue
            row = json.loads(source_rows[entry["line"]])
            response = row.get("response")
            if not isinstance(response, str):
                skipped["no_response_row"] += 1
                continue
            raw_path = raw_dir / (entry["candidate_id"].replace(":", "_") + ".json")
            item = None
            if raw_path.exists():
                item = json.loads(raw_path.read_text()).get("item")
            token_ids = mapper.tokenizer(response, add_special_tokens=False)["input_ids"]
            result = C.credit_for_candidate(
                token_ids=token_ids,
                formal_statement=formals.get(entry["statement_id"] or "", "") or "",
                item=item,
                mapper=mapper,
                candidate_id=entry["candidate_id"],
            )
            expected = offline_positions(entry["token_credit"])
            n += 1
            results.append(result)
            if result["positions"] != expected:
                mismatches.append(
                    {
                        "candidate_id": entry["candidate_id"],
                        "online": {str(k): v for k, v in result["positions"].items()},
                        "offline": {str(k): v for k, v in expected.items()},
                    }
                )
    return {
        "artifact_type": "v5_r001_online_replay",
        "purpose": "online credit chain vs frozen Phase-B token_credit (CPU only, no verifier)",
        "replayed": len(results),
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "skipped": skipped,
        "counters": C.summarize(results),
    }, results


def advantage_probe(results: list[dict[str, Any]], seed: int = 20260927) -> dict[str, Any]:
    """One synthetic group through the tensor stage, λ = 0 and λ = 1, from replayed positions."""

    import torch

    from tinylean_rl.rl.grpo import compute_grpo_outcome_advantage

    valid = [r for r in results if r["valid"] == 1]
    with_credit = [r for r in valid if r["positions"]]
    group = (with_credit or valid)[:4]
    if len(group) < 2:
        return {"probe": "insufficient_replayed_rows"}
    torch.manual_seed(seed)
    width = max((max(r["positions"]) + 1 for r in group if r["positions"]), default=16)
    rewards = torch.zeros(len(group), width)
    mask = torch.ones(len(group), width)
    for row, result in enumerate(group):
        rewards[row, width - 1] = 1.0 if result["status"] == "SUCCESS" else 0.0
    index = ["probe"] * len(group)
    credit = [r["positions"] for r in group]
    reference, _ = compute_grpo_outcome_advantage(
        rewards.clone(), mask.clone(), index, norm_adv_by_std_in_grpo=False
    )
    zero, _, keep_zero, _ = compute_process_advantage(
        rewards.clone(), mask.clone(), index, [1] * len(group), credit, lambda_process=0.0
    )
    one, _, keep_one, stats_one = compute_process_advantage(
        rewards.clone(), mask.clone(), index, [1] * len(group), credit, lambda_process=LAMBDA_PRIMARY
    )
    deltas = {
        str(row): float((one[row] - zero[row]).abs().max())
        for row in range(len(group))
        if credit[row]
    }
    return {
        "probe": "advantage_tensor",
        "rows": len(group),
        "n_valid": len(valid),
        "n_valid_with_credit": len(with_credit),
        "rows_with_credit_in_probe": sum(1 for r in group if r["positions"]),
        "width": width,
        "lambda0_equals_v1_bit_for_bit": bool(torch.equal(zero, reference)),
        "row_keep_identical": keep_zero.tolist() == keep_one.tolist() == [1.0] * len(group),
        "expected_credit_tokens": sum(len(r["positions"]) for r in group),
        "credit_tokens_applied": stats_one.credit_tokens,
        "dropped_by_mask": stats_one.dropped_by_mask,
        "out_of_range": stats_one.out_of_range,
        "max_abs_delta_per_row_with_credit": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report, results = replay(args.limit)
    report["advantage_tensor_probe"] = advantage_probe(results)
    print(json.dumps({k: report[k] for k in ("replayed", "n_mismatches", "skipped")}, indent=1))
    print(json.dumps(report["counters"], indent=1))
    print(json.dumps(report["advantage_tensor_probe"], indent=1))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"[replay] wrote {out}")
    return 0 if report["n_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
