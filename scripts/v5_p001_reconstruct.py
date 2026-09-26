#!/usr/bin/env python3
"""V5-P001 historical surface reconstruction (READ-ONLY, CPU only).

Re-derives the primary analysis surface for V5-P001 from the raw V1 RLVR rollout
dumps and verifies it against the frozen V3 provenance manifest before any
process analysis may begin (owner §4-§6: "No process analysis may begin until
binary reconstruction passes").

What is re-derived, with parsing assumptions copied verbatim from the repo's
canonical analyzers (``scripts/v3_data_audit.py``):

  * group = maximal contiguous run of records with byte-identical ``input``
    inside one step file (files split on ``"\\n"`` only);
  * label = ``classify(score_sum, size)`` -> ALL_FAIL / MIXED / ALL_SUCCESS;
  * contamination = a candidate whose ``tool_feedback`` starts with
    ``# System Error:``; any contaminated candidate excludes the whole group
    from the *primary* surface (kept in the secondary raw table);
  * statement join = FORMAL_BLOCK_RE + normalize on ``input`` -> statement_id
    (deduped parquet map) -> component_id (frozen V2 family-component registry).

This script never writes to the rollout directories, never calls a model, and
never runs the Lean oracle. It emits the candidate *index* (hashes + locators)
that Phase B consumes; the response text itself stays in the raw files.

Outputs (owner §33): ``experiments/manifests/v5/v5_historical_surface.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pyarrow.parquet as pq
import v5_p001_spec as S

FORMAL_BLOCK_RE = re.compile(r"# Formal Statement:\s*\n```lean4\n(.*?)\n```", re.DOTALL)


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def is_system_error(tool_feedback: Any) -> bool:
    return isinstance(tool_feedback, str) and tool_feedback.lstrip().startswith(S.SYSTEM_ERROR_MARKER)


def classify(score_sum: int, size: int) -> str:
    if score_sum == 0:
        return S.GROUP_ALL_FAIL
    if score_sum >= size:
        return S.GROUP_ALL_SUCCESS
    return S.GROUP_MIXED


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# frozen provenance: file hashes
# --------------------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    return json.loads(S.V1_SOURCES_MANIFEST.read_text())


def verify_sources(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify every rollout file against the frozen V3 manifest, then hash the dirs."""

    report: dict[str, Any] = {"seeds": {}, "all_files_match": True}
    for seed, directory in S.SEED_DIRS.items():
        frozen = manifest["seeds"][seed]
        expected = {entry["name"]: entry["sha256"] for entry in frozen["files"]}
        files = sorted(directory.glob("*.jsonl"), key=lambda p: int(p.stem))
        actual = {path.name: sha256_file(path) for path in files}
        mismatches = [
            name
            for name in sorted(set(expected) | set(actual))
            if expected.get(name) != actual.get(name)
        ]
        concat = hashlib.sha256(
            "".join(actual[name] for name in sorted(actual, key=lambda n: int(Path(n).stem))).encode()
        ).hexdigest()
        report["seeds"][seed] = {
            "directory": frozen["directory"],
            "n_files": len(files),
            "n_files_expected": frozen["n_files"],
            "files_hash_mismatches": mismatches,
            "dir_concat_sha256": concat,
            "dir_concat_expected": frozen["dir_concat_sha256"],
            "dir_concat_match": concat == frozen["dir_concat_sha256"],
        }
        if mismatches or concat != frozen["dir_concat_sha256"] or len(files) != frozen["n_files"]:
            report["all_files_match"] = False
    used_concat = hashlib.sha256(
        "".join(report["seeds"][seed]["dir_concat_sha256"] for seed in S.SEED_DIRS).encode()
    ).hexdigest()
    report["used_dirs_concat_sha256"] = used_concat
    report["used_dirs_concat_expected"] = manifest["seeds_summary"]["used_dirs_concat_sha256"]
    report["used_dirs_concat_match"] = used_concat == manifest["seeds_summary"]["used_dirs_concat_sha256"]
    if not report["used_dirs_concat_match"]:
        report["all_files_match"] = False
    return report


# --------------------------------------------------------------------------------------
# reference maps (dataset statements, family components)
# --------------------------------------------------------------------------------------


def load_statement_map() -> dict[str, str]:
    """normalized formal statement -> statement_id (dedupe by statement_id)."""

    table = pq.read_table(S.DATASET, columns=["statement_id", "formal_statement"])
    out: dict[str, str] = {}
    for statement_id, formal in zip(
        table.column("statement_id").to_pylist(), table.column("formal_statement").to_pylist()
    ):
        out.setdefault(normalize(formal), statement_id)
    return out


def load_component_map() -> dict[str, str]:
    """statement_id -> component_id (invert the frozen registry)."""

    registry = json.loads(S.REGISTRY.read_text())
    out: dict[str, str] = {}
    for component in registry["components"]:
        for member in component["member_statement_ids"]:
            out[member] = component["component_id"]
    return out


# --------------------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------------------


def read_records(path: Path) -> tuple[list[dict[str, Any]], list[int], int]:
    """Parse one step file; returns (records, source line numbers, n_unparsable)."""

    records: list[dict[str, Any]] = []
    lines: list[int] = []
    unparsable = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            unparsable += 1
            continue
        records.append(record)
        lines.append(line_number)
    return records, lines, unparsable


def sentinel_kind(pred: str) -> str:
    if pred == S.SENTINEL_NO_PROOF:
        return "no_proof"
    if pred == S.SENTINEL_NO_STATEMENT:
        return "no_statement"
    return "code"


def reconstruct_seed(
    seed: str, statement_map: dict[str, str], component_map: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = S.SEED_DIRS[seed]
    diag = {
        "files": 0,
        "records": 0,
        "unparsable_lines": 0,
        "irregular_blocks": 0,
        "no_formal_block": 0,
        "statement_unmapped": 0,
        "component_unmapped": 0,
    }
    groups: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda p: int(p.stem)):
        diag["files"] += 1
        step = int(path.stem)
        records, lines, unparsable = read_records(path)
        diag["records"] += len(records)
        diag["unparsable_lines"] += unparsable
        i = 0
        group_index = 0
        while i < len(records):
            block_input = records[i]["input"]
            j = i
            while j < len(records) and records[j]["input"] == block_input:
                j += 1
            members = list(zip(records[i:j], lines[i:j]))
            size = len(members)
            if size != S.GROUP_N:
                diag["irregular_blocks"] += 1
            scores = [float(rec["score"]) for rec, _ in members]
            accs = [float(rec["acc"]) for rec, _ in members]
            group_key = f"{seed}:{step:04d}:g{group_index:02d}"
            match = FORMAL_BLOCK_RE.search(block_input)
            if not match:
                diag["no_formal_block"] += 1
            statement_text = normalize(match.group(1)) if match else None
            statement_id = statement_map.get(statement_text) if statement_text else None
            component_id = component_map.get(statement_id) if statement_id else None
            if statement_text and statement_id is None:
                diag["statement_unmapped"] += 1
            if statement_id and component_id is None:
                diag["component_unmapped"] += 1
            candidates = []
            for slot, (record, line_number) in enumerate(members):
                pred = record.get("pred")
                pred = pred if isinstance(pred, str) else ""
                infra = is_system_error(record.get("tool_feedback", ""))
                candidates.append(
                    {
                        "candidate_id": f"{group_key}:c{slot:02d}",
                        "slot": slot,
                        "file": path.name,
                        "line": line_number,
                        "score": scores[slot],
                        "acc": accs[slot],
                        "format_error": record.get("format_error", "No error."),
                        "has_format_error": record.get("format_error", "No error.") != "No error.",
                        "infra": infra,
                        "pred_kind": sentinel_kind(pred),
                        "pred_sha256": S.sha256_text(pred),
                        "response_sha256": S.sha256_text(record.get("response", "")),
                        "tool_feedback_sha256": S.sha256_text(str(record.get("tool_feedback", ""))),
                    }
                )
            groups.append(
                {
                    "group_key": group_key,
                    "seed": seed,
                    "step": step,
                    "group_index": group_index,
                    "size": size,
                    "score_sum": int(sum(scores)),
                    "acc_sum": int(sum(accs)),
                    "label": classify(int(sum(scores)), size),
                    "n_infra": sum(1 for cand in candidates if cand["infra"]),
                    "contaminated": any(cand["infra"] for cand in candidates),
                    "n_format_error": sum(1 for cand in candidates if cand["has_format_error"]),
                    "n_code": sum(1 for cand in candidates if cand["pred_kind"] == "code"),
                    "n_sentinel": sum(1 for cand in candidates if cand["pred_kind"] != "code"),
                    "statement_id": statement_id,
                    "component_id": component_id,
                    "candidates": candidates,
                }
            )
            group_index += 1
            i = j
    return groups, diag


# --------------------------------------------------------------------------------------
# verification summary
# --------------------------------------------------------------------------------------


def label_counts(groups: list[dict[str, Any]]) -> dict[str, int]:
    counts = defaultdict(int)
    for group in groups:
        counts[group["label"]] += 1
    return {
        S.GROUP_ALL_FAIL: counts[S.GROUP_ALL_FAIL],
        S.GROUP_MIXED: counts[S.GROUP_MIXED],
        S.GROUP_ALL_SUCCESS: counts[S.GROUP_ALL_SUCCESS],
    }


def seed_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(groups)
    labels = label_counts(groups)
    primary = [group for group in groups if not group["contaminated"]]
    return {
        "n_groups": total,
        "labels": labels,
        "contaminated_groups": total - len(primary),
        "contaminated_candidates": sum(group["n_infra"] for group in groups),
        "primary_groups": len(primary),
        "primary_labels": label_counts(primary),
        "candidates": sum(group["size"] for group in groups),
        "candidates_code": sum(group["n_code"] for group in groups),
        "candidates_sentinel": sum(group["n_sentinel"] for group in groups),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(S.HISTORICAL_SURFACE))
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="debug only: do not fail when a rollout file disagrees with the frozen manifest",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest()
    sources = verify_sources(manifest)
    for seed, info in sources["seeds"].items():
        print(
            f"[sources] {seed}: {info['n_files']} files, hashes_match={not info['files_hash_mismatches']}, "
            f"dir_concat_match={info['dir_concat_match']}"
        )

    statement_map = load_statement_map()
    component_map = load_component_map()

    all_groups: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    for seed in S.SEED_DIRS:
        groups, seed_diag = reconstruct_seed(seed, statement_map, component_map)
        per_seed[seed] = seed_summary(groups)
        diag[seed] = seed_diag
        all_groups.extend(groups)

    pooled = seed_summary(all_groups)
    primary = [group for group in all_groups if not group["contaminated"]]
    verification = {
        "n_groups": len(all_groups),
        "n_groups_expected": S.EXPECTED_GROUPS_TOTAL,
        "n_candidates": sum(group["size"] for group in all_groups),
        "labels": label_counts(all_groups),
        "labels_expected": {
            S.GROUP_ALL_FAIL: S.EXPECTED_ALL_FAIL,
            S.GROUP_MIXED: S.EXPECTED_MIXED,
            S.GROUP_ALL_SUCCESS: S.EXPECTED_ALL_SUCCESS,
        },
        "primary_groups": len(primary),
        "primary_groups_expected": S.EXPECTED_GROUPS_PRIMARY,
        "primary_labels": label_counts(primary),
        "contaminated_groups": len(all_groups) - len(primary),
        "contaminated_candidates": pooled["contaminated_candidates"],
        "structural_diag": diag,
        "sources": sources,
    }
    checks = {
        "group_count": len(all_groups) == S.EXPECTED_GROUPS_TOTAL,
        "candidate_count": sum(group["size"] for group in all_groups) == S.EXPECTED_GROUPS_TOTAL * S.GROUP_N,
        "labels": verification["labels"]
        == {
            S.GROUP_ALL_FAIL: S.EXPECTED_ALL_FAIL,
            S.GROUP_MIXED: S.EXPECTED_MIXED,
            S.GROUP_ALL_SUCCESS: S.EXPECTED_ALL_SUCCESS,
        },
        "primary_group_count": len(primary) == S.EXPECTED_GROUPS_PRIMARY,
        "file_hashes": sources["all_files_match"],
        "regular_blocks": all(entry["irregular_blocks"] == 0 for entry in diag.values()),
        "formal_blocks": all(entry["no_formal_block"] == 0 for entry in diag.values()),
        "statement_join": all(entry["statement_unmapped"] == 0 for entry in diag.values()),
        "component_join": all(entry["component_unmapped"] == 0 for entry in diag.values()),
        "unparsable_lines": all(entry["unparsable_lines"] == 0 for entry in diag.values()),
    }
    verification["checks"] = checks
    verification["passed"] = all(checks.values())

    artifact = {
        "artifact": "v5_historical_surface",
        "experiment": S.EXPERIMENT_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "primary V5-P001 surface: existing V1 RLVR rollouts (seed1/2/3) only, re-derived "
            "from raw artifacts and verified against the frozen V3 provenance manifest"
        ),
        "parsing_assumptions": {
            "group_regroup_by": "input equality + contiguity (scripts/v3_data_audit.py)",
            "jsonl_split": "\\n only",
            "label_rule": "score_sum: 0 -> ALL_FAIL, >= size -> ALL_SUCCESS, else MIXED",
            "contamination": f"tool_feedback starts with {S.SYSTEM_ERROR_MARKER!r} excludes the group",
            "statement_join": "FORMAL_BLOCK_RE + normalize -> statement_id -> component_id",
        },
        "inputs": {
            "dataset": str(S.DATASET.relative_to(ROOT)),
            "dataset_sha256": sha256_file(S.DATASET),
            "registry": str(S.REGISTRY.relative_to(ROOT)),
            "registry_sha256": sha256_file(S.REGISTRY),
            "v1_sources_manifest": str(S.V1_SOURCES_MANIFEST.relative_to(ROOT)),
            "v1_sources_manifest_sha256": sha256_file(S.V1_SOURCES_MANIFEST),
        },
        "verification": verification,
        "per_seed": per_seed,
        "pooled": pooled,
        "formal_statements": {
            "n_distinct": len({group["statement_id"] for group in all_groups if group["statement_id"]}),
            "n_components": len({group["component_id"] for group in all_groups if group["component_id"]}),
        },
        "groups": all_groups,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=1) + "\n")

    print(json.dumps({"pooled": pooled, "verification": {k: v for k, v in verification.items() if k != "sources"}}, indent=1))
    if not verification["passed"]:
        print("BINARY_RECONSTRUCTION_FAILED")
        if not args.skip_hash_verification:
            return 1
    print("BINARY_RECONSTRUCTION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
