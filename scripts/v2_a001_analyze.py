#!/usr/bin/env python
"""V2-A001 - apply the preregistered Track-A checkpoint-selection rule.

Reads the seven ``v2_a001_<label>.json`` checkpoint-evaluation artifacts (frozen
A3-primary selection set, 512 theorems x 4 samples) and applies the decision
procedure preregistered in ``experiments/manifests/v2/V2-A001.yaml``:

- R3 (degradation override, evaluated first): the default incumbent
  (``seed1_step60``) counts as degraded when the theorem-level paired bootstrap
  of ``Delta(default - theta0)`` has ``ci_low < 0``. The selection then moves to
  the candidate with the largest point estimate against theta0 among those whose
  ``ci_low(Delta(C - theta0)) >= -1.0 pp`` (not detectably worse than the
  anchor); if no candidate qualifies, the least-bad point estimate is selected
  and tagged ``R3-least-bad``.
- R1 (strict superiority): otherwise, a candidate replaces the default only if
  ``ci_low(Delta(C - default)) > 0``; the largest such point estimate wins.
- R2 (default): otherwise the default is retained.

Freeze/owner rule: ``ci_low(Delta(S - theta0)) > 0`` means an established gain
-> freeze-ready for A4. Otherwise the script emits ``OWNER_DECISION_REQUIRED``
with the paired evidence; the single Candidate-2 training run is never started
automatically (explicit owner approval is required, see the manifest).

Tie-break everywhere: larger point estimate, then smaller global step, then
lexicographic label. All statistics are theorem-level paired; deltas are kept in
fraction units internally and reported in percentage points (x100) next to the
raw values, matching the E018/E019/E023 convention.

``decide`` / ``classify_outcome`` / ``theorem_counts_from_records`` are pure and
unit-tested in ``tests/test_v2_a001_analyze.py``; loading is fail-closed
(schema, protocol settings and statement alignment are all validated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.p3c_stats import (
    mcnemar_exact,
    paired_bootstrap,
    win_tie_loss,
)

EXPERIMENT = "V2-A001"
MODELS = ("base", "step10", "step20", "step30", "seed1_step60", "seed2_step60", "seed3_step60")
ANCHOR = "base"  # theta0; reference anchor, never a theta_RL* candidate
DEFAULT = "seed1_step60"  # current incumbent / default candidate
CANDIDATES = tuple(m for m in MODELS if m != ANCHOR)
GLOBAL_STEPS = {
    "base": 0,
    "step10": 10,
    "step20": 20,
    "step30": 30,
    "seed1_step60": 60,
    "seed2_step60": 60,
    "seed3_step60": 60,
}

BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_ALPHA = 0.05
# R3 eligibility margin: candidate is "not detectably worse than theta0" while
# ci_low(Delta(C - theta0)) >= -1.0 pp, i.e. >= -0.01 in fraction units.
R3_ELIGIBLE_CI_LOW = -0.01

SET_REL = "experiments/manifests/v2/v2_a001_selection_set.json"
SET_SHA256 = "f429ddd7c628c3ce36b9ee7309c1fc68c819a0e03d20b38f5fd44112d7853e86"
EXPECTED_SELECTION_SEED = 20260919
EXPECTED_N_THEOREMS = 512
EXPECTED_SETTINGS = {
    "theorems": 512,
    "samples_per_theorem": 4,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 4096,
    "seed_base": 20260917,
    "seed_group_size": 8,
}
MANIFEST_REL = "experiments/manifests/v2/V2-A001.yaml"


class AnalysisError(RuntimeError):
    """Raised when an evaluation artifact deviates from the preregistration."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def theorem_counts_from_records(
    records: list[dict], n_theorems: int, samples_per_theorem: int
) -> list[int]:
    """Per-theorem verified-candidate counts; fail-closed on any coverage gap."""

    seen: set[tuple[int, int]] = set()
    counts = [0] * n_theorems
    for record in records:
        index = record["theorem_index"]
        sample = record["sample_index"]
        if not (isinstance(index, int) and 0 <= index < n_theorems):
            raise AnalysisError(f"theorem_index out of range: {index!r}")
        if not (isinstance(sample, int) and 0 <= sample < samples_per_theorem):
            raise AnalysisError(f"sample_index out of range: {sample!r}")
        if (index, sample) in seen:
            raise AnalysisError(f"duplicate record for theorem {index} sample {sample}")
        seen.add((index, sample))
        counts[index] += int(record["verified"])
    expected = n_theorems * samples_per_theorem
    if len(seen) != expected:
        raise AnalysisError(f"record coverage gap: {len(seen)} of {expected} (theorem, sample) pairs")
    return counts


def _stat_pp(stats: dict) -> dict:
    out = dict(stats)
    out["mean_delta_pp"] = round(stats["mean_delta"] * 100.0, 6)
    out["median_delta_pp"] = round(stats["median_delta"] * 100.0, 6)
    out["ci_low_pp"] = round(stats["ci_low"] * 100.0, 6)
    out["ci_high_pp"] = round(stats["ci_high"] * 100.0, 6)
    return out


def paired_deltas(counts_a: list[int], counts_b: list[int], samples_per_theorem: int) -> list[float]:
    return [(a - b) / samples_per_theorem for a, b in zip(counts_a, counts_b)]


def contrast(counts_a: list[int], counts_b: list[int], samples_per_theorem: int) -> dict:
    stats = paired_bootstrap(
        paired_deltas(counts_a, counts_b, samples_per_theorem),
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        alpha=BOOTSTRAP_ALPHA,
    )
    return _stat_pp(stats)


def _tie_key(mean_delta: float, label: str) -> tuple:
    """Pre-registered tie-break: point estimate desc, global step asc, label asc."""

    return (-mean_delta, GLOBAL_STEPS[label], label)


def decide(vs_anchor: dict, vs_default: dict) -> dict:
    """Apply the preregistered selection rule to precomputed contrast statistics.

    ``vs_anchor`` / ``vs_default`` map every candidate label to
    ``{"mean_delta", "ci_low", ...}`` in fraction units, the output of
    ``paired_bootstrap`` on theorem-level deltas. Pure function.
    """

    for label in CANDIDATES:
        if label not in vs_anchor:
            raise AnalysisError(f"missing vs_anchor contrast for {label!r}")
        if label not in vs_default:
            raise AnalysisError(f"missing vs_default contrast for {label!r}")

    incumbent_degraded = vs_anchor[DEFAULT]["ci_low"] < 0.0
    if incumbent_degraded:
        eligible = [c for c in CANDIDATES if vs_anchor[c]["ci_low"] >= R3_ELIGIBLE_CI_LOW]
        if eligible:
            rule = "R3"
            pool = eligible
        else:
            rule = "R3-least-bad"
            pool = list(CANDIDATES)
        selected = min(pool, key=lambda c: _tie_key(vs_anchor[c]["mean_delta"], c))
    else:
        superior = [c for c in CANDIDATES if c != DEFAULT and vs_default[c]["ci_low"] > 0.0]
        if superior:
            rule = "R1"
            selected = min(superior, key=lambda c: _tie_key(vs_default[c]["mean_delta"], c))
        else:
            rule = "R2"
            selected = DEFAULT

    return {
        "rule": rule,
        "selected": selected,
        "incumbent_degraded": incumbent_degraded,
        "r3_eligible_candidates": sorted(
            c for c in CANDIDATES if vs_anchor[c]["ci_low"] >= R3_ELIGIBLE_CI_LOW
        ),
        "no_candidate_established_gain": all(vs_anchor[c]["ci_low"] <= 0.0 for c in CANDIDATES),
        "tie_break": "point estimate desc, global step asc, label asc",
    }


def classify_outcome(selected_vs_anchor: dict) -> dict:
    """Freeze / owner-decision classification for the selected model vs theta0."""

    mean_pp = selected_vs_anchor["mean_delta"] * 100.0
    low_pp = selected_vs_anchor["ci_low"] * 100.0
    high_pp = selected_vs_anchor["ci_high"] * 100.0
    base = {
        "point_estimate_pp": round(mean_pp, 6),
        "ci_low_pp": round(low_pp, 6),
        "ci_high_pp": round(high_pp, 6),
    }
    if low_pp > 0.0:
        return {
            **base,
            "recommendation": "FREEZE_READY",
            "candidate2_eligible": False,
            "reason": "established_gain_over_theta0",
        }
    if high_pp < 0.0:
        reason = "degraded_vs_theta0"
    elif mean_pp < 1.0:
        reason = "null_no_detectable_gain"
    else:
        reason = "borderline_subthreshold_positive"
    return {
        **base,
        "recommendation": "OWNER_DECISION_REQUIRED",
        "candidate2_eligible": True,
        "reason": reason,
    }


def load_model_artifact(label: str, path: Path, set_theorems: list[dict], set_path: Path) -> dict:
    """Fail-closed load of one evaluation artifact; returns counts + traceability."""

    if not path.exists():
        raise AnalysisError(f"evaluation artifact missing: {path} (run scripts/run_v2_a001.sh first)")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "p3c_checkpoint_eval":
        raise AnalysisError(f"{path}: unexpected artifact_type {artifact.get('artifact_type')!r}")
    if artifact.get("checkpoint") != label:
        raise AnalysisError(f"{path}: checkpoint {artifact.get('checkpoint')!r} != {label!r}")
    if Path(artifact.get("fixed_set", "")).resolve() != set_path.resolve():
        raise AnalysisError(f"{path}: fixed set {artifact.get('fixed_set')!r} != {set_path!s}")
    if artifact.get("selection_seed") != EXPECTED_SELECTION_SEED:
        raise AnalysisError(f"{path}: selection_seed {artifact.get('selection_seed')!r}")
    settings = artifact.get("settings", {})
    for key, expected in EXPECTED_SETTINGS.items():
        if settings.get(key) != expected:
            raise AnalysisError(f"{path}: settings[{key}] = {settings.get(key)!r}, expected {expected!r}")

    records = artifact.get("records", [])
    if len(records) != EXPECTED_N_THEOREMS * EXPECTED_SETTINGS["samples_per_theorem"]:
        raise AnalysisError(f"{path}: {len(records)} records, expected {EXPECTED_N_THEOREMS * 4}")
    expected_ids = [theorem["statement_id"] for theorem in set_theorems]
    for record in records:
        if expected_ids[record["theorem_index"]] != record["statement_id"]:
            raise AnalysisError(
                f"{path}: statement_id mismatch at theorem_index {record['theorem_index']}"
            )

    counts = theorem_counts_from_records(
        records, EXPECTED_N_THEOREMS, EXPECTED_SETTINGS["samples_per_theorem"]
    )
    return {
        "artifact": str(path),
        "artifact_sha256": file_sha256(path),
        "created_at_utc": artifact.get("created_at_utc"),
        "global_step": artifact.get("global_step"),
        "model_dir": artifact.get("model_dir"),
        "summary": artifact.get("summary"),
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the preregistered V2-A001 selection rule.")
    parser.add_argument("--set", default=SET_REL, help="frozen A3-primary selection set (sha256 pinned)")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out", default="experiments/results/v2_a001_analysis.json")
    args = parser.parse_args()

    set_path = Path(args.set)
    if not set_path.is_absolute():
        set_path = ROOT / set_path
    sha_now = file_sha256(set_path)
    if sha_now != SET_SHA256:
        raise AnalysisError(f"selection set sha256 {sha_now} != preregistered {SET_SHA256}")
    selection_set = json.loads(set_path.read_text(encoding="utf-8"))
    set_theorems = selection_set["theorems"]
    if len(set_theorems) != EXPECTED_N_THEOREMS:
        raise AnalysisError(f"selection set has {len(set_theorems)} theorems, expected 512")

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    loaded = {}
    for label in MODELS:
        loaded[label] = load_model_artifact(
            label, results_dir / f"v2_a001_{label}.json", set_theorems, set_path
        )

    counts = {label: loaded[label]["counts"] for label in MODELS}
    samples = EXPECTED_SETTINGS["samples_per_theorem"]
    vs_anchor = {c: contrast(counts[c], counts[ANCHOR], samples) for c in CANDIDATES}
    vs_default = {c: contrast(counts[c], counts[DEFAULT], samples) for c in CANDIDATES}

    decision = decide(vs_anchor, vs_default)
    selected = decision["selected"]
    decision["selected_vs_anchor"] = {
        "paired_bootstrap": vs_anchor[selected],
        "win_tie_loss": win_tie_loss(counts[ANCHOR], counts[selected]),
        "mcnemar_exact": mcnemar_exact(
            [c >= 1 for c in counts[ANCHOR]], [c >= 1 for c in counts[selected]]
        ),
    }
    decision["selected_vs_default"] = {
        "paired_bootstrap": vs_default[selected],
        "win_tie_loss": win_tie_loss(counts[DEFAULT], counts[selected]),
        "mcnemar_exact": mcnemar_exact(
            [c >= 1 for c in counts[DEFAULT]], [c >= 1 for c in counts[selected]]
        ),
    }
    outcome = classify_outcome(vs_anchor[selected])

    manifest_path = ROOT / MANIFEST_REL
    analysis = {
        "artifact_type": "v2_a001_analysis",
        "experiment": EXPERIMENT,
        "track": "A",
        "manifest": MANIFEST_REL,
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
        "selection_set": {"path": str(set_path), "sha256": sha_now},
        "n_theorems": EXPECTED_N_THEOREMS,
        "samples_per_theorem": samples,
        "statistics": {
            "unit": "theorem-level paired delta of c_i/n (fraction; pp = x100)",
            "methods": ["paired_bootstrap", "win_tie_loss", "mcnemar_exact"],
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_alpha": BOOTSTRAP_ALPHA,
        },
        "models": {
            label: {
                "artifact": loaded[label]["artifact"],
                "artifact_sha256": loaded[label]["artifact_sha256"],
                "created_at_utc": loaded[label]["created_at_utc"],
                "global_step": loaded[label]["global_step"],
                "model_dir": loaded[label]["model_dir"],
                "verified_candidates": sum(counts[label]),
                "mean_c_over_n": round(sum(counts[label]) / (EXPECTED_N_THEOREMS * samples), 6),
                "theorems_solved_at_least_1": sum(1 for c in counts[label] if c >= 1),
                "per_theorem_counts": counts[label],
            }
            for label in MODELS
        },
        "contrasts_vs_anchor": vs_anchor,
        "contrasts_vs_default": vs_default,
        "decision": decision,
        "outcome": outcome,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[V2-A001] rule={decision['rule']} selected={selected}")
    print(f"[V2-A001] outcome={outcome['recommendation']} reason={outcome['reason']}")
    print(
        f"[V2-A001] outcome point/CI pp: {outcome['point_estimate_pp']:+.3f} "
        f"[{outcome['ci_low_pp']:+.3f}, {outcome['ci_high_pp']:+.3f}]"
    )
    for label in CANDIDATES:
        stat = vs_anchor[label]
        print(
            f"  {label:>14}: vs theta0 {stat['mean_delta_pp']:+.3f} pp "
            f"[{stat['ci_low_pp']:+.3f}, {stat['ci_high_pp']:+.3f}]"
        )
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
