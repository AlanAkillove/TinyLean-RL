#!/usr/bin/env python3
"""V5-P001 canonical analyzer (runs exactly once, after the Phase-B freeze).

Reads the frozen process labels, applies the preregistered gates in the frozen
order, and writes ``experiments/manifests/v5/V5-P001_results.json`` plus the
``V5_P001_RESULT`` report. It never touches the oracle, the model or the data
generation path; all statistical semantics are frozen in
``docs/v5/V5-P001_preregistration.md`` and ``scripts/v5_p001_spec.py`` before
the first process label existed (owner §22, §29, §30, §34).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_p001_spec as S
import v5_process_oracle as O
from v5_p001_process_run import RunPaths, build_plan, code_stamp

RECOVERABLE = "recoverable"
CENSORED = "censored"


# --------------------------------------------------------------------------------------
# inputs and integrity gates
# --------------------------------------------------------------------------------------


class AnalyzerError(RuntimeError):
    """A frozen input is missing, mutated or inconsistent: stop, do not analyze."""


def load_inputs(run_dir: Path) -> dict[str, Any]:
    paths = RunPaths(run_dir)
    for required in (paths.run_meta, paths.labels, paths.raw_manifest, paths.freeze):
        if not required.exists():
            raise AnalyzerError(f"missing frozen input {required}")
    for required in (S.HISTORICAL_SURFACE, S.ORACLE_VALIDATION):
        if not required.exists():
            raise AnalyzerError(f"missing frozen input {required}")
    surface = json.loads(S.HISTORICAL_SURFACE.read_text(encoding="utf-8"))
    validation = json.loads(S.ORACLE_VALIDATION.read_text(encoding="utf-8"))
    freeze = json.loads(paths.freeze.read_text(encoding="utf-8"))
    run_meta = json.loads(paths.run_meta.read_text(encoding="utf-8"))
    if freeze.get("passed") is not True:
        raise AnalyzerError("freeze did not pass: refusing to analyze")
    if validation.get("summary", {}).get("all_validated") is not True:
        raise AnalyzerError("fixture validation did not pass")
    if validation.get("summary", {}).get("all_deterministic") is not True:
        raise AnalyzerError("fixture validation is not deterministic")
    if S.sha256_file(paths.labels) != freeze["labels"]["sha256"]:
        raise AnalyzerError("labels file changed after the freeze")
    if S.sha256_file(paths.raw_manifest) != freeze["raw"]["manifest_sha256"]:
        raise AnalyzerError("raw manifest changed after the freeze")
    stamp = code_stamp()
    for name, artifact in (("freeze", freeze), ("run_meta", run_meta)):
        if artifact.get("code_stamp") != stamp:
            raise AnalyzerError(f"{name} code stamp differs from this analyzer's stamp")
    if run_meta.get("debug_limit") is not None:
        raise AnalyzerError("run_meta records a debug limit: refusing to analyze")
    if freeze["coverage"]["n_missing"]:
        raise AnalyzerError(f"labels are incomplete: {freeze['coverage']['n_missing']} missing")
    validation_path = paths.validation
    if validation_path.exists():
        validation_artifact = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_artifact.get("passed") is not True:
            raise AnalyzerError("structural validation did not pass")
    else:
        raise AnalyzerError("structural validation artifact is missing")
    planned = build_plan(surface)
    records = []
    seen: set[str] = set()
    for line in paths.labels.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        candidate_id = record["candidate_id"]
        if candidate_id in seen:
            raise AnalyzerError(f"duplicate label record {candidate_id}")
        seen.add(candidate_id)
        records.append(record)
    planned_ids = {entry["candidate_id"] for entry in planned}
    if len(records) != len(planned):
        raise AnalyzerError(f"{len(records)} records for {len(planned)} planned candidates")
    if seen != planned_ids:
        raise AnalyzerError(f"{len(seen - planned_ids)} unplanned label records")
    return {
        "paths": paths,
        "surface": surface,
        "freeze": freeze,
        "run_meta": run_meta,
        "validation": json.loads(validation_path.read_text(encoding="utf-8")),
        "planned": planned,
        "records": records,
    }


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["group_key"], []).append(record)
    return groups


def candidate_class(record: dict[str, Any]) -> str:
    """Censoring trichotomy used by every denominator in this report."""

    oracle = record.get("oracle")
    if record["process_status"] == "FORMAT_NO_CODE":
        return "no_code"
    if oracle and oracle["infra"]:
        return CENSORED
    return "conclusive"


# --------------------------------------------------------------------------------------
# E1 / E2
# --------------------------------------------------------------------------------------


def compute_e1(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tactic -> first-token mapping rate among position-eligible parsed nodes."""

    eligible = mapped = exact = contained = 0
    outside = not_in_response = response_only = ambiguous = unmapped = 0
    for record in records:
        if candidate_class(record) != "conclusive":
            continue
        for tactic in record["tactics"]:
            mapping = tactic["mapping"]
            status = mapping["status"]
            if status == "outside_response":
                outside += 1
                continue
            if status == "response_only":
                response_only += 1
                continue
            if mapping.get("span_in_response") is not True:
                not_in_response += 1
                continue
            eligible += 1
            if status == "exact":
                exact += 1
                mapped += 1
            elif status == "contained":
                contained += 1
                mapped += 1
            elif status == "ambiguous":
                ambiguous += 1
            else:
                unmapped += 1
    rate = mapped / eligible if eligible else None
    return {
        "definition": (
            "numerator = parsed tactic nodes whose span is verbatim in the generated response and "
            "whose first token is mapped (exact or contained); denominator = parsed tactic nodes "
            "whose span is verbatim in the generated response"
        ),
        "eligible_tactics": eligible,
        "mapped_tactics": mapped,
        "n_exact": exact,
        "n_contained": contained,
        "n_ambiguous": ambiguous,
        "n_unmapped": unmapped,
        "n_outside_response": outside,
        "n_span_not_in_response": not_in_response,
        "n_response_only": response_only,
        "rate": rate,
        "threshold": S.GATE_E1_MIN,
        "passed": rate is not None and rate >= S.GATE_E1_MIN,
    }


def compute_e2(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Conclusive-oracle fraction among code-extractable primary candidates."""

    code = [r for r in records if candidate_class(r) != "no_code"]
    conclusive = [r for r in code if candidate_class(r) == "conclusive"]
    censored = [r for r in code if candidate_class(r) == CENSORED]
    rate = len(conclusive) / len(code) if code else None
    return {
        "definition": "conclusive outcomes / code-extractable primary candidates",
        "n_code": len(code),
        "n_conclusive": len(conclusive),
        "n_censored": len(censored),
        "censored_share": (len(censored) / len(code)) if code else None,
        "rate": rate,
        "threshold": S.GATE_E2_MIN,
        "passed": rate is not None and rate >= S.GATE_E2_MIN,
    }


# --------------------------------------------------------------------------------------
# primary estimand, bootstrap, per-seed, quartiles
# --------------------------------------------------------------------------------------


def all_fail_groups(
    groups: dict[str, list[dict[str, Any]]], surface: dict[str, Any]
) -> list[dict[str, Any]]:
    """Primary all-fail groups with their evaluation status."""

    surface_groups = {
        group["group_key"]: group for group in surface["groups"] if not group["contaminated"]
    }
    out = []
    for group_key, records in groups.items():
        info = surface_groups[group_key]
        if info["label"] != S.GROUP_ALL_FAIL:
            continue
        classes = [candidate_class(record) for record in records]
        n_conclusive = classes.count("conclusive")
        n_censored = classes.count(CENSORED)
        n_no_code = classes.count("no_code")
        if n_conclusive == 0 and n_no_code == 0:
            status = "excluded_all_censored"
        else:
            status = "evaluable"
        max_tactics = max(
            [record["n_tactics"] for record in records if candidate_class(record) == "conclusive"],
            default=0,
        )
        out.append(
            {
                "group_key": group_key,
                "seed": info["seed"],
                "component_id": info["component_id"],
                "statement_id": info["statement_id"],
                "status": status,
                "n_conclusive": n_conclusive,
                "n_censored": n_censored,
                "n_no_code": n_no_code,
                "group_max_tactics": max_tactics,
                "recoverable": any(
                    candidate_class(record) == "conclusive" and O.structured_recoverable(record)
                    for record in records
                ),
            }
        )
    out.sort(key=lambda entry: entry["group_key"])
    return out


def rate(population: list[dict[str, Any]]) -> float | None:
    if not population:
        return None
    return sum(1 for entry in population if entry["recoverable"]) / len(population)


def component_bootstrap(
    population: list[dict[str, Any]], *, n_resamples: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Family/component-cluster percentile bootstrap of the group-level rate."""

    n_resamples = n_resamples or int(S.BOOTSTRAP["n_resamples"])
    seed = seed if seed is not None else S.BOOTSTRAP["seed"]
    clusters: dict[str, list[dict[str, Any]]] = {}
    for entry in population:
        clusters.setdefault(entry["component_id"] or entry["statement_id"], []).append(entry)
    keys = sorted(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        draws = [clusters[keys[rng.randrange(len(keys))]] for _ in range(len(keys))]
        flat = [entry for draw in draws for entry in draw]
        value = rate(flat)
        if value is not None:
            samples.append(value)
    samples.sort()
    lower_index = int(S.BOOTSTRAP["alpha"] / 2 * len(samples))
    upper_index = min(len(samples) - 1, int((1 - S.BOOTSTRAP["alpha"] / 2) * len(samples)))
    return {
        "n_clusters": len(keys),
        "n_resamples": n_resamples,
        "seed": seed,
        "method": S.BOOTSTRAP["method"],
        "point": rate(population),
        "ci_lower": samples[lower_index] if samples else None,
        "ci_upper": samples[upper_index] if samples else None,
        "mean": statistics.fmean(samples) if samples else None,
        "median": statistics.median(samples) if samples else None,
        "n_groups": len(population),
        "n_recoverable": sum(1 for entry in population if entry["recoverable"]),
    }


def per_seed(population: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for seed in sorted({entry["seed"] for entry in population}):
        subset = [entry for entry in population if entry["seed"] == seed]
        out[seed] = {
            "n_groups": len(subset),
            "n_recoverable": sum(1 for entry in subset if entry["recoverable"]),
            "rate": rate(subset),
            "threshold": S.GATE_G2_PER_SEED_MIN,
            "passed": (rate(subset) or 0.0) >= S.GATE_G2_PER_SEED_MIN,
        }
    return out


def quartiles(population: list[dict[str, Any]], n_bins: int = 4) -> list[dict[str, Any]]:
    """Equal-count length bins over (group_max_tactics, group_key)."""

    ordered = sorted(population, key=lambda entry: (entry["group_max_tactics"], entry["group_key"]))
    out = []
    size = len(ordered)
    for index in range(n_bins):
        start = index * size // n_bins
        stop = (index + 1) * size // n_bins
        bin_entries = ordered[start:stop]
        if not bin_entries:
            out.append({"bin": index + 1, "n_groups": 0, "passed": False})
            continue
        value = rate(bin_entries)
        out.append(
            {
                "bin": index + 1,
                "n_groups": len(bin_entries),
                "tactic_min": bin_entries[0]["group_max_tactics"],
                "tactic_max": bin_entries[-1]["group_max_tactics"],
                "n_zero_code": sum(1 for entry in bin_entries if entry["n_conclusive"] == 0),
                "n_recoverable": sum(1 for entry in bin_entries if entry["recoverable"]),
                "rate": value,
                "threshold": S.GATE_G3_RATE_MIN,
                "n_guard_ok": len(bin_entries) >= S.GATE_QUARTILE_MIN_N,
                "passed": len(bin_entries) >= S.GATE_QUARTILE_MIN_N
                and (value or 0.0) > S.GATE_G3_RATE_MIN,
            }
        )
    return out


# --------------------------------------------------------------------------------------
# mechanism, token credit, format decomposition, sensitivity
# --------------------------------------------------------------------------------------


def candidate_mechanism(
    population: list[dict[str, Any]], records_by_group: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    recovering = [entry for entry in population if entry["recoverable"]]
    best_prefixes = []
    prefixes_hist: dict[str, int] = {}
    failure_kinds: dict[str, int] = {}
    statuses: dict[str, int] = {}
    n_recovering_candidates = 0
    for entry in population:
        best = 0
        for record in records_by_group[entry["group_key"]]:
            if candidate_class(record) != "conclusive":
                continue
            if entry["recoverable"] and O.structured_recoverable(record):
                n_recovering_candidates += 1
                statuses[record["process_status"]] = statuses.get(record["process_status"], 0) + 1
                counts = O.verified_prefix_count(record)
                best = max(best, counts)
                key = str(min(counts, 10))
                prefixes_hist[key] = prefixes_hist.get(key, 0) + 1
                kind = record["failure_kind"]
                failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
        best_prefixes.append(best)
    return {
        "n_recovering_groups": len(recovering),
        "n_recovering_candidates": n_recovering_candidates,
        "recovering_candidate_statuses": statuses,
        "recovering_candidate_failure_kinds": failure_kinds,
        "verified_prefix_hist_recovering_candidates": dict(sorted(prefixes_hist.items())),
        "best_prefix_per_recovering_group": {
            "median": statistics.median(best_prefixes) if best_prefixes else None,
            "max": max(best_prefixes) if best_prefixes else None,
        },
        "n_groups_with_two_or_more_verified_prefix": sum(
            1 for value in best_prefixes if value >= 2
        ),
        "any_active_groups": sum(
            1
            for entry in population
            if any(
                candidate_class(record) == "conclusive" and O.any_active(record)
                for record in records_by_group[entry["group_key"]]
            )
        ),
    }


def token_credit_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    positions: dict[str, list[int]] = {"d1": [], "d2": [], "success": []}
    conflicts = blamed_mappable = blamed_total = 0
    span_not_in_response = 0
    for record in records:
        if candidate_class(record) != "conclusive":
            continue
        credit = record["token_credit"]
        for label, values in positions.items():
            values.extend(pair[0] for pair in credit[f"{label}_token_positions"])
        conflicts += credit["n_credit_conflicts"]
        span_not_in_response += credit["n_span_not_in_response"]
        if credit["blamed_mappable"] is not None:
            blamed_total += 1
            blamed_mappable += int(credit["blamed_mappable"])

    def summary(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": ordered[0],
            "p25": ordered[len(ordered) // 4],
            "median": statistics.median(ordered),
            "p75": ordered[3 * len(ordered) // 4],
            "max": ordered[-1],
            "share_first_256": sum(1 for value in ordered if value < 256) / len(ordered),
        }

    return {
        "d1_tokens": summary(positions["d1"]),
        "d2_tokens": summary(positions["d2"]),
        "success_tokens": summary(positions["success"]),
        "n_credit_conflicts": conflicts,
        "n_spans_not_in_response": span_not_in_response,
        "blamed_mappable_rate": (blamed_mappable / blamed_total) if blamed_total else None,
        "canonical_d1": S.D1_CANONICAL,
        "canonical_d2": S.D2_CANONICAL,
    }


def format_decomposition(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    failure_kinds: dict[str, int] = {}
    blame_kinds: dict[str, int] = {}
    for record in records:
        statuses[record["process_status"]] = statuses.get(record["process_status"], 0) + 1
        if record["process_status"] not in ("FORMAT_NO_CODE", "PROCESS_ORACLE_INFRA"):
            key = str(record["failure_kind"])
            failure_kinds[key] = failure_kinds.get(key, 0) + 1
            blame = str(record["blame_kind"])
            blame_kinds[blame] = blame_kinds.get(blame, 0) + 1
    n = len(records)
    return {
        "process_status": statuses,
        "process_status_share": {key: value / n for key, value in statuses.items()},
        "failure_kind": failure_kinds,
        "blame_kind": blame_kinds,
        "n_candidates": n,
    }


def sensitivity(
    population: list[dict[str, Any]], credit: dict[str, Any]
) -> dict[str, Any]:
    """d1/d2 value settings (owner §23): the label set is invariant by construction."""

    n_d1 = credit["d1_tokens"]["n"]
    n_d2 = credit["d2_tokens"]["n"]
    out = {}
    for name, (d1, d2) in S.SENSITIVITY_SETTINGS.items():
        out[name] = {
            "d1": d1,
            "d2": d2,
            "recovery_rate": rate(population),
            "credit_mass": {
                "d1_tokens": n_d1,
                "d2_tokens": n_d2,
                "total_phi": n_d1 * d1 + n_d2 * d2,
            },
            "labels_identical_to_canonical": True,
            "note": (
                "d1/d2 are credit values, not labels: structured_recoverable only tests "
                "mappability, so every classification is invariant across settings by construction"
            ),
        }
    return out


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------


def classify(
    *,
    e1: dict[str, Any],
    e2: dict[str, Any],
    g1: dict[str, Any],
    g2: dict[str, Any],
    g3: list[dict[str, Any]],
    population: list[dict[str, Any]],
    all_fail: list[dict[str, Any]],
) -> dict[str, Any]:
    """Frozen gate order: E1/E2 -> GO -> G1 -> G2 -> G3 (owner §22)."""

    n_quartiles_passed = sum(1 for entry in g3 if entry.get("passed"))
    failed_gates = []
    if not e1["passed"]:
        failed_gates.append("E1")
    if not e2["passed"]:
        failed_gates.append("E2")
    if not g1["passed"]:
        failed_gates.append("G1")
    if not all(entry["passed"] for entry in g2.values()):
        failed_gates.append("G2")
    if n_quartiles_passed < S.GATE_G3_QUARTILES_MIN:
        failed_gates.append("G3")

    seed_min_evaluable = 1.0
    for seed, entry in g2.items():
        total = sum(1 for item in all_fail if item["seed"] == seed)
        seed_min_evaluable = min(seed_min_evaluable, entry["n_groups"] / total if total else 1.0)
    censored_share = e2["censored_share"] or 0.0
    inconclusive_trigger = None
    if not e1["passed"]:
        inconclusive_trigger = "E1 below the frozen mapping floor"
    elif not e2["passed"]:
        inconclusive_trigger = "E2 below the frozen oracle-decision floor"
    elif censored_share > 0.05:
        inconclusive_trigger = f"censored share {censored_share:.4f} above 0.05"
    elif seed_min_evaluable < 0.5:
        inconclusive_trigger = f"a seed keeps only {seed_min_evaluable:.2f} evaluable all-fail groups"

    if inconclusive_trigger:
        classification = "INCONCLUSIVE_PROCESS_ORACLE"
    elif not failed_gates:
        classification = "PROCESS_SIGNAL_GO"
    elif not g1["passed"]:
        classification = "NO_PROCESS_RECOVERY"
    elif not all(entry["passed"] for entry in g2.values()):
        classification = "PROCESS_SIGNAL_SEED_UNSTABLE"
    elif n_quartiles_passed < S.GATE_G3_QUARTILES_MIN:
        classification = "PROCESS_SIGNAL_LENGTH_CONFOUNDED"
    else:
        classification = "INCONCLUSIVE_PROCESS_ORACLE"
    return {
        "classification": classification,
        "failed_gates": failed_gates,
        "inconclusive_trigger": inconclusive_trigger,
        "n_quartiles_passed": n_quartiles_passed,
        "seed_min_evaluable_share": seed_min_evaluable,
        "gate_order": ["E1", "E2", "GO", "G1", "G2", "G3"],
        "n_evaluable_all_fail_groups": len(population),
    }


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def build_report(inputs: dict[str, Any]) -> dict[str, Any]:
    records = inputs["records"]
    surface = inputs["surface"]
    freeze = inputs["freeze"]
    groups = group_records(records)
    records_by_group = groups

    e1 = compute_e1(records)
    e2 = compute_e2(records)
    all_fail = all_fail_groups(groups, surface)
    population = [entry for entry in all_fail if entry["status"] == "evaluable"]
    excluded = [entry for entry in all_fail if entry["status"] != "evaluable"]
    g1 = component_bootstrap(population)
    g1["threshold"] = S.GATE_G1_RATE_MIN
    g1["ci_lower_threshold"] = S.GATE_G1_CI_LOWER_MIN
    g1["passed"] = (
        (g1["point"] or 0.0) >= S.GATE_G1_RATE_MIN
        and (g1["ci_lower"] or 0.0) > S.GATE_G1_CI_LOWER_MIN
    )
    g2 = per_seed(population)
    g3 = quartiles(population)
    credit = token_credit_summary(records)
    classification = classify(
        e1=e1, e2=e2, g1=g1, g2=g2, g3=g3, population=population, all_fail=all_fail
    )
    candidate_level = {
        "n_code_all_fail_candidates": sum(
            1 for record in records if record["group_label"] == S.GROUP_ALL_FAIL
            and candidate_class(record) != "no_code"
        ),
        "n_recoverable_candidates": sum(
            1
            for record in records
            if record["group_label"] == S.GROUP_ALL_FAIL
            and candidate_class(record) == "conclusive"
            and O.structured_recoverable(record)
        ),
    }
    candidate_level["rate"] = (
        candidate_level["n_recoverable_candidates"] / candidate_level["n_code_all_fail_candidates"]
        if candidate_level["n_code_all_fail_candidates"]
        else None
    )
    return {
        "artifact": "V5_P001_results",
        "experiment": S.EXPERIMENT_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "offline process-reward recoverability audit over the frozen V1 surface",
        "provenance": {
            "code_stamp": freeze["code_stamp"],
            "labels": {
                "path": freeze["labels"]["path"],
                "sha256": freeze["labels"]["sha256"],
                "n_records": freeze["labels"]["n_records"],
            },
            "raw_manifest_sha256": freeze["raw"]["manifest_sha256"],
            "freeze_sha256": S.sha256_file(inputs["paths"].freeze),
            "validation_sha256": S.sha256_file(inputs["paths"].validation),
            "historical_surface_sha256": S.sha256_file(S.HISTORICAL_SURFACE),
            "oracle_endpoint": inputs["run_meta"]["oracle"]["endpoint"],
            "oracle_image_digest": inputs["run_meta"]["oracle"]["image_digest"],
            "tokenizer": inputs["run_meta"]["tokenizer_identity"],
            "fixture_set_sha256": inputs["validation"].get("fixture_set_sha256"),
        },
        "process_oracle": {
            "fixtures": inputs["validation"]["summary"],
            "re_derivation": inputs["validation"]["checks"],
            "infrastructure": freeze["infrastructure"],
        },
        "historical_surface": {
            "n_groups": surface["verification"]["n_groups"],
            "n_groups_primary": surface["verification"]["primary_groups"],
            "labels": surface["verification"]["labels"],
            "primary_labels": surface["verification"]["primary_labels"],
            "contaminated_groups": surface["verification"]["contaminated_groups"],
        },
        "gates": {
            "E1": e1,
            "E2": e2,
            "G1": g1,
            "G2": g2,
            "G3": {"quartiles": g3, "n_passed": classification["n_quartiles_passed"]},
        },
        "primary_recovery": {
            "unit": "primary all-fail group (existence of >=1 structured-recoverable candidate)",
            "n_primary_all_fail_groups": len(all_fail),
            "n_evaluable_groups": len(population),
            "n_excluded_groups_all_censored": len(excluded),
            "n_recoverable_groups": g1["n_recoverable"],
            "rate": g1["point"],
            "ci_lower": g1["ci_lower"],
            "ci_upper": g1["ci_upper"],
            "candidate_level_secondary": candidate_level,
            "censored_candidates_in_population": sum(entry["n_censored"] for entry in population),
            "no_code_candidates_in_population": sum(entry["n_no_code"] for entry in population),
        },
        "by_seed": g2,
        "length_robustness": {
            "sort_key": "group_max_tactics then group_key, equal-count bins",
            "quartiles": g3,
            "zero_code_groups": sum(1 for entry in population if entry["n_conclusive"] == 0),
        },
        "candidate_mechanism": candidate_mechanism(population, records_by_group),
        "token_credit": credit,
        "format_decomposition": format_decomposition(records),
        "sensitivity": sensitivity(population, credit),
        "FINAL_CLASSIFICATION": classification,
        "permitted_claim": (
            "Offline process-label recoverability on the frozen V1 surface only; no training, "
            "no generation, no capability or gradient claim."
        ),
        "limitations": [
            "single historical surface (three GRPO seeds of one sub-billion model)",
            "labels are single-run oracle outcomes; infra outcomes are censored, not failed",
            "group-level RecoveryRate is an existence predicate, not a per-candidate rate",
            "the d1/d2 credit surface was never applied to a model in V5-P001",
        ],
        "compute": {
            "new_model_generation": 0,
            "training": 0,
            "oracle_submissions": freeze["infrastructure"]["n_submissions"],
            "oracle_censored": freeze["infrastructure"]["n_infra_censored"],
            "container_recoveries": freeze["infrastructure"]["n_recoveries"],
        },
        "if_GO": (
            {
                "V5_R001_DRAFT_CREATED": "NO (drafting is a separate owner-authorized step)",
                "V5_R001_TRAINING_LAUNCHED": "NO",
            }
            if classification["classification"] == "PROCESS_SIGNAL_GO"
            else None
        ),
        "SEALED_RESERVE_TOUCHED": 0,
    }


def print_report(report: dict[str, Any]) -> None:
    primary = report["primary_recovery"]
    classification = report["FINAL_CLASSIFICATION"]["classification"]
    print("V5_P001_RESULT")
    print(f"  final_classification        {classification}")
    print(f"  failed_gates                {report['FINAL_CLASSIFICATION']['failed_gates']}")
    print(f"  E1 mapping                  {report['gates']['E1']['rate']} "
          f"(>= {S.GATE_E1_MIN})")
    print(f"  E2 oracle decision          {report['gates']['E2']['rate']} "
          f"(>= {S.GATE_E2_MIN})")
    print(f"  primary RecoveryRate        {primary['rate']} "
          f"CI [{primary['ci_lower']}, {primary['ci_upper']}] over {primary['n_evaluable_groups']} groups")
    for seed, entry in sorted(report["by_seed"].items()):
        print(f"  {seed} rate                  {entry['rate']} over {entry['n_groups']} groups")
    for quartile in report["length_robustness"]["quartiles"]:
        print(f"  Q{quartile['bin']} tactics {quartile.get('tactic_min')}-{quartile.get('tactic_max')} "
              f"rate {quartile.get('rate')} ({quartile['n_groups']} groups)")
    print(f"  candidate-level (secondary) {primary['candidate_level_secondary']}")
    credit = report["token_credit"]
    print(f"  credit tokens               d1={credit['d1_tokens']['n']} "
          f"d2={credit['d2_tokens']['n']} success={credit['success_tokens']['n']} "
          f"conflicts={credit['n_credit_conflicts']}")
    print(f"  statuses                    {report['format_decomposition']['process_status']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(S.PROCESS_RUN_DIR))
    parser.add_argument("--out", default=str(S.PROCESS_RESULTS))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    inputs = load_inputs(Path(args.run_dir))
    report = build_report(inputs)
    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"[v5p001] wrote {out_path}")
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalyzerError",
    "build_report",
    "candidate_class",
    "component_bootstrap",
    "compute_e1",
    "compute_e2",
    "load_inputs",
    "quartiles",
]
