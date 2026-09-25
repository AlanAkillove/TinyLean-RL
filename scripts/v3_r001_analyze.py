#!/usr/bin/env python3
"""V3-R001 analyzer -- score the frozen gate against the raw rollout. It decides nothing else.

Owner 2026-09-24 review sections 10-18. The analyzer reads exactly five things: the frozen
preregistration objects (via `v3_r001_spec.load_frozen`), the frozen sample, the frozen q_B2, the
frozen B1 scores and the formal raw rollout. It never retrains, never re-selects the block, never
re-orders the sample and never re-derives a threshold. Every statistic it reports comes from the
functions in `scripts/v3_r001_gate.py` -- the same code that produced the frozen artifact -- so a
number cannot change between the freeze and the analysis without a diff in that file.

The order of operations is load-bearing and is the order the owner fixed:

    1. every group must be finalized (an INCOMPLETE group aborts the run -- section 7);
    2. the outcome-D data guard runs BEFORE any gate is evaluated;
    3. the pooled gate (P1 exact test AND P2 enrichment+CI) on the FROZEN block;
    4. the co-primary within-synthetic gate (S1, S2);
    5. `pooled_gate()` maps the four cells to A/B/C/D mechanically -- there is no override;
    6. descriptives, per-source diagnostics and lambda diagnostics, none of which is a gate.

B2 > B1 is NOT a gate: the arms are reported side by side and a negative delta is reported as a
negative delta, never as "B2 failed".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v3_r001_spec as S
from v3_d001_lib import average_precision, brier, component_bootstrap, topk_enrichment
from v3_r001_gate import (
    ALPHA,
    BOOT_N_REP,
    BOOT_SEED,
    CI_LOWER_BOUND,
    ENRICHMENT_FLOOR,
    LAMBDA_GRID,
    MIN_ANALYZED_FRACTION,
    MIN_POSITIVES,
    attainable_alpha,
    critical_function,
    exact_p,
    mixture_policy,
    pooled_gate,
)

RESULTS_DEFAULT = "experiments/manifests/v3/V3-R001_results.json"
RAW_DEFAULT = "runs/v3_r001/rollout/v3_r001_raw_rollout.jsonl"
SUMMARY_DEFAULT = "runs/v3_r001/rollout/v3_r001_run_summary.json"
TOP_FRACTIONS = (0.10, 0.20, 0.30)


# --- labels --------------------------------------------------------------------------------------

def labels_from_groups(frozen: S.Frozen, groups: dict) -> list[dict]:
    """One row per frozen theorem: y in {0, 1, None}, with INCOMPLETE treated as a fatal input.

    A missing y is an excluded unit, never a zero (owner decision 5). An INCOMPLETE group means the
    rollout itself is unfinished, which is not a statistical finding but a broken input -- so it
    aborts instead of being quietly dropped (owner section 7).
    """
    rows = []
    for t in sorted(frozen.theorems, key=lambda r: r["rank_by_q_B2"]):
        sid = t["statement_id"]
        recs = groups.get(sid, [])
        if not recs:
            fin = {"group_status": S.GROUP_INCOMPLETE, "y": None, "n_candidates": 0,
                   "n_pos_score": None, "n_infra_candidates": None,
                   "reason": "no candidate rows at all for a frozen-sample theorem"}
        else:
            fin = S.finalize_group(recs)
        if fin["group_status"] == S.GROUP_INCOMPLETE:
            raise S.FrozenViolation(
                f"{sid} (rank {t['rank_by_q_B2']}) is INCOMPLETE: {fin['reason']}. The analyzer "
                "refuses to interpret a rollout whose groups are not all finalized -- re-run the "
                "group with the runner's --resume instead of analyzing around it.")
        rows.append({
            "statement_id": sid, "component_id": t["component_id"],
            "theorem_rank": t["rank_by_q_B2"],
            "formal_sample_rank": t["formal_sample_rank"], "source": t["source"],
            "in_frozen_block": bool(t["in_top20pct_block"]),
            "q_B2": float(t["q_B2_controller"]), "q_B1": float(t["q_B1_handcrafted"]),
            "y": fin["y"], "group_status": fin["group_status"],
            "n_pos_score": fin["n_pos_score"], "n_infra_candidates": fin["n_infra_candidates"],
            "n_candidates": fin["n_candidates"],
            "n_verified": sum(1 for r in recs if r["verified"]),
            "n_format_error": sum(1 for r in recs
                                  if r["verify_status"] == S.FORMAT_ERROR),
            "n_infra": sum(1 for r in recs if S.is_infra_candidate(r)),
            "generated_tokens_mean": (round(float(np.mean([r["generated_tokens"] for r in recs])), 1)
                                      if recs else None),
            "truncated": sum(1 for r in recs if r["truncated"]),
        })
    return rows


def check_raw_provenance(frozen: S.Frozen, groups: dict, summary: dict | None) -> list[str]:
    """The raw file must be the rollout this design predicted, from this model, at these settings."""
    problems: list[str] = []
    if summary is not None:
        # the four hashes the runner stamped before it generated anything, re-read here
        for key, expected, what in (
                ("frozen_settings_sha256", S.sha(S.FROZEN_SETTINGS), "the frozen settings block"),
                ("sample_sha256", frozen.formal["sample_sha256"], "the frozen sample"),
                ("block_sha256", frozen.formal["top20pct_block"]["sha256"], "the frozen block"),
                ("gate_sha256", frozen.file_hashes[S.GATE], "the frozen gate file"),
                ("formal_sample_sha256_on_disk", frozen.file_hashes[S.FORMAL_SAMPLE],
                 "the frozen sample file")):
            got = summary.get(key)
            if got is None:
                problems.append(f"the run summary carries no {key}")
            elif got != expected:
                problems.append(f"the run summary's {key}={str(got)[:16]}... is not {what}'s "
                                f"{expected[:16]}...")
    all_rows = [r for rows in groups.values() for r in rows]
    if not all_rows:
        problems.append("no candidate rows")
        return problems
    if {r["model_sha256"] for r in all_rows} != {frozen.theta0["weights_sha256"]}:
        problems.append("a candidate row was generated by a model other than the frozen theta0")
    if {r["experiment_id"] for r in all_rows} != {S.EXPERIMENT_ID}:
        problems.append("a candidate row carries another experiment id")
    frozen_rank = {t["statement_id"]: t["formal_sample_rank"] for t in frozen.theorems}
    bad_rank = sorted({(r["statement_id"], r.get("formal_sample_rank")) for r in all_rows
                       if r.get("formal_sample_rank") != frozen_rank.get(r["statement_id"])})
    if bad_rank:
        problems.append(f"{len(bad_rank)} candidate rows carry a formal_sample_rank that is not the "
                        f"frozen draw-order rank (first three: {bad_rank[:3]}); the generation seeds "
                        "of such a row cannot be the frozen schedule")
    bad_seed = [(r["statement_id"], r["sample_index"], r["seed"]) for r in all_rows
                if r.get("formal_sample_rank") == frozen_rank.get(r["statement_id"])
                and r["seed"] != S.candidate_seed(r["formal_sample_rank"], r["sample_index"])]
    if bad_seed:
        problems.append(f"{len(bad_seed)} candidate rows do not carry the frozen seed formula "
                        f"(first three: {bad_seed[:3]})")
    off = sorted({r["statement_id"] for r in all_rows}
                 - {t["statement_id"] for t in frozen.theorems})
    if off:
        problems.append(f"{len(off)} candidate rows are outside the frozen sample")
    sealed = {c["component_id"] for c in frozen.reserve["components"]}
    hit = sorted({r["component_id"] for r in all_rows} & sealed)
    if hit:
        problems.append(f"{len(hit)} candidate rows come from the SEALED reserve: {hit[:3]}")
    return problems


# --- gate cells ----------------------------------------------------------------------------------

def _enrichment(y: np.ndarray, in_block: np.ndarray) -> float:
    """(informative in the frozen block) / (informative in the analyzed set); NaN if undefined."""
    prevalence = y.mean()
    if prevalence <= 0 or in_block.sum() == 0:
        return float("nan")
    return float(y[in_block].mean() / prevalence)


def gate_cell(frozen: S.Frozen, rows: list[dict], stratum: str | None = None) -> dict:
    """One gate cell (pooled or within a source stratum), evaluated on the FROZEN block.

    Membership of the block is frozen and comes from `frozen_block_for`: within a stratum the ranking
    is the frozen q ranking of that stratum, so the source label never enters the selection (owner
    decision 9). Censoring can only remove members from a block that is already fixed; the block is
    never re-taken after the outcome (owner decision 13). `m_variant_if_the_block_were_recomputed`
    reports what the frozen gate artifact's alternative rule would have used, as a disclosure and
    nothing more -- the gate reads `m_frozen_block_analyzed`.

    Statistics are the frozen module's own functions: `critical_function` (the rejection boundary),
    `exact_p` (the conditional hypergeometric tail) and `attainable_alpha` (the honest size of the
    discrete rule at the design prevalence).
    """
    sub = [r for r in rows if stratum is None or r["source"] == stratum]
    analyzed = [r for r in sub if r["y"] is not None]
    block = set(S.frozen_block_for(frozen, stratum))
    n = len(analyzed)
    y = np.array([int(r["y"]) for r in analyzed], dtype=int)
    in_block = np.array([r["statement_id"] in block for r in analyzed], dtype=bool)
    k = int(y.sum())
    m = int(in_block.sum())
    x = int(y[in_block].sum())
    c = critical_function(n, m, ALPHA) if n and m else None
    boundary = c[k] if c is not None else math.inf
    pi = (frozen.gate["pooled_gate"]["P1"]["pi_design_anchor"] if stratum is None
          else frozen.gate["within_synthetic_gate"]["S1"]["pi_from_V1_synthetic_rate"])
    enrichment = _enrichment(y, in_block) if (n and m) else float("nan")
    cell = {
        "stratum": stratum or "pooled", "n_nominal": len(sub), "n_analyzed": n,
        "n_censored": len(sub) - n, "m_frozen_block_analyzed": m,
        "k_analyzed_positives": k, "x_block_positives": x,
        "analyzed_positives": k, "block_positives": x,
        "block_positive_rate": (round(x / m, 4) if m else None),
        "prevalence": (round(float(y.mean()), 4) if n else None),
        "pi_design_anchor": pi,
        "enrichment": (round(enrichment, 4) if math.isfinite(enrichment) else None),
        "identifiable": bool(c is not None and math.isfinite(boundary)),
        "rejection_boundary": (int(boundary) if math.isfinite(boundary) else None),
        "enrichment_ci_lower": None, "enrichment_ci_upper": None,
        "pass_P1": None, "pass_P2": None,
        "m_frozen_block_nominal": len(block),
        "m_variant_if_the_block_were_recomputed": S.block_size(n),
        "recomputation_note": ("the gate uses the FROZEN block membership (owner amendment A, "
                               "2026-09-25): N = analyzed, K = positives in the analyzed set, "
                               "m = analyzed members of the frozen block, x = positives among them. "
                               "The block is never re-taken, never topped up from labels and never "
                               "repaired for a censored theorem; m_variant_* is the size a recomputed "
                               "top-20% would have had and is a descriptive disclosure that never "
                               "enters the decision."),
    }
    if not cell["identifiable"]:
        cell["why_not"] = ("no rejection region exists at this (N_analyzed, m, k): this co-primary "
                           "is INCONCLUSIVE-BY-DATA, never a FAIL and never a silent fallback to the "
                           "pooled result (owner decision 9)")
        return cell
    lo, hi, _vals = component_bootstrap(
        [{"component_id": r["component_id"]} for r in analyzed], np.zeros(n),
        lambda idx: _enrichment(y[idx], in_block[idx]),
        n_rep=BOOT_N_REP, seed=BOOT_SEED, alpha=0.05)
    cell.update({
        "reject_if_at_least": int(boundary),
        "exact_p_one_sided": round(float(exact_p(n, m, k, x)), 6),
        "conditional_level_at_this_k": round(float(exact_p(n, m, k, int(boundary))), 5),
        "implied_enrichment_at_the_boundary": round((boundary / m) / (k / n), 4) if k else None,
        "attainable_alpha_at_design_pi": round(float(attainable_alpha(n, m, pi)), 4),
        "nominal_alpha": ALPHA,
        "pass_P1": bool(x >= int(boundary)),
        "enrichment_ci_lower": round(float(lo), 4), "enrichment_ci_upper": round(float(hi), 4),
        "pass_P2": bool(math.isfinite(enrichment) and enrichment >= ENRICHMENT_FLOOR
                        and lo > CI_LOWER_BOUND),
        "bootstrap": {"n_rep": BOOT_N_REP, "seed": BOOT_SEED, "unit": "family component",
                      "interval": "2.5 / 97.5 percentile", "ci_lower_bound_required": CI_LOWER_BOUND,
                      "point_estimate_floor": ENRICHMENT_FLOOR},
    })
    return cell


def gate_taxonomy_input(cell: dict) -> dict:
    """The exact shape `pooled_gate()` compares, with no None where it does arithmetic.

    A cell with no rejection region gets an unreachable boundary (one more than the block could
    possibly contain) so the frozen taxonomy rule reads it as a failing test instead of crashing, and
    an undefined enrichment becomes NaN so both comparisons are False. `identifiable` is what
    actually classifies the cell; the report itself keeps the honest nulls.
    """
    def num(v):
        return float(v) if isinstance(v, (int, float)) else float("nan")

    return {
        "block_positives": cell["block_positives"],
        "reject_if_at_least": (cell["rejection_boundary"] if cell["identifiable"]
                               else cell["m_frozen_block_analyzed"] + 1),
        "enrichment": num(cell["enrichment"]),
        "enrichment_ci_lower": num(cell["enrichment_ci_lower"]),
        "analyzed_positives": cell["analyzed_positives"],
        "n_analyzed": cell["n_analyzed"],
        "n_nominal": cell["n_nominal"],
        "identifiable": cell["identifiable"],
    }


# --- descriptives and diagnostics ----------------------------------------------------------------

def descriptives(rows: list[dict]) -> dict:
    """Prevalence and ranking quality. Comparative, never a gate (owner decision 10)."""
    analyzed = [r for r in rows if r["y"] is not None]
    y = np.array([int(r["y"]) for r in analyzed])
    q2 = np.array([r["q_B2"] for r in analyzed])
    q1 = np.array([r["q_B1"] for r in analyzed])
    out: dict = {"n_analyzed": len(analyzed), "n_excluded_censored": len(rows) - len(analyzed),
                 "positives": int(y.sum()), "prevalence": (round(float(y.mean()), 4)
                                                           if len(analyzed) else None)}
    if len(analyzed) < 2 or y.sum() == 0:
        out["note"] = ("too few analyzed positives for a ranking metric; AUPRC/enrichment/Brier are "
                       "undefined and are not reported as zero")
        return out
    for tag, p in (("B2_controller", q2), ("B1_handcrafted", q1)):
        out[f"auprc_{tag}"] = round(float(average_precision(y, p)), 4)
        out[f"brier_{tag}"] = round(float(brier(y, p)), 4)
        out[f"enrichment_{tag}"] = {f"top{int(f * 100)}pct":
                                    round(float(topk_enrichment(y, p, f)), 3) for f in TOP_FRACTIONS}
    out["auprc_prevalence_baseline"] = round(float(y.mean()), 4)
    out["delta_auprc_B2_minus_B1"] = round(out["auprc_B2_controller"] - out["auprc_B1_handcrafted"], 4)
    out["delta_enrichment_top20_B2_minus_B1"] = round(
        out["enrichment_B2_controller"]["top20pct"] - out["enrichment_B1_handcrafted"]["top20pct"], 3)
    out["gate_relevance"] = ("B2 > B1 is not a gate and a non-positive delta does not make B2 "
                             "'failed'; it is the D001 level-2 comparison, reported for continuity")
    return out


def source_diagnostics(rows: list[dict]) -> dict:
    """Per-source counts and, only where decidable, per-source ranking quality."""
    out = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        analyzed = [r for r in sub if r["y"] is not None]
        pos = sum(int(r["y"]) for r in analyzed)
        d = {"n_frozen": len(sub), "n_analyzed": len(analyzed), "n_censored": len(sub) - len(analyzed),
             "positives": pos,
             "prevalence": (round(pos / len(analyzed), 4) if analyzed else None)}
        if pos >= MIN_POSITIVES and len(analyzed) - pos >= 1:
            y = np.array([int(r["y"]) for r in analyzed])
            d["auprc_B2"] = round(float(average_precision(y, np.array([r["q_B2"] for r in analyzed]))), 4)
            d["auprc_B1"] = round(float(average_precision(y, np.array([r["q_B1"] for r in analyzed]))), 4)
            d["enrichment_B2_top20"] = round(float(topk_enrichment(
                y, np.array([r["q_B2"] for r in analyzed]), S.TOP_FRACTION)), 3)
            d["status"] = "COMPUTED"
        else:
            d["status"] = (f"DESCRIPTIVE_ONLY: fewer than {MIN_POSITIVES} positives in this source, "
                           "so a ranking metric here would be noise")
        out[src] = d
    return out


def mixture_diagnostics(frozen: S.Frozen, rows: list[dict]) -> dict:
    """Owner decision 12: lambda in {0.5, 0.8} as diagnostics only, with strict positivity enforced.

    `expected_IGR` is the quantity the frozen gate artifact left null: sum_i P_lambda(i) * y_i. It is
    computable only now that labels exist, and no lambda is selected as "best" by this report.

    Two policies are reported per lambda: the frozen one over all 128 theorems, whose weight hash
    MUST equal the gate artifact's pin (that equality is the check that this analyzer is looking at
    the same q ranking that was frozen), and the policy renormalized onto the analyzed subset, which
    is the only vector an IGR on observed labels can mean.
    """
    analyzed = [r for r in rows if r["y"] is not None]
    y = np.array([int(r["y"]) for r in analyzed], dtype=float)
    q_all = [r["q_B2"] for r in sorted(rows, key=lambda r: r["theorem_rank"])]
    q_sub = [r["q_B2"] for r in analyzed]
    values = frozen.gate["sampler_diagnostics"]["values"]
    out = {"not_a_gate": True,
           "formula": frozen.gate["sampler_diagnostics"]["formula"],
           "selection_rule": ("none. Both lambdas are reported; the launch choice is the owner's and "
                              "is never made on a number produced here"),
           "epsilon_zero": frozen.gate["sampler_diagnostics"]["epsilon_zero"]}
    for lam in LAMBDA_GRID:
        pol_all = mixture_policy(q_all, lam)
        pin = values[f"lambda={lam}"]["policy_weights_sha256"]
        if pol_all["min_probability"] <= 0.0:
            raise S.FrozenViolation(f"lambda={lam} put a frozen sampling probability at or below "
                                    "zero -- the epsilon=0 failure mode the owner abandoned")
        if pol_all["policy_weights_sha256"] != pin:
            raise S.FrozenViolation(
                f"lambda={lam}: the policy weights recomputed from the frozen q "
                f"({pol_all['policy_weights_sha256'][:16]}...) do not hash to the gate artifact's "
                f"pin ({pin[:16]}...) -- the analyzer is looking at a different ranking than froze")
        p_sub = np.array(policy_vector(q_sub, lam)) if q_sub else np.array([])
        if p_sub.size and not np.all(p_sub > 0.0):
            raise S.FrozenViolation(f"lambda={lam} produced a non-positive analyzed-subset probability")
        out[f"lambda={lam}"] = {
            "frozen_policy_over_all_128": {**{k: pol_all[k] for k in
                                              ("n", "min_probability", "max_probability",
                                               "effective_sample_size", "entropy_bits",
                                               "policy_weights_sha256")},
                                           "matches_the_frozen_gate_pin": True,
                                           "expected_IGR_note": ("stays null here: it needs a label "
                                                                 "for every one of the 128 theorems, "
                                                                 "and censored groups have none")},
            "renormalized_on_the_analyzed_subset": {
                "n": int(p_sub.size),
                "min_probability": (round(float(p_sub.min()), 8) if p_sub.size else None),
                "max_probability": (round(float(p_sub.max()), 8) if p_sub.size else None),
                "effective_sample_size": (round(float(1.0 / (p_sub ** 2).sum()), 3)
                                          if p_sub.size else None),
                "entropy_bits": (round(float(-(p_sub * np.log2(p_sub)).sum()), 4)
                                 if p_sub.size else None),
                "all_probabilities_strictly_positive": bool(np.all(p_sub > 0.0)) if p_sub.size else None,
                "expected_IGR": round(float((p_sub * y).sum()), 4) if p_sub.size else None,
                "unweighted_mean_informativeness": (round(float(y.mean()), 4) if y.size else None),
            },
            "positivity_invariant": frozen.gate["sampler_diagnostics"]["positivity_invariant"],
        }
    return out


def policy_vector(q: list[float], lam: float) -> list[float]:
    """The mixture weights themselves (gate's `mixture_policy` reports summary statistics only)."""
    a = np.asarray(q, dtype=float)
    return list((1.0 - lam) / a.size + lam * a / a.sum())


# --- main ---------------------------------------------------------------------------------------

def not_evaluated(why: str) -> dict:
    """A gate that was never reached. Explicit, so 'absent' can never read as 'failed'."""
    return {"status": "NOT_EVALUATED", "why": why}


def analyze(frozen: S.Frozen, groups: dict, summary: dict | None) -> dict:
    """The whole analyzer, in the owner's fixed order: labels, provenance, data guard, gates, taxonomy."""
    rows = labels_from_groups(frozen, groups)
    problems = check_raw_provenance(frozen, groups, summary)
    if problems:
        raise S.FrozenViolation("raw rollout provenance failed: " + "; ".join(problems))

    analyzed = [r for r in rows if r["y"] is not None]
    n_analyzed, positives = len(analyzed), sum(int(r["y"]) for r in analyzed)
    minimum = math.ceil(MIN_ANALYZED_FRACTION * S.N_NOMINAL)
    guard = {
        "rule": (f"N_analyzed >= {MIN_ANALYZED_FRACTION} * {S.N_NOMINAL} = {minimum} AND "
                 f"positives >= {MIN_POSITIVES} (owner decision 7-D), checked BEFORE any gate "
                 "is evaluated"),
        "n_analyzed": n_analyzed, "minimum_required": minimum,
        "positives": positives, "positives_minimum_required": MIN_POSITIVES,
        "pass": bool(n_analyzed >= minimum and positives >= MIN_POSITIVES),
    }
    if guard["pass"]:
        pooled = gate_cell(frozen, rows)
        synth = gate_cell(frozen, rows, stratum="synthetic")
        verdict = pooled_gate(gate_taxonomy_input(pooled), gate_taxonomy_input(synth))
    else:
        # the guard is not a gate, so a failing guard does not become a NO-GO: the design simply did
        # not produce enough decidable data to answer, and nothing downstream is computed at all.
        pooled = not_evaluated(guard["rule"])
        synth = not_evaluated(guard["rule"])
        verdict = {"outcome": "D_INCONCLUSIVE-BY-DATA",
                   "why": f"data guard failed before any gate: n_analyzed={n_analyzed} "
                          f"(needs {minimum}), positives={positives} (needs {MIN_POSITIVES})",
                   "P1_exact_test": None, "P2_enrichment_and_CI": None,
                   "S1_exact_test_within_synthetic": None,
                   "S2_enrichment_and_CI_within_synthetic": None, "r002_eligible": False}
    result = {
        "artifact_type": "v3_r001_results",
        "status": ("COMPUTED — the first outcome data on this design; the gate, the sample, the "
                   "block and this analyzer's code were frozen before it existed"),
        "authorized_by": "owner decisions 2026-09-24 and the V3-R001 preregistration",
        "question": ("Do frozen Kimina representations prospectively identify reward-informative "
                     "RLVR groups on previously unseen families?"),
        "prohibited_readings": [
            "not a claim that the controller improves RL, improves theorem proving or saves compute",
            "not a B2>B1 claim: the arm comparison is descriptive (owner decision 10)",
            "not evidence about the 93 sealed reserve components",
            "not a licence to re-draw, re-rank, re-fit or redefine the label rule (owner 25)",
        ],
        "frozen": {
            "prereg_commit": S.PREREG_COMMIT, "sample_key": S.SAMPLE_KEY,
            "sample_sha256": frozen.formal["sample_sha256"],
            "block_sha256": frozen.formal["top20pct_block"]["sha256"],
            "block_m": len(frozen.block_ids),
            "gate_sha256": frozen.file_hashes[S.GATE],
            "formal_sample_file_sha256": frozen.file_hashes[S.FORMAL_SAMPLE],
            "theta0_weights_sha256": frozen.theta0["weights_sha256"],
            "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
            "script_versions": S.script_version_hash("scripts/v3_r001_analyze.py",
                                                     "scripts/v3_r001_spec.py",
                                                     "scripts/v3_r001_gate.py",
                                                     "scripts/v3_r001_rollout.py"),
            "raw_artifact_sha256": (summary or {}).get("raw_artifact_sha256"),
        },
        "data_guard_outcome_D_first": guard,
        "pooled_gate": pooled,
        "within_synthetic_gate": synth,
        "outcome": verdict,
        "outcome_taxonomy_definition": frozen.gate["outcome_taxonomy"],
        "gate_check": {
            "P1": verdict.get("P1_exact_test"), "P2": verdict.get("P2_enrichment_and_CI"),
            "S1": verdict.get("S1_exact_test_within_synthetic"),
            "S2": verdict.get("S2_enrichment_and_CI_within_synthetic"),
            "rule": ("A needs P1 and P2 and (S1 and S2 within an identifiable synthetic stratum); "
                     "pooled pass with a synthetic fail is B; pooled fail is C; under the data guard, "
                     "or with an unidentifiable co-primary, D. Computed by pooled_gate() in the "
                     "frozen gate module -- this analyzer does not restate that rule."),
        },
        "descriptives": descriptives(rows),
        "source_diagnostics": source_diagnostics(rows),
        "mixture_diagnostics": mixture_diagnostics(frozen, rows),
        "per_theorem": rows,
        "group_status_counts": {st: sum(1 for r in rows if r["group_status"] == st)
                                for st in S.GROUP_STATUSES},
    }
    return result


def _brief(cell: dict, keys: tuple) -> dict:
    return ({k: cell.get(k) for k in keys} if "status" not in cell else cell)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--summary", default=SUMMARY_DEFAULT,
                    help="the runner's summary; pass 'none' to skip its hash cross-check")
    ap.add_argument("--out", default=RESULTS_DEFAULT)
    ap.add_argument("--check-only", action="store_true",
                    help="load, verify and report the outcome; write nothing")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.is_absolute():
        raw = ROOT / raw
    summary = None
    if args.summary.lower() != "none":
        path = Path(args.summary)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"FATAL: no run summary at {path}; pass --summary none only if you "
                             "accept that the runner's frozen-settings hash goes uncross-checked")
        summary = json.loads(path.read_text(encoding="utf-8"))

    frozen = S.load_frozen(verify_files=True)
    groups = S.read_groups(raw)
    result = analyze(frozen, groups, summary)
    print(json.dumps({
        "outcome": result["outcome"]["outcome"],
        "why": result["outcome"]["why"],
        "data_guard": result["data_guard_outcome_D_first"],
        "pooled": _brief(result["pooled_gate"],
                         ("n_analyzed", "m_frozen_block_analyzed", "k_analyzed_positives",
                          "x_block_positives", "rejection_boundary", "exact_p_one_sided",
                          "conditional_level_at_this_k", "attainable_alpha_at_design_pi",
                          "enrichment", "enrichment_ci_lower", "pass_P1", "pass_P2")),
        "within_synthetic": _brief(result["within_synthetic_gate"],
                                   ("n_analyzed", "identifiable", "rejection_boundary",
                                    "x_block_positives", "enrichment", "enrichment_ci_lower")),
        "group_status_counts": result["group_status_counts"],
        "prevalence": result["descriptives"].get("prevalence"),
    }, indent=2, ensure_ascii=False))
    if args.check_only:
        print("[check-only] nothing written")
        return 0
    out = Path(args.out)
    S.write_json_atomic(out if out.is_absolute() else ROOT / out, result)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
