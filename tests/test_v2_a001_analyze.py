"""Tests for the V2-A001 preregistered selection-rule analyzer.

Pure-function tests only: synthetic contrast statistics and tiny synthetic
artifacts (no GPU, no Lean server, no real evaluation data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v2_a001_analyze as analyzer


def stats(mean: float, ci_low: float, ci_high: float | None = None) -> dict:
    return {"mean_delta": mean, "ci_low": ci_low, "ci_high": mean if ci_high is None else ci_high}


def contrast_map(**overrides) -> dict:
    """All candidates at zero; override any label with a (mean, low, high) tuple."""

    base = {label: stats(0.0, 0.0, 0.0) for label in analyzer.CANDIDATES}
    for label, values in overrides.items():
        base[label] = stats(*values)
    return base


def anchor_map(**overrides) -> dict:
    """All candidates clearly ineligible (below -1 pp); override explicitly."""

    base = {label: stats(-0.05, -0.06, -0.04) for label in analyzer.CANDIDATES}
    for label, values in overrides.items():
        base[label] = stats(*values)
    return base


def test_r2_keeps_default_when_nothing_is_strictly_better():
    decision = analyzer.decide(contrast_map(), contrast_map())
    assert decision["rule"] == "R2"
    assert decision["selected"] == analyzer.DEFAULT
    assert decision["incumbent_degraded"] is False
    assert decision["no_candidate_established_gain"] is True


def test_r1_adopts_the_strictly_better_candidate():
    vs_default = contrast_map(step20=(0.02, 0.001, 0.04))
    decision = analyzer.decide(contrast_map(), vs_default)
    assert decision["rule"] == "R1"
    assert decision["selected"] == "step20"


def test_r1_requires_ci_low_strictly_positive():
    vs_default = contrast_map(step20=(0.02, 0.0, 0.04))
    decision = analyzer.decide(contrast_map(), vs_default)
    assert decision["rule"] == "R2"
    assert decision["selected"] == analyzer.DEFAULT


def test_r1_tie_break_smaller_step_then_label():
    vs_default = contrast_map(step10=(0.02, 0.005, 0.04), step20=(0.02, 0.005, 0.04))
    assert analyzer.decide(contrast_map(), vs_default)["selected"] == "step10"
    vs_default = contrast_map(step30=(0.02, 0.005, 0.04), seed2_step60=(0.02, 0.005, 0.04))
    assert analyzer.decide(contrast_map(), vs_default)["selected"] == "step30"
    vs_default = contrast_map(seed2_step60=(0.02, 0.005, 0.04), seed3_step60=(0.02, 0.005, 0.04))
    assert analyzer.decide(contrast_map(), vs_default)["selected"] == "seed2_step60"


def test_r3_degradation_override_takes_precedence_over_r1():
    vs_anchor = anchor_map(
        **{
            analyzer.DEFAULT: (-0.02, -0.03, -0.005),  # degraded: ci_low < 0
            "step20": (-0.02, -0.02, -0.01),  # strictly better than default, but below the -1 pp margin
            "step10": (-0.004, -0.005, -0.002),  # eligible
        }
    )
    vs_default = contrast_map(step20=(0.01, 0.002, 0.02))  # R1 alone would pick step20
    decision = analyzer.decide(vs_anchor, vs_default)
    assert decision["rule"] == "R3"
    assert decision["selected"] == "step10"
    assert decision["incumbent_degraded"] is True


def test_r3_eligibility_boundary_is_minus_one_point():
    # ci_low exactly at -0.01 (-1.0 pp) is eligible ...
    vs_anchor = anchor_map(
        **{analyzer.DEFAULT: (-0.03, -0.04, -0.02), "step10": (-0.01, -0.01, -0.005)}
    )
    decision = analyzer.decide(vs_anchor, contrast_map())
    assert decision["rule"] == "R3"
    assert decision["selected"] == "step10"

    # ... anything below the margin is not.
    vs_anchor = anchor_map(
        **{analyzer.DEFAULT: (-0.03, -0.04, -0.02), "step10": (-0.02, -0.0101, -0.005)}
    )
    decision = analyzer.decide(vs_anchor, contrast_map())
    assert decision["rule"] == "R3-least-bad"
    assert decision["selected"] == "step10"  # least-bad point estimate


def test_degraded_flag_requires_strictly_negative_ci_low():
    vs_anchor = contrast_map(**{analyzer.DEFAULT: (-0.001, 0.0, 0.002)})
    decision = analyzer.decide(vs_anchor, contrast_map())
    assert decision["incumbent_degraded"] is False
    assert decision["rule"] == "R2"


def test_no_candidate_established_gain_flag():
    vs_anchor = contrast_map(step30=(0.01, 0.001, 0.02))
    decision = analyzer.decide(vs_anchor, contrast_map())
    assert decision["no_candidate_established_gain"] is False


def test_decide_fails_closed_on_missing_contrasts():
    complete = contrast_map()
    incomplete = {k: v for k, v in complete.items() if k != "step20"}
    with pytest.raises(analyzer.AnalysisError):
        analyzer.decide(incomplete, complete)
    with pytest.raises(analyzer.AnalysisError):
        analyzer.decide(complete, incomplete)


def test_classify_outcome_matrix():
    got = analyzer.classify_outcome(stats(0.02, 0.001, 0.04))
    assert got["recommendation"] == "FREEZE_READY"
    assert got["candidate2_eligible"] is False

    got = analyzer.classify_outcome(stats(0.005, -0.01, 0.02))
    assert got["recommendation"] == "OWNER_DECISION_REQUIRED"
    assert got["reason"] == "null_no_detectable_gain"
    assert got["candidate2_eligible"] is True

    assert analyzer.classify_outcome(stats(-0.02, -0.04, -0.005))["reason"] == "degraded_vs_theta0"
    assert (
        analyzer.classify_outcome(stats(0.015, -0.005, 0.035))["reason"]
        == "borderline_subthreshold_positive"
    )


def test_classify_outcome_zero_ci_low_is_not_freeze_ready():
    got = analyzer.classify_outcome(stats(0.005, 0.0, 0.02))
    assert got["recommendation"] == "OWNER_DECISION_REQUIRED"
    assert got["candidate2_eligible"] is True


def test_theorem_counts_from_records_aggregates_and_validates():
    records = [
        {"theorem_index": index, "sample_index": sample, "verified": sample < (index % 3)}
        for index in range(3)
        for sample in range(4)
    ]
    assert analyzer.theorem_counts_from_records(records, 3, 4) == [0, 1, 2]
    with pytest.raises(analyzer.AnalysisError, match="coverage gap"):
        analyzer.theorem_counts_from_records(records[:-1], 3, 4)
    with pytest.raises(analyzer.AnalysisError, match="duplicate"):
        analyzer.theorem_counts_from_records(records + [records[0]], 3, 4)
    with pytest.raises(analyzer.AnalysisError, match="theorem_index"):
        analyzer.theorem_counts_from_records(
            [{"theorem_index": 3, "sample_index": 0, "verified": 1}], 3, 4
        )


def test_paired_deltas_and_contrast_units():
    counts_a = [2, 2, 0, 0]
    counts_b = [0, 0, 0, 0]
    assert analyzer.paired_deltas(counts_a, counts_b, 4) == [0.5, 0.5, 0.0, 0.0]
    got = analyzer.contrast(counts_a, counts_b, 4)
    assert got["n"] == 4
    assert got["mean_delta"] == pytest.approx(0.25)
    assert got["mean_delta_pp"] == pytest.approx(25.0)


def _write_artifact(tmp_path, label, *, fixed_set, n=2, samples=4, temperature=1.0, prefix="stmt"):
    records = [
        {
            "theorem_index": index,
            "sample_index": sample,
            "verified": int(sample == 0),
            "statement_id": f"{prefix}-{index}",
        }
        for index in range(n)
        for sample in range(samples)
    ]
    artifact = {
        "artifact_type": "p3c_checkpoint_eval",
        "checkpoint": label,
        "fixed_set": str(fixed_set),
        "selection_seed": analyzer.EXPECTED_SELECTION_SEED,
        "settings": {
            "theorems": n,
            "samples_per_theorem": samples,
            "temperature": temperature,
            "top_p": 1.0,
            "max_new_tokens": 4096,
            "seed_base": 20260917,
            "seed_group_size": 8,
        },
        "summary": {},
        "records": records,
        "created_at_utc": "2026-09-19T00:00:00+00:00",
        "global_step": 0,
        "model_dir": "models/weights/kimina_distill_0_6b",
    }
    path = tmp_path / f"v2_a001_{label}.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _shrink_expectations(monkeypatch, n=2, samples=4):
    monkeypatch.setattr(analyzer, "EXPECTED_N_THEOREMS", n)
    monkeypatch.setitem(analyzer.EXPECTED_SETTINGS, "theorems", n)
    monkeypatch.setitem(analyzer.EXPECTED_SETTINGS, "samples_per_theorem", samples)


def test_load_model_artifact_happy_path_and_fail_closed(tmp_path, monkeypatch):
    _shrink_expectations(monkeypatch)
    set_path = tmp_path / "set.json"
    theorems = [{"statement_id": f"stmt-{index}"} for index in range(2)]

    good = _write_artifact(tmp_path, "base", fixed_set=set_path)
    loaded = analyzer.load_model_artifact("base", good, theorems, set_path)
    assert loaded["counts"] == [1, 1]
    assert loaded["artifact_sha256"]

    bad_temp = _write_artifact(tmp_path, "step10", fixed_set=set_path, temperature=0.6)
    with pytest.raises(analyzer.AnalysisError, match="temperature"):
        analyzer.load_model_artifact("step10", bad_temp, theorems, set_path)

    bad_statement = _write_artifact(tmp_path, "step20", fixed_set=set_path, prefix="other")
    with pytest.raises(analyzer.AnalysisError, match="statement_id"):
        analyzer.load_model_artifact("step20", bad_statement, theorems, set_path)


def test_load_model_artifact_missing_file(tmp_path, monkeypatch):
    _shrink_expectations(monkeypatch)
    with pytest.raises(analyzer.AnalysisError, match="missing"):
        analyzer.load_model_artifact(
            "base", tmp_path / "v2_a001_base.json", [{"statement_id": "stmt-0"}], tmp_path / "set.json"
        )


def test_load_family_clusters_maps_and_counts(tmp_path):
    registry = {
        "components": [
            {"component_id": "fc-a", "member_statement_ids": ["s1", "s2", "s9"]},
            {"component_id": "fc-b", "member_statement_ids": ["s3"]},
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    set_theorems = [{"statement_id": sid} for sid in ["s3", "s1", "s2"]]
    clusters, info = analyzer.load_family_clusters(set_theorems, registry_path)
    assert clusters == [[1, 2], [0]]
    assert info["n_clusters"] == 2
    assert info["n_theorems"] == 3
    assert info["unmapped_theorems"] == 0
    assert info["registry_sha256"]


def test_load_family_clusters_counts_unmapped(tmp_path):
    registry = {"components": [{"component_id": "fc-a", "member_statement_ids": ["s1"]}]}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    _clusters, info = analyzer.load_family_clusters(
        [{"statement_id": "s1"}, {"statement_id": "s-unknown"}], registry_path
    )
    assert info["n_clusters"] == 1
    assert info["unmapped_theorems"] == 1


def test_load_family_clusters_missing_registry(tmp_path):
    with pytest.raises(analyzer.AnalysisError, match="missing"):
        analyzer.load_family_clusters([], tmp_path / "nope.json")


def test_family_cluster_contrast_pp_fields():
    counts_a = [4, 3, 2, 1]
    counts_b = [0, 0, 0, 0]
    clusters = [[0, 1], [2], [3]]
    contrast = analyzer.cluster_contrast(counts_a, counts_b, 4, clusters)
    assert contrast["n_clusters"] == 3
    assert contrast["n_theorems"] == 4
    assert contrast["mean_delta"] == pytest.approx((4 + 3 + 2 + 1) / 4 / 4)
    assert contrast["mean_delta_pp"] == pytest.approx(contrast["mean_delta"] * 100)
    assert contrast["ci_low_pp"] <= contrast["mean_delta_pp"] <= contrast["ci_high_pp"]


def test_real_registry_covers_the_frozen_selection_set():
    selection_set = json.loads((analyzer.ROOT / analyzer.SET_REL).read_text(encoding="utf-8"))
    _clusters, info = analyzer.load_family_clusters(
        selection_set["theorems"], analyzer.ROOT / analyzer.FAMILY_REGISTRY_REL
    )
    assert info["unmapped_theorems"] == 0
    assert info["n_theorems"] == analyzer.EXPECTED_N_THEOREMS
    assert info["n_clusters"] == 392
