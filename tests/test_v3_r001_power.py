"""Tests for the V3-R001 power analysis (§17 step 5).

The gate is being frozen on these numbers, so the arithmetic has to be checked against something
independent: the critical values are cross-checked against scipy's one-sided Fisher exact test, and
the closed-form power is cross-checked against a direct Monte-Carlo of the same experiment. Also
guards the infeasibility logic, because "enrichment 4 in the top half" is not an effect size, it is
an impossibility, and a power curve that returns 1.0 there would silently overstate the design.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v3_r001_power.py"
ARTIFACT = ROOT / "experiments" / "manifests" / "v3" / "V3-R001_power.json"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("v3_r001_power", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_critical_values_match_scipy_fisher_on_every_total(mod):
    """c[k] is the exact rejection boundary of the one-sided hypergeometric test: check both sides."""
    N, m = 60, 12
    c = np.asarray(mod.critical_values(N, m))
    for k in range(1, N + 1):
        lo, hi = max(0, k - (N - m)), min(k, m)
        for x in range(lo, hi + 1):
            table = [[x, m - x], [k - x, (N - m) - (k - x)]]
            if min(min(r) for r in table) < 0:
                continue
            p = fisher_exact(table, alternative="greater")[1]
            should_reject = p <= mod.ALPHA
            assert should_reject == (x >= c[k]), (k, x, p, c[k])


@pytest.mark.parametrize("N,m,pi,e", [(56, 11, 0.09, 2.5), (128, 26, 0.14, 3.5),
                                      (192, 38, 0.06, 2.0), (221, 111, 0.17, 1.5)])
def test_exact_power_matches_a_direct_monte_carlo(mod, N, m, pi, e):
    """Re-derive the experiment by simulation: the closed form must agree within MC error."""
    rng = np.random.default_rng(20260924)
    reps = 60_000
    r = mod.exact_power(N, m, pi, e)
    assert r["feasible"] is True
    y = np.concatenate([rng.random((reps, m)) < r["pi_sel"],
                        rng.random((reps, N - m)) < r["pi_rest"]], axis=1)
    k_tot = y.sum(axis=1)
    k_sel = y[:, :m].sum(axis=1)
    c = np.asarray(mod.critical_values(N, m))
    mc = float(np.mean(k_sel >= c[k_tot.astype(int)]))
    se = (mc * (1 - mc) / reps) ** 0.5
    assert abs(mc - r["power"]) < 4 * se + 1e-3, (mc, r["power"], se)


def test_null_hypothesis_reproduces_the_test_size(mod):
    """At enrichment 1.0 the 'power' is the type-I error, and it must sit at or below alpha."""
    for N, m, pi in [(56, 11, 0.09), (128, 26, 0.14), (221, 44, 0.17)]:
        size = mod.exact_power(N, m, pi, 1.0)["power"]
        assert 0.0 < size <= mod.ALPHA


def test_incoherent_effect_sizes_are_refused_not_overstated(mod):
    """E <= 1/f is arithmetic, not a design choice: beyond it the unselected block is negative."""
    r = mod.exact_power(N=100, m=50, pi=0.14, e_true=3.0)     # needs 42% inside a half sample
    assert r["power"] is None and r["feasible"] is False
    assert "negative informative rate" in r["why_infeasible"]
    # and the search refuses to report a power for an effect that cannot exist at that prevalence
    assert mod.min_detectable_e(100, 50, 0.06).startswith("unreachable")
    assert isinstance(mod.min_detectable_e(221, 44, 0.06), float)


def test_the_null_hypothesis_is_always_a_coherent_effect_size(mod):
    """E = 1 must never be classified impossible: that is what made the first version lie."""
    for N, m, pi in [(56, 11, 0.09), (128, 26, 0.14), (221, 111, 0.06), (86, 43, 0.17)]:
        r = mod.exact_power(N, m, pi, 1.0)
        assert r["feasible"] is True
        assert 0.0 < r["power"] <= mod.ALPHA


def test_power_increases_with_sample_size_at_a_fixed_design_point(mod):
    """Monotonicity is not guaranteed cell-by-cell (discrete level), but a 4x sample must help."""
    lo = mod.exact_power(56, 11, 0.14, 3.0)["power"]
    hi = mod.exact_power(221, 44, 0.14, 3.0)["power"]
    assert lo < hi


# -------------------------------------------------------------------------------------------
# the committed artifact
# -------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def art():
    if not ARTIFACT.exists():
        pytest.skip("power artifact not built yet")
    return json.loads(ARTIFACT.read_text())


def test_anchors_come_from_the_frozen_d001_artifact(mod, art):
    d001 = json.loads((ROOT / "experiments/manifests/v3/V3-D001_results.json").read_text())
    a = art["prevalence_anchors"]
    assert a["group_prevalence_V1"] == d001["dataset"]["prevalence"]
    assert a["d001_auprc"]["B2"] == d001["B2_block18_PRIMARY"]["auprc"]
    assert a["d001_auprc"]["delta"] == d001["B2_block18_PRIMARY"]["delta_auprc_vs_B1"]
    assert a["d001_topk_enrichment"]["B2"]["top20"] == d001["B2_block18_PRIMARY"]["top20_enrichment"]
    assert a["source_prevalence_V1"]["synthetic"] > a["source_prevalence_V1"]["human"]


def test_prevalence_prediction_uses_the_real_pool_source_mix(mod, art):
    """The predicted clean-pool prevalence must be recomputable from the committed pool artifact."""
    pool = json.loads((ROOT / "experiments/manifests/v3/V3-R001_family_clean_pool.json").read_text())
    per = art["prevalence_anchors"]["source_prevalence_V1"]
    for reading, pred in art["prevalence_anchors"]["predicted_clean_pool_prevalence"].items():
        mix = pool["pools_by_reading"][reading]["sources_of_one_per_component_eligible"]
        if not mix:
            assert pred == {"n_candidates": 0}
            continue
        n = sum(mix.values())
        assert pred["n_candidates"] == n
        assert pred["predicted_prevalence_from_V1_source_rates"] == pytest.approx(
            sum(mix[s] / n * per[s] for s in mix), abs=1e-4)


def test_the_unpowered_comparison_is_reported_as_unpowered(art):
    """§14 lets power, not wishfulness, decide whether B2 > B1 significance enters the gate."""
    d = art["delta_auprc_verdict"]
    assert all(v["would_exclude_zero_if_delta_equalled_D001_point_estimate"] is False
               for v in d["by_N"].values())
    assert "must NOT require B2 > B1 significance" in d["conclusion"]
    assert d["d001_reference"]["n_components"] == 433
    # the required delta grows as the sample shrinks: an explicit, checkable monotonicity
    half = {int(k): v["approx_ci_halfwidth_on_delta_auprc"] for k, v in d["by_N"].items()}
    assert half[56] > half[128] > half[221]
    assert half[56] > d["d001_reference"]["mean_delta_auprc"]


def test_grid_reports_its_own_realized_level(art):
    """Every cell must state the level it actually tests at, since it is below alpha at these N."""
    for N, blocks in art["power_grid_by_N_and_block"].items():
        for blk, cells in blocks.items():
            for pi, c in cells.items():
                size = c["realized_size_of_the_exact_test_under_H0"]
                assert 0.0 < size <= 0.05, (N, blk, pi, size)
                assert c["enrichment_ceiling_N_over_m"] == pytest.approx(
                    int(N) / c["selected_block_size"], abs=0.02)


def _md_tables(path: Path) -> list[dict]:
    """Each markdown table in `path` as {"header": [...], "rows": {first cell: remaining cells}}.

    The doc is the owner's decision record, so its numbers are re-read from the artifact here rather
    than trusted. A table's header is the selector, which also pins the doc's structure.
    """
    tables: list[dict] = []
    for block in re.findall(r"(?:^\|.*\|\s*$\n?)+", path.read_text(), re.MULTILINE):
        rows = [[c.strip().replace("**", "").replace("`", "")
                 for c in ln.strip("|").split("|")] for ln in block.strip().splitlines()]
        rows = [r for r in rows if not set("".join(r)) <= set("-: ")]   # drop the separator row
        tables.append({"header": rows[0], "rows": {r[0]: r[1:] for r in rows[1:]}})
    return tables


def _table(tables, header: list[str]) -> dict[str, list[str]]:
    hits = [t["rows"] for t in tables if t["header"] == header]
    assert len(hits) == 1, f"expected exactly one table with header {header}, found {len(hits)}"
    return hits[0]


def test_the_narrative_quotes_the_artifact_and_not_its_own_memory(art):
    """Every number in the four tables of docs/v3/V3-R001_power.md must be the artifact's number."""
    tables = _md_tables(ROOT / "docs" / "v3" / "V3-R001_power.md")
    grid = art["power_grid_by_N_and_block"]
    pis = ["0.06", "0.09", "0.14", "0.17"]

    mde = _table(tables, ["pool N"] + [f"π={p}" for p in pis])
    assert set(mde) == set(grid), "the doc's MDE table must carry exactly the pooled sample sizes"
    for N, row in mde.items():
        for cell, pi in zip(row, pis, strict=True):
            want = grid[N]["top20"][pi]["min_detectable_enrichment_at_80pct_power"]
            assert cell == f"{want:.2f}", (N, pi, cell, want)

    verdicts = _table(tables, ["reading", "expected positives in the selected block",
                               "power at E=3.47", "N needed for 80% power", "verdict"])
    per = art["design_verdicts_by_pool_reading"]["by_reading"]
    for label, name in [("owner_literal_wider_v1", "owner_literal_wider_v1"),
                        ("owner_literal", "owner_literal"), ("consumed_only", "consumed_only")]:
        key = next(k for k in verdicts if k.startswith(label + " ("))
        row, v = verdicts[key], per[name]
        _, cap, prev = re.fullmatch(r"(\S+) \((\d+), π ([\d.]+)\)", key).groups()
        assert int(cap) == v["pool_capacity"]
        assert float(prev) == pytest.approx(v["predicted_prevalence"], abs=1e-3)
        assert float(row[0]) == pytest.approx(v["expected_positives_in_block"], abs=5e-3)
        assert float(row[1].lstrip("~")) == pytest.approx(
            v["power_at_D001_observed_top20_enrichment"], abs=5e-3)
        assert int(row[2]) == v["N_needed_for_80pct_power_at_that_effect"]
        assert row[3].startswith("not usable") == (not v["usable"]), row

    halves = _table(tables, ["N", "approximate ΔAUPRC half-width",
                             "would D001's point delta clear zero?"])
    assert set(halves) == set(art["delta_auprc_verdict"]["by_N"])
    for N, row in halves.items():
        v = art["delta_auprc_verdict"]["by_N"][N]
        assert float(row[0]) == pytest.approx(v["approx_ci_halfwidth_on_delta_auprc"], abs=5e-4)
        assert row[1] == ("yes" if v["would_exclude_zero_if_delta_equalled_D001_point_estimate"]
                          else "no")

    anchors = _table(tables, ["quantity", "value", "source"])
    a = art["prevalence_anchors"]

    def _row(prefix: str) -> list[str]:
        return anchors[next(k for k in anchors if k.startswith(prefix))]

    def _num(prefix: str, col: int = 0) -> float:
        return float(re.match(r"[+-]?\d+\.?\d*", _row(prefix)[col]).group())

    assert _num("group prevalence") == pytest.approx(a["group_prevalence_V1"], abs=5e-5)
    assert _num("infra-censoring rate") == pytest.approx(a["infra_censoring_rate"], abs=5e-5)
    assert _num("theorem-level") == pytest.approx(
        float(re.search(r"[\d.]+", a["theorem_level_rate_note"]).group()), abs=5e-5)
    assert [float(x) for x in _row("B2 top10")[0].split(" / ")] == [
        pytest.approx(a["d001_topk_enrichment"]["B2"][k], abs=5e-4)
        for k in ("top10", "top20", "top30")]
    assert [float(x) for x in _row("AUPRC B2")[0].split(" / ")][:2] == [
        pytest.approx(a["d001_auprc"]["B2"], abs=5e-5), pytest.approx(a["d001_auprc"]["B1"], abs=5e-5)]

    mixes = _table(tables, ["reading", "pool capacity", "source mix (auto / human / synth)",
                            "predicted prevalence"])
    pred = a["predicted_clean_pool_prevalence"]
    assert set(mixes) == set(pred) - {"owner_literal_all_reservations_honored",
                                      "registry_labels_maximal"}
    for reading, row in mixes.items():
        assert int(row[0]) == pred[reading]["n_candidates"]
        assert row[1] == " / ".join(str(pred[reading]["source_mix"][s])
                                    for s in ("autoformalizer", "human", "synthetic"))
        assert float(row[2]) == pytest.approx(
            pred[reading]["predicted_prevalence_from_V1_source_rates"], abs=1e-3)


def test_the_recommended_design_is_the_one_the_grid_supports(art):
    """The doc recommends N=128 from consumed_only; pin the three numbers that recommendation rests on."""
    g = art["power_grid_by_N_and_block"]["128"]["top20"]
    assert g["0.14"]["min_detectable_enrichment_at_80pct_power"] == pytest.approx(2.3264, abs=1e-3)
    assert g["0.14"]["realized_size_of_the_exact_test_under_H0"] == pytest.approx(0.0274, abs=1e-3)
    assert g["0.14"]["expected_analyzable_N_after_censoring"] == pytest.approx(122.0, abs=0.1)
    assert art["design_verdicts_by_pool_reading"]["by_reading"]["owner_literal_wider_v1"]["usable"] is False
    sizes = [c["realized_size_of_the_exact_test_under_H0"]
             for blocks in art["power_grid_by_N_and_block"].values()
             for cells in blocks.values() for c in cells.values()]
    assert min(sizes) > 0.004 and max(sizes) < 0.034          # the doc quotes this range


def test_the_registry_entry_points_at_real_files_and_agrees(art):
    """registry.yaml is the index the owner reads first; it must resolve and must not disagree."""
    reg = yaml.safe_load((ROOT / "experiments/manifests/v3/registry.yaml").read_text())
    e = reg["prepreregistration_audits"]["V3-R001-power-analysis"]
    for key in ["script", "tests", "result", "narrative"]:
        assert (ROOT / e[key]).exists(), key
    assert (ROOT / e["result"]).samefile(ARTIFACT)
    per = art["design_verdicts_by_pool_reading"]["by_reading"]
    assert per["owner_literal_wider_v1"]["usable"] is False
    assert "MUST NOT require B2 > B1 significance" in e["delta_auprc_verdict"]
    assert "consumed_only" in e["recommendation"]
    assert "usable: false" in e["finding"]


def test_status_and_prohibitions(art):
    assert "no rollout" in art["status"]
    assert any("no rollout" in p for p in art["prohibitions_respected"])
    assert any("no frozen artifact modified" in p for p in art["prohibitions_respected"])
    # the gate recommendation must name the thing §14 asked about
    assert art["gate_recommendation"]["recommended_shape"]["does_not_require"].startswith(
        "B2 > B1 significance")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
