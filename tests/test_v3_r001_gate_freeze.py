"""V3-R001 §17 step 8 — the frozen allocation, the gate and the preregistration must agree.

These tests are the mechanical half of "preregistered": they pin that the formal sample is the one
the power calculation used, that the 128/93 partition is exact and sealed, that the boundary the
document quotes is the boundary the frozen code computes, that the document's numbers are the
artifact's numbers, and that the outcome taxonomy cannot be argued about later.

CPU-only. No rollout, no GPU, no verifier, no label.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_r001_gate as G
import v3_r001_prescore as P

FORMAL_REL = "experiments/manifests/v3/v3_r001_formal_sample.json"
RESERVE_REL = "experiments/manifests/v3/v3_final_holdout_reserve.json"
GATE_REL = "experiments/manifests/v3/V3-R001_gate.json"
POOL_REL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
PRED_REL = "experiments/manifests/v3/V3-R001_predictions.json"
DOC_REL = "docs/v3/V3-R001_preregistration.md"
TRACK_C_SENTENCE = (
    "V2 Track C reserved families were released by owner for V3 because Track C was never executed "
    "and no outcomes from those families were observed. This is a governance change in reservation "
    "status, not a reuse of previously evaluated data.")


def read(rel: str):
    return json.loads((ROOT / rel).read_text())


def _doc():
    return (ROOT / DOC_REL).read_text()


def _doc_flat():
    """The document with its line structure removed: a sentence may be wrapped or blockquoted in
    markdown, and that must not be a reason for a hash or a frozen number to read as missing."""
    return " ".join(line.lstrip("> ").strip() for line in _doc().splitlines())


def _draw_order(formal, pred):
    """The manifest lists theorems by q rank; the frozen hashes are over the DRAW order."""
    order = {c: i for i, c in
             enumerate(pred["prospective_samples"][formal["predictions_sample_key"]]
                       ["component_ids"])}
    return sorted(formal["theorems"], key=lambda t: order[t["component_id"]])


@pytest.fixture(scope="module")
def formal():
    return read(FORMAL_REL)


@pytest.fixture(scope="module")
def reserve():
    return read(RESERVE_REL)


@pytest.fixture(scope="module")
def gate():
    return read(GATE_REL)


@pytest.fixture(scope="module")
def pool():
    return read(POOL_REL)


@pytest.fixture(scope="module")
def pred():
    return read(PRED_REL)


# ---------------------------------------------------------------- the formal sample is the §14 sample
def test_formal_sample_is_exactly_the_power_calculation_sample(formal, pred):
    """The load-bearing check: the 128 were not picked again, they are the ones power was computed on."""
    key = formal["predictions_sample_key"]
    frozen = pred["prospective_samples"][key]
    pool_comps = sorted(frozen["component_ids"])
    recomputed = P.draw_components(pool_comps, G.N_NOMINAL, seed=P.DRAW_SEED)
    assert recomputed == list(frozen["component_ids"])
    assert G.sha({"reading": "consumed_only", "seed": P.DRAW_SEED, "components": recomputed}) == \
        frozen["sample_sha256"] == formal["sample_sha256"]
    ident = formal["identity_with_the_power_sample"]
    assert ident["recomputed_sample_sha256"] == ident["committed_sample_sha256"]
    assert ident["recomputed_component_order_matches"] is True
    for arm in ("q_B2", "q_B1"):
        assert ident[f"{arm}_vector_sha256"] == ident[f"committed_{arm}_vector_sha256"]
    # and the vectors themselves, recomputed from the manifest, hash to the committed values
    drawn = _draw_order(formal, pred)
    assert G.sha([t["q_B2_controller"] for t in drawn]) == frozen["q_B2_vector_sha256"]
    assert G.sha([t["q_B1_handcrafted"] for t in drawn]) == frozen["q_B1_vector_sha256"]
    assert len({t["statement_id"] for t in formal["theorems"]}) == G.N_NOMINAL


def test_the_two_freeze_scripts_hash_identically():
    """A frozen object must not acquire a different commitment depending on which script hashed it."""
    payload = {"reading": "consumed_only", "seed": 20260924, "components": ["fc-a", "fc-b"]}
    assert G.sha(payload) == P.sha(payload) == hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def test_no_candidate_sample_hash_was_invented(formal, pred):
    """Every hash quoted in the formal manifest exists in the committed predictions artifact."""
    s = formal["sample_sha256"]
    assert s in {v["sample_sha256"] for v in pred["prospective_samples"].values()}
    assert formal["coverage"]["n_missing_q"] == 0


# --------------------------------------------------------------------- the 128 / 93 partition
def test_partition_is_exact_disjoint_and_covering(formal, reserve, pool):
    pool_comps = sorted(pool["pools_by_reading"]["consumed_only"]["candidate_component_ids"])
    fc = [t["component_id"] for t in formal["theorems"]]
    rc = [c["component_id"] for c in reserve["components"]]
    assert len(pool_comps) == 221
    assert sorted(fc + rc) == pool_comps
    assert not set(fc) & set(rc)
    fs = {t["statement_id"] for t in formal["theorems"]}
    rs = {c["statement_id"] for c in reserve["components"]}
    assert len(fs) == 128 and len(rs) == 93 and not fs & rs
    # the manifest's own claims are true, not just present
    drawn = _draw_order(formal, pool and read(PRED_REL))
    assert formal["membership_hashes"]["component_ids_sha256"] == G.sha(
        [t["component_id"] for t in drawn])
    assert formal["membership_hashes"]["statement_ids_sha256"] == G.sha(
        [t["statement_id"] for t in drawn])
    assert reserve["membership_hashes"]["component_ids_sha256"] == G.sha(rc)
    assert (formal["membership_hashes"]["pool_hash_of_this_reading"]
            == reserve["membership_hashes"]["pool_hash_of_this_reading"]
            == pool["pools_by_reading"]["consumed_only"]["pool_hash"])


def test_reserve_membership_is_the_complement_of_the_draw_not_a_q_based_choice(formal, reserve):
    """Owner decision 2: membership may not have been influenced by a q value."""
    drawn = {t["component_id"] for t in formal["theorems"]}
    assert all(c["component_id"] not in drawn for c in reserve["components"])
    assert "MINUS the frozen N=128 formal sample" in reserve["membership_rule"]["how"]
    assert reserve["membership_rule"]["formal_sample_sha256"] == formal["sample_sha256"]


def test_reserve_is_sealed_with_the_owner_s_purpose(reserve):
    assert reserve["status"] == "SEALED"
    assert reserve["purpose"] == "future final family-clean capability evaluation"
    forbidden = " ".join(reserve["seal"]["forbidden_uses"]).lower()
    for use in ("rollout", "fitting", "sampler", "power", "analysis", "hyperparameter"):
        assert use in forbidden
    assert "q value" in forbidden
    assert any("use, not existence" in v for v in reserve["seal"].values()
               if isinstance(v, str)) or "binds USE, not existence" in json.dumps(reserve)


def test_governance_carries_the_owner_s_track_c_wording_verbatim(formal):
    gov = formal["governance"]
    assert gov["owner_release_of_track_c"] == TRACK_C_SENTENCE
    assert TRACK_C_SENTENCE in _doc_flat()
    assert gov["contamination_definition"] == "consumed_only"
    assert gov["allocation"] == {"r001_components": 128, "final_reserve_components": 93,
                                 "reserve_manifest": RESERVE_REL}
    assert gov["v2_history_untouched"]


# --------------------------------------------------------------------------- the frozen sample
def test_one_theorem_per_component_and_every_theorem_scored(formal):
    t = formal["theorems"]
    assert len(t) == 128 and len({x["component_id"] for x in t}) == 128
    assert formal["coverage"]["any_historical_label"] is False
    for x in t:
        for k in ("q_B2_controller", "q_B1_handcrafted", "q_B2_fold_ensemble", "q_B1_fold_ensemble"):
            assert 0.0 <= x[k] <= 1.0
        assert x["has_historical_label"] is False
        assert x["step_norm_used"] == 0.0            # owner §6: pre-update / initial policy


def test_top20_block_is_a_frozen_function_of_frozen_q(formal):
    m = G.block_size(G.N_NOMINAL)
    assert m == 26 == formal["top20pct_block"]["size"]
    order = sorted(formal["theorems"], key=lambda x: (-x["q_B2_controller"], x["statement_id"]))
    assert [x["statement_id"] for x in order[:m]] == formal["top20pct_block"]["statement_ids"]
    assert [x["rank_by_q_B2"] for x in order] == list(range(1, 129))
    assert sum(x["in_top20pct_block"] for x in formal["theorems"]) == m
    assert formal["top20pct_block"]["sha256"] == G.sha(formal["top20pct_block"]["statement_ids"])


def test_every_candidate_prompt_fits_the_context_the_rollout_will_use(formal):
    ctx = formal["context_check"]
    worst = max(t["prompt_token_count"] for t in formal["theorems"])
    assert worst == ctx["max_prompt_token_count"]
    assert ctx["max_prompt_token_count"] + ctx["max_response_tokens"] <= ctx["max_model_len"]
    assert ctx["fits"] is True


def test_source_distribution_of_the_sample_is_recorded(formal, pred):
    mix = {}
    for t in formal["theorems"]:
        mix[t["source"]] = mix.get(t["source"], 0) + 1
    assert dict(sorted(mix.items())) == formal["source_distribution"]
    assert mix == pred["prospective_samples"][formal["predictions_sample_key"]]["source_mix"]


# ------------------------------------------------------------------------- P1: exactness of P1
@pytest.mark.parametrize("k", list(range(2, 41)))
def test_p1_boundary_and_the_exact_p_agree(k):
    """The rule is 'p <= 0.05'; the table is 'x >= c'. They must be the same rule."""
    N, m = G.N_NOMINAL, G.block_size(G.N_NOMINAL)
    c = G.critical_function(N, m)[k]
    if c == float("inf"):
        pytest.skip(f"no rejection region at k={k}")
    assert G.exact_p(N, m, k, int(c)) <= G.ALPHA
    assert G.exact_p(N, m, k, int(c) - 1) > G.ALPHA


def test_the_boundary_function_is_monotone_where_it_matters():
    N, m = G.N_NOMINAL, G.block_size(G.N_NOMINAL)
    c = G.critical_function(N, m)
    finite = [(k, c[k]) for k in range(2, 116) if c[k] != float("inf")]
    assert all(finite[i][1] <= finite[i + 1][1] for i in range(len(finite) - 1))


def test_attainable_alpha_is_reported_and_below_the_nominal_level(gate):
    p1 = gate["pooled_gate"]["P1"]
    a = p1["attainable_alpha_unconditional_at_pi"]
    assert 0.0 < a < p1["nominal_alpha"]
    # recomputation from the frozen rule, not a copied number
    assert a == pytest.approx(G.attainable_alpha(128, 26, p1["pi_design_anchor"]), abs=1e-4)
    s1 = gate["within_synthetic_gate"]["S1"]
    assert s1["attainable_alpha_unconditional_at_pi"] < p1["nominal_alpha"]


def test_the_gate_artifact_is_reproducible_from_the_manifests(gate, formal):
    src_pi = read("experiments/manifests/v3/V3-R001_power.json")[
        "prevalence_anchors"]["source_prevalence_V1"]
    exp = G.expected_positives_by_source(formal["theorems"], {k: float(v) for k, v in src_pi.items()})
    assert exp["predicted_prevalence_pi"] == gate["pooled_gate"]["P1"]["pi_design_anchor"]
    assert exp["source_counts"] == formal["source_distribution"]
    synth = [t for t in formal["theorems"] if t["source"] == "synthetic"]
    assert len(synth) == gate["within_synthetic_gate"]["S1"]["n"]
    assert G.block_size(len(synth)) == gate["within_synthetic_gate"]["S1"]["m_top20"]
    assert gate["pooled_gate"]["P1"]["reject_if_block_positives_at_least"] == 7
    assert gate["within_synthetic_gate"]["S1"]["reject_if_block_positives_at_least"] == 6


def test_design_only_samples_are_every_other_candidate(pred, gate, formal):
    never = gate["candidate_samples_never_to_analyze"]
    assert set(never) == set(pred["prospective_samples"]) - {formal["predictions_sample_key"]}
    assert len(never) == 9
    for key, v in never.items():
        assert v["status"] == "DESIGN-ONLY / NEVER ANALYZE"
        assert v["sample_sha256"] == pred["prospective_samples"][key]["sample_sha256"]
    assert formal["predictions_sample_key"] not in never


# ------------------------------------------------------------------------- the taxonomy is code
def _cells(*, pooled_x, pooled_e, pooled_ci, synth_x, synth_e, synth_ci, n_analyzed=128,
           positives=18, identifiable=True):
    pooled = {"block_positives": pooled_x, "reject_if_at_least": 7, "enrichment": pooled_e,
              "enrichment_ci_lower": pooled_ci, "n_analyzed": n_analyzed, "n_nominal": 128,
              "analyzed_positives": positives}
    synth = {"block_positives": synth_x, "reject_if_at_least": 6, "enrichment": synth_e,
             "enrichment_ci_lower": synth_ci, "identifiable": identifiable}
    return pooled, synth


def _c(**kw):
    return kw


@pytest.mark.parametrize("kwargs,expected", [
    (_c(pooled_x=7, pooled_e=2.0, pooled_ci=1.2, synth_x=6, synth_e=2.2, synth_ci=1.1),
     "A_GO-SEMANTIC"),
    (_c(pooled_x=7, pooled_e=2.0, pooled_ci=1.2, synth_x=5, synth_e=1.2, synth_ci=0.8),
     "B_SOURCE-DRIVEN-ONLY"),
    (_c(pooled_x=7, pooled_e=1.4, pooled_ci=1.2, synth_x=6, synth_e=2.2, synth_ci=1.1),
     "C_NO-GO"),                     # P2 fails on the point estimate, so the pooled gate fails
    (_c(pooled_x=7, pooled_e=2.0, pooled_ci=0.95, synth_x=6, synth_e=2.2, synth_ci=1.1),
     "C_NO-GO"),                     # and on the interval alone
    (_c(pooled_x=6, pooled_e=3.0, pooled_ci=1.5, synth_x=6, synth_e=2.0, synth_ci=1.1),
     "C_NO-GO"),
    (_c(pooled_x=9, pooled_e=4.0, pooled_ci=2.0, synth_x=0, synth_e=0.0, synth_ci=0.0,
        positives=4), "D_INCONCLUSIVE-BY-DATA"),
    (_c(pooled_x=9, pooled_e=4.0, pooled_ci=2.0, synth_x=0, synth_e=0.0, synth_ci=0.0,
        n_analyzed=100), "D_INCONCLUSIVE-BY-DATA"),
    (_c(pooled_x=7, pooled_e=2.0, pooled_ci=1.2, synth_x=0, synth_e=0.0, synth_ci=0.0,
        identifiable=False), "D_INCONCLUSIVE-BY-DATA"),
])
def test_outcome_taxonomy_resolves_without_discretion(kwargs, expected):
    """Which of these is right is arithmetic; the point is that it is not a judgement call later."""
    pooled, synth = _cells(**kwargs)
    out = G.pooled_gate(pooled, synth)
    assert out["outcome"] == expected
    assert out["r002_eligible"] == (expected == "A_GO-SEMANTIC")


def test_only_outcome_A_opens_the_r002_door(gate):
    assert gate["outcome_taxonomy"]["A_GO-SEMANTIC"].count("V3-R002") == 1
    assert "NO expensive RL" not in gate["outcome_taxonomy"]["A_GO-SEMANTIC"]
    for k in ("B_SOURCE-DRIVEN-ONLY", "C_NO-GO", "D_INCONCLUSIVE-BY-DATA"):
        assert k in gate["outcome_taxonomy"]
    assert "NOT a NO-GO" in gate["outcome_taxonomy"]["D_INCONCLUSIVE-BY-DATA"]
    assert "switching to another candidate sample" in json.dumps(
        gate["inconclusive_by_data"]["never_permitted"])


def test_enrichment_ci_is_a_gate_component_not_a_decoration():
    """P2 is ANDed: a significant but tiny enrichment must not pass."""
    pooled, synth = _cells(pooled_x=9, pooled_e=1.6, pooled_ci=0.9, synth_x=6, synth_e=2.0,
                           synth_ci=1.1)
    out = G.pooled_gate(pooled, synth)
    assert out["P1_exact_test"] is True and out["P2_enrichment_and_CI"] is False
    assert out["outcome"] == "C_NO-GO"


# ------------------------------------------------------------------------- diagnostics, epsilon
def test_mixture_policy_is_strictly_positive_and_normalized(formal, gate):
    q = [t["q_B2_controller"] for t in formal["theorems"]]
    for lam, cell in gate["sampler_diagnostics"]["values"].items():
        recomputed = G.mixture_policy(q, cell["lambda"])
        assert recomputed["min_probability"] > 0.0, lam
        assert recomputed["effective_sample_size"] <= len(q)
        assert recomputed["policy_weights_sha256"] == cell["policy_weights_sha256"]
        assert recomputed["expected_IGR"] is None       # no outcome exists yet
    assert gate["sampler_diagnostics"]["epsilon_zero"] == "PERMANENTLY ABANDONED (owner directive §13)"


def test_an_epsilon_zero_sampler_cannot_pass_this_code():
    """The bug the owner retired: a policy that zeroes an eligible theorem must raise."""
    with pytest.raises(SystemExit):
        G.mixture_policy([0.5, 0.0, 0.2], lam=1.0)     # lambda=1 and a zero q => P(i)=0
    with pytest.raises(SystemExit):
        G.mixture_policy([0.0, 0.0, 0.0], lam=0.8)     # degenerate q: no policy at all


def test_b2_beats_b1_is_recorded_as_not_a_gate(gate):
    ng = gate["not_a_gate"]["b2_vs_b1_significance"]
    assert ng["status"] == "NOT A GATE"
    assert "AUPRC_prevalence" in ng["reported_instead"]
    assert "NOT read as a controller failure" in ng["interpretation_rule"]


def test_the_gate_refuses_a_sample_that_touches_the_sealed_reserve(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    good_formal = read(FORMAL_REL)
    reserve = read(RESERVE_REL)
    tampered = json.loads(json.dumps(good_formal))
    tampered["theorems"][0]["component_id"] = reserve["components"][0]["component_id"]
    monkeypatch.setattr(G, "load", lambda rel: {
        FORMAL_REL: tampered, RESERVE_REL: reserve, POOL_REL: read(POOL_REL),
        PRED_REL: read(PRED_REL), "experiments/manifests/v3/V3-R001_power.json": read(
            "experiments/manifests/v3/V3-R001_power.json")}[rel])
    monkeypatch.setattr(sys, "argv", ["v3_r001_gate.py", "--out", str(tmp_path / "gate.json")])
    with pytest.raises(SystemExit) as e:
        G.main()
    assert "SEALED reserve" in str(e.value)
    assert not (tmp_path / "gate.json").exists()          # fail closed, write nothing
    assert "wrote" not in capsys.readouterr().out


# ------------------------------------------------------------------- the document cannot drift
def test_the_preregistration_quotes_the_frozen_gate_numbers():
    d = _doc_flat()
    g = read(GATE_REL)
    p1, s1 = g["pooled_gate"]["P1"], g["within_synthetic_gate"]["S1"]
    assert f"π = {p1['pi_design_anchor']}" in d
    assert f"x ≥ {p1['reject_if_block_positives_at_least']}" in d
    assert f"{p1['implied_enrichment_at_the_boundary']}" in d
    assert f"**{p1['attainable_alpha_unconditional_at_pi']}**" in d
    assert f"{p1['min_detectable_enrichment_at_80pct_power']}" in d
    assert f"{s1['attainable_alpha_unconditional_at_pi']}" in d
    assert f"{s1['min_detectable_enrichment_at_80pct_power']}" in d
    assert f"x ≥ {s1['reject_if_block_positives_at_least']}" in d
    assert f"{s1['n']}" in d and f"{s1['m_top20']}" in d


def test_the_preregistration_carries_the_frozen_hashes():
    d = _doc_flat()
    f = read(FORMAL_REL)
    for h in (f["sample_sha256"], f["top20pct_block"]["sha256"],
              f["identity_with_the_power_sample"]["q_B2_vector_sha256"],
              f["membership_hashes"]["component_ids_sha256"],
              read(RESERVE_REL)["membership_hashes"]["component_ids_sha256"]):
        assert h in d
    assert "DESIGN-ONLY / NEVER ANALYZE" in d


def test_the_preregistration_states_the_rollout_parameters_and_the_host():
    d = _doc_flat()
    for token in ("n = 8", "temperature / top_p | 1.0 / 1.0", "max response tokens | 4096",
                  "fly122", "RTX 3080", "0 < Σ_{j=1..8} score_ij < 8",
                  "missing / excluded", "NOT AUTHORIZED"):
        assert token in d, token
    assert re.search(r"seed_base = 20260924", d)


def test_the_preregistration_does_not_claim_results():
    d = _doc().lower()
    for phrase in ("we found that r001", "the rollout showed", "outcome a was", "result: a_go"):
        assert phrase not in d
    assert "not launched" in d


REG_REL = "experiments/manifests/v3/registry.yaml"


def _sha256_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@pytest.mark.parametrize("rel,pin_key", [
    (FORMAL_REL, "formal_sample_sha256"),
    (RESERVE_REL, "final_reserve_sha256"),
    (GATE_REL, "gate_sha256"),
])
def test_the_registry_pin_is_the_actual_file_hash(rel, pin_key):
    """A preregistration hash that no one can re-check is decoration: the pinned sha256 has to be
    the byte hash of the committed artifact."""
    import yaml

    reg = yaml.safe_load((ROOT / REG_REL).read_text())
    entry = reg["prepreregistration_audits"]["V3-R001-formal-sample-and-final-reserve"]
    assert _sha256_of(rel) == entry["artifacts"][pin_key]


@pytest.mark.parametrize("rel", [FORMAL_REL, RESERVE_REL, GATE_REL])
def test_the_pinned_artifacts_have_no_wall_clock_or_host_field(rel):
    """Re-running a freeze script must not change a frozen artifact's hash. A created_at, host or
    git_revision field would make the pinned sha256 unverifiable by anyone but the original machine
    at the original minute."""
    raw = (ROOT / rel).read_text()
    for field in ("created_at_utc", "git_revision", '"host"'):
        assert field not in raw, (rel, field)
    assert "pure function" in raw
