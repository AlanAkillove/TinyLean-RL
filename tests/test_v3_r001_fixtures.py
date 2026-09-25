"""V3-R001 §19 -- the seven outcome fixtures, run against the frozen design on synthetic labels.

The owner's requirement is that the analyzer's behaviour is *demonstrated before* any real outcome
exists, not argued afterwards. Each fixture below builds a full 128-theorem synthetic rollout -- one
group of eight raw candidate rows per frozen theorem, with the frozen seed formula and the frozen
theta0 model hash -- feeds it to the real `v3_r001_analyze.analyze()`, and asserts the outcome the
preregistration promises:

    A  GO-SEMANTIC              pooled gate passes AND the within-synthetic co-primary passes
    B  SOURCE-DRIVEN-ONLY       pooled passes, within-synthetic fails
    C  NO-GO                    pooled fails
    D  INCONCLUSIVE-BY-DATA     the data guard fires BEFORE any gate is evaluated
    E  infra censoring          never becomes score=0 and never becomes an all-fail group
    F  block-hash tamper        the analyzer aborts
    G  sample-manifest tamper   the runner aborts before it generates anything

Labels are written as group outcomes ("pos"/"neg"/"censored"), never as scores, so a fixture cannot
accidentally test a score the analyzer invented. The block memberships come from the frozen
manifest, and the choice of which theorems to mark positive is made from the frozen q ranking only.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v3_r001_analyze as A
import v3_r001_rollout as R
import v3_r001_spec as S

# how many of the eight candidates of a "pos" group verify: y = 1 iff 0 < n_pos < 8, so a positive
# group must be a MIXED group. All-eight-verified is y = 0 by the label rule, not a positive.
N_POS_PER_POSITIVE_GROUP = 3


@pytest.fixture(scope="module")
def frozen() -> S.Frozen:
    """The real frozen design (file-level pins are covered by tests/test_v3_r001_gate_freeze.py)."""
    return S.load_frozen(verify_files=False)


@pytest.fixture(scope="module")
def design(frozen):
    """The frozen q-ordered theorem lists the fixtures select labels from."""
    pooled_block = S.frozen_block_for(frozen)
    synth_block = S.frozen_block_for(frozen, "synthetic")
    by_rank = [t["statement_id"] for t in sorted(frozen.theorems, key=lambda t: t["rank_by_q_B2"])]
    assert set(synth_block) <= set(pooled_block), (
        "the frozen top-20% block is entirely synthetic in this sample, so the within-synthetic "
        "co-primary is a strict subset of it; if that ever stops holding the fixtures below need "
        "rethinking, not patching")
    return {"pooled_block": pooled_block, "synth_block": synth_block, "by_rank": by_rank,
            "synth": [t["statement_id"] for t in frozen.theorems if t["source"] == "synthetic"],
            "non_synth": [t["statement_id"] for t in frozen.theorems
                          if t["source"] != "synthetic"]}


def _row(frozen: S.Frozen, theorem: dict, sample_index: int, status: str) -> dict:
    """One raw candidate row, exactly as schema-faithful as the runner promises to write."""
    score = S.candidate_score(status)
    return {
        "schema_version": S.SCHEMA_VERSION, "run_id": "fixture-000",
        "experiment_id": S.EXPERIMENT_ID, "theorem_rank": theorem["rank_by_q_B2"],
        "formal_sample_rank": theorem["formal_sample_rank"],
        "statement_id": theorem["statement_id"], "component_id": theorem["component_id"],
        "name": theorem["statement_id"][:8], "source": theorem["source"],
        "sample_index": sample_index,
        "seed": S.candidate_seed(theorem["formal_sample_rank"], sample_index),
        "prompt_sha256": "0" * 64, "frozen_prompt_sha256": "0" * 64,
        "formal_statement_sha256": "0" * 64,
        "model_sha256": frozen.theta0["weights_sha256"], "model_revision": frozen.theta0["revision"],
        "completion_text": "<think>fixture</think>", "completion_sha256": "0" * 64,
        "generated_tokens": 512, "truncated": False,
        "format_ok": status != S.FORMAT_ERROR, "verified": score == 1, "score": score,
        "acc": 1 if score == 1 else 0,
        "verify_status": status, "verifier_error_category": "none" if score == 1 else status,
        "lean_message": "", "generation_time": 1.0, "verification_time": 1.0,
        "host": "fixture", "gpu": "fixture", "created_at": "2026-09-24T00:00:00+00:00",
    }


def groups_for(frozen: S.Frozen, labels: dict) -> dict:
    """statement_id -> its eight candidate rows, for a label map of pos / neg / censored."""
    by_stmt = {t["statement_id"]: t for t in frozen.theorems}
    unknown = set(labels) - set(by_stmt)
    assert not unknown, f"a fixture labelled a theorem outside the frozen sample: {sorted(unknown)}"
    groups = {}
    for sid, kind in labels.items():
        t = by_stmt[sid]
        if kind == "pos":
            statuses = (["verified"] * N_POS_PER_POSITIVE_GROUP
                        + ["lean_error"] * (S.N_SAMPLES - N_POS_PER_POSITIVE_GROUP))
        elif kind == "neg":
            statuses = ["lean_error"] * S.N_SAMPLES
        elif kind == "censored":
            statuses = ["lean_error"] * (S.N_SAMPLES - 1) + ["verifier_timeout"]
        elif kind == "format_error":
            statuses = ["format_error"] * S.N_SAMPLES
        else:
            raise ValueError(kind)
        groups[sid] = [_row(frozen, t, i, st) for i, st in enumerate(statuses)]
    return groups


def label_map(design: dict, positives: list, censored: list | None = None,
              negatives: str = "rest") -> dict:
    labels = {sid: "neg" for sid in design["by_rank"]} if negatives == "rest" else {}
    for sid in positives:
        labels[sid] = "pos"
    for sid in censored or []:
        labels[sid] = "censored"
    return labels


def run(frozen, groups) -> dict:
    return A.analyze(frozen, groups, summary=None)


# --- the block structure the fixtures are written against ---------------------------------------

def test_the_frozen_pooled_block_is_entirely_synthetic(frozen, design) -> None:
    """A structural fact four of the fixtures rely on, asserted rather than assumed.

    If the pooled top-20% ever stops being a superset of the synthetic top-20%, fixture B's whole
    premise (pooled enrichment that is not within-stratum enrichment) has to be rebuilt.
    """
    assert len(design["pooled_block"]) == 26 and len(design["synth_block"]) == 9
    assert set(design["synth_block"]) <= set(design["pooled_block"])
    assert all({t["statement_id"]: t["source"] for t in frozen.theorems}[s] == "synthetic"
               for s in design["pooled_block"])
    assert len(design["synth"]) == 45 and len(design["by_rank"]) == 128


# --- A: GO-SEMANTIC -------------------------------------------------------------------------------

def test_fixture_A_pooled_and_within_synthetic_both_pass(frozen, design) -> None:
    """The controller's top synthetic theorems really are the informative ones: outcome A."""
    positives = design["synth_block"] + design["pooled_block"][9:12]
    result = run(frozen, groups_for(frozen, label_map(design, positives)))
    outcome = result["outcome"]
    assert outcome["outcome"] == "A_GO-SEMANTIC", json.dumps(result["pooled_gate"], indent=2)
    assert outcome["r002_eligible"] is True
    pooled, synth = result["pooled_gate"], result["within_synthetic_gate"]
    assert pooled["pass_P1"] and pooled["pass_P2"] and synth["pass_P1"] and synth["pass_P2"]
    assert pooled["x_block_positives"] >= pooled["rejection_boundary"]
    assert pooled["enrichment"] >= A.ENRICHMENT_FLOOR and pooled["enrichment_ci_lower"] > 1.0
    assert synth["identifiable"] is True
    assert result["data_guard_outcome_D_first"]["pass"] is True


# --- B: SOURCE-DRIVEN-ONLY ------------------------------------------------------------------------

def test_fixture_B_pooled_passes_but_within_synthetic_fails(frozen, design) -> None:
    """Enrichment carried by the stratum as a whole, not by the controller's own top-9 of it."""
    positives = design["pooled_block"][9:19]
    assert not (set(positives) & set(design["synth_block"]))
    result = run(frozen, groups_for(frozen, label_map(design, positives)))
    assert result["outcome"]["outcome"] == "B_SOURCE-DRIVEN-ONLY", json.dumps(
        {"pooled": result["pooled_gate"], "synth": result["within_synthetic_gate"]}, indent=2)
    assert result["outcome"]["r002_eligible"] is False
    assert result["pooled_gate"]["pass_P1"] and result["pooled_gate"]["pass_P2"]
    assert result["within_synthetic_gate"]["block_positives"] == 0
    assert result["within_synthetic_gate"]["pass_P1"] is False
    assert result["within_synthetic_gate"]["pass_P2"] is False


# --- C: NO-GO -------------------------------------------------------------------------------------

def test_fixture_C_pooled_gate_fails(frozen, design) -> None:
    """Positives exist, but not where the controller pointed: the prospective claim is wrong.

    18 theorems come out informative and only 2 of them sit in the frozen top-20% block, against a
    boundary of 7 at k=18. That is a genuine NO-GO, not a thin-data artifact: the gate has enough
    decidable units and enough positives to decide, and it decides against the controller.
    """
    pooled_block = set(design["pooled_block"])
    in_block = [s for s in design["pooled_block"] if s not in set(design["synth_block"])][:2]
    outside = [s for s in design["by_rank"] if s not in pooled_block][:16]
    positives = in_block + outside
    assert len(set(positives) & pooled_block) == 2 and len(positives) == 18
    result = run(frozen, groups_for(frozen, label_map(design, positives)))
    pooled = result["pooled_gate"]
    assert result["outcome"]["outcome"] == "C_NO-GO", json.dumps(pooled, indent=2)
    assert pooled["pass_P1"] is False and pooled["x_block_positives"] < pooled["rejection_boundary"]
    assert pooled["exact_p_one_sided"] > A.ALPHA
    assert result["data_guard_outcome_D_first"]["pass"] is True, "C is a decision, not a shortfall"
    assert result["outcome"]["r002_eligible"] is False


# --- D: INCONCLUSIVE-BY-DATA ----------------------------------------------------------------------

def test_fixture_D_too_few_positives_never_reads_as_a_no_go(frozen, design) -> None:
    """4 positives is below the guard: the gate is not evaluated, and the run is not a FAIL."""
    result = run(frozen, groups_for(frozen, label_map(design, design["synth_block"][:4])))
    assert result["data_guard_outcome_D_first"]["pass"] is False
    assert result["outcome"]["outcome"] == "D_INCONCLUSIVE-BY-DATA"
    assert result["pooled_gate"]["status"] == "NOT_EVALUATED"
    assert result["within_synthetic_gate"]["status"] == "NOT_EVALUATED"
    assert result["outcome"]["P1_exact_test"] is None, "an unevaluated gate must not report a verdict"


def test_fixture_D_censoring_beyond_twenty_percent(frozen, design) -> None:
    """The other arm of the same guard: too little decidable data, whatever its label says."""
    positives = design["synth_block"]
    censored = [s for s in design["by_rank"] if s not in set(positives)][:30]
    result = run(frozen, groups_for(frozen, label_map(design, positives, censored=censored)))
    guard = result["data_guard_outcome_D_first"]
    assert guard["n_analyzed"] == 98 and guard["pass"] is False
    assert result["outcome"]["outcome"] == "D_INCONCLUSIVE-BY-DATA"


def test_an_incomplete_group_aborts_the_analyzer(frozen, design) -> None:
    """Owner 7: a group the runner never finished is a broken input, not a statistical finding."""
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    victim = design["by_rank"][40]
    groups[victim] = groups[victim][:5]
    with pytest.raises(S.FrozenViolation, match="INCOMPLETE"):
        run(frozen, groups)
    del groups[victim]
    with pytest.raises(S.FrozenViolation, match="INCOMPLETE"):
        run(frozen, groups)


# --- E: infra censoring is never a zero -----------------------------------------------------------

def test_fixture_E_infra_censoring_leaves_the_label_missing_never_zero(frozen, design) -> None:
    """A verifier outage removes a unit; it may not silently become a hard negative."""
    positives = design["synth_block"]
    censored = design["pooled_block"][9:13]
    result = run(frozen, groups_for(frozen, label_map(design, positives, censored=censored)))
    rows = {r["statement_id"]: r for r in result["per_theorem"]}
    for sid in censored:
        assert rows[sid]["group_status"] == S.GROUP_INFRA_CENSORED
        assert rows[sid]["y"] is None, "a censored group must have no label at all"
        assert rows[sid]["n_infra"] == 1 and rows[sid]["n_pos_score"] is None
    guard = result["data_guard_outcome_D_first"]
    assert guard["n_analyzed"] == 124, "the four censored units are excluded, not counted as zeros"
    assert guard["positives"] == 9, "prevalence is computed on the decidable units only"
    pooled = result["pooled_gate"]
    assert pooled["n_censored"] == 4 and pooled["k_analyzed_positives"] == 9
    assert pooled["m_frozen_block_analyzed"] == 22, "the frozen block shrank by its censored members"
    assert pooled["m_variant_if_the_block_were_recomputed"] != 22, (
        "the re-taken-top-20% variant is disclosed precisely because it is NOT what the gate used")
    raw = [r for sid in censored for r in groups_for(
        frozen, {sid: "censored"})[sid]]
    assert sum(1 for r in raw if r["verify_status"] == "verifier_timeout") == 4
    assert all(r["score"] is None for r in raw if r["verify_status"] == "verifier_timeout")
    assert result["group_status_counts"][S.GROUP_INFRA_CENSORED] == 4


def test_a_hand_written_zero_on_an_infra_row_is_rejected(frozen) -> None:
    """The schema itself refuses the all-fail-by-accident bug, before any statistic is computed."""
    row = _row(frozen, frozen.theorems[0], 0, "lean_error")
    row["verify_status"] = "verifier_timeout"
    with pytest.raises(S.FrozenViolation, match="contradicts verify_status"):
        S.validate_row(row)
    row["score"] = None
    row["verified"] = False
    S.validate_row(row)                      # the same row is legal once the score is honestly null
    with pytest.raises(S.FrozenViolation, match="outside the frozen B0 vocabulary"):
        S.validate_row({**row, "verify_status": "system_error"})


def test_a_format_error_is_a_real_zero_not_censoring(frozen, design) -> None:
    """No extractable proof is the model's failure, so it counts; only the verifier's does not."""
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    sid = design["non_synth"][0]
    t = {x["statement_id"]: x for x in frozen.theorems}[sid]
    groups[sid] = [_row(frozen, t, i, S.FORMAT_ERROR) for i in range(S.N_SAMPLES)]
    result = run(frozen, groups)
    row = {r["statement_id"]: r for r in result["per_theorem"]}[sid]
    assert row["group_status"] == S.GROUP_COMPLETE and row["y"] == 0
    assert row["n_format_error"] == 8 and row["n_infra"] == 0
    assert result["data_guard_outcome_D_first"]["n_analyzed"] == 128


# --- F: a tampered block hash aborts the analyzer -------------------------------------------------

def _tampered_manifest(tmp_path: Path, mutate) -> str:
    """Write a copy of the frozen sample manifest with one deliberate edit; return its path."""
    real = json.loads((ROOT / S.FORMAL_SAMPLE).read_text(encoding="utf-8"))
    tampered = copy.deepcopy(real)
    mutate(tampered)
    path = tmp_path / "v3_r001_formal_sample.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    return str(path)


def test_fixture_F_block_hash_tamper_aborts_every_consumer(tmp_path, monkeypatch, frozen) -> None:
    """Membership of the top-20% block is a frozen hash, so editing it cannot go unnoticed."""
    def poke(manifest):
        manifest["top20pct_block"]["sha256"] = "f" * 64

    path = _tampered_manifest(tmp_path, poke)
    monkeypatch.setattr(S, "FORMAL_SAMPLE", path)
    with pytest.raises(S.FrozenViolation, match="block_hash_recomputes"):
        S.load_frozen(verify_files=False)
    # and the analyzer's own entry point is that call, so on a tampered design it must abort before
    # it can read a raw rollout or write anything: the canonical results artifact is never touched
    monkeypatch.setattr(sys, "argv", [
        "v3_r001_analyze.py", "--raw", str(tmp_path / "absent.jsonl"), "--summary", "none",
        "--out", str(tmp_path / "never_written.json"), "--check-only",
    ])
    with pytest.raises(S.FrozenViolation, match="block_hash_recomputes"):
        A.main()
    assert not (tmp_path / "never_written.json").exists()


def test_fixture_F_block_membership_tamper_is_caught_too(tmp_path, monkeypatch) -> None:
    """Dropping a theorem out of the block changes its hash AND breaks the q-top check."""
    def poke(manifest):
        manifest["top20pct_block"]["statement_ids"] = manifest["top20pct_block"]["statement_ids"][1:]
        manifest["top20pct_block"]["size"] = 25

    monkeypatch.setattr(S, "FORMAL_SAMPLE", _tampered_manifest(tmp_path, poke))
    with pytest.raises(S.FrozenViolation, match="block_size_is_frozen|block_hash_recomputes"):
        S.load_frozen(verify_files=False)


# --- G: a tampered sample manifest aborts the runner before generation ---------------------------

def _fake_env() -> dict:
    """A clean, fly122-shaped environment: the host interlock is not what these tests are about."""
    return {"hostname": "ubuntu", "ips": "10.3.25.122/24", "git_revision": "0" * 40,
            "branch": S.BRANCH, "status_porcelain": "",
            "gpu_names": S.FORMAL_GPU_NAME, "gpu_compute_apps": "", "prereg_is_ancestor": True}


@pytest.mark.parametrize("poke,expect", [
    (lambda m: m["theorems"][0].__setitem__("component_id", "tampered-component"),
     "draw_pairing_matches_the_manifest"),
    (lambda m: m["theorems"][0].__setitem__("q_B2_controller", 0.999), "q_B2_vector_hash"),
    (lambda m: m["theorems"][0].__setitem__("rank_by_q_B2", 7), "ranks_are_consecutive"),
    (lambda m: m["theorems"][0].__setitem__("has_historical_label", True), "no_historical_label"),
    (lambda m: m.__setitem__("sample_sha256", "a" * 64), "sample_sha"),
])
def test_fixture_G_runner_aborts_before_generation(tmp_path, monkeypatch, poke, expect) -> None:
    """Whatever the tamper, the runner must exit as a preflight abort with nothing generated.

    `expect` names the check that should catch it; the assertion is only that some frozen check fires
    (any of them may be first), and that the runner never reaches the prompt surface, let alone a
    model or a verifier.
    """
    monkeypatch.setattr(S, "FORMAL_SAMPLE", _tampered_manifest(tmp_path, poke))
    monkeypatch.setattr(S, "collect_env", _fake_env)

    def never(*_a, **_k):
        raise AssertionError("the runner touched the prompt surface after a failed preflight")

    monkeypatch.setattr(R, "load_statement_texts", never)
    monkeypatch.setattr(R, "build_plan", never)
    monkeypatch.setattr(R, "make_session", never)
    with pytest.raises(S.FrozenViolation, match=expect):
        S.load_frozen(verify_files=False)
    assert R.main(["--dry-run", "--out-dir", str(tmp_path / "rollout")]) == 3
    assert not (tmp_path / "rollout").exists() or not list((tmp_path / "rollout").iterdir())


def test_the_launch_interlock_blocks_a_formal_run(tmp_path) -> None:
    """READY_FOR_OWNER_LAUNCH = NO is a rule in code, not only a line in the document."""
    assert R.launch_interlock(R.parse_args(["--dry-run"])) is None
    assert R.launch_interlock(R.parse_args(["--i-have-owner-launch-authorization"])) is None
    blocked = R.launch_interlock(R.parse_args([]))
    assert blocked is not None and "READY_FOR_OWNER_LAUNCH = NO" in blocked


def test_the_provenance_check_names_a_foreign_model(frozen, design) -> None:
    """Rows from another checkpoint cannot be analyzed as if they came from theta0."""
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    groups[design["by_rank"][60]][0]["model_sha256"] = "e" * 64
    with pytest.raises(S.FrozenViolation, match="other than the frozen theta0"):
        run(frozen, groups)


def test_the_provenance_check_names_a_broken_seed(frozen, design) -> None:
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    groups[design["by_rank"][61]][2]["seed"] += 1
    with pytest.raises(S.FrozenViolation, match="frozen seed formula"):
        run(frozen, groups)


def test_a_reserve_component_in_the_raw_artifact_aborts(frozen, design) -> None:
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    sealed = frozen.reserve["components"][0]["component_id"]
    groups[design["by_rank"][62]][0]["component_id"] = sealed
    with pytest.raises(S.FrozenViolation, match="SEALED reserve"):
        run(frozen, groups)


def test_merging_two_runs_into_one_group_is_refused_on_read(tmp_path, frozen, design) -> None:
    """Owner 8: a group is one run's eight candidates; a ninth row is corruption, not extra data."""
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    path = tmp_path / "raw.jsonl"
    rows = [r for group in groups.values() for r in group]
    S.append_rows_durable(path, rows + [dict(rows[0], run_id="second-run")])
    with pytest.raises(S.FrozenViolation, match="candidates from different runs have been merged"):
        S.read_groups(path)


def test_the_lambda_diagnostics_are_reproduced_and_stay_diagnostics(frozen, design) -> None:
    """Owner 12: the mixture policy is reported, pinned to the frozen hash, and selects nothing."""
    result = run(frozen, groups_for(frozen, label_map(design, design["synth_block"])))
    diag = result["mixture_diagnostics"]
    assert diag["not_a_gate"] is True and set(diag) >= {"lambda=0.5", "lambda=0.8"}
    for lam in A.LAMBDA_GRID:
        block = diag[f"lambda={lam}"]
        assert block["frozen_policy_over_all_128"]["matches_the_frozen_gate_pin"] is True
        sub = block["renormalized_on_the_analyzed_subset"]
        assert sub["all_probabilities_strictly_positive"] is True
        assert sub["min_probability"] > 0.0
        assert sub["expected_IGR"] is not None, "labels exist now, so the gate's null must fill in"
    assert "best" not in json.dumps(diag).lower()


def test_the_b2_vs_b1_delta_is_reported_but_never_gates(frozen, design) -> None:
    """A non-positive delta must not be able to produce a 'B2 failed' reading anywhere."""
    result = run(frozen, groups_for(frozen, label_map(design, design["synth_block"])))
    d = result["descriptives"]
    assert {"auprc_B2_controller", "auprc_B1_handcrafted", "delta_auprc_B2_minus_B1",
            "auprc_prevalence_baseline"} <= set(d)
    assert d["enrichment_B2_controller"]["top20pct"] > 1.0
    blob = json.dumps(result)
    assert "B2 failed" not in blob and "B2_FAILED" not in blob
    assert "not a gate" in blob.lower()


def test_per_source_diagnostics_refuse_a_metric_on_four_positives(frozen, design) -> None:
    """Owner 18: a stratum too thin to rank is DESCRIPTIVE_ONLY, never a number that looks real."""
    positives = design["synth_block"] + design["non_synth"][:4]
    result = run(frozen, groups_for(frozen, label_map(design, positives)))
    src = result["source_diagnostics"]
    assert src["synthetic"]["status"] == "COMPUTED"
    assert src["human"]["positives"] == 0
    assert src["human"]["status"].startswith("DESCRIPTIVE_ONLY")
    assert "auprc_B2" not in src["human"]


# --- owner amendment A (2026-09-25): the frozen block is never recomputed ------------------------

def test_amendment_A_censoring_shrinks_the_block_and_never_tops_it_up(frozen, design) -> None:
    """The block loses a censored member; no theorem outside the frozen block may replace it."""
    from v3_r001_gate import ALPHA, critical_function

    inside = set(design["pooled_block"])
    outside = [s for s in design["by_rank"] if s not in inside]
    labels = label_map(design, positives=outside[:10])
    labels.update({s: "censored" for s in design["pooled_block"][:4]})
    rows = A.labels_from_groups(frozen, groups_for(frozen, labels))
    cell = A.gate_cell(frozen, rows)
    assert cell["m_frozen_block_nominal"] == 26
    assert cell["m_frozen_block_analyzed"] == 22, "four frozen block members were censored"
    assert cell["m_variant_if_the_block_were_recomputed"] == S.block_size(cell["n_analyzed"])
    assert cell["x_block_positives"] == 0, "an outside theorem was counted into the block"
    assert cell["identifiable"] and cell["reject_if_at_least"] == int(critical_function(
        cell["n_analyzed"], cell["m_frozen_block_analyzed"], ALPHA)[cell["k_analyzed_positives"]])
    assert "never re-taken" in cell["recomputation_note"]
    assert "descriptive" in cell["recomputation_note"]


def test_amendment_A_the_gate_artifact_states_the_frozen_block_semantics() -> None:
    """The deprecated wording is retired in the frozen artifact itself, not only in a document."""
    gate = json.loads((ROOT / "experiments/manifests/v3/V3-R001_gate.json").read_text())
    fbs = gate["frozen_block_semantics"]
    assert fbs["pooled"]["m"] == "|B_analyzed|" and fbs["pooled"]["N"] == "|A|"
    assert "never recomputed" in fbs["invariants"][0]
    assert "never enters a gate" in fbs["invariants"][-1]
    rule = gate["pooled_gate"]["P1"]["recomputation_rule"]
    assert "deprecated" in rule and "round(0.20 * N_analyzed)" in rule
    assert "PRE-OUTCOME CLARIFICATION" in fbs["amendment"]


# --- owner amendment B (2026-09-25): the generation seed stream may not read q -------------------

def test_amendment_B_the_rank_is_the_frozen_draw_order_not_the_q_rank(frozen) -> None:
    ranks = {t["statement_id"]: t["formal_sample_rank"] for t in frozen.theorems}
    assert sorted(ranks.values()) == list(range(1, S.N_NOMINAL + 1))
    draw_order = [s for s, _ in sorted(ranks.items(), key=lambda kv: kv[1])]
    assert draw_order == list(frozen.draw_order), "the seed rank must be the frozen draw order"
    assert draw_order != [t["statement_id"] for t in
                          sorted(frozen.theorems, key=lambda t: t["rank_by_q_B2"])], (
        "the draw order coincides with the q rank in this sample, so this test could not see a "
        "regression to rank_by_q_B2 -- it needs a different sample, not a weaker assertion")


def test_amendment_B_generation_seeds_do_not_move_when_q_moves(frozen) -> None:
    """If a seed read the q-derived rank, re-deriving that rank from permuted q would move it."""
    before = R.seed_schedule(frozen.theorems)
    flipped = [dict(t, q_B2_controller=-t["q_B2_controller"],
                    q_B1_handcrafted=-t["q_B1_handcrafted"],
                    q_B2_fold_ensemble=-t["q_B2_fold_ensemble"],
                    q_B1_fold_ensemble=-t["q_B1_fold_ensemble"],
                    source={"synthetic": "human"}.get(t["source"], "synthetic"))
               for t in frozen.theorems]
    for pos, t in enumerate(sorted(flipped, key=lambda t: (-t["q_B2_controller"],
                                                           t["statement_id"]))):
        t["rank_by_q_B2"] = pos + 1                      # what a re-run of the freeze would store
    assert [t["statement_id"] for t in sorted(flipped, key=lambda t: t["rank_by_q_B2"])] != \
        [t["statement_id"] for t in sorted(frozen.theorems, key=lambda t: t["rank_by_q_B2"])]
    assert R.seed_schedule(flipped) == before, "a generation seed moved when q moved"


def test_amendment_B_all_1024_generation_seeds_are_unique(frozen) -> None:
    schedule = R.seed_schedule(frozen.theorems)
    seeds = [s for entry in schedule.values() for s in entry]
    assert len(schedule) == S.N_NOMINAL
    assert len(seeds) == S.EXPECTED_TOTAL_CANDIDATES == 1024
    assert len(set(seeds)) == 1024
    assert min(seeds) == S.SEED_BASE
    assert max(seeds) == S.SEED_BASE + (S.N_NOMINAL - 1) * S.SEED_GROUP_SIZE + S.N_SAMPLES - 1


def test_a_row_without_the_draw_rank_is_rejected_by_the_schema(frozen, design) -> None:
    row = groups_for(frozen, label_map(design, design["synth_block"]))[design["by_rank"][0]][0]
    del row["formal_sample_rank"]
    with pytest.raises(S.FrozenViolation, match="missing schema fields"):
        S.validate_row(row)


def test_the_provenance_check_names_a_foreign_draw_rank(frozen, design) -> None:
    groups = groups_for(frozen, label_map(design, design["synth_block"]))
    row = groups[design["by_rank"][61]][0]
    row["formal_sample_rank"] = 129 - row["formal_sample_rank"]      # never the same rank
    with pytest.raises(S.FrozenViolation, match="draw-order rank"):
        run(frozen, groups)
