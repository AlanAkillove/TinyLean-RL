"""Tests for V3-R001 §17 step 7 -- the theta0 extraction gate and the prospective pre-scoring.

Nothing here needs a GPU, a tokenizer or a rollout. Two things are checked.

(1) The pure helpers of scripts/v3_r001_prescore.py: the draw that freezes the sample (it must be
    one deterministic uniform order, so every candidate N is a nested prefix of every larger one and
    the owner's eventual N cannot silently reshuffle a smaller sample), the per-source-stratum power
    readout that decides what the gate is allowed to ask for (it must call the committed exact-power
    machinery, not re-derive it, and it must call an under-positive stratum UNREACHABLE rather than
    reporting a null), and the design-row builder, which must place the block-18 vector and the
    step_norm = 0 slot exactly where the frozen D001 controller reads them (owner §6).

(2) The fail-closed guards of the same script, by running it against deliberately broken inputs in a
    tmp dir: a scoring run that would pre-score a different statement set than §5 froze, score a
    family V1 already touched, or write a NON-FORMAL artifact into experiments/manifests/ has to
    refuse. Each refusal is asserted to produce no output file, since the point of a guard that
    fires after the write is nothing.

Also the extraction script's provenance declarations, checked against the committed V3-D001
metadata: same checkpoint hash, same pre-registered layers, and a prompt renderer that feeds the
chat template WITH the generation prompt and strips specials the way the rollout dumps record them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PRESCORE = SCRIPTS / "v3_r001_prescore.py"
EXTRACT = SCRIPTS / "v3_r001_extract.py"
POOL_ARTIFACT = ROOT / "experiments/manifests/v3/V3-R001_family_clean_pool.json"
D001_META = ROOT / "experiments/manifests/v3/V3-D001_theta0_reps_meta.json"
PREDICTIONS = ROOT / "experiments/manifests/v3/V3-R001_predictions.json"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def P():
    """scripts/v3_r001_prescore.py (numpy/pandas only -- torch is not imported at module level)."""
    return _load("v3_r001_prescore_under_test", PRESCORE)


@pytest.fixture(scope="module")
def E():
    pytest.importorskip("torch")
    return _load("v3_r001_extract_under_test", EXTRACT)


@pytest.fixture(scope="module")
def R():
    import v3_d001_run

    return v3_d001_run


@pytest.fixture(scope="module")
def pool():
    return json.loads(POOL_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def art():
    if not PREDICTIONS.exists():
        pytest.skip("formal R001 predictions not yet frozen (owner §17 step 7 runs on fly122)")
    return json.loads(PREDICTIONS.read_text())


# --------------------------------------------------------------------------------------------------
# the frozen sample draw (owner §7)
# --------------------------------------------------------------------------------------------------

def test_draw_is_one_uniform_order_shared_by_every_candidate_N(P):
    ids = [f"fc-{i:04d}" for i in range(300)]
    full = P.draw_components(ids, len(ids))
    assert sorted(full) == sorted(ids), "the draw must permute, never invent or drop components"
    for n in (56, 64, 86, 128, 192, 221):
        assert P.draw_components(ids, n) == full[:n], f"N={n} is not a prefix of the frozen order"


def test_draw_order_is_exactly_the_documented_hash_key(P):
    ids = [f"fc-{i:04d}" for i in range(50)]
    expect = sorted(ids, key=lambda c: hashlib.sha256(f"{P.DRAW_SEED}|{c}".encode()).hexdigest())
    assert P.draw_components(ids, 50) == expect
    assert P.DRAW_SEED == 20260924, "the seed is the owner-directive date and is part of the freeze"


def test_draw_ignores_input_order_and_python_hash_randomization(P):
    """SHA256 ordering instead of a library RNG: the same sample must come out on any machine."""
    ids = [f"fc-{i:04d}" for i in range(120)]
    shuffled = list(np.random.default_rng(7).permutation(ids))
    assert P.draw_components(shuffled, 40) == P.draw_components(ids, 40)

    snippet = (
        "import importlib.util,json,sys;"
        "s=importlib.util.spec_from_file_location('m',sys.argv[1]);m=importlib.util.module_from_spec(s);"
        "sys.modules['m']=m;s.loader.exec_module(m);"
        "print(json.dumps(m.draw_components([f'fc-{i:04d}' for i in range(120)],40)))"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-c", snippet, str(PRESCORE)], capture_output=True,
                           text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed})
        seen.add(r.stdout.strip())
    assert len(seen) == 1, "the frozen sample depends on PYTHONHASHSEED"


def test_draw_units_are_the_pool_components_and_their_frozen_representatives(P, pool):
    """The sampling unit and the bootstrap unit must be the same component, and the theorem drawn
    for it must be the one the pool artifact named -- otherwise one-theorem-per-family is a claim
    about two different things."""
    for reading, p in pool["pools_by_reading"].items():
        comps = sorted(p["candidate_component_ids"])
        if not comps:
            continue
        for c in P.draw_components(comps, min(30, len(comps))):
            assert p["component_to_candidate"][c] in pool["extraction_union"], f"{reading}/{c}"


# --------------------------------------------------------------------------------------------------
# per-stratum power (owner §11 / §14): the numbers that cap what the gate may demand
# --------------------------------------------------------------------------------------------------

def test_stratum_power_delegates_to_the_frozen_power_machinery(P):
    import v3_r001_power as PW

    n, pi = 45, 0.3444
    r = P.stratum_power(n, pi)
    assert r["testable"] is True
    m = r["block_size_top20pct"]
    assert m == round(0.20 * n) == 9
    assert r["realized_size_of_the_exact_test_under_H0"] == round(
        PW.exact_power(n, m, pi, 1.0)["power"], 4)
    assert r["power_at_enrichment_2.5"] == round(PW.exact_power(n, m, pi, 2.5)["power"], 4)
    assert r["min_detectable_enrichment_at_80pct_power"] == round(PW.min_detectable_e(n, m, pi), 4)
    assert r["expected_positives_in_block"] == round(m * pi, 2)


def test_a_stratum_that_cannot_hold_positives_is_called_unreachable_not_null(P):
    """human at pi=0.0215 in an N=128 sample: no coherent enrichment reaches 80% power. Saying so is
    the honest answer; reporting a low power as a RESULT would turn an absence of signal into an
    absence of evidence for the controller."""
    r = P.stratum_power(25, 0.0215)
    assert r["testable"] is True
    mde = r["min_detectable_enrichment_at_80pct_power"]
    assert isinstance(mde, str) and "unreachable" in mde, mde
    assert r["expected_positives_in_block"] < 1


@pytest.mark.parametrize("n,block_fraction", [(1, 0.20), (4, 1.00)])
def test_a_cohort_that_a_top_block_cannot_split_is_not_testable(P, n, block_fraction):
    r = P.stratum_power(n, 0.34, block_fraction=block_fraction)
    assert r["testable"] is False and "whole cohort" in r["why"]


# --------------------------------------------------------------------------------------------------
# the frozen scoring surface (owner §6: block 18 + step_norm = 0, no refit, no undefined q)
# --------------------------------------------------------------------------------------------------

def test_step_norm_slot_is_the_pre_update_value_the_directive_names(P):
    assert P.STEP_NORM_USED == 0.0


def test_candidate_rows_place_block18_and_step_norm_where_the_head_reads_them(P, R):
    import v3_d001_lib as L

    rep = {lay: np.full(R.REP_DIM, float(lay)) for lay in (9, 18, 27)}
    prompt = ("# Task\nprove the following.\n# Formal Statement:\n```lean4\n"
              "theorem t (n : Nat) : n + 0 = n\n```\n")
    Xb1, Xb2 = P.candidate_matrix(["a"], {"a": (17, rep)}, {"a": "synthetic"}, {"a": prompt})

    assert Xb2.shape == (1, R.REP_DIM + 1)
    assert np.array_equal(Xb2[0, :R.REP_DIM], rep[R.PRIMARY_LAYER]), "B2 must read block 18"
    assert Xb2[0, R.REP_DIM] == P.STEP_NORM_USED, "the last B2 column is the step_norm slot"
    assert Xb1.shape == (1, len(L.B1_FEATURES))
    assert Xb1[0, L.B1_FEATURES.index("step_norm")] == 0.0
    assert Xb1[0, L.B1_FEATURES.index("prompt_token_count")] == 17.0
    onehot = {s: Xb1[0, len(L.B1_NUMERIC) + i] for i, s in enumerate(L.B1_SOURCE_LEVELS)}
    assert onehot["synthetic"] == 1.0 and sum(onehot.values()) == 1.0, "exactly one source is active"


def test_a_candidate_without_a_formal_block_is_refused_not_scored_blank(P, R):
    rep = {lay: np.zeros(R.REP_DIM) for lay in (9, 18, 27)}
    with pytest.raises(SystemExit, match="Formal Statement"):
        P.candidate_matrix(["a"], {"a": (1, rep)}, {"a": "human"}, {"a": "# Task\nno block here\n"})


# --------------------------------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------------------------------

def test_sha_is_key_order_invariant_but_sequence_sensitive(P):
    assert P.sha({"a": 1, "b": 2}) == P.sha({"b": 2, "a": 1})
    assert P.sha([1, 2]) != P.sha([2, 1]), "the q-vector hash must depend on the ordering"


def test_quantiles_are_ordered_rounded_and_empty_safe(P):
    q = P.quantiles([3.0, 1.0, 2.0, 4.0])
    assert q["n"] == 4 and q["min"] == 1.0 and q["max"] == 4.0
    assert q["mean"] == 2.5 and q["median"] == 2.5
    keys = ("min", "p25", "median", "p75", "p90", "max")
    assert [q[k] for k in keys] == sorted(q[k] for k in keys)
    assert P.quantiles([]) == {"n": 0}


def test_load_reps_maps_ids_to_per_layer_vectors(P, tmp_path):
    layers = [9, 18, 27]
    reps = np.arange(2 * len(layers) * 1024, dtype=np.float32).reshape(2, len(layers), 1024)
    p = tmp_path / "r.npz"
    np.savez(p, statement_ids=np.array(["x", "y"], dtype=object), token_lens=np.array([3, 4]),
             reps=reps, layers=np.array(layers))
    got = P.load_reps(p)
    assert got["ids"] == ["x", "y"] and got["layers"] == layers
    tl, vec = got["map"]["y"]
    assert tl == 4 and np.array_equal(vec[18], reps[1, 1])
    assert set(vec) == set(layers)


# --------------------------------------------------------------------------------------------------
# the extraction script's provenance declarations (no GPU: constants and the renderer only)
# --------------------------------------------------------------------------------------------------

def test_extraction_uses_the_same_checkpoint_and_layers_as_d001(E, R):
    meta = json.loads(D001_META.read_text())
    assert E.THETA0_WEIGHTS_SHA == meta["theta0_weights_sha256_expected"]
    assert E.PREREG_LAYERS == sorted(meta["pre_registered_layers_0based"])
    assert sorted(E.PREREG_LAYERS) == sorted(R.LAYERS), "extraction and the D001 fit must agree on layers"
    assert R.PRIMARY_LAYER in E.PREREG_LAYERS


def test_prompt_of_renders_with_generation_prompt_and_strips_specials(E):
    calls = {}

    class FakeTok:
        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
            calls["msgs"] = msgs
            calls["add_generation_prompt"] = add_generation_prompt
            return {"input_ids": [[1, 2, 3]]}              # transformers encoding-style return

        def decode(self, ids, skip_special_tokens=False):
            calls["ids"], calls["skip_special_tokens"] = ids, skip_special_tokens
            return "RENDERED"

    out = E.prompt_of(FakeTok(), [{"role": "system", "content": "s"},
                                  {"role": "user", "content": "u"}])
    assert out == "RENDERED"
    assert calls["add_generation_prompt"] is True, "theta0 saw a statement plus a generation prompt, never a prefix"
    assert calls["ids"] == [1, 2, 3], "a nested encoding must be flattened before decode"
    assert calls["skip_special_tokens"] is True
    assert [m["content"] for m in calls["msgs"]] == ["s", "u"], "messages must reach the template intact"


# --------------------------------------------------------------------------------------------------
# fail-closed guards: broken inputs must abort BEFORE anything is written
# --------------------------------------------------------------------------------------------------

def _cand_files(tmp_path, union, *, layers=(9, 18, 27), recon=True, recipe=True):
    """A synthetic extraction output: real ids, meaningless reps. Enough to drive the guards."""
    ids = sorted(union)
    lay = list(layers)
    reps = np.zeros((len(ids), len(lay), 1024), dtype=np.float32)
    np.savez_compressed(tmp_path / "cand.npz", statement_ids=np.array(ids, dtype=object),
                        token_lens=np.full(len(ids), 123), reps=reps, layers=np.array(lay))
    meta = {"formality": "NON-FORMAL (tooling check only)",
            "prompt_reconstruction_vs_rollout_inputs": {"pass": recon},
            "extraction_recipe_check": {"pass": recipe}}
    (tmp_path / "cand_meta.json").write_text(json.dumps(meta))
    return tmp_path / "cand.npz", tmp_path / "cand_meta.json"


def _prescore(tmp_path, expect, must_not_exist=None, *flags):
    cmd = [sys.executable, str(PRESCORE), *flags]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode != 0, f"expected a refusal; stderr={r.stderr[-2000:]}"
    assert expect in r.stderr, f"expected {expect!r} in stderr, got: {r.stderr[-2000:]}"
    if must_not_exist is not None:
        assert not Path(must_not_exist).exists(), "a refused scoring must not leave an artifact"
    return r


def test_missing_candidate_reps_is_a_hard_stop_not_a_silent_pool(tmp_path):
    _prescore(tmp_path, "candidate reps missing", tmp_path / "o.json",
              "--cand-reps", str(tmp_path / "nope.npz"), "--cand-meta", str(tmp_path / "nope.json"),
              "--allow-non-formal", "--out", str(tmp_path / "o.json"))


def test_non_formal_reps_are_refused_unless_explicitly_allowed(tmp_path, pool):
    reps, meta = _cand_files(tmp_path, pool["extraction_union"])
    _prescore(tmp_path, "NON-FORMAL", tmp_path / "o.json", "--cand-reps", str(reps),
              "--cand-meta", str(meta), "--out", str(tmp_path / "o.json"))


def test_a_non_formal_scoring_cannot_land_in_experiments(tmp_path, pool):
    reps, meta = _cand_files(tmp_path, pool["extraction_union"])
    decoy = ROOT / "experiments/manifests/v3/V3-R001_predictions_GUARDTEST.json"
    try:
        _prescore(tmp_path, "must not land in experiments/manifests", decoy, "--cand-reps", str(reps),
                  "--cand-meta", str(meta), "--allow-non-formal", "--out", str(decoy))
    finally:
        decoy.unlink(missing_ok=True)


@pytest.mark.parametrize("field,expect", [("recon", "prompt-reconstruction"),
                                          ("recipe", "recipe-identity")])
def test_unverified_extraction_provenance_is_not_scored(tmp_path, pool, field, expect):
    reps, meta = _cand_files(tmp_path, pool["extraction_union"], **{field: False})
    _prescore(tmp_path, expect, tmp_path / "o.json", "--cand-reps", str(reps), "--cand-meta",
              str(meta), "--allow-non-formal", "--out", str(tmp_path / "o.json"))


def test_scoring_a_different_statement_set_than_5_froze_is_refused(tmp_path, pool):
    reps, meta = _cand_files(tmp_path, pool["extraction_union"][:-1])
    _prescore(tmp_path, "refusing to score a different set", tmp_path / "o.json", "--cand-reps",
              str(reps), "--cand-meta", str(meta), "--allow-non-formal", "--out",
              str(tmp_path / "o.json"))


def test_a_layer_surface_that_drifted_from_d001_is_refused(tmp_path, pool):
    """Block 27 is the robustness arm the pre-registration names; an extraction that silently omits
    it must be refused at load time, not indexed into a ValueError several guards later."""
    reps, meta = _cand_files(tmp_path, pool["extraction_union"], layers=(9, 18))
    _prescore(tmp_path, "the pre-registered surface needs", tmp_path / "o.json", "--cand-reps",
              str(reps), "--cand-meta", str(meta), "--allow-non-formal", "--out",
              str(tmp_path / "o.json"))


def test_a_candidate_from_a_family_v1_already_touched_is_refused(tmp_path, pool):
    """The pool is family-clean BY CLAIM; this guard is what makes the claim load-bearing. A fake
    labelled D001 rep file stands in for the gitignored real one, so the test needs no host state."""
    hist_id = "0113d3bd-4cbc-4cda-8be4-e6f9fbbf02cf"      # a statement V1/D001 actually rolled out
    np.savez_compressed(tmp_path / "hist.npz", statement_ids=np.array([hist_id], dtype=object),
                        token_lens=np.array([200]), reps=np.zeros((1, 3, 1024), dtype=np.float32),
                        layers=np.array([9, 18, 27]))
    (tmp_path / "hist_meta.json").write_text(json.dumps({"reps_content_sha256": "0" * 64,
                                                         "formality": "FORMAL (guard test)",
                                                         "host": {}}))
    tampered = json.loads(json.dumps(pool))
    tampered["extraction_union"] = sorted(pool["extraction_union"][:-1] + [hist_id])
    (tmp_path / "pool.json").write_text(json.dumps(tampered))
    reps, meta = _cand_files(tmp_path, tampered["extraction_union"])
    _prescore(tmp_path, "already appear in the labelled D001 rep set", tmp_path / "o.json",
              "--cand-reps", str(reps), "--cand-meta", str(meta), "--pool", str(tmp_path / "pool.json"),
              "--d001-reps", str(tmp_path / "hist.npz"), "--d001-meta", str(tmp_path / "hist_meta.json"),
              "--allow-non-formal", "--out", str(tmp_path / "o.json"))


# --------------------------------------------------------------------------------------------------
# the committed artifact, once the fly122 formal scoring exists
# --------------------------------------------------------------------------------------------------

def test_the_frozen_head_is_the_head_that_earned_the_d001_go(art):
    rep = art["controller"]["frozen_protocol_replication"]
    for arm in ("B1", "B2"):
        assert rep[arm]["mismatched_folds"] == []
        assert rep[arm]["folds_reproduced"] == rep[arm]["n_folds"] == 5, arm
    assert "never updated" in art["controller"]["no_refit_promise"]


def test_every_candidate_has_a_usable_q_and_they_are_all_prospective(art, pool):
    cs = art["candidate_scores"]
    assert len(cs) == len(pool["extraction_union"]) == art["coverage"]["n_candidates"]
    assert art["coverage"]["n_missing_q"] == 0 and art["coverage"]["n_with_q"] == len(cs)
    for sid, c in cs.items():
        assert c["has_historical_label"] is False, sid
        assert c["step_norm_used"] == 0.0, sid
        for k in ("q_B2_controller", "q_B1_handcrafted", "q_B2_fold_ensemble", "q_B1_fold_ensemble"):
            assert 0.0 <= c[k] <= 1.0 and np.isfinite(c[k]), f"{sid} {k} undefined -- owner §6 forbids it"


def test_frozen_samples_are_one_per_component_and_nested_in_the_frozen_order(art, pool):
    for key, s in art["prospective_samples"].items():
        assert len(set(s["component_ids"])) == s["N"] == len(s["statement_ids"]), key
        assert set(s["statement_ids"]) <= set(pool["extraction_union"]), key
        assert set(s["component_ids"]) <= set(
            pool["pools_by_reading"][s["reading"]]["candidate_component_ids"]), key
    by_reading: dict[str, list] = {}
    for s in art["prospective_samples"].values():
        by_reading.setdefault(s["reading"], []).append((s["N"], s["component_ids"]))
    for reading, lst in by_reading.items():
        lst.sort()
        for (_, small), (_, big) in itertools.pairwise(lst):
            assert big[:len(small)] == small, f"{reading}: samples are not nested prefixes"


def test_source_concentration_is_recorded_before_any_outcome(art):
    pre = art["pre_outcome_source_concentration"]
    assert sum(pre["candidates_by_source"].values()) == len(art["candidate_scores"])
    top = round(0.20 * len(art["candidate_scores"]))
    for arm in ("B2", "B1"):
        assert sum(pre["top20pct_block_source_mix_of_union"][arm].values()) == top
        assert set(pre["mean_q_by_source"][arm]) == set(pre["candidates_by_source"])
    assert art["gate_design_from_power"]["b2_vs_b1_significance_required"] is False
    assert "descriptive" in pre["consequence"], "an unpowered stratum must be promised as descriptive"
