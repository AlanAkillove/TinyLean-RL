"""Tests for the V3-D001 dependency-free library (deterministic grouped CV + metrics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import (
    B1_FEATURES,
    B1_SOURCE_LEVELS,
    average_precision,
    brier,
    ece_equal_mass,
    extract_b1,
    fit_logistic,
    predict_proba_logistic,
    roc_auc,
    stratified_group_kfold,
    topk_enrichment,
)


def test_group_disjointness_hard():
    # 30 groups, 20 rows each; a group must never straddle folds
    k = 5
    groups = np.repeat(np.arange(30), 20)
    y = (np.arange(len(groups)) % 7 == 0).astype(int)
    fold = stratified_group_kfold(y, groups, k, seed=20260923)
    fg = {g: set(fold[groups == g]) for g in np.unique(groups)}
    assert all(len(s) == 1 for s in fg.values()), "a component leaked across folds"
    assert set(fold.tolist()) <= set(range(k))


def test_determinism_same_seed():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, 400)
    y = (rng.random(400) < 0.15).astype(int)
    a = stratified_group_kfold(y, groups, 5, seed=20260923)
    b = stratified_group_kfold(y, groups, 5, seed=20260923)
    assert np.array_equal(a, b)
    # different seed -> (almost surely) a different, still-valid partition
    c = stratified_group_kfold(y, groups, 5, seed=1)
    assert not np.array_equal(a, c)


def test_no_statement_level_split():
    # every row of a group shares the fold (guards against per-row stratification regressions)
    groups = np.array(["a", "a", "b", "b", "b", "c", "d", "d"])
    y = np.array([1, 0, 1, 1, 0, 0, 0, 1])
    fold = stratified_group_kfold(y, groups, 2, seed=20260923)
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1


def test_stratification_balances_positives():
    # imbalanced labels; per-fold positive count should be within tolerance of equal split
    groups = np.repeat(np.arange(100), 10)
    y = np.zeros(len(groups), dtype=int)
    y[::5] = 1  # 20% positives, spread across groups
    k = 5
    fold = stratified_group_kfold(y, groups, k, seed=20260923)
    pos_per_fold = np.array([y[fold == f].sum() for f in range(k)])
    assert pos_per_fold.max() - pos_per_fold.min() <= 6
    assert pos_per_fold.sum() == y.sum()


def test_single_group_not_larger_than_fold_guard():
    # a dominant group must still land entirely in one fold
    groups = np.array([0] * 50 + [1, 2, 3, 4, 5])
    y = np.array([1] * 50 + [0, 1, 0, 1, 0])
    fold = stratified_group_kfold(y, groups, 3, seed=20260923)
    assert len(set(fold[groups == 0])) == 1


def test_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.1])
    assert roc_auc(y, p) == 1.0
    assert average_precision(y, p) == 1.0
    assert abs(brier(y, p) - np.mean((p - y) ** 2)) < 1e-12
    # perfect ranking enrichment: top 50% are the two positives -> 2x prevalence
    assert abs(topk_enrichment(y, p, 0.5) - 2.0) < 1e-9
    assert 0.0 <= ece_equal_mass(y, p, bins=2) <= 1.0


def test_metrics_auprc_partial():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.7, 0.3, 0.2])
    # sorted by p desc -> labels [1,0,1,0]; AP = (1/2)*(1 + 2/3)
    assert abs(average_precision(y, p) - (1 / 2 * (1.0 + 2 / 3))) < 1e-9
    # ROC AUC = 0.75 for this classic case
    assert abs(roc_auc(y, p) - 0.75) < 1e-9


def test_logistic_recovers_sign_and_calibration_direction():
    rng = np.random.default_rng(7)
    n, d = 500, 3
    X = rng.normal(size=(n, d))
    w_true = np.array([0.0, 2.0, -2.0, 0.5])
    logit = w_true[0] + X @ w_true[1:]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    w = fit_logistic(X, y, C=1.0)
    assert w[1] > 0 and w[2] < 0  # sign recovered
    p = predict_proba_logistic(X, w)
    # better-than-chance separation
    assert roc_auc((y > 0.5).astype(int), p) > 0.75


def test_extract_b1_shape_determinism_and_source_onehot():
    formal = ("theorem foo (a b : Nat) (h : a > 0) : \u2203 c, a + b = c := by\n  sorry")
    v = extract_b1(formal, prompt_token_count=42, source="synthetic", step_norm=0.5)
    assert v.shape == (len(B1_FEATURES),)
    assert np.all(np.isfinite(v))
    # deterministic: identical inputs -> identical vector
    v2 = extract_b1(formal, prompt_token_count=42, source="synthetic", step_norm=0.5)
    assert np.array_equal(v, v2)
    # pass-through numeric features land in the right slots
    idx = {n: i for i, n in enumerate(B1_FEATURES)}
    assert v[idx["prompt_token_count"]] == 42.0
    assert v[idx["step_norm"]] == 0.5
    assert v[idx["has_existential"]] == 1.0  # \u2203 present
    # source one-hot: exactly one active level, correct one set
    onehot = [v[idx[f"source={s}"]] for s in B1_SOURCE_LEVELS]
    assert sum(onehot) == 1.0 and onehot[B1_SOURCE_LEVELS.index("synthetic")] == 1.0
    # unknown source -> all-zero one-hot (no fabricated level)
    v3 = extract_b1(formal, 42, "unknown-x", 0.5)
    assert sum(v3[idx[f"source={s}"]] for s in B1_SOURCE_LEVELS) == 0.0
    # monotone in structural complexity: more binders/parens -> n_paren grows
    rich = "theorem bar (x : Nat) (y : Nat) (h : (x > 0) \u2192 (y > 0)) : \u2203 z, (x + y = z) \u2192 true := by\n  sorry"
    vr = extract_b1(rich, 42, "human", 0.5)
    assert vr[idx["n_paren"]] > v[idx["n_paren"]]
    assert vr[idx["n_arrows"]] > v[idx["n_arrows"]]

