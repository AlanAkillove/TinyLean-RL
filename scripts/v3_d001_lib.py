#!/usr/bin/env python3
"""V3-D001 core library (dependency-free: numpy only).

The fly122 formal node has NO scikit-learn, and sklearn is not a project dependency.
Owner directive §3 permits a deterministic StratifiedGroupKFold *equivalent* with
tests. This module therefore implements, in numpy, everything V3-D001 needs so that
fly90 and fly122 produce bit-identical fold / fit / metric results:

  * group-record construction (label = score-sum informative; infra censoring policy)
  * deterministic StratifiedGroupKFold equivalent (group unit = family component)
  * L2-regularised logistic regression via Newton/IRLS  (no class_weight)
  * AUPRC (average precision), AUROC, Brier, equal-mass ECE, top-k enrichment
  * family-component bootstrap

No randomness other than the explicit, seeded RNG below. No network, no file writes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 1b. B1 handcrafted difficulty features (FROZEN; statement-only, no rollout signal)
# ---------------------------------------------------------------------------
# Fixed, interpretable, deterministic. None of these may touch reward/verifier/rollout.
B1_NUMERIC = [
    "formal_char_count", "formal_line_count", "formal_word_count",
    "n_paren", "n_binders", "n_commas", "n_quant_syms", "n_arrows",
    "has_existential", "prompt_token_count", "step_norm",
]
B1_SOURCE_LEVELS = ["synthetic", "autoformalizer", "human"]
B1_FEATURES = B1_NUMERIC + [f"source={s}" for s in B1_SOURCE_LEVELS]

_QUANT = set("∀∃∃!Σλ⨅⨆→")
_ARROWS = ["->", "→", "↔", "<->", "⇒"]


def _signature(text: str) -> str:
    """Text of the theorem between the declaration and the ':' conclusion (binders live here)."""
    m = re.search(r"(?:theorem|lemma|example)\s+\S+\s*(.*?)\s*:", text, re.DOTALL)
    return m.group(1) if m else ""


def extract_b1(formal_text: str, prompt_token_count: int, source: str | None, step_norm: float) -> np.ndarray:
    sig = _signature(formal_text)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_']*", formal_text)
    n_arrows = sum(formal_text.count(a) for a in _ARROWS)
    feats = {
        "formal_char_count": float(len(formal_text)),
        "formal_line_count": float(formal_text.count("\n") + 1),
        "formal_word_count": float(len(words)),
        "n_paren": float(formal_text.count("(")),
        "n_binders": float(sig.count("(")),
        "n_commas": float(sig.count(",")),
        "n_quant_syms": float(sum(1 for ch in formal_text if ch in _QUANT)),
        "n_arrows": float(n_arrows),
        "has_existential": 1.0 if ("∃" in formal_text or "Exists" in formal_text) else 0.0,
        "prompt_token_count": float(prompt_token_count),
        "step_norm": float(step_norm),
    }
    vec = [feats[n] for n in B1_NUMERIC]
    vec += [1.0 if source == s else 0.0 for s in B1_SOURCE_LEVELS]
    return np.asarray(vec, dtype=float)


# ---------------------------------------------------------------------------
# 1. Record construction  (identical parsing to scripts/v3_data_audit.py)
# ---------------------------------------------------------------------------
FORMAL_BLOCK_RE = re.compile(r"# Formal Statement:\s*\n```lean4\n(.*?)\n```", re.DOTALL)
SEED_DIRS = {"seed1": "runs/p3b_pilot/rollout_data",
             "seed2": "runs/m1_seed2/rollout_data",
             "seed3": "runs/m1_seed3/rollout_data"}
N = 8
STEPS = 60


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _is_syserr(tf: str) -> bool:
    return isinstance(tf, str) and tf.lstrip().startswith("# System Error:")


def formal_text(prompt_text: str) -> str:
    """Normalised Lean4 body from the '# Formal Statement:' fenced block (B1 input, statement-only)."""
    m = FORMAL_BLOCK_RE.search(prompt_text)
    return _normalize(m.group(1)) if m else ""


def _load_statement_map(root: Path, dataset_rel: str) -> dict[str, str]:
    t = pq.read_table(root / dataset_rel, columns=["statement_id", "formal_statement", "source"])
    sid = t.column("statement_id").to_pylist()
    fst = t.column("formal_statement").to_pylist()
    src = t.column("source").to_pylist()
    m: dict[str, tuple[str, str]] = {}
    for s, f, c in zip(sid, fst, src):
        m.setdefault(_normalize(f), (s, c))
    return m


def _load_component_map(root: Path, registry_rel: str) -> dict[str, str]:
    reg = json.loads((root / registry_rel).read_text())
    m: dict[str, str] = {}
    for comp in reg["components"]:
        for member in comp["member_statement_ids"]:
            m[member] = comp["component_id"]
    return m


def build_records(root: Path, dataset_rel: str, registry_rel: str) -> dict:
    """Reconstruct all 720 groups and return {valid, infra_censored, meta}.

    Each record: {seed, step, statement_id, component_id, source, prompt_text,
                  y_score, y_acc, n_infra_candidates, n_pos_score, step_norm, feats(B1 later)}
    Primary valid set excludes any group with >=1 infra-censored candidate.
    """
    sid_map = _load_statement_map(root, dataset_rel)
    comp_map = _load_component_map(root, registry_rel)
    all_groups: list[dict] = []
    prompt_text_by_stmt: dict[str, str] = {}  # statement_id -> prompt (identical across copies)
    for seed, rel in SEED_DIRS.items():
        for fp in sorted((root / rel).glob("*.jsonl"), key=lambda p: int(p.stem)):
            step = int(fp.stem)
            recs = [json.loads(l) for l in fp.read_text(encoding="utf-8").split("\n") if l.strip()]
            i = 0
            while i < len(recs):
                inp = recs[i]["input"]
                j, scores, accs, nsys = i, [], [], 0
                while j < len(recs) and recs[j]["input"] == inp:
                    scores.append(float(recs[j]["score"]))
                    accs.append(float(recs[j]["acc"]))
                    if _is_syserr(recs[j].get("tool_feedback", "")):
                        nsys += 1
                    j += 1
                mm = FORMAL_BLOCK_RE.search(inp)
                norm = _normalize(mm.group(1)) if mm else None
                sid, source = sid_map.get(norm, (None, None))
                comp = comp_map.get(sid)
                prompt_text_by_stmt.setdefault(sid, inp)
                all_groups.append({
                    "seed": seed, "step": step, "statement_id": sid, "component_id": comp,
                    "source": source, "n_group": len(scores), "n_pos_score": int(sum(scores)),
                    "n_pos_acc": int(sum(accs)), "n_infra_candidates": nsys, "size": len(scores),
                })
                i = j
    for g in all_groups:
        g["y_score"] = 1 if 0 < g["n_pos_score"] < g["size"] else 0
        g["y_acc"] = 1 if 0 < g["n_pos_acc"] < g["size"] else 0
        g["step_norm"] = g["step"] / STEPS
        g["infra_tainted"] = g["n_infra_candidates"] > 0
    valid = [g for g in all_groups if not g["infra_tainted"]]
    censored = [g for g in all_groups if g["infra_tainted"]]
    # secondary partial-identifiability: censored groups whose OBSERVED candidates already
    # contain both a 0 and a 1 are mathematically informative regardless of the censored cells.
    partial_identifiable = sum(1 for g in censored if 0 < g["n_pos_score"] < g["size"])
    meta = {
        "total_groups": len(all_groups),
        "valid_groups": len(valid),
        "infra_censored_groups": len(censored),
        "censored_partial_identifiable": partial_identifiable,
        "components": len({g["component_id"] for g in valid}),
        "statements": len({g["statement_id"] for g in valid}),
        "informative_rate_valid": round(sum(g["y_score"] for g in valid) / len(valid), 4),
    }
    return {"valid": valid, "infra_censored": censored, "prompts": prompt_text_by_stmt, "meta": meta}


# ---------------------------------------------------------------------------
# 2. Deterministic StratifiedGroupKFold equivalent
# ---------------------------------------------------------------------------
def stratified_group_kfold(y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Return per-sample fold label in 0..k-1. Hard group disjointness; class-aware.

    Equivalent in spirit to sklearn StratifiedGroupKFold: place the largest groups first
    (rarest-class groups first), then greedily assign each group to the fold whose
    resulting (positive, negative, total) profile best matches the per-fold target.
    Fully deterministic given (y, groups, k, seed) — ties broken by a seeded shuffle then
    by fold index. No sample of a group ever appears in two folds.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows_by_group = {g: np.where(groups == g)[0] for g in uniq}
    pos_by_group = {g: int(y[idx].sum()) for g, idx in rows_by_group.items()}
    size_by_group = {g: len(idx) for g, idx in rows_by_group.items()}
    total_pos = int(y.sum())
    total_n = len(y)
    tgt_pos = total_pos / k
    tgt_neg = (total_n - total_pos) / k
    tgt_tot = total_n / k

    rng = np.random.default_rng(seed)
    order = uniq.copy()
    rng.shuffle(order)  # seeded tie-break among equally-prioritised groups
    # rarest class first: sort groups by (#positives) then by (#size) — largest positive-
    # deficit and largest groups placed earliest (stable via the pre-shuffle above)
    order = sorted(order.tolist(), key=lambda g: (-pos_by_group[g] / max(size_by_group[g], 1), -size_by_group[g]))

    fold_pos = np.zeros(k)
    fold_neg = np.zeros(k)
    fold_tot = np.zeros(k)
    fold_used = [set() for _ in range(k)]
    assign: dict = {}
    # Balanced partition via a convex (quadratic) load potential: minimising the marginal
    # squared load spreads groups onto the least-loaded fold instead of over-filling one.
    # (An absolute-deviation-from-target cost makes empty folds look maximally expensive and
    #  over-concentrates; the quadratic form is the standard LPT/balanced-partition greedy.)
    tp = max(tgt_pos, 1e-9)
    tn = max(tgt_neg, 1e-9)
    tt = max(tgt_tot, 1e-9)
    for g in order:
        np_ = pos_by_group[g]
        nn_ = size_by_group[g] - np_
        best_fold, best_score = None, None
        for f in range(k):
            cost = ((fold_pos[f] + np_) ** 2) / tp + ((fold_neg[f] + nn_) ** 2) / tn \
                   + ((fold_tot[f] + size_by_group[g]) ** 2) / tt
            if best_score is None or cost < best_score - 1e-12:
                best_score, best_fold = cost, f
        fold_pos[best_fold] += np_
        fold_neg[best_fold] += nn_
        fold_tot[best_fold] += size_by_group[g]
        fold_used[best_fold].add(g)
        assign[g] = best_fold
    fold_of_row = np.empty(len(y), dtype=int)
    for g, f in assign.items():
        fold_of_row[rows_by_group[g]] = f
    return fold_of_row


def outer_inner_folds(records: list[dict], n_outer: int, n_inner: int, seed: int) -> dict:
    """Assign each record an outer fold and a per-outer-fold inner fold (group-disjoint)."""
    y = np.array([r["y_score"] for r in records])
    groups = np.array([r["component_id"] for r in records])
    outer = stratified_group_kfold(y, groups, n_outer, seed)
    inner = [None] * len(records)
    for f in range(n_outer):
        tr = np.where(outer != f)[0]
        iy = y[tr]
        ig = groups[tr]
        if len(np.unique(ig)) < n_inner:
            sub = np.zeros(len(tr), dtype=int)
        else:
            sub = stratified_group_kfold(iy, ig, n_inner, seed + 1000 + f)
        for local, gi in enumerate(tr):
            inner[gi] = int(sub[local])
    return {
        "outer_fold": [int(x) for x in outer],
        "inner_fold": [int(x) for x in inner],
        "seed": seed, "n_outer": n_outer, "n_inner": n_inner,
    }


# ---------------------------------------------------------------------------
# 3. L2 logistic regression (Newton / IRLS)
# ---------------------------------------------------------------------------
def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float, max_iter: int = 50, tol: float = 1e-7) -> np.ndarray:
    """Regularised logistic regression: min  sum nll  + (1/(2C))||w||^2 . Returns w (with bias)."""
    X = np.hstack([np.ones((len(X), 1)), X])
    d = X.shape[1]
    lam = 1.0 / max(C, 1e-12)
    w = np.zeros(d)
    for _ in range(max_iter):
        eta = X @ w
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (p - y) + lam * np.concatenate([[0.0], w[1:]])
        W = np.clip(p * (1 - p), 1e-6, None)
        H = (X * W[:, None]).T @ X + lam * np.eye(d)
        H[0, 0] -= lam  # no penalty on bias
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.max(np.abs(step)) < tol:
            w = w_new
            break
        w = w_new
    return w


def predict_proba_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    eta = np.clip(Xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    pos = y_sorted.sum()
    neg = len(y_sorted) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tpr_prev = fpr_prev = 0.0
    auc = 0.0
    tp = fp = 0
    i = 0
    while i < len(y_sorted):
        j = i
        while j < len(y_sorted) and p[order[j]] == p[order[i]]:
            tp += y_sorted[j]
            fp += 1 - y_sorted[j]
            j += 1
        tpr = tp / pos
        fpr = fp / neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr
        i = j
    return float(auc)


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    pos = y_sorted.sum()
    if pos == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    precision = tp / k
    ap = float((precision * y_sorted).sum() / pos)
    return ap


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece_equal_mass(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    order = np.argsort(p)
    e = 0.0
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if len(idx) == 0:
            continue
        e += len(idx) / n * abs(p[idx].mean() - y[idx].mean())
    return float(e)


def topk_enrichment(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    n = len(y)
    k = max(1, round(frac * n))
    order = np.argsort(-p)
    sel = order[:k]
    prevalence = y.mean()
    if prevalence == 0:
        return float("nan")
    return float(y[sel].mean() / prevalence)


# ---------------------------------------------------------------------------
# 5. Family-component bootstrap
# ---------------------------------------------------------------------------
def component_bootstrap(records: list[dict], p_oof: np.ndarray, metric_fn, n_rep: int = 10000, seed: int = 20260923,
                        alpha: float = 0.05):
    comps = np.array([r["component_id"] for r in records])
    uniq = np.unique(comps)
    comp_rows = {c: np.where(comps == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    m = len(uniq)
    for _ in range(n_rep):
        draw = rng.integers(0, m, m)
        idx = np.concatenate([comp_rows[uniq[d]] for d in draw])
        vals.append(metric_fn(idx))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)), vals
