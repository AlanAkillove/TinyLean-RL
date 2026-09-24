"""Tests for the V3-R001 deployment-pool audit's sampler-replay logic.

Only `replicate_sampler()` is under test: it is the part that turns "V1 sampled filtered parquet
rows with RandomSampler(data.seed)" from a reading of the source into a falsifiable claim, so its
comparison rules (per-step multiset, not raw sequence) must be correct. Runs on a synthetic 6-row
"filtered dataframe"; no parquet, tokenizer, GPU or rollout involved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v3_r001_pool_audit.py"


def _load(monkeypatch, batch: int, steps: int):
    spec = importlib.util.spec_from_file_location("v3_r001_pool_audit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "TRAIN_BATCH_SIZE", batch, raising=True)
    monkeypatch.setattr(mod, "V1_STEPS", steps, raising=True)
    return mod


def _replay(mod, kept_order, data_seed, realized, order_within_step):
    logs = {"seed1": {"resolved_data_config": {"data_seed_key_present": data_seed is not None,
                                               "data_seed_value": data_seed}}}
    recs = []
    for step, stmts in realized.items():
        seq = sorted(stmts) if order_within_step == "sorted" else list(stmts)
        for s in seq:
            recs.append({"seed": "seed1", "step": step, "statement_id": s})
    return mod.replicate_sampler(logs, kept_order, recs, filter_recomputed=True)


def _truth(monkeypatch, data_seed):
    """Build history that DID come from the account, then check the replay says so."""
    mod = _load(monkeypatch, batch=2, steps=2)
    kept = ["s0", "s1", "s2", "s3", "s4", "s5"]
    perm = torch.randperm(len(kept), generator=torch.Generator().manual_seed(data_seed)).tolist()[:4]
    drawn = [kept[i] for i in perm]
    realized = {1: drawn[0:2], 2: drawn[2:4]}
    return mod, kept, realized


def test_replay_confirms_an_account_that_generated_the_history(monkeypatch):
    mod, kept, realized = _truth(monkeypatch, 7)
    out = _replay(mod, kept, 7, realized, order_within_step="sorted")
    assert out["performed"] is True
    assert out["filtered_rows_used"] == 6
    assert out["draws_expected_per_seed"] == 4
    assert out["seeds_tested"] == 1
    assert out["seeds_where_set_matches"] == 1
    assert out["seeds_where_batching_matches"] == 1
    assert out["seeds_where_realized_equals_sorted_replay"] == 1
    ps = out["per_seed"]["seed1"]
    assert ps["extra_in_replay_not_realized"] == []
    assert ps["extra_in_realized_not_replay"] == []


def test_within_step_order_is_not_treated_as_a_mismatch(monkeypatch):
    """build_records() sorts each step's group; that must not look like a failed replay."""
    mod, kept, realized = _truth(monkeypatch, 7)
    scrambled = {s: list(reversed(v)) for s, v in realized.items()}
    out = _replay(mod, kept, 7, scrambled, order_within_step="unsorted")
    assert out["seeds_where_batching_matches"] == 1
    assert out["per_seed"]["seed1"]["every_step_holds_the_same_4_theorems"] is True


def test_wrong_data_seed_is_reported_as_a_failure(monkeypatch):
    mod, kept, realized = _truth(monkeypatch, 7)
    out = _replay(mod, kept, 8, realized, order_within_step="sorted")
    assert out["seeds_where_set_matches"] == 0
    assert out["seeds_where_batching_matches"] == 0
    ps = out["per_seed"]["seed1"]
    assert ps["extra_in_replay_not_realized"] or ps["extra_in_realized_not_replay"]


def test_absent_data_seed_key_replays_with_the_documented_default_1(monkeypatch):
    """main_ppo.py uses data_config.get("seed", 1) when the key is absent; seed1's log has no key."""
    mod, kept, realized = _truth(monkeypatch, 1)
    out = _replay(mod, kept, None, realized, order_within_step="sorted")
    assert out["seeds_where_set_matches"] == 1
    ps = out["per_seed"]["seed1"]
    assert ps["data_seed_configured"] is False
    assert ps["data_seed_used_in_replay"] == 1


def test_moved_statement_in_one_step_breaks_only_the_per_step_test(monkeypatch):
    mod, kept, realized = _truth(monkeypatch, 7)
    realized = {k: list(v) for k, v in realized.items()}
    realized[1][0], realized[2][0] = realized[2][0], realized[1][0]
    out = _replay(mod, kept, 7, realized, order_within_step="sorted")
    assert out["seeds_where_set_matches"] == 1          # same 4 rows overall
    assert out["seeds_where_batching_matches"] == 0     # but not the same step composition


def test_skipped_tokenize_cannot_replay(monkeypatch):
    mod = _load(monkeypatch, batch=2, steps=2)
    out = mod.replicate_sampler({}, None, [], filter_recomputed=False)
    assert out["performed"] is False
    assert "cannot be replayed" in out["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
