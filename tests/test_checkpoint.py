"""Tests for the VERL/FSDP checkpoint conversion helpers."""

from __future__ import annotations

import pytest
import torch

from tinylean_rl.inference.checkpoint import (
    compare_state_dicts_exact,
    export_hf_weights,
    load_safetensors_state_dict,
    state_dict_diff_report,
    tensor_diff_stats,
    verify_key_compat,
)


def _state_dict(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.full((4, 4), 1.0 + offset),
        "lm_head.weight": torch.eye(4),
    }


def test_key_compat_accepts_identical_sets_and_rejects_mismatches():
    verify_key_compat(_state_dict(), set(_state_dict().keys()), "ok")
    with pytest.raises(ValueError):
        verify_key_compat(_state_dict(), {"model.embed_tokens.weight"}, "missing")


def test_export_roundtrip_is_bitwise_identical(tmp_path):
    state = _state_dict()
    output = tmp_path / "model.safetensors"
    export_hf_weights(state, output)
    reloaded = load_safetensors_state_dict(output)
    report = compare_state_dicts_exact(reloaded, state)
    assert report["identical"] is True
    assert report["equal_keys"] == len(state)


def test_exact_compare_detects_a_mismatch(tmp_path):
    reference = _state_dict()
    candidate = dict(reference)
    candidate["model.embed_tokens.weight"] = candidate["model.embed_tokens.weight"] + 0.5
    report = compare_state_dicts_exact(candidate, reference)
    assert report["identical"] is False
    assert report["first_mismatch"] == "model.embed_tokens.weight"


def test_tensor_diff_stats_known_values():
    a = torch.ones(2, 2)
    b = torch.full((2, 2), 0.5)
    stats = tensor_diff_stats(a, b)
    assert stats["max_abs"] == pytest.approx(0.5)
    assert stats["mean_abs"] == pytest.approx(0.5)
    # ||a - b|| / ||b|| = 1.0 here (both tensors have equal Frobenius norm)
    assert stats["rel_l2"] == pytest.approx(1.0)


def test_state_dict_diff_report_tracks_changed_keys():
    reference = _state_dict()
    candidate = dict(reference)
    candidate["model.embed_tokens.weight"] = reference["model.embed_tokens.weight"] + 0.25
    report = state_dict_diff_report(candidate, reference)
    assert report["keys"] == 2
    assert report["changed_keys"] == 1
    assert report["aggregate"]["max_abs"] == pytest.approx(0.25)
    assert report["aggregate"]["rel_l2"] > 0
    assert "model.embed_tokens.weight" in report["highlight_tensors"]
