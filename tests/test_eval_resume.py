"""Tests for the resumable chunked-evaluation state validation."""

from __future__ import annotations

import hashlib
import json

import pytest

from tinylean_rl.evaluation.resume import (
    ResumeError,
    chunk_start_offset,
    file_sha256,
    load_resume_state,
)


def _settings(**overrides):
    base = {
        "checkpoint": "base",
        "fixed_set_sha256": "abc",
        "theorems": 8,
        "samples_per_theorem": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 4096,
        "chunk_theorems": 4,
        "seed_base": 20260917,
    }
    base.update(overrides)
    return base


def _records(processed: int, samples: int = 2):
    return [
        {"theorem_index": index, "sample_index": sample}
        for index in range(processed)
        for sample in range(samples)
    ]


def _write_partial(tmp_path, settings, processed, records=None):
    path = tmp_path / "out.partial.json"
    payload = {
        "records": _records(processed) if records is None else records,
        "processed_theorems": processed,
        "settings": settings,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_resume_state_accepts_matching_partial(tmp_path):
    settings = _settings()
    state = load_resume_state(_write_partial(tmp_path, settings, 4), settings)
    assert state.processed_theorems == 4
    assert len(state.records) == 8


def test_load_resume_state_rejects_settings_mismatch(tmp_path):
    path = _write_partial(tmp_path, _settings(), 4)
    with pytest.raises(ResumeError, match="different settings"):
        load_resume_state(path, _settings(temperature=0.6))


def test_load_resume_state_rejects_record_count_mismatch(tmp_path):
    path = _write_partial(tmp_path, _settings(), 4, records=_records(3))
    with pytest.raises(ResumeError, match="inconsistent"):
        load_resume_state(path, _settings())


def test_load_resume_state_rejects_coverage_gap(tmp_path):
    records = _records(4)
    records[5] = {"theorem_index": 0, "sample_index": 1}  # duplicate index 0; index 2 incomplete
    path = _write_partial(tmp_path, _settings(), 4, records=records)
    with pytest.raises(ResumeError, match="cover"):
        load_resume_state(path, _settings())


def test_load_resume_state_rejects_missing_file(tmp_path):
    with pytest.raises(ResumeError, match="not found"):
        load_resume_state(tmp_path / "absent.partial.json", _settings())


def test_chunk_start_offset_rules():
    assert chunk_start_offset(0, 4, 8) == 0
    assert chunk_start_offset(4, 4, 8) == 4
    assert chunk_start_offset(8, 4, 8) == 8
    with pytest.raises(ResumeError):
        chunk_start_offset(3, 4, 8)
    with pytest.raises(ResumeError):
        chunk_start_offset(9, 4, 8)


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello")
    assert file_sha256(path) == hashlib.sha256(b"hello").hexdigest()
