"""Tests for the E024 MiniF2F evaluator and paired analyzer.

Synthetic fixtures only (no GPU, no Lean server): MiniF2F schema loading and
the 244-row guard, the canonical seed schedule, prompt construction, artifact
completeness and the analyzer pairing guards (statement alignment, seed
schedule, samples-per-theorem consistency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import e024_minif2f_analyze as analyzer
import e024_minif2f_eval as evaluator
from promptset_rollout_probe import SYSTEM_PROMPT, USER_TEMPLATE

REAL_MINIF2F = ROOT / "data" / "raw" / "minif2f_hf" / "data" / "train-00000-of-00001.parquet"


def _write_parquet(path: Path, count: int) -> None:
    table = pa.table(
        {
            "name": [f"thm_{index:03d}" for index in range(count)],
            "informal_prefix": [f"/-- informal {index} -/" for index in range(count)],
            "formal_statement": [
                f"import Mathlib\n\ntheorem thm_{index:03d} : True := by\n"
                for index in range(count)
            ],
        }
    )
    parquet.write_table(table, path)


def _records(
    prefix: str, count: int, samples: int, verified_table: dict[int, int]
) -> list[dict]:
    records = []
    for index in range(count):
        statement = f"{prefix}-{index:03d}"
        for sample in range(samples):
            records.append(
                {
                    "theorem_index": index,
                    "sample_index": sample,
                    "statement_id": statement,
                    "sampling_seed": evaluator.seed_for(index, sample),
                    "verified": sample < verified_table.get(index, 0),
                }
            )
    return records


def _artifact(records: list[dict], samples: int, count: int) -> dict:
    return {
        "artifact_type": "e024_minif2f_eval",
        "settings": {"theorems": count, "samples_per_theorem": samples},
        "records": records,
    }


class _CaptureTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict] | None = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.messages = messages
        return "\n".join(message["content"] for message in messages)


def test_minif2f_schema_loading(tmp_path: Path) -> None:
    path = tmp_path / "minif2f.parquet"
    _write_parquet(path, evaluator.MINIF2F_EXPECTED_TEST_SIZE)
    rows = evaluator.load_minif2f_rows(path)
    assert len(rows) == 244
    assert [row["theorem_index"] for row in rows] == list(range(244))
    assert rows[0]["statement_id"] == "thm_000"
    assert rows[0]["formal_statement"].endswith(":= by\n")


def test_row_count_guard(tmp_path: Path) -> None:
    path = tmp_path / "minif2f.parquet"
    _write_parquet(path, 3)
    with pytest.raises(SystemExit):
        evaluator.load_minif2f_rows(path)


def test_schema_column_guard(tmp_path: Path) -> None:
    path = tmp_path / "missing_columns.parquet"
    parquet.write_table(pa.table({"name": ["a"], "formal_statement": ["b"]}), path)
    with pytest.raises(SystemExit):
        evaluator.load_minif2f_rows(path, expected_size=0)


def test_seed_schedule() -> None:
    assert evaluator.seed_for(0, 0) == 20260917
    assert evaluator.seed_for(5, 3) == 20260917 + 5 * 8 + 3
    theorems = [{"theorem_index": 0}, {"theorem_index": 1}]
    schedule = evaluator.candidate_schedule(theorems, 4)
    assert len(schedule) == 8
    assert [entry["sample_index"] for entry in schedule] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert schedule[5]["sampling_seed"] == evaluator.seed_for(1, 1)
    assert len({entry["sampling_seed"] for entry in schedule}) == 8


def test_prompt_construction() -> None:
    row = {
        "theorem_index": 0,
        "statement_id": "t",
        "name": "t",
        "informal_prefix": "INFORMAL_MARKER",
        "formal_statement": "FORMAL_MARKER := by",
    }
    tokenizer = _CaptureTokenizer()
    prompt = evaluator.build_prompt_text(tokenizer, evaluator.prompt_fields(row))
    assert "INFORMAL_MARKER" in prompt
    assert "FORMAL_MARKER := by" in prompt
    assert tokenizer.messages is not None
    assert tokenizer.messages[0]["content"] == SYSTEM_PROMPT
    expected_user = USER_TEMPLATE.format(
        informal_problem="INFORMAL_MARKER", formal_statement="FORMAL_MARKER := by"
    )
    assert tokenizer.messages[1]["content"] == expected_user


def test_artifact_completeness() -> None:
    record = {key: None for key in evaluator.REQUIRED_RECORD_KEYS}
    artifact = {
        "artifact_type": "e024_minif2f_eval",
        "experiment": "E024",
        "host": "h",
        "gpu": "g",
        "git_revision": "r",
        "model_label": "theta0",
        "model_path": "/m",
        "dataset": "/d",
        "dataset_revision": "rev",
        "split": "test",
        "settings": {},
        "summary": {},
        "resources": {},
        "records": [record],
        "created_at_utc": "now",
    }
    assert evaluator.validate_artifact_schema(artifact) == []
    incomplete = dict(artifact)
    del incomplete["host"]
    problems = evaluator.validate_artifact_schema(incomplete)
    assert any("host" in problem for problem in problems)
    bad_record = dict(record)
    del bad_record["taxonomy"]
    bad_artifact = dict(artifact, records=[bad_record])
    problems = evaluator.validate_artifact_schema(bad_artifact)
    assert any("taxonomy" in problem for problem in problems)


def test_analyzer_alignment_pass_and_fail() -> None:
    left = _artifact(_records("stmt", 4, 4, {0: 1, 1: 0}), 4, 4)
    right = _artifact(_records("stmt", 4, 4, {0: 2, 1: 0}), 4, 4)
    analyzer.verify_pairing(left, right)
    shifted = _artifact(_records("different", 4, 4, {}), 4, 4)
    with pytest.raises(SystemExit):
        analyzer.verify_pairing(left, shifted)


def test_samples_per_theorem_mismatch_guard() -> None:
    left = _artifact(_records("stmt", 4, 4, {}), 4, 4)
    right = _artifact(_records("stmt", 4, 8, {}), 8, 4)
    with pytest.raises(SystemExit):
        analyzer.verify_pairing(left, right)


def test_seed_schedule_guard() -> None:
    left = _artifact(_records("stmt", 4, 4, {}), 4, 4)
    tampered = _artifact(
        [
            dict(record, sampling_seed=record["sampling_seed"] + 1)
            for record in _records("stmt", 4, 4, {})
        ],
        4,
        4,
    )
    with pytest.raises(SystemExit):
        analyzer.verify_pairing(left, tampered)


def test_paired_statistics_and_agreement() -> None:
    base = {0: 0, 1: 2, 2: 1}
    treat = {0: 1, 1: 2, 2: 0}
    stats = analyzer.paired_statistics(base, treat, 4)
    assert stats["theta0_verified"] == 3
    assert stats["seed1_verified"] == 3
    assert stats["candidate_delta"] == 0
    assert stats["win_tie_loss"] == {"win": 1, "tie": 1, "loss": 1}
    assert stats["mcnemar"]["newly_solved"] == 1
    assert stats["mcnemar"]["newly_lost"] == 1
    assert stats["theorem_level_bootstrap"]["n"] == 3
    agreement = analyzer.agreement_counts(base, treat)
    assert agreement == {
        "both_solved": 1,
        "only_theta0_solved": 1,
        "only_seed1_solved": 1,
        "neither_solved": 0,
    }


def test_adjudication_credit_correction(tmp_path: Path) -> None:
    records = _records("stmt", 2, 4, {0: 1})
    adjudication = {
        "candidates": [
            {
                "theorem_index": 0,
                "sample_index": 2,
                "final_class": "verified_on_recheck",
            },
            {
                "theorem_index": 1,
                "sample_index": 0,
                "final_class": "deterministic_lean_failure",
            },
        ]
    }
    path = tmp_path / "adjudication.json"
    path.write_text(json.dumps(adjudication), encoding="utf-8")
    credits = analyzer.adjudication_credits(path)
    assert credits == {(0, 2): 1}
    corrected = analyzer.apply_credits(
        analyzer.counts_by_theorem(records), records, credits
    )
    assert corrected[0] == 2
    assert corrected[1] == 0


@pytest.mark.skipif(not REAL_MINIF2F.exists(), reason="local MiniF2F parquet missing")
def test_real_minif2f_file() -> None:
    rows = evaluator.load_minif2f_rows(REAL_MINIF2F)
    assert len(rows) == 244
    assert len({row["statement_id"] for row in rows}) == 244
    assert all(row["formal_statement"].rstrip().endswith(":= by") for row in rows)
