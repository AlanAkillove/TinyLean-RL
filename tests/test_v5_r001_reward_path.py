"""Fixture tests for the R001 online reward path (owner §2, §4, §14; draft §7.4).

The reward function (``scripts/v5_r001_reward.py``) is the only place where online rollouts meet
the oracle. These tests pin its contract with a *fake* oracle client and the frozen tokenizer:

* the two hard guards (ids required, row alignment);
* sentinel preds never reach the verifier and stay a valid ``FORMAT_NO_CODE`` record (V1 skipped
  exactly those strings) - not an infra censor;
* an infrastructure item censors the candidate (``valid = 0``, never a failure), asks
  ``require_healthy``, and fails closed when the oracle itself is down;
* the V1 outcome surface is reconstructed per row: ``score = proof_rw * format_rw`` and
  ``acc = proof_rw``, with the frozen feedback strings;
* S0's static identity probe is bound to the frozen endpoint/phi/lambda/sentinels.

The format gate is frozen V1 code that the reward only *calls*; the score-combination test stubs
its verdict, and two tests run the real gate end-to-end on a canonical response (think block with
matching ``tactics`` blocks + one ``lean4`` block, the historical V1 shape).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "third_party/kimina-prover-rl/recipe/kimina_prover_rl"
for _path in (ROOT / "scripts", ROOT / "src", RECIPE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_p001_spec as S
import v5_process_oracle as O
import v5_r001_reward as R
from kimina_prover_rl.reward.proof_utils import FormatError

from tinylean_rl.rl.process_credit import LAMBDA_PRIMARY, PHI
from tinylean_rl.verifier.policy import VerifierUnhealthyError, VerifyOutcome

TOKENIZER_DIR = ROOT / "models/weights/kimina_distill_0_6b"

FORMAL = "theorem demo : True := by\n  sorry"
PRED = "theorem demo : True := by\n  simp"
#: The historical V1 shape: a think block with matching ``tactics`` blocks, then the ``lean4``
#: block. Passes the frozen :class:`FormatReward` gate (verified in
#: ``test_verified_candidate_scores_one_and_carries_credit``) and extracts exactly ``PRED``.
RESPONSE = """<think>
We need to prove the statement. The goal follows by `simp`.

```tactics
theorem demo : True := by
```

Then close the goal with the canonical tactic.

```tactics
simp
```
</think>
```lean4
theorem demo : True := by
  simp
```
"""
#: ``simp`` sits on pred line 2, columns 2-6 (infotree ranges are numbered in ``pred`` coordinates).
SPAN_SIMP = (2, 2, 2, 6)


@pytest.fixture(scope="module")
def mapper():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("frozen tokenizer not present")
    return O.TokenMapper()


@pytest.fixture(autouse=True)
def _frozen_globals(monkeypatch):
    """Fresh module caches per test: no client, no mapper, no archive handle."""

    monkeypatch.setattr(R, "_CLIENT", None)
    monkeypatch.setattr(R, "_MAPPER", None)
    monkeypatch.setattr(R, "_FORMAT", None)
    yield
    if R._ARCHIVE is not None:
        R._ARCHIVE.close()
    R._ARCHIVE = None
    R._ARCHIVE_PATH = None


class FakeOracle:
    """The oracle surface the reward path uses, without any traffic."""

    def __init__(self, item=None, *, healthy=True):
        self.endpoint = "http://fake-oracle:8020"
        self.calls: list[dict] = []
        self.health_checks = 0
        self._item = item
        self._healthy = healthy

    def verify_raw(self, pred, custom_id):
        self.calls.append({"pred": pred, "custom_id": custom_id})
        item = self._item
        classified = O.classify_result_item(item)
        return {
            "item": item,
            "classified": classified,
            "attempts": [{"attempt": 0, "outcome": classified.outcome.value}],
            "seconds": 0.01,
        }

    def require_healthy(self):
        self.health_checks += 1
        if not self._healthy:
            raise VerifierUnhealthyError("oracle canary failed")


class StubGate:
    """A ``FormatReward`` whose verdict is fixed (the real gate is pinned by V1's own tests)."""

    def __init__(self, error):
        self.error = error
        self.calls: list[tuple] = []

    def check_format_error(self, messages, formal_statement):
        self.calls.append((messages, formal_statement))
        return self.error, None, None


def tactic_node(span, name="Lean.Parser.Tactic.simp", pp="simp"):
    return {
        "node": {
            "name": name,
            "stx": {
                "range": {
                    "start": {"line": span[0], "column": span[1]},
                    "finish": {"line": span[2], "column": span[3]},
                },
                "pp": pp,
            },
            "goalsBefore": ["g"],
            "goalsAfter": [],
        }
    }


def verdict_item(tactics=()):
    return {
        "custom_id": "fixture",
        "response": {"messages": [], "sorries": [], "infotree": list(tactics)},
    }


def ids_of(text, mapper):
    return [int(token_id) for token_id in mapper.tokenizer(text, add_special_tokens=False)["input_ids"]]


def run_reward(monkeypatch, mapper, oracle, rows, *, prompt=None, gate=None):
    """Call the reward with the frozen identities patched to the fixtures."""

    monkeypatch.setattr(R, "token_mapper", lambda: mapper)
    monkeypatch.setattr(R, "oracle_client", lambda: oracle)
    if gate is not None:
        monkeypatch.setattr(R, "format_gate", lambda: gate)
    return R.reward(
        data_sources=["fixture"] * len(rows),
        solution_strs=["ignored"] * len(rows),
        ground_truths=[formal for _, formal, _ in rows],
        extra_infos=[
            {
                "formal_statement": formal,
                "index": index,
                "prompt": prompt if prompt is not None else [{"role": "user", "content": "prove it"}],
            }
            for index, formal, _ in rows
        ],
        token_ids=[ids_of(text, mapper) for _, _, text in rows],
    )


def summary_line(capsys) -> dict:
    payload = capsys.readouterr().out.strip().split("R001_REWARD ", 1)[1]
    return json.loads(payload)


def test_reward_requires_token_ids(monkeypatch, mapper) -> None:
    monkeypatch.setattr(R, "token_mapper", lambda: mapper)
    with pytest.raises(RuntimeError, match="token ids"):
        R.reward(
            data_sources=["fixture"],
            solution_strs=["ignored"],
            ground_truths=[FORMAL],
            extra_infos=[{"formal_statement": FORMAL}],
            token_ids=None,
        )


def test_reward_rejects_row_mismatch(monkeypatch, mapper) -> None:
    monkeypatch.setattr(R, "token_mapper", lambda: mapper)
    with pytest.raises(RuntimeError, match="row mismatch"):
        R.reward(
            data_sources=["fixture", "fixture"],
            solution_strs=["a", "b"],
            ground_truths=[FORMAL, FORMAL],
            extra_infos=[{"formal_statement": FORMAL}, {"formal_statement": FORMAL}],
            token_ids=[ids_of(RESPONSE, mapper)],
        )


def test_sentinel_rows_never_reach_the_oracle(monkeypatch, mapper, capsys) -> None:
    text = "<think>no lean block here</think>"
    oracle = FakeOracle()
    out = run_reward(monkeypatch, mapper, oracle, [(0, FORMAL, text)])

    assert oracle.calls == [] and oracle.health_checks == 0
    assert out[0]["pred"] == S.SENTINEL_NO_PROOF
    assert out[0]["score"] == 0.0 and out[0]["acc"] == 0.0
    assert out[0]["process_credit"] == {"valid": 1, "status": "FORMAT_NO_CODE", "positions": {}}
    assert out[0]["process_counters"]["no_code"] == 1
    assert out[0]["process_counters"]["infra_censored"] == 0
    assert out[0]["tool_feedback"] == "filtered proof."
    summary = summary_line(capsys)
    assert summary["submitted"] == 0 and summary["rows"] == 1
    assert summary["infra_censored"] == 0 and summary["statuses"] == {"FORMAT_NO_CODE": 1}


def test_verified_candidate_scores_one_and_carries_credit(monkeypatch, mapper, capsys) -> None:
    oracle = FakeOracle(verdict_item([tactic_node(SPAN_SIMP)]))
    out = run_reward(monkeypatch, mapper, oracle, [(0, FORMAL, RESPONSE)])

    assert len(oracle.calls) == 1 and oracle.calls[0]["pred"] == PRED
    assert oracle.calls[0]["custom_id"] == "r001-0-0"
    assert oracle.health_checks == 0
    row = out[0]
    assert row["format_error"] == FormatError.NONE.value  # the real frozen gate, end-to-end
    assert row["score"] == 1.0 and row["acc"] == 1.0
    assert row["tool_feedback"] == "valid proof found."
    assert row["process_credit"]["valid"] == 1
    assert row["process_credit"]["status"] == "SUCCESS"
    assert row["process_credit"]["positions"] == {60: "success"}
    assert row["process_counters"]["mappable_tactics"] == 1
    assert row["process_counters"]["contained"] == 1

    summary = summary_line(capsys)
    assert summary["submitted"] == 1 and summary["verified"] == 1
    assert summary["positions_total"] == 1
    assert summary["endpoint"] == oracle.endpoint


def test_format_error_zeroes_the_score_but_not_acc(monkeypatch, mapper) -> None:
    oracle = FakeOracle(verdict_item([tactic_node(SPAN_SIMP)]))
    gate = StubGate(FormatError.NO_VALID_THINK_BLOCK)
    out = run_reward(monkeypatch, mapper, oracle, [(0, FORMAL, RESPONSE)], gate=gate)

    assert gate.calls, "the frozen gate must be consulted"
    messages, formal = gate.calls[0]
    assert messages[-1] == {"role": "assistant", "content": RESPONSE}
    assert formal == FORMAL
    assert out[0]["score"] == 0.0
    assert out[0]["acc"] == 1.0
    assert out[0]["format_error"] == FormatError.NO_VALID_THINK_BLOCK.value
    assert out[0]["process_credit"]["valid"] == 1  # the credit channel is independent of format


def test_infrastructure_item_censors_and_checks_health(monkeypatch, mapper, tmp_path, capsys) -> None:
    archive_dir = tmp_path / "archive"
    monkeypatch.setenv(R.ORACLE_ARCHIVE_ENV, str(archive_dir))
    item = {"custom_id": "fixture", "error": "timed out"}
    oracle = FakeOracle(item)
    out = run_reward(monkeypatch, mapper, oracle, [(3, FORMAL, RESPONSE)])

    assert oracle.health_checks == 1, "an infrastructure item must trigger the fail-close canary"
    row = out[0]
    assert row["score"] == 0.0 and row["acc"] == 0.0
    assert row["process_credit"]["valid"] == 0
    assert row["process_credit"]["status"] == VerifyOutcome.VERIFIER_TIMEOUT.value
    assert row["process_credit"]["positions"] == {}
    assert row["process_counters"]["infra_censored"] == 0, "the summary counts it, not the per-row map"
    assert row["tool_feedback"] == VerifyOutcome.VERIFIER_TIMEOUT.value
    assert row["pred"] == PRED

    files = list(archive_dir.glob("*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["custom_id"] == "r001-3-0" and record["row"] == 0
    assert record["outcome"] == VerifyOutcome.VERIFIER_TIMEOUT.value
    assert record["pred_sha256"] == S.sha256_text(PRED)
    assert record["item"] == item
    assert record["attempts"] == [{"attempt": 0, "outcome": VerifyOutcome.VERIFIER_TIMEOUT.value}]

    summary = summary_line(capsys)
    assert summary["archive"] == str(files[0])
    assert summary["infra_censored"] == 1 and summary["verified"] == 0


def test_unhealthy_oracle_aborts_the_step(monkeypatch, mapper) -> None:
    oracle = FakeOracle({"error": "timed out"}, healthy=False)
    with pytest.raises(VerifierUnhealthyError):
        run_reward(monkeypatch, mapper, oracle, [(0, FORMAL, RESPONSE)])
    assert oracle.health_checks == 1


def test_static_identity_binds_the_frozen_stack(monkeypatch, mapper) -> None:
    monkeypatch.setattr(R, "token_mapper", lambda: mapper)
    monkeypatch.setenv(R.ORACLE_ENDPOINT_ENV, "http://127.0.0.1:8020")
    identity = R.static_identity()

    assert identity["endpoint"] == "http://127.0.0.1:8020"
    assert identity["endpoint_env"] == "http://127.0.0.1:8020"
    assert identity["oracle_container"] == S.ORACLE_INFRA["container"]
    assert identity["oracle_image_digest"] == S.ORACLE_INFRA["image_digest"]
    assert identity["phi"] == dict(PHI)
    assert identity["lambda_primary"] == LAMBDA_PRIMARY
    assert identity["sentinels"] == list(S.PRED_SENTINELS)
    assert identity["tokenizer_expected_vocab"] == S.TOKENIZER_EXPECTED_VOCAB
    assert identity["tokenizer_dir"] == str(S.TOKENIZER_DIR)
