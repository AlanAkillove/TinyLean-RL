"""Fixtures for the R001 online credit chain (draft §7.4, §10.4 S3).

Two layers:

* pure fixtures (no artifacts, no verifier) pinning the online-only rules: the id round-trip gate,
  the payload/summary shape, no-code and infra handling, and the tensor stage fed by replayed
  positions (λ = 0 bit-for-bit V1, λ = 1 credit at the mapped positions only);
* a **replay** fixture over the frozen Phase-B artifacts: every stored candidate whose response
  bytes are on disk is pushed through the online chain and must reproduce the frozen
  ``token_credit`` positions exactly. This is the no-drift guarantee between the offline oracle
  that produced ``runs/v5_p001_process/v5_p001_process_labels.jsonl`` and the online path the
  training loop will use. It is skipped when the (gitignored) raw artifacts are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_process_oracle as O
import v5_r001_online_credit as C

from tinylean_rl.rl.grpo import compute_grpo_outcome_advantage
from tinylean_rl.rl.process_credit import LAMBDA_PRIMARY, compute_process_advantage

TOKENIZER_DIR = ROOT / "models/weights/kimina_distill_0_6b"
LABELS = ROOT / "runs/v5_p001_process/v5_p001_process_labels.jsonl"
RAW_DIR = ROOT / "runs/v5_p001_process/raw"


@pytest.fixture(scope="module")
def mapper():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("frozen tokenizer not present")
    return O.TokenMapper()


def _verdict_item(messages=(), sorries=(), infotree=()) -> dict:
    return {"custom_id": "fixture", "response": {"messages": list(messages), "sorries": list(sorries), "infotree": list(infotree)}}


def test_strip_special_ids_removes_only_specials(mapper) -> None:
    specials = mapper.tokenizer.all_special_ids
    ids = [10, 20]
    probe = [*ids, *specials[:2], 30]
    assert C.strip_special_ids(probe, mapper.tokenizer) == [*ids, 30]


#: A real non-canonical id sequence for the frozen tokenizer: decoding it and re-encoding the
#: text yields ``[..., 35884, 66256]`` instead of ``[..., 127383]`` (a BPE merge the original ids
#: did not take). Exactly the situation the round-trip gate exists for.
NON_ROUNDTRIP_IDS = [110250, 10612, 67873, 134027, 127383]


def test_round_trip_mismatch_drops_credit_and_counts(mapper) -> None:
    _, lattice, status = C.decode_and_lattice(NON_ROUNDTRIP_IDS, mapper=mapper)
    assert status == C.RT_MISMATCH and lattice is None

    result = C.credit_for_candidate(
        token_ids=NON_ROUNDTRIP_IDS,
        formal_statement="theorem t : True := by sorry",
        item=_verdict_item(),
        mapper=mapper,
        candidate_id="fixture:mismatch",
    )
    assert result["status"] == C.RT_MISMATCH
    assert result["positions"] == {}
    assert result["valid"] == 1  # the outcome channel is untouched by the credit censor
    assert result["counters"]["retokenization_mismatch"] == 1
    assert result["counters"]["credit_tokens"] == 0


def test_round_trip_gate_passes_on_canonical_ids(mapper) -> None:
    text = "```lean4\nby\n  sorry\n```"
    ids = mapper.tokenizer(text, add_special_tokens=False)["input_ids"]
    decoded, lattice, status = C.decode_and_lattice(ids, mapper=mapper)
    assert status == C.RT_OK and lattice is not None
    assert decoded == text
    assert list(lattice["ids"]) == ids


def test_missing_item_and_infra_are_censored(mapper) -> None:
    """An *extractable* pred whose verifier item is missing/transport-broken is infra-censored."""

    text = "```lean4\ntheorem t : True := by sorry\n```"
    ids = mapper.tokenizer(text, add_special_tokens=False)["input_ids"]
    for item, detail in (
        (None, "no_item"),
        ({"error": "timed out"}, "infrastructure"),  # transport-style item error -> infrastructure
    ):
        result = C.credit_for_candidate(
            token_ids=ids,
            formal_statement="theorem t : True := by sorry",
            item=item,
            mapper=mapper,
            candidate_id="fixture:infra",
        )
        assert result["valid"] == 0
        assert result["positions"] == {}
        assert result["detail"] == detail
        assert result["counters"]["retokenization_mismatch"] == 0


def test_sentinel_without_item_is_a_valid_no_code_record(mapper) -> None:
    """V1-skipped sentinels never touch the verifier: valid FORMAT_NO_CODE, no censoring."""

    for text in ("<think>nothing here</think>", "```lean4\nby\n  sorry\n```"):
        ids = mapper.tokenizer(text, add_special_tokens=False)["input_ids"]
        result = C.credit_for_candidate(
            token_ids=ids,
            formal_statement="theorem t : True := by sorry",
            item=None,
            mapper=mapper,
            candidate_id="fixture:sentinel",
        )
        assert result["valid"] == 1, text
        assert result["status"] == "FORMAT_NO_CODE"
        assert result["positions"] == {}
        assert result["counters"]["no_code"] == 1
        assert result["counters"]["retokenization_mismatch"] == 0


def test_no_code_response_is_a_valid_zero_outcome_without_credit(mapper) -> None:
    text = "<think>no lean block here</think>"
    ids = mapper.tokenizer(text, add_special_tokens=False)["input_ids"]
    result = C.credit_for_candidate(
        token_ids=ids,
        formal_statement="theorem t : True := by sorry",
        item=_verdict_item(),
        mapper=mapper,
        candidate_id="fixture:no-code",
    )
    assert result["valid"] == 1
    assert result["status"] == "FORMAT_NO_CODE"
    assert result["positions"] == {}
    assert result["counters"]["no_code"] == 1


def test_payload_and_summary_shapes(mapper) -> None:
    ok = {
        "valid": 1,
        "status": "SUCCESS",
        "positions": {3: "success", 5: "d2"},
        "counters": {
            "mappable_tactics": 2,
            "credit_tokens": 2,
            "ambiguous": 0,
            "outside_response": 0,
            "span_not_in_response": 0,
            "unmapped": 0,
            "retokenization_mismatch": 0,
            "conflicts": 0,
            "no_code": 0,
        },
    }
    censored = {
        "valid": 0,
        "status": "PROCESS_ORACLE_INFRA",
        "positions": {},
        "counters": dict(ok["counters"], credit_tokens=0, mappable_tactics=0),
    }
    payload = C.payload_of(ok)
    assert payload == {"valid": 1, "status": "SUCCESS", "positions": {3: "success", 5: "d2"}}
    summary = C.summarize([ok, censored])
    assert summary["candidates"] == 2 and summary["valid"] == 1
    assert summary["infra_censored"] == 1
    assert summary["credit_tokens"] == 2
    assert summary["statuses"] == {"SUCCESS": 1, "PROCESS_ORACLE_INFRA": 1}


def test_replayed_positions_drive_the_advantage_tensor(mapper) -> None:
    """§10.4's last arrow: credit positions -> advantage tensor, λ = 0 vs λ = 1."""

    width = 12
    rewards = torch.zeros(2, width)
    rewards[0, width - 1] = 1.0
    mask = torch.ones(2, width)
    index = ["g", "g"]
    credit = [{2: "success", 4: "d1"}, {1: "d2"}]
    reference, _ = compute_grpo_outcome_advantage(
        rewards.clone(), mask.clone(), index, norm_adv_by_std_in_grpo=False
    )
    zero, _, _, _ = compute_process_advantage(
        rewards.clone(), mask.clone(), index, [1, 1], credit, lambda_process=0.0
    )
    one, _, _, stats = compute_process_advantage(
        rewards.clone(), mask.clone(), index, [1, 1], credit, lambda_process=LAMBDA_PRIMARY
    )
    assert torch.equal(zero, reference)
    assert one[0, 2].item() == pytest.approx(zero[0, 2].item() + (1.0 - 0.5))
    assert one[1, 1].item() == pytest.approx(zero[1, 1].item() + (-0.10 - 0.5))
    assert one[0, width - 1].item() == zero[0, width - 1].item()
    assert stats.credit_tokens == 3
    assert one[0].abs().sum().item() > 0.0


def _replay_candidates(limit: int) -> list[tuple[dict, str, str, dict]]:
    """(labels entry, response text, formal statement, raw item) for stored candidates."""

    import v5_p001_process_run as PR

    manifest = json.loads(O.S.V1_SOURCES_MANIFEST.read_text())
    formals = PR.load_formal_by_statement_id()
    cache: dict[tuple[str, str], list[str]] = {}
    out: list[tuple[dict, str, str, dict]] = []
    with LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            if len(out) >= limit:
                break
            entry = json.loads(line)
            raw_path = RAW_DIR / (entry["candidate_id"].replace(":", "_") + ".json")
            if not raw_path.exists():
                continue
            key = (entry["seed"], entry["file"])
            if key not in cache:
                source = ROOT / manifest["seeds"][entry["seed"]]["directory"] / entry["file"]
                cache[key] = source.read_text(encoding="utf-8").split("\n")
            rows = cache[key]
            if entry["line"] >= len(rows):
                continue
            response = json.loads(rows[entry["line"]]).get("response")
            if not isinstance(response, str):
                continue
            item = json.loads(raw_path.read_text()).get("item")
            out.append((entry, response, formals.get(entry["statement_id"] or "", "") or "", item))
    return out


def test_stored_phase_b_replay_is_position_exact(mapper) -> None:
    if not LABELS.exists() or not RAW_DIR.exists():
        pytest.skip("frozen Phase-B artifacts not present")
    candidates = _replay_candidates(limit=40)
    if len(candidates) < 10:
        pytest.skip("not enough stored candidates with raw items and response bytes")
    checked_with_tactics = 0
    for entry, response, formal, item in candidates:
        token_ids = mapper.tokenizer(response, add_special_tokens=False)["input_ids"]
        result = C.credit_for_candidate(
            token_ids=token_ids,
            formal_statement=formal,
            item=item,
            mapper=mapper,
            candidate_id=entry["candidate_id"],
        )
        expected: dict[int, str] = {}
        for key, label in (
            ("d1_token_positions", "d1"),
            ("d2_token_positions", "d2"),
            ("success_token_positions", "success"),
        ):
            for position in entry["token_credit"][key]:
                expected[int(position[0])] = label
        assert result["positions"] == expected, entry["candidate_id"]
        if result["counters"]["mappable_tactics"]:
            checked_with_tactics += 1
    assert checked_with_tactics >= 5  # the fixture must include mappable tactics, not just no-ops
