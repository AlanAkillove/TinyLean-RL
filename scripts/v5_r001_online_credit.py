#!/usr/bin/env python3
"""R001 online credit chain: training response ids -> frozen process credit.

Scientific content (draft §7.1-§7.4, owner directive §2-§4): this module *reuses* the frozen V5
process oracle (`scripts/v5_process_oracle.py`) and the frozen credit semantics
(`src/tinylean_rl/rl/process_credit.py`) on the online batch. Nothing new is invented here:

* ``derive_facts`` is called with exactly the arguments Phase B used (``pred``, ``response_text``,
  ``formal_statement``, the verifier item, the token lattice), so tactic extraction, blame, labels
  and the tactic -> response-offset mapping are the *same code* that produced
  ``runs/v5_p001_process/v5_p001_process_labels.jsonl``;
* ``assign_positions`` is the frozen collision/label -> position rule, pinned against the offline
  oracle's ``_token_credit`` by ``tests/test_v5_r001_process_credit.py``.

What is online-specific (draft §7.4) and lives only here:

1. the token lattice is built from the *decoded training response* and must reproduce the training
   ids exactly (modulo special tokens, which ``skip_special_tokens=True`` decoding removes). Any
   disagreement sets the candidate's process term to zero and counts ``retokenization_mismatch`` -
   nothing is silently shifted;
2. the response text is the decoded completion (offset 0 = first generated token), which is the
   frame the credit positions must live in (``dp_actor.py:247``);
3. an infrastructure outcome censors the candidate (``valid = 0``) with no process credit at all.
   A candidate that was never submitted because its ``pred`` is a sentinel (V1 skipped exactly
   these two strings) is *not* censored: it is the frozen ``FORMAT_NO_CODE`` record - a valid
   zero-outcome candidate with no process tokens (``no_code`` counter).

The per-candidate result is a small payload that the reward path stores in
``non_tensor_batch["process_credit"]`` and the advantage wrapper
(``src/tinylean_rl/rl/r001_advantage.py``) consumes:

    {"valid": 1|0, "status": str, "positions": {token_position: label}}

``positions`` keys are indices into the batch ``responses`` row (0 = first generated token), which
is exactly the frame ``compute_process_advantage`` writes into ``advantages``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
_UPSTREAM = ROOT / "third_party" / "kimina-prover-rl" / "recipe" / "kimina_prover_rl"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

import v5_p001_spec as S
import v5_process_oracle as O
from kimina_prover_rl.reward.proof_utils import extract_proof_from_text

from tinylean_rl.rl.process_credit import CreditStats, assign_positions

#: Lattice round-trip outcomes (the online-only check of draft §7.4).
RT_OK = "ok"
RT_MISMATCH = "retokenization_mismatch"

#: Status recorded for an infrastructure outcome or a missing verifier item.
STATUS_INFRA = "PROCESS_ORACLE_INFRA"


def strip_special_ids(token_ids: Sequence[int], tokenizer: Any) -> list[int]:
    """The ids that survive ``skip_special_tokens=True`` decoding, in order."""

    specials = set(tokenizer.all_special_ids or [])
    return [int(token_id) for token_id in token_ids if int(token_id) not in specials]


def decode_and_lattice(
    token_ids: Sequence[int], *, mapper: Any
) -> tuple[str, dict[str, Any] | None, str]:
    """Decode the training ids and build the lattice, gated on exact id round-trip.

    Returns ``(response_text, lattice_or_None, status)``. ``status`` is ``RT_OK`` only when
    re-tokenizing the decoded text reproduces the non-special training ids element for element; on
    mismatch the lattice is discarded so no credit can be derived from a different tokenization.
    """

    core = strip_special_ids(token_ids, mapper.tokenizer)
    text = mapper.tokenizer.decode(core, skip_special_tokens=False)
    lattice = mapper.lattice(text)
    if list(lattice["ids"]) != core:
        return text, None, RT_MISMATCH
    return text, lattice, RT_OK


def credit_for_candidate(
    *,
    token_ids: Sequence[int],
    formal_statement: str,
    item: Mapping[str, Any] | None,
    mapper: Any,
    candidate_id: str = "",
    attempts: int = 0,
    seconds: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The full online chain for one candidate (pure CPU; the verifier call happens upstream).

    ``item`` is the raw ``/verify`` result body (the shape stored under ``item`` in
    ``runs/v5_p001_process/raw/*.json``). Returns the payload dict
    (``valid``/``status``/``positions``) plus the mapping-stage counters and the frozen oracle
    record.

    ``item is None`` has one meaning per pred kind, mirroring the frozen Phase-B runner
    (``v5_p001_process_run.py::process_one``): a sentinel pred (no code to submit, V1 skipped the
    same two strings) yields the valid ``FORMAT_NO_CODE`` record with ``no_code`` counted, while
    an extractable pred that produced no conclusive item is an infrastructure censor
    (``valid = 0``).
    """

    stats = CreditStats()
    text, lattice, rt_status = decode_and_lattice(token_ids, mapper=mapper)
    pred = extract_proof_from_text(text, formal_statement)
    bases: dict[str, Any] = {
        "candidate_id": candidate_id,
        "response_text": text,
        "positions": {},
        "counters": stats.as_dict(),
    }

    if item is None and pred in S.PRED_SENTINELS:
        record = O.derive_facts(
            candidate_id=candidate_id,
            pred=pred,
            response_text=text,
            formal_statement=formal_statement,
            oracle_result=None,
            token_lattice=None,
            meta=dict(meta or {}),
        )
        stats.no_code += 1
        bases.update(
            {
                "valid": 1,
                "status": record["process_status"],
                "detail": record["failure_kind"],
                "record": record,
                "counters": stats.as_dict(),
            }
        )
        return bases

    if item is None:
        bases.update({"valid": 0, "status": STATUS_INFRA, "detail": "no_item", "record": None})
        return bases

    classified = O.classify_result_item(item)
    if classified.outcome.is_infrastructure:
        bases.update(
            {
                "valid": 0,
                "status": classified.outcome.value,
                "detail": "infrastructure",
                "record": None,
            }
        )
        return bases

    oracle_result = {
        "item": dict(item),
        "classified": classified,
        "attempts": [{"attempt": 0, "outcome": classified.outcome.value} for _ in range(attempts)],
        "seconds": seconds,
    }

    if rt_status != RT_OK:
        # The process term is zeroed and counted; the outcome channel is unaffected.
        record = O.derive_facts(
            candidate_id=candidate_id,
            pred=pred,
            response_text=text,
            formal_statement=formal_statement,
            oracle_result=oracle_result,
            token_lattice=None,
            meta=dict(meta or {}),
        )
        stats.retokenization_mismatch += 1
        bases.update(
            {
                "valid": 1,
                "status": RT_MISMATCH,
                "detail": record["process_status"],
                "record": record,
                "counters": stats.as_dict(),
            }
        )
        return bases

    record = O.derive_facts(
        candidate_id=candidate_id,
        pred=pred,
        response_text=text,
        formal_statement=formal_statement,
        oracle_result=oracle_result,
        token_lattice=lattice,
        meta=dict(meta or {}),
    )
    if not record["code"]["extractable"]:
        stats.no_code += 1
    positions, stats = assign_positions(record["tactics"], stats=stats)
    bases.update(
        {
            "valid": 1,
            "status": record["process_status"],
            "detail": record["failure_kind"],
            "record": record,
            "positions": positions,
            "counters": stats.as_dict(),
        }
    )
    return bases


def payload_of(result: Mapping[str, Any]) -> dict[str, Any]:
    """The compact per-row payload written to ``non_tensor_batch["process_credit"]``."""

    return {
        "valid": int(result["valid"]),
        "status": str(result["status"]),
        "positions": {int(position): str(label) for position, label in result["positions"].items()},
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the S3 counter set (owner §14) over per-candidate results.

    ``infra_censored`` is counted from ``valid == 0``; the mapping-stage counters come from the
    frozen ``CreditStats`` of each candidate, so no counter is incremented twice.
    """

    totals: dict[str, int] = {
        "candidates": 0,
        "valid": 0,
        "infra_censored": 0,
        "mapped": 0,
        "credit_tokens": 0,
        "ambiguous": 0,
        "outside_response": 0,
        "span_not_in_response": 0,
        "unmapped": 0,
        "retokenization_mismatch": 0,
        "conflict": 0,
        "no_code": 0,
    }
    statuses: dict[str, int] = {}
    for result in results:
        totals["candidates"] += 1
        totals["valid"] += int(result["valid"])
        totals["infra_censored"] += 1 - int(result["valid"])
        counters = result["counters"]
        totals["mapped"] += int(counters["mappable_tactics"])
        totals["credit_tokens"] += int(counters["credit_tokens"])
        totals["ambiguous"] += int(counters["ambiguous"])
        totals["outside_response"] += int(counters["outside_response"])
        totals["span_not_in_response"] += int(counters["span_not_in_response"])
        totals["unmapped"] += int(counters["unmapped"])
        totals["retokenization_mismatch"] += int(counters["retokenization_mismatch"])
        totals["conflict"] += int(counters["conflicts"])
        totals["no_code"] += int(counters["no_code"])
        status = str(result["status"])
        statuses[status] = statuses.get(status, 0) + 1
    totals["statuses"] = statuses
    return totals
