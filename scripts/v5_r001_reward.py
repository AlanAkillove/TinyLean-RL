#!/usr/bin/env python3
"""R001 online reward function (owner directive §2, §4, §14-§15; draft §7.4).

Loaded by the pinned trainer as the ``custom_module`` reward function
(``+custom_reward_function.path=<this file>``), and fed each row's *generated token ids* by
``scripts/v5_r001_manager.py`` (the ``r001_batch`` manager) - the frozen process credit is defined
on the training token frame, so the ids the model actually produced must be the ones that reach
``derive_facts``.

Per candidate, in this order, all against frozen code:

1. decode the generated ids and gate them on the exact id round-trip (``decode_and_lattice``): any
   disagreement zeroes the process term and counts ``retokenization_mismatch``;
2. the V1 outcome surface: ``score = proof_rw * format_rw`` where ``format_rw`` is the frozen
   ``FormatReward`` gate and ``proof_rw`` comes from the frozen oracle's verdict (owner §5);
3. the process term: a sentinel ``pred`` is the frozen ``FORMAT_NO_CODE`` record (V1 skipped
   exactly these two strings, never submitted) with probability ``no_code`` counted; anything else
   is submitted to the dedicated oracle with ``infotree`` requested, exactly like Phase B;
4. infrastructure outcomes are *censored* (``valid = 0``), never counted as failures and never
   entering ``g_bar`` (§4). They fail **closed** instead: if the oracle's own canary says the
   instance is down, ``VerifierUnhealthyError`` aborts the step (the same contract as the frozen
   Phase-B runner).

Both arms run *this* function with the same endpoint, timeouts and ``MAX_REPLS`` (§15); the only
difference between control and treatment is ``algorithm.r001_lambda`` in the advantage wrapper.

Debug/ops hooks (never part of the objective):

* ``TINYLEAN_R001_ORACLE_ENDPOINT`` overrides the frozen endpoint (S0/S3 wiring checks);
* ``TINYLEAN_R001_ORACLE_ARCHIVE`` (directory) appends one JSONL record per submitted candidate -
  the raw verifier surface of the *online* rollouts, which the frozen Phase-B archive cannot
  provide;
* every call prints one ``R001_REWARD`` JSON line with the §14 counter set.

Run ``python scripts/v5_r001_reward.py --check`` for the static identity probe (S0; no GPU, no
verifier traffic).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_p001_spec as S
import v5_process_oracle as O
import v5_r001_online_credit as C
from kimina_prover_rl.reward.format_reward import FormatReward
from kimina_prover_rl.reward.proof_utils import FormatError, extract_proof_from_text

from tinylean_rl.rl.process_credit import LAMBDA_PRIMARY, PHI

ORACLE_ENDPOINT_ENV = "TINYLEAN_R001_ORACLE_ENDPOINT"
ORACLE_ARCHIVE_ENV = "TINYLEAN_R001_ORACLE_ARCHIVE"

_CLIENT: Any = None
_MAPPER: Any = None
_FORMAT: Any = None
_ARCHIVE: TextIO | None = None
_ARCHIVE_PATH: Path | None = None


def oracle_endpoint() -> str:
    """The frozen oracle endpoint, overridable only for engineering wiring checks."""

    return os.environ.get(ORACLE_ENDPOINT_ENV) or S.ORACLE_INFRA["endpoint"]


def oracle_client() -> Any:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = O.OracleClient(endpoint=oracle_endpoint())
    return _CLIENT


def token_mapper() -> Any:
    global _MAPPER
    if _MAPPER is None:
        _MAPPER = O.TokenMapper()
    return _MAPPER


def format_gate() -> Any:
    global _FORMAT
    if _FORMAT is None:
        _FORMAT = FormatReward()
    return _FORMAT


def format_error_of(response_text: str, formal_statement: str, prompt: Any) -> str:
    """The frozen V1 format gate on the decoded response (messages = prompt + assistant turn)."""

    messages = [dict(message) for message in (prompt or [])]
    messages.append({"role": "assistant", "content": response_text})
    error, _, _ = format_gate().check_format_error(messages, formal_statement)
    return error.value if hasattr(error, "value") else str(error)


def archive_path() -> Path | None:
    """Where the online oracle surface is appended, or ``None`` when archiving is off."""

    directory = os.environ.get(ORACLE_ARCHIVE_ENV)
    if not directory:
        return None
    global _ARCHIVE, _ARCHIVE_PATH
    if _ARCHIVE is None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        _ARCHIVE_PATH = target / f"pid-{os.getpid()}.jsonl"
        _ARCHIVE = _ARCHIVE_PATH.open("a", encoding="utf-8")
    return _ARCHIVE_PATH


def _archive(raw: dict[str, Any], *, custom_id: str, pred: str, row: int) -> None:
    if archive_path() is None or _ARCHIVE is None:
        return
    classified = raw.get("classified")
    record = {
        "custom_id": custom_id,
        "row": row,
        "pred_sha256": S.sha256_text(pred),
        "outcome": getattr(getattr(classified, "outcome", None), "value", None),
        "attempts": raw.get("attempts"),
        "seconds": raw.get("seconds"),
        "item": raw.get("item"),
    }
    _ARCHIVE.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _ARCHIVE.flush()


def tool_feedback_of(result: dict[str, Any], outcome: int | None) -> str:
    """The compact V1 feedback stand-in (multiturn is frozen OFF; this is diagnostics only)."""

    if result["counters"].get("no_code"):
        return "filtered proof."  # V1's exact string for a never-submitted candidate
    if outcome == 1:
        return "valid proof found."  # V1's exact string for a verified proof
    return str(result.get("status", "unknown"))


def reward(
    data_sources: Any = None,
    solution_strs: Any = None,
    ground_truths: Any = None,
    extra_infos: Any = None,
    token_ids: Any = None,
    *,
    endpoint: str | None = None,
    return_dict: bool = True,
    **context: Any,
) -> list[dict[str, Any]]:
    """One dict per row; ``score`` is the V1 outcome reward, the rest are frozen extras.

    ``return_dict`` is accepted for configuration compatibility with the V1 surface (the manager
    always consumes dicts); ``context`` is any extra ``custom_reward_function.reward_kwargs`` and is
    only echoed in the summary line, so a misconfigured key stays visible instead of silent.
    """

    if token_ids is None:
        raise RuntimeError(
            "R001 reward needs the generated token ids; use reward_manager=r001_batch "
            "(scripts/v5_r001_manager.py) - the plain batch manager does not pass them"
        )
    rows = len(solution_strs or [])
    if rows == 0:
        return []
    if len(token_ids) != rows or len(extra_infos or []) != rows:
        raise RuntimeError(
            f"R001 reward row mismatch: {rows} responses, {len(token_ids)} id rows, "
            f"{len(extra_infos or [])} extra_infos"
        )

    client = oracle_client() if endpoint is None else O.OracleClient(endpoint=endpoint)
    mapper = token_mapper()
    results: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    submitted = 0
    verified = 0
    started = time.perf_counter()

    for row in range(rows):
        extra = extra_infos[row] or {}
        formal = str(extra.get("formal_statement", "") or "")
        ids = [int(token_id) for token_id in token_ids[row]]
        text, _, _ = C.decode_and_lattice(ids, mapper=mapper)
        pred = extract_proof_from_text(text, formal)
        index = extra.get("index", row)
        custom_id = f"r001-{index}-{row}"
        source = None if data_sources is None else str(data_sources[row])

        item: dict[str, Any] | None = None
        attempts = 0
        seconds: float | None = None
        if pred not in S.PRED_SENTINELS:
            raw = client.verify_raw(pred, custom_id=custom_id)
            attempts = len(raw.get("attempts") or [])
            seconds = raw.get("seconds")
            item = raw["item"]
            submitted += 1
            if raw["classified"].outcome.is_infrastructure:
                # Candidate-specific censor unless the oracle itself is down, in which case the
                # canary raises and the step stops (frozen Phase-B contract, owner §4).
                client.require_healthy()
            _archive(raw, custom_id=custom_id, pred=pred, row=row)

        result = C.credit_for_candidate(
            token_ids=ids,
            formal_statement=formal,
            item=item,
            mapper=mapper,
            candidate_id=custom_id,
            attempts=attempts,
            seconds=seconds,
            meta={"online": True, "row": row, "data_source": source},
        )
        results.append(result)
        payload = C.payload_of(result)
        outcome = (result.get("record") or {}).get("global_outcome")
        proof_rw = 1.0 if outcome == 1 else 0.0
        verified += int(proof_rw == 1.0)
        format_error = format_error_of(text, formal, extra.get("prompt"))
        out.append(
            {
                "score": proof_rw if format_error == FormatError.NONE.value else 0.0,
                "acc": proof_rw,
                "pred": pred,
                "format_error": format_error,
                "tool_feedback": tool_feedback_of(result, outcome),
                "response": text,
                "process_credit": payload,
                "process_counters": dict(result["counters"]),
            }
        )

    elapsed = time.perf_counter() - started
    summary = C.summarize(results)
    summary.update(
        {
            "rows": rows,
            "submitted": submitted,
            "verified": verified,
            "positions_total": sum(len(result["positions"]) for result in results),
            "seconds": round(elapsed, 2),
            "seconds_per_row": round(elapsed / rows, 3),
            "endpoint": client.endpoint,
            "archive": str(archive_path()) if archive_path() else None,
            "extra_kwargs": sorted(context),
        }
    )
    print("R001_REWARD " + json.dumps(summary, sort_keys=True, default=str), flush=True)
    return out


def static_identity() -> dict[str, Any]:
    """S0 static probe: the frozen identities this path must be bound to (no verifier traffic)."""

    mapper = token_mapper()
    return {
        "endpoint": oracle_endpoint(),
        "endpoint_env": os.environ.get(ORACLE_ENDPOINT_ENV),
        "oracle_container": S.ORACLE_INFRA["container"],
        "oracle_image_digest": S.ORACLE_INFRA["image_digest"],
        "max_repls": S.ORACLE_INFRA["max_repls"],
        "server_timeout_s": S.ORACLE_INFRA["server_timeout_s"],
        "tokenizer_dir": str(S.TOKENIZER_DIR),
        "tokenizer_identity": mapper.identity,
        "tokenizer_expected_vocab": S.TOKENIZER_EXPECTED_VOCAB,
        "phi": dict(PHI),
        "lambda_primary": LAMBDA_PRIMARY,
        "sentinels": list(S.PRED_SENTINELS),
        "archive": str(archive_path()) if archive_path() else None,
    }


def main(argv: list[str]) -> int:
    if argv[1:] != ["--check"]:
        print("usage: python scripts/v5_r001_reward.py --check", file=sys.stderr)
        return 2
    print(json.dumps(static_identity(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
