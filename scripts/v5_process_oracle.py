#!/usr/bin/env python3
"""V5 process oracle -- Lean tactic-level verification for TinyLean-RL V5-P001.

Read-only instrument. It re-verifies one historical candidate exactly as the V1
pipeline submitted it (``pred`` verbatim, the exact string
``extract_proof_from_text`` produced), asks the pinned Kimina server for the
elaboration *info tree*, and derives:

* the ordered list of tactic nodes with exact source spans,
* the first-error (or first-sorry) propagation blame index,
* paper-compatible tactic labels (d1 = -0.05 before the first error,
  d2 = -0.10 at the first error and after; success = 1),
* the character-level mapping tactic -> submitted code -> generated response,
* the first generated token of each tactic under the frozen Kimina tokenizer.

Nothing here trains, generates or writes to a model. The canonical whole-proof
verifier semantics (``tinylean_rl.verifier.policy``) are reused unmodified; this
module only *adds* the infotree-reading layer on top of the same endpoint.

Position conventions of the pinned server (calibrated in Phase A, see
``docs/v5/process_oracle_design.md``): every reported line is 1-based and every
column 0-based, and both messages and info-tree ranges live in the *frame* --
the submitted code minus a maximal leading prefix of blank/import lines.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
_UPSTREAM = ROOT / "third_party" / "kimina-prover-rl" / "recipe" / "kimina_prover_rl"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

import httpx
import v5_p001_spec as S
from kimina_prover_rl.reward.proof_utils import (
    extract_first_theorem_statement,
    extract_proof_from_text,
    is_index_commented,
    normalize_formal_statement,
)

from tinylean_rl.verifier.kimina import verify_code
from tinylean_rl.verifier.policy import (
    CANARY_PROOF,
    VerificationSession,
    VerifyOutcome,
    classify_result_item,
    classify_transport_error,
)

LEAN4_BLOCK_RE = re.compile(r"```lean4\n(.*?)\n```", re.DOTALL)
HEADER_LINE_RE = re.compile(r"\s*import\b")
THEOREM_BY_RE = re.compile(r":=\s*by")

#: Server placeholder for a syntax node whose pretty-printer failed (seen on
#: error-recovery nodes after a parser error). Diagnostic only, never a status.
UNPRINTABLE_PP = "<failed to pretty print>"

#: Mapping statuses that count as an exact tactic -> first-token mapping: the
#: tactic's first character lies inside exactly one tokenizer token (``exact``
#: when the token also starts there; ``contained`` when a BPE token merges the
#: leading whitespace of the tactic, which is the common case for indented
#: tactic blocks).
MAPPABLE_STATUSES = ("exact", "contained")


# --------------------------------------------------------------------------------------
# text layer: frame construction and generated-tail location
# --------------------------------------------------------------------------------------


def strip_header(code: str) -> tuple[str, int]:
    """Split the submitted code into (frame, prefix_chars).

    The pinned server strips a maximal leading prefix of lines that are blank
    or ``import`` lines before it builds the REPL file; all reported positions
    are relative to what remains. Replicated client-side, verified by the
    fixture spans (``frame[start:finish]`` must equal the expected tactic text).
    """

    lines = code.split("\n")
    head = 0
    while head < len(lines) and (lines[head].strip() == "" or HEADER_LINE_RE.match(lines[head])):
        head += 1
    prefix_chars = sum(len(line) + 1 for line in lines[:head])
    return code[prefix_chars:], prefix_chars


def frame_offset(frame: str, line: int, column: int) -> int:
    """Character offset of a 1-based-line / 0-based-column position in the frame."""

    lines = frame.split("\n")
    if not 1 <= line <= len(lines):
        raise PositionError(f"line {line} outside the frame (1..{len(lines)})")
    text = lines[line - 1]
    if not 0 <= column <= len(text):
        raise PositionError(f"column {column} outside line {line} (0..{len(text)})")
    return sum(len(entry) + 1 for entry in lines[: line - 1]) + column


class PositionError(ValueError):
    """A server-reported position does not fit the client-side frame."""


def locate_generated_tail(response: str, formal_statement: str) -> dict[str, Any]:
    """Locate the generated proof tail inside the model response.

    Mirrors ``extract_proof_from_text`` exactly (same block regex, same reversed
    search, same ``:= by`` search after the theorem statement), but additionally
    reports the absolute character offset of the tail inside ``response`` so a
    server span can be mapped back to the generated text.
    """

    normalized = normalize_formal_statement(formal_statement)
    theorem_statement = extract_first_theorem_statement(normalized)
    out: dict[str, Any] = {
        "normalized_formal": normalized,
        "normalized_formal_sha256": S.sha256_text(normalized),
        "theorem_statement": theorem_statement,
        "found": False,
        "tail_offset_in_response": None,
        "tail": None,
        "pred_recomputed": None,
        "n_lean4_blocks": 0,
    }
    if not theorem_statement:
        return out
    blocks = list(LEAN4_BLOCK_RE.finditer(response))
    out["n_lean4_blocks"] = len(blocks)
    for block in reversed(blocks):
        code = block.group(1)
        if theorem_statement not in code:
            continue
        theorem_start = code.find(theorem_statement)
        if is_index_commented(code, theorem_start):
            continue
        match = THEOREM_BY_RE.search(code[theorem_start:])
        if not match:
            continue
        tail_start_in_code = theorem_start + match.end()
        tail_abs = block.start(1) + tail_start_in_code
        tail = response[tail_abs : block.end(1)]
        out.update(
            {
                "found": True,
                "tail_offset_in_response": tail_abs,
                "tail": tail,
                "pred_recomputed": normalized + tail,
            }
        )
        return out
    return out


# --------------------------------------------------------------------------------------
# oracle response layer: infotree -> tactics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tactic:
    """One tactic node parsed out of the Lean elaboration info tree."""

    index: int
    name: str
    start_line: int
    start_col: int
    finish_line: int
    finish_col: int
    goals_before: Any
    goals_after: Any
    pp: str | None


def _goal_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def flatten_infotree(tree: Any) -> list[tuple[int, dict]]:
    """Depth-first flatten of the info tree into ``(depth, node)`` pairs."""

    out: list[tuple[int, dict]] = []

    def walk(node: Any, depth: int) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, depth)
            return
        if not isinstance(node, dict):
            return
        out.append((depth, node))
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                walk(child, depth + 1)

    walk(tree, 0)
    return out


def extract_tactics(tree: Any) -> tuple[list[Tactic], dict[str, int]]:
    """Collect tactic nodes (sorted by source start) and extraction diagnostics.

    Exact duplicates (same name and same source span) are collapsed: the same
    syntactic node can be reachable twice in the elaboration tree, and a tactic
    sequence must not credit the same node twice.
    """

    stats = {
        "nodes_total": 0,
        "tactic_named": 0,
        "wrappers_skipped": 0,
        "synthetic_skipped": 0,
        "malformed_skipped": 0,
        "duplicates_removed": 0,
    }
    raw: list[tuple[Any, ...]] = []
    for _depth, node in flatten_infotree(tree):
        stats["nodes_total"] += 1
        info = node.get("node")
        if not isinstance(info, dict):
            stats["malformed_skipped"] += 1
            continue
        name = info.get("name")
        if not isinstance(name, str) or not name.startswith(S.TACTIC_NAME_PREFIX):
            continue
        stats["tactic_named"] += 1
        if name in S.TACTIC_WRAPPER_NAMES:
            stats["wrappers_skipped"] += 1
            continue
        stx = info.get("stx") or {}
        rng = stx.get("range") or {}
        start, finish = rng.get("start"), rng.get("finish")
        if not isinstance(start, dict) or not isinstance(finish, dict):
            stats["malformed_skipped"] += 1
            continue
        if rng.get("synthetic"):
            stats["synthetic_skipped"] += 1
            continue
        try:
            key = (
                int(start["line"]),
                int(start["column"]),
                int(finish["line"]),
                int(finish["column"]),
                name,
            )
        except (KeyError, TypeError, ValueError):
            stats["malformed_skipped"] += 1
            continue
        raw.append(key + (info,))
    raw.sort(key=lambda entry: entry[:5])
    deduped: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in raw:
        key = entry[:5]
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        deduped.append(entry)
    tactics = [
        Tactic(
            index=index,
            name=entry[4],
            start_line=entry[0],
            start_col=entry[1],
            finish_line=entry[2],
            finish_col=entry[3],
            goals_before=entry[5].get("goalsBefore"),
            goals_after=entry[5].get("goalsAfter"),
            pp=(entry[5].get("stx") or {}).get("pp"),
        )
        for index, entry in enumerate(deduped)
    ]
    stats["tactics"] = len(tactics)
    return tactics, stats


def _contains(tactic: Tactic, line: int, column: int) -> bool:
    return (tactic.start_line, tactic.start_col) <= (line, column) < (
        tactic.finish_line,
        tactic.finish_col,
    )


#: One message range: inclusive start, optional end (both 1-based line /
#: 0-based column, finish exclusive, exactly as the server reports them).
MsgRange = tuple[int, int, int | None, int | None]


def blame_index(
    tactics: list[Tactic], ranges: list[MsgRange]
) -> tuple[int | None, int, str | None]:
    """Attribute each message range to a tactic; earliest blamed tactic overall.

    Attribution rules (frozen; see docs/v5/process_oracle_design.md):

    1. ``contains``: the innermost tactic node whose span contains the message
       start. This is the exact rule for tactic-local errors.
    2. ``range_last``: when no node contains the start, use the message *range*
       (Lean reports block-level failures such as ``unsolved goals`` with a
       range covering the whole failing tactic sequence, starting right after
       ``by``): the last tactic node that *starts* inside the range is blamed,
       i.e. the point where elaboration stopped.
    3. unattributed: the message counts towards ``n_unmapped`` and credits no
       tactic.

    Returns ``(blamed_index, n_unmapped, blame_kind)``; ``blamed_index`` is the
    earliest attributed tactic in source order (the proof's first error).
    """

    blamed: list[int] = []
    kinds: list[str] = []
    unmapped = 0
    for start_line, start_col, end_line, end_col in ranges:
        containers = [t.index for t in tactics if _contains(t, start_line, start_col)]
        if containers:
            best = max(
                containers,
                key=lambda index: (
                    tactics[index].start_line,
                    tactics[index].start_col,
                    -tactics[index].finish_line,
                    -tactics[index].finish_col,
                ),
            )
            blamed.append(best)
            kinds.append("contains")
            continue
        if end_line is not None and end_col is not None:
            inside = [
                t.index
                for t in tactics
                if (start_line, start_col) <= (t.start_line, t.start_col) < (end_line, end_col)
            ]
            if inside:
                blamed.append(max(inside))
                kinds.append("range_last")
                continue
        unmapped += 1
    if not blamed:
        return None, unmapped, None
    first = min(blamed)
    return first, unmapped, kinds[blamed.index(first)]


def _as_ranges(entries: list[Any]) -> list[MsgRange]:
    out: list[MsgRange] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pos = entry.get("pos") or {}
        line, column = pos.get("line"), pos.get("column")
        if not isinstance(line, int) or not isinstance(column, int):
            continue
        end = entry.get("endPos") or {}
        end_line, end_col = end.get("line"), end.get("column")
        if not isinstance(end_line, int) or not isinstance(end_col, int):
            end_line, end_col = None, None
        out.append((line, column, end_line, end_col))
    return out


def _error_ranges(response: dict) -> list[MsgRange]:
    return _as_ranges(
        [
            message
            for message in response.get("messages") or []
            if isinstance(message, dict) and message.get("severity") == "error"
        ]
    )


def _sorry_ranges(response: dict) -> list[MsgRange]:
    return _as_ranges([entry for entry in response.get("sorries") or []])


# --------------------------------------------------------------------------------------
# token layer: tactic -> generated response -> first generated token
# --------------------------------------------------------------------------------------


class TokenMapper:
    """Frozen Kimina tokenizer wrapper used for first-token process credit."""

    def __init__(self, tokenizer_dir: Path = S.TOKENIZER_DIR) -> None:
        from transformers import AutoTokenizer

        self.tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir), use_fast=True)
        # ``vocab_size`` is the base BPE vocabulary (151643); ``len(tokenizer)``
        # additionally counts the 26 added special tokens (151669). The frozen
        # identity is the base vocab + the tokenizer.json/config bytes.
        self.vocab_size = self.tokenizer.vocab_size
        self.identity = {
            "dir": str(self.tokenizer_dir.relative_to(ROOT)),
            "class": type(self.tokenizer).__name__,
            "vocab_size": self.vocab_size,
            "len_with_added_tokens": len(self.tokenizer),
            "add_special_tokens": S.TOKENIZER_ADD_SPECIAL_TOKENS,
            "tokenizer_json_sha256": S.sha256_file(self.tokenizer_dir / "tokenizer.json"),
            "tokenizer_config_sha256": S.sha256_file(self.tokenizer_dir / "tokenizer_config.json"),
        }
        if self.vocab_size != S.TOKENIZER_EXPECTED_VOCAB:
            raise RuntimeError(
                f"tokenizer vocab {self.vocab_size} != frozen {S.TOKENIZER_EXPECTED_VOCAB}"
            )

    def lattice(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=S.TOKENIZER_ADD_SPECIAL_TOKENS,
            return_offsets_mapping=True,
        )
        ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        return {
            "ids": ids,
            "offsets": offsets,
            "n_tokens": len(ids),
            "roundtrip_ok": self.tokenizer.decode(ids) == text,
        }

    @staticmethod
    def first_token_at(lattice: dict[str, Any], char_offset: int) -> dict[str, Any] | None:
        """The unique token whose character span contains ``char_offset``."""

        offsets = lattice["offsets"]
        for index, (start, end) in enumerate(offsets):
            if start <= char_offset < end:
                return {
                    "token_index": index,
                    "token_id": lattice["ids"][index],
                    "token_start": start,
                    "token_end": end,
                    "exact": start == char_offset,
                }
            if start > char_offset:
                break
        return None


# --------------------------------------------------------------------------------------
# candidate-level derivation
# --------------------------------------------------------------------------------------


def _classify_status(
    *, extractable: bool, infra: bool, global_outcome: int | None, n_tactics: int,
    blamed: int | None,
) -> str:
    if not extractable:
        return "FORMAT_NO_CODE"
    if infra:
        return "PROCESS_ORACLE_INFRA"
    if global_outcome == 1:
        return "SUCCESS"
    if n_tactics == 0 or blamed is None:
        return "PARSE_OR_SYNTAX_FAILURE"
    if blamed == 0:
        return "FIRST_TACTIC_FAILURE"
    return "PREFIX_BEARING_FAILURE"


def derive_facts(
    *,
    candidate_id: str,
    pred: str,
    response_text: str,
    formal_statement: str,
    oracle_result: dict[str, Any] | None,
    token_lattice: dict[str, Any] | None,
    d1: float = S.D1_CANONICAL,
    d2: float = S.D2_CANONICAL,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the canonical per-candidate process record.

    ``oracle_result`` is ``None`` for candidates whose ``pred`` is a sentinel
    (no code was ever submitted): those get no tactic credit of any kind.
    """

    record: dict[str, Any] = dict(meta or {})
    record["candidate_id"] = candidate_id
    record["d1"], record["d2"] = d1, d2

    locate = locate_generated_tail(response_text, formal_statement)
    extractable = pred not in S.PRED_SENTINELS
    frame, prefix_chars = strip_header(pred) if extractable else ("", 0)
    record["code"] = {
        "extractable": extractable,
        "sentinel": None if extractable else pred,
        "pred_sha256": S.sha256_text(pred),
        "prefix_chars": prefix_chars,
        "n_lines": len(pred.split("\n")) if extractable else 0,
        "frame_sha256": S.sha256_text(frame) if extractable else None,
    }
    record["tail"] = {
        "located": locate["found"],
        "offset_in_response": locate["tail_offset_in_response"],
        "n_lean4_blocks": locate["n_lean4_blocks"],
        "pred_recomputed_matches": (locate["pred_recomputed"] == pred)
        if locate["found"] and extractable
        else None,
        "normalized_formal_chars": len(locate["normalized_formal"]),
    }
    if extractable and locate["found"] and locate["pred_recomputed"] != pred:
        # The extraction replication disagrees with the stored pred: keep the
        # candidate (the stored pred is the historical code) but flag it.
        record["code"]["pred_recompute_mismatch"] = True

    if not extractable:
        record.update(
            {
                "oracle": None,
                "global_outcome": None,
                "process_status": "FORMAT_NO_CODE",
                "failure_kind": None,
                "n_tactics": 0,
                "blamed_index": None,
                "blame_kind": None,
                "n_error_messages": 0,
                "n_unmapped_error_messages": 0,
                "n_sorries": 0,
                "n_unmapped_sorries": 0,
                "tactics": [],
                "token_credit": _empty_token_credit(),
            }
        )
        return record

    if oracle_result is None:
        raise ValueError("extractable candidate needs an oracle result")

    classified = oracle_result["classified"]
    item = oracle_result["item"]
    body = (item or {}).get("response") if isinstance(item, dict) else None
    body = body if isinstance(body, dict) else {}
    infra = bool(classified.outcome.is_infrastructure)
    global_outcome = None if infra else int(classified.outcome is VerifyOutcome.VERIFIED)

    tactics, extraction_stats = extract_tactics(body.get("infotree"))
    error_ranges = _error_ranges(body)
    sorry_ranges = _sorry_ranges(body)

    blamed, n_unmapped_errors, blame_kind = blame_index(tactics, error_ranges)
    sorry_blamed, n_unmapped_sorries, sorry_kind = blame_index(tactics, sorry_ranges)
    if blamed is None and global_outcome == 0 and sorry_ranges:
        blamed, blame_kind = sorry_blamed, sorry_kind

    if global_outcome == 0:
        if error_ranges and sorry_ranges:
            failure_kind: str | None = "error+sorry"
        elif error_ranges:
            failure_kind = "error"
        elif sorry_ranges:
            failure_kind = "sorry"
        elif not infra:
            failure_kind = "no_error_message"
        else:
            failure_kind = None
    else:
        failure_kind = None

    status = _classify_status(
        extractable=True,
        infra=infra,
        global_outcome=global_outcome,
        n_tactics=len(tactics),
        blamed=blamed,
    )

    blamed_pp_unprintable: bool | None = None
    if blamed is not None:
        blamed_pp = tactics[blamed].pp
        blamed_pp_unprintable = blamed_pp in (None, UNPRINTABLE_PP)

    tactic_records: list[dict[str, Any]] = []
    for tactic in tactics:
        errors_inside = sum(1 for rng in error_ranges if _contains(tactic, rng[0], rng[1]))
        sorries_inside = sum(1 for rng in sorry_ranges if _contains(tactic, rng[0], rng[1]))
        mapping = _map_tactic(
            tactic, frame=frame, prefix_chars=prefix_chars,
            normalized_formal_chars=len(locate["normalized_formal"]),
            tail_offset=locate["tail_offset_in_response"], response_text=response_text,
            token_lattice=token_lattice,
        )
        if infra:
            label = None
        elif global_outcome == 1:
            label = "success"
        elif blamed is None:
            label = None
        else:
            label = "d1" if tactic.index < blamed else "d2"
        tactic_records.append(
            {
                "i": tactic.index,
                "name": tactic.name,
                "start": [tactic.start_line, tactic.start_col],
                "finish": [tactic.finish_line, tactic.finish_col],
                "pp": (tactic.pp or "")[:200],
                "goals_before_n": _goal_count(tactic.goals_before),
                "goals_after_n": _goal_count(tactic.goals_after),
                "error_hits": errors_inside,
                "sorry_hits": sorries_inside,
                "locally_verified": bool(errors_inside == 0 and sorries_inside == 0),
                "blamed": bool(tactic.index == blamed),
                "label": label,
                "mapping": mapping,
            }
        )

    record.update(
        {
            "oracle": {
                "outcome": classified.outcome.value,
                "infra": infra,
                "redeclaration": bool(classified.redeclaration),
                "message": classified.message[:300],
                "attempts": len(oracle_result.get("attempts") or []),
                "seconds": oracle_result.get("seconds"),
                "raw_item_sha256": S.sha256_text(S.canonical_json(item)) if item else None,
                "extraction_stats": extraction_stats,
            },
            "global_outcome": global_outcome,
            "process_status": status,
            "failure_kind": failure_kind,
            "n_tactics": len(tactics),
            "blamed_index": blamed,
            "blame_kind": blame_kind,
            "blamed_pp_unprintable": blamed_pp_unprintable,
            "n_error_messages": len(error_ranges),
            "n_unmapped_error_messages": n_unmapped_errors,
            "n_sorries": len(sorry_ranges),
            "n_unmapped_sorries": n_unmapped_sorries,
            "tactics": tactic_records,
            "token_credit": _token_credit(tactic_records, infra=infra, blamed=blamed),
        }
    )
    return record


def _empty_token_credit() -> dict[str, Any]:
    return {
        "n_mapped": 0,
        "n_exact": 0,
        "n_contained": 0,
        "n_ambiguous": 0,
        "n_outside_response": 0,
        "n_span_not_in_response": 0,
        "n_unmapped": 0,
        "n_credit_tokens": 0,
        "n_credit_conflicts": 0,
        "d1_token_positions": [],
        "d2_token_positions": [],
        "success_token_positions": [],
    }


def _map_tactic(
    tactic: Tactic,
    *,
    frame: str,
    prefix_chars: int,
    normalized_formal_chars: int,
    tail_offset: int | None,
    response_text: str,
    token_lattice: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map one tactic span to a char offset and the first generated token.

    Statuses: ``exact`` (the token starts exactly at the tactic's first
    character), ``contained`` (exactly one token contains that character, BPE
    merged the leading whitespace into it), ``ambiguous`` (more than one token
    would cover it), ``outside_response`` (the span sits in the prompt-derived
    formal-statement head), ``response_only`` (no tokenizer lattice available),
    ``unmapped`` (no token covers the offset).
    """

    out: dict[str, Any] = {
        "status": "unmapped",
        "pred_offset": None,
        "response_offset": None,
        "token_index": None,
        "token_id": None,
        "token_start": None,
        "token_end": None,
        "exact": None,
    }
    try:
        span_start = frame_offset(frame, tactic.start_line, tactic.start_col)
        span_end = frame_offset(frame, tactic.finish_line, tactic.finish_col)
    except PositionError:
        return out
    pred_offset = prefix_chars + span_start
    out["pred_offset"] = pred_offset
    out["span_chars"] = span_end - span_start
    out["span_sha256"] = S.sha256_text(frame[span_start:span_end])
    if pred_offset < normalized_formal_chars:
        # Span sits in the prompt-derived formal-statement head: it is not part
        # of the generated response, so it can carry no token-level credit.
        out["status"] = "outside_response"
        return out
    if tail_offset is None:
        return out
    response_offset = tail_offset + (pred_offset - normalized_formal_chars)
    out["response_offset"] = response_offset
    slice_end = response_offset + (span_end - span_start)
    if slice_end > len(response_text):
        return out
    out["span_in_response"] = response_text[response_offset:slice_end] == frame[span_start:span_end]
    if token_lattice is None:
        out["status"] = "response_only"
        return out
    token = TokenMapper.first_token_at(token_lattice, response_offset)
    if token is None:
        return out
    if token["token_start"] > response_offset:
        out["status"] = "ambiguous"  # defensive: offsets never start after the point
        return out
    out.update(
        {
            "status": "exact" if token["exact"] else "contained",
            "token_index": token["token_index"],
            "token_id": token["token_id"],
            "token_start": token["token_start"],
            "token_end": token["token_end"],
            "exact": token["exact"],
        }
    )
    return out


#: Credit precedence when two tactics map to the same first token (a composite
#: tactic such as ``constructor <;> trivial`` and its first child share it):
#: the error region wins over the verified prefix, which wins over success.
_CREDIT_PRECEDENCE = {"d2": 3, "d1": 2, "success": 1}


def _token_credit(
    tactic_records: list[dict[str, Any]], *, infra: bool, blamed: int | None
) -> dict[str, Any]:
    """The frozen per-token credit surface on tactic first-tokens (owner §16)."""

    credit = _empty_token_credit()
    assigned: dict[int, tuple[int, str]] = {}
    conflicts = 0
    for tactic in tactic_records:
        mapping = tactic["mapping"]
        status = mapping["status"]
        if status in MAPPABLE_STATUSES and mapping.get("span_in_response") is True:
            credit["n_mapped"] += 1
            credit["n_exact" if status == "exact" else "n_contained"] += 1
            token_index, token_id = int(mapping["token_index"]), int(mapping["token_id"])
            label = tactic["label"]
            if label is None:
                continue
            previous = assigned.get(token_index)
            if previous is not None and previous[1] != label:
                conflicts += 1
                if _CREDIT_PRECEDENCE[label] <= _CREDIT_PRECEDENCE[previous[1]]:
                    continue
            assigned[token_index] = (token_id, label)
        elif status in MAPPABLE_STATUSES:
            credit["n_span_not_in_response"] += 1
            credit["n_unmapped"] += 1
        elif status == "outside_response":
            credit["n_outside_response"] += 1
        elif status != "response_only":
            credit["n_unmapped"] += 1
    for token_index in sorted(assigned):
        token_id, label = assigned[token_index]
        credit[f"{label}_token_positions"].append([token_index, token_id])
    credit["n_credit_tokens"] = len(assigned)
    credit["n_credit_conflicts"] = conflicts
    if infra or blamed is None:
        credit["blamed_mappable"] = None
    else:
        credited = (
            tactic_records[blamed]["mapping"]["status"] in MAPPABLE_STATUSES
            and tactic_records[blamed]["mapping"].get("span_in_response") is True
        )
        credit["blamed_mappable"] = credited
    return credit


def structured_recoverable(record: dict[str, Any]) -> bool:
    """Owner §11: both d1 and d2 credit on mappable first tokens.

    A candidate is structured-recoverable when it failed with a blamed tactic,
    at least one *locally verified* tactic strictly before that blame has a
    mappable first token, and the blamed tactic itself has a mappable first
    token. The trivial "every parsed tactic is immediately erroneous" case is
    excluded by the strict inequality.
    """

    if record["process_status"] not in ("PREFIX_BEARING_FAILURE", "FIRST_TACTIC_FAILURE"):
        return False
    blamed = record["blamed_index"]
    if blamed is None or blamed == 0 or blamed >= len(record["tactics"]):
        return False

    def credited(tactic: dict[str, Any]) -> bool:
        return (
            tactic["mapping"]["status"] in MAPPABLE_STATUSES
            and tactic["mapping"].get("span_in_response") is True
        )

    pre_ok = any(
        tactic["locally_verified"] and credited(tactic) for tactic in record["tactics"][:blamed]
    )
    return bool(pre_ok and credited(record["tactics"][blamed]))


def any_active(record: dict[str, Any]) -> bool:
    """Owner §12 (broad diagnostic): a failed candidate with a nonempty prefix."""

    return (
        record["process_status"] in ("PREFIX_BEARING_FAILURE", "FIRST_TACTIC_FAILURE")
        and record["blamed_index"] is not None
        and record["blamed_index"] >= 1
    )


def verified_prefix_count(record: dict[str, Any]) -> int:
    blamed = record["blamed_index"]
    if blamed is None:
        return 0
    return sum(1 for tactic in record["tactics"][:blamed] if tactic["locally_verified"])


# --------------------------------------------------------------------------------------
# transport: the dedicated oracle instance
# --------------------------------------------------------------------------------------


class OracleClient:
    """Bounded, fail-close client for the dedicated V5 oracle instance."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        server_timeout: int | None = None,
        record_events: bool = True,
    ) -> None:
        infra = S.ORACLE_INFRA
        self.endpoint = (endpoint or infra["endpoint"]).rstrip("/")
        self.session = VerificationSession(
            base_url=self.endpoint,
            server_timeout=float(server_timeout or infra["server_timeout_s"]),
            client_slack=float(infra["client_slack_s"]),
            batch_size=1,
            max_single_retries=int(infra["max_single_retries"]),
            canary_timeout=float(infra["warm_canary_timeout_s"]),
            canary_retries=int(infra["canary_retries"]),
        )
        self.record_events = record_events

    # -- health ---------------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.endpoint}/health", timeout=10.0, trust_env=False)
            return {"ok": response.status_code == 200, "status": response.status_code,
                    "body": str(response.text)[:200]}
        except httpx.HTTPError as exc:
            return {"ok": False, "status": f"transport error: {type(exc).__name__}: {exc}"}

    def warmup(self, *, timeout_s: int | None = None) -> dict[str, Any]:
        """Cold canary: the first request after a (re)start re-imports Mathlib."""

        budget = int(timeout_s or S.ORACLE_INFRA["canary_timeout_s"])
        started = time.perf_counter()
        try:
            decoded = verify_code(
                CANARY_PROOF,
                custom_id="v5-oracle-cold-canary",
                base_url=self.endpoint,
                timeout=budget + int(S.ORACLE_INFRA["client_slack_s"]),
                server_timeout=budget,
            )
            items = decoded.get("results") or decoded.get("codes") or []
            classified = classify_result_item(items[0] if items else None)
        except httpx.HTTPError as exc:
            classified = classify_transport_error(exc)
        ok = classified.outcome is VerifyOutcome.VERIFIED
        result = {
            "ok": ok,
            "outcome": classified.outcome.value,
            "message": classified.message[:300],
            "seconds": round(time.perf_counter() - started, 2),
            "budget_s": budget,
            "canary_sha256": S.sha256_text(CANARY_PROOF),
        }
        if not ok:
            raise RuntimeError(f"oracle cold canary failed: {result}")
        return result

    def verify_raw(self, proof: str, custom_id: str) -> dict[str, Any]:
        """One candidate: bounded attempts, raw item retained, fail-close upstream.

        A server-confirmed timeout is final (retrying would burn the same
        budget); other infrastructure outcomes are retried in isolation up to
        the frozen bound and stay censored if they persist.
        """

        attempts: list[dict[str, Any]] = []
        item: dict[str, Any] | None = None
        classified = classify_result_item(None)
        started_all = time.perf_counter()
        for attempt in range(self.session.max_single_retries + 1):
            started = time.perf_counter()
            item = None
            try:
                decoded = verify_code(
                    proof,
                    custom_id=custom_id if attempt == 0 else f"{custom_id}-iso{attempt}",
                    base_url=self.endpoint,
                    timeout=self.session.client_timeout,
                    server_timeout=int(self.session.server_timeout),
                )
                items = decoded.get("results") or decoded.get("codes") or []
                item = items[0] if items else None
                classified = classify_result_item(item)
            except httpx.HTTPError as exc:
                classified = classify_transport_error(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "outcome": classified.outcome.value,
                    "message": classified.message[:300],
                    "seconds": round(time.perf_counter() - started, 2),
                    "item_sha256": S.sha256_text(S.canonical_json(item)) if item else None,
                }
            )
            if classified.redeclaration:
                continue
            if (
                classified.outcome.is_conclusive
                or classified.outcome is VerifyOutcome.VERIFIER_TIMEOUT
            ):
                break
            if attempt < self.session.max_single_retries:
                time.sleep(self.session.retry_backoff_seconds)
        return {
            "item": item,
            "classified": classified,
            "attempts": attempts,
            "seconds": round(time.perf_counter() - started_all, 2),
        }

    def require_healthy(self) -> None:
        """Fail-close gate used after every infrastructure event."""

        if not self.session.canary_probe():
            from tinylean_rl.verifier.policy import VerifierUnhealthyError

            raise VerifierUnhealthyError(
                f"oracle canary failed on {self.endpoint}; stop and restart the instance"
            )


# --------------------------------------------------------------------------------------
# engineering fixtures (owner §8): 12 required classes, run >= 2x
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    fid: str
    klass: str
    formal_statement: str
    response: str
    #: True = valid, False = failed, None = never submitted (no extractable code).
    expect_global: bool | None
    expect_status: str
    expect_n_tactics: int
    expect_blamed: int | None
    expect_tactic_texts: tuple[str, ...]
    #: Optional extra assertions (None = not checked).
    expect_pp_unprintable: bool | None = None
    expect_sentinel: str | None = None
    notes: str = ""


def _response(statement: str, body: str, think: str = "Plan the Lean proof.") -> str:
    return f"<think>\n{think}\n</think>\n\n```lean4\n{statement} := by{body}\n```\n"


F1_FORMAL = "import Mathlib\n\nexample (a b c : ℕ) (h1 : a = b) (h2 : b = c) : a = c := by sorry"
F2_FORMAL = "import Mathlib\n\nexample : ℕ := by sorry"
F3_FORMAL = "import Mathlib\n\nexample (a : ℕ) : ℕ := by sorry"
F4_FORMAL = "import Mathlib\n\nexample : True ∧ True := by sorry"
F5_FORMAL = "import Mathlib\n\nexample : True := by sorry"
F6_FORMAL = "import Mathlib\n\nexample : Inhabited Empty := by sorry"
F7_FORMAL = "import Mathlib\n\nexample (a : ℕ) : a = a := by sorry"
F8_FORMAL = "import Mathlib\n\nexample : True ∧ True := by sorry"
F9_FORMAL = "import Mathlib\n\nexample (n : ℕ) : n = n := by sorry"
F10_FORMAL = "import Mathlib\n\nexample : True := by sorry"
F11_FORMAL = "import Mathlib\n\nexample : True := by sorry"
F12_FORMAL = "import Mathlib\n\nexample : (1 : ℕ) = 1 := by sorry"
F13_FORMAL = "import Mathlib\n\nexample : True := by sorry"

FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        fid="F01_fully_successful",
        klass="fully_successful_tactic_proof",
        formal_statement=F1_FORMAL,
        response=_response(
            "example (a b c : ℕ) (h1 : a = b) (h2 : b = c) : a = c",
            "\n  rw [h1]\n  exact h2",
        ),
        expect_global=True,
        expect_status="SUCCESS",
        expect_n_tactics=2,
        expect_blamed=None,
        expect_tactic_texts=("rw [h1]", "exact h2"),
    ),
    Fixture(
        fid="F02_error_first_tactic",
        klass="error_in_first_tactic",
        formal_statement=F2_FORMAL,
        response=_response("example : ℕ", "\n  exact True.intro"),
        expect_global=False,
        expect_status="FIRST_TACTIC_FAILURE",
        expect_n_tactics=1,
        expect_blamed=0,
        expect_tactic_texts=("exact True.intro",),
    ),
    Fixture(
        fid="F03_prefix_then_type_error",
        klass="valid_prefix_then_later_type_error",
        formal_statement=F3_FORMAL,
        response=_response("example (a : ℕ) : ℕ", "\n  have h : ℕ := 0\n  exact True.intro"),
        expect_global=False,
        expect_status="PREFIX_BEARING_FAILURE",
        expect_n_tactics=2,
        expect_blamed=1,
        expect_tactic_texts=("have h : ℕ := 0", "exact True.intro"),
    ),
    Fixture(
        fid="F04_prefix_then_unsolved_goals",
        klass="valid_prefix_then_unsolved_goals",
        formal_statement=F4_FORMAL,
        response=_response("example : True ∧ True", "\n  refine ⟨?_, ?_⟩\n  trivial"),
        expect_global=False,
        expect_status="PREFIX_BEARING_FAILURE",
        expect_n_tactics=2,
        expect_blamed=1,
        expect_tactic_texts=("refine ⟨?_, ?_⟩", "trivial"),
        notes=(
            "a real tactic elaborated, a goal stayed open; Lean reports 'unsolved goals' with a "
            "range covering the whole tactic block (start right after 'by'), so blame lands on the "
            "last tactic in the range (blame_kind=range_last) and the verified prefix keeps d1"
        ),
    ),
    Fixture(
        fid="F05_unknown_identifier",
        klass="unknown_identifier",
        formal_statement=F5_FORMAL,
        response=_response("example : True", "\n  exact unknown_xyz_123"),
        expect_global=False,
        expect_status="FIRST_TACTIC_FAILURE",
        expect_n_tactics=1,
        expect_blamed=0,
        expect_tactic_texts=("exact unknown_xyz_123",),
    ),
    Fixture(
        fid="F06_typeclass_failure",
        klass="typeclass_failure",
        formal_statement=F6_FORMAL,
        response=_response("example : Inhabited Empty", "\n  infer_instance"),
        expect_global=False,
        expect_status="FIRST_TACTIC_FAILURE",
        expect_n_tactics=1,
        expect_blamed=0,
        expect_tactic_texts=("infer_instance",),
    ),
    Fixture(
        fid="F07_nested_by_block",
        klass="nested_by_block",
        formal_statement=F7_FORMAL,
        response=_response("example (a : ℕ) : a = a", "\n  have h : a = a := by rfl\n  exact h"),
        expect_global=True,
        expect_status="SUCCESS",
        expect_n_tactics=3,
        expect_blamed=None,
        expect_tactic_texts=("have h : a = a := by rfl", "rfl", "exact h"),
    ),
    Fixture(
        fid="F08_combinator",
        klass="semicolon_combinator_tactic",
        formal_statement=F8_FORMAL,
        response=_response("example : True ∧ True", "\n  constructor <;> trivial"),
        expect_global=True,
        expect_status="SUCCESS",
        expect_n_tactics=3,
        expect_blamed=None,
        expect_tactic_texts=("constructor", "constructor <;> trivial", "trivial"),
        notes=(
            "the combinator node and its children are all tactic nodes; exact-duplicate nodes are "
            "collapsed by (name, start, finish). The composite and its first child share a first "
            "token (credit precedence d2 > d1 > success)"
        ),
    ),
    Fixture(
        fid="F09_case_style_block",
        klass="all_goals_case_style_block",
        formal_statement=F9_FORMAL,
        response=_response(
            "example (n : ℕ) : n = n",
            "\n  induction n with\n  | zero => rfl\n  | succ k ih => rfl",
        ),
        expect_global=True,
        expect_status="SUCCESS",
        expect_n_tactics=3,
        expect_blamed=None,
        expect_tactic_texts=("induction n with\n  | zero => rfl\n  | succ k ih => rfl", "rfl", "rfl"),
        notes=(
            "branch bodies are nested tactic nodes; the induction node is reachable twice in the "
            "elaboration tree and is collapsed by (name, start, finish)"
        ),
    ),
    Fixture(
        fid="F10_multiline_tactic",
        klass="multi_line_tactic",
        formal_statement=F10_FORMAL,
        response=_response("example : True", "\n  exact\n    True.intro"),
        expect_global=True,
        expect_status="SUCCESS",
        expect_n_tactics=1,
        expect_blamed=None,
        expect_tactic_texts=("exact\n    True.intro",),
    ),
    Fixture(
        fid="F11_syntax_error",
        klass="syntax_parser_failure",
        formal_statement=F11_FORMAL,
        response=_response("example : True", "\n  exact (True.intro"),
        expect_global=False,
        expect_status="FIRST_TACTIC_FAILURE",
        expect_n_tactics=1,
        expect_blamed=0,
        expect_tactic_texts=("exact (True.intro",),
        expect_pp_unprintable=True,
        notes=(
            "an unbalanced parenthesis is still recovered into one tactic node by the server's "
            "parser; the first (and only) tactic is blamed, its pp is unprintable, and no verified "
            "tactic precedes it, so no credit is fabricated. Recorded for the syntax-damage "
            "diagnostic (blamed_pp_unprintable)"
        ),
    ),
    Fixture(
        fid="F12_term_style_not_extractable",
        klass="term_style_proof",
        formal_statement=F12_FORMAL,
        response=(
            "<think>\nTerm proof.\n</think>\n\n"
            "```lean4\nexample : (1 : ℕ) = 1 := rfl\n```\n"
        ),
        expect_global=None,
        expect_status="FORMAT_NO_CODE",
        expect_n_tactics=0,
        expect_blamed=None,
        expect_tactic_texts=(),
        expect_sentinel=S.SENTINEL_NO_PROOF,
        notes=(
            "term-style proofs contain no ':= by', so the historical extractor returns its "
            "sentinel: no code was ever submitted, and no tactic credit may be fabricated"
        ),
    ),
    Fixture(
        fid="F13_empty_proof_body",
        klass="empty_proof_body",
        formal_statement=F13_FORMAL,
        response=_response("example : True", ""),
        expect_global=False,
        expect_status="PARSE_OR_SYNTAX_FAILURE",
        expect_n_tactics=0,
        expect_blamed=None,
        expect_tactic_texts=(),
        notes="submitted code ends right after 'by': failure with no tactic to blame",
    ),
)


def fixture_set_sha256(fixtures: tuple[Fixture, ...] = FIXTURES) -> str:
    payload = [
        {
            "fid": fixture.fid,
            "klass": fixture.klass,
            "formal_statement": fixture.formal_statement,
            "response": fixture.response,
            "expect_global": fixture.expect_global,
            "expect_status": fixture.expect_status,
            "expect_n_tactics": fixture.expect_n_tactics,
            "expect_blamed": fixture.expect_blamed,
            "expect_tactic_texts": list(fixture.expect_tactic_texts),
            "expect_pp_unprintable": fixture.expect_pp_unprintable,
            "expect_sentinel": fixture.expect_sentinel,
        }
        for fixture in fixtures
    ]
    return S.sha256_text(S.canonical_json(payload))


def run_fixture_once(
    fixture: Fixture, client: OracleClient, mapper: TokenMapper, run_tag: str
) -> dict[str, Any]:
    """Submit one fixture and derive its facts (run 1 or run 2)."""

    pred = extract_proof_from_text(fixture.response, fixture.formal_statement)
    lattice = mapper.lattice(fixture.response)
    oracle_result = None
    if pred not in S.PRED_SENTINELS:
        oracle_result = client.verify_raw(pred, custom_id=f"v5-fix-{fixture.fid}-{run_tag}")
        if oracle_result["classified"].outcome.is_infrastructure:
            client.require_healthy()
    record = derive_facts(
        candidate_id=f"fixture::{fixture.fid}",
        pred=pred,
        response_text=fixture.response,
        formal_statement=fixture.formal_statement,
        oracle_result=oracle_result,
        token_lattice=lattice,
    )
    record["fixture_run"] = run_tag
    record["pred"] = pred
    return record


def validate_fixture(fixture: Fixture, record: dict[str, Any]) -> list[str]:
    """Semantic + span-level checks; returns a list of failure strings."""

    failures: list[str] = []
    if record["process_status"] != fixture.expect_status:
        failures.append(f"status {record['process_status']} != {fixture.expect_status}")
    expected_outcome = None if fixture.expect_global is None else int(fixture.expect_global)
    if record["global_outcome"] != expected_outcome:
        failures.append(f"global_outcome {record['global_outcome']} != {expected_outcome}")
    if record["n_tactics"] != fixture.expect_n_tactics:
        failures.append(f"n_tactics {record['n_tactics']} != {fixture.expect_n_tactics}")
    if record["blamed_index"] != fixture.expect_blamed:
        failures.append(f"blamed {record['blamed_index']} != {fixture.expect_blamed}")
    frame, _prefix = strip_header(record["pred"])
    for tactic, expected in zip(record["tactics"], fixture.expect_tactic_texts):
        start = frame_offset(frame, *tactic["start"])
        finish = frame_offset(frame, *tactic["finish"])
        if frame[start:finish] != expected:
            failures.append(f"tactic {tactic['i']} span text {frame[start:finish]!r} != {expected!r}")
    if len(record["tactics"]) != len(fixture.expect_tactic_texts):
        failures.append(
            f"span text count {len(record['tactics'])} != {len(fixture.expect_tactic_texts)}"
        )
    for tactic in record["tactics"]:
        if (
            tactic["mapping"]["status"] in MAPPABLE_STATUSES
            and tactic["mapping"].get("span_in_response") is not True
        ):
            failures.append(f"tactic {tactic['i']} mapped span is not verbatim in the response")
    if record["tail"]["located"] and record["tail"]["pred_recomputed_matches"] is not True:
        failures.append("recomputed pred does not match the historical extraction")
    if (
        fixture.expect_pp_unprintable is not None
        and record["blamed_pp_unprintable"] != fixture.expect_pp_unprintable
    ):
        failures.append(
            f"blamed_pp_unprintable {record['blamed_pp_unprintable']} "
            f"!= {fixture.expect_pp_unprintable}"
        )
    if (
        fixture.expect_sentinel is not None
        and record["code"]["sentinel"] != fixture.expect_sentinel
    ):
        failures.append(f"sentinel {record['code']['sentinel']!r} != {fixture.expect_sentinel!r}")
    return failures


def run_fixture_validation(
    client: OracleClient, mapper: TokenMapper, *, runs: int = 2
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "experiment": S.EXPERIMENT_ID,
        "purpose": "engineering fixture validation of the process oracle (owner §8, §21)",
        "oracle_endpoint": client.endpoint,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": mapper.identity,
        "fixture_set_sha256": fixture_set_sha256(),
        "runs_per_fixture": runs,
        "health": client.health(),
        "per_fixture": [],
    }
    n_ok = 0
    for fixture in FIXTURES:
        entry: dict[str, Any] = {
            "fid": fixture.fid,
            "klass": fixture.klass,
            "notes": fixture.notes,
            "expectations": {
                "global": fixture.expect_global,
                "status": fixture.expect_status,
                "n_tactics": fixture.expect_n_tactics,
                "blamed_index": fixture.expect_blamed,
                "tactic_texts": list(fixture.expect_tactic_texts),
                "pp_unprintable": fixture.expect_pp_unprintable,
                "sentinel": fixture.expect_sentinel,
            },
            "runs": [],
        }
        hashes = []
        for run_index in range(runs):
            record = run_fixture_once(fixture, client, mapper, f"r{run_index + 1}")
            projection = {
                key: value
                for key, value in record.items()
                if key not in {"oracle", "fixture_run", "pred"}
            }
            hashes.append(S.sha256_text(S.canonical_json(projection)))
            entry["runs"].append(record)
        entry["deterministic"] = len(set(hashes)) == 1
        entry["projection_hashes"] = hashes
        failures = validate_fixture(fixture, entry["runs"][0])
        if not entry["deterministic"]:
            failures.append("nondeterministic across runs")
        entry["validation_failures"] = failures
        entry["ok"] = not failures
        n_ok += int(entry["ok"])
        out["per_fixture"].append(entry)
    classes = sorted({fixture.klass for fixture in FIXTURES})
    out["class_coverage"] = {
        klass: [fixture.fid for fixture in FIXTURES if fixture.klass == klass] for klass in classes
    }
    out["summary"] = {
        "n_fixtures": len(FIXTURES),
        "n_ok": n_ok,
        "n_failed": len(FIXTURES) - n_ok,
        "all_deterministic": all(entry["deterministic"] for entry in out["per_fixture"]),
        "all_validated": n_ok == len(FIXTURES),
    }
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    warm = sub.add_parser("warmup", help="cold canary against the dedicated oracle instance")
    warm.add_argument("--endpoint", default=None)

    fx = sub.add_parser("fixtures", help="run the 12-class fixture validation (>=2x each)")
    fx.add_argument("--out", default=str(S.ORACLE_VALIDATION))
    fx.add_argument("--runs", type=int, default=2)
    fx.add_argument("--endpoint", default=None)

    one = sub.add_parser("verify", help="derive facts for one submitted code string")
    one.add_argument("--pred-file", required=True)
    one.add_argument("--response-file", required=True)
    one.add_argument("--formal-file", required=True)
    one.add_argument("--endpoint", default=None)

    args = parser.parse_args(argv)

    if args.command == "warmup":
        client = OracleClient(endpoint=args.endpoint)
        print(json.dumps(client.warmup(), indent=2))
        return 0

    if args.command == "fixtures":
        client = OracleClient(endpoint=args.endpoint)
        print(f"[warmup] {client.warmup()}")
        mapper = TokenMapper()
        validation = run_fixture_validation(client, mapper, runs=args.runs)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(validation, indent=1) + "\n")
        for entry in validation["per_fixture"]:
            status = "OK " if entry["ok"] else "FAIL"
            observed = entry["runs"][0]
            print(
                f"  {status} {entry['fid']:26s} {entry['klass']:34s} "
                f"status={observed['process_status']:26s} tactics={observed['n_tactics']} "
                f"blamed={observed['blamed_index']}"
            )
            for failure in entry["validation_failures"]:
                print(f"       - {failure}")
        print(json.dumps(validation["summary"], indent=2))
        return 0 if validation["summary"]["all_validated"] else 1

    if args.command == "verify":
        client = OracleClient(endpoint=args.endpoint)
        mapper = TokenMapper()
        pred = Path(args.pred_file).read_text()
        response = Path(args.response_file).read_text()
        formal = Path(args.formal_file).read_text()
        record = derive_facts(
            candidate_id="cli::verify",
            pred=pred,
            response_text=response,
            formal_statement=formal,
            oracle_result=client.verify_raw(pred, custom_id="cli-verify"),
            token_lattice=mapper.lattice(response),
        )
        print(json.dumps(record, indent=1, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
