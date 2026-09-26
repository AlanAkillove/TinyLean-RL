"""V5-P001 fixtures: process oracle, token mapping, Phase-B runner, analyzer.

The fixtures fabricate *oracle items* (what the pinned Lean server would answer),
never the design: positions, spans, projections and expected labels come from the
frozen spec (``scripts/v5_p001_spec.py``) and the frozen design
(``docs/v5/process_oracle_design.md``).  Two conventions are load-bearing
throughout: server lines are 1-based, server columns 0-based, and both messages
and info-tree ranges live in the *frame* -- the submitted code minus a maximal
leading prefix of blank/``import`` lines.

Token-mapping tests use a synthetic one-token-per-character lattice where the
logic of the mapping layer is under test (statuses, containment, credit
precedence); the frozen tokenizer itself has its own test and is skipped when the
weights are absent.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "third_party" / "kimina-prover-rl" / "recipe" / "kimina_prover_rl",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_p001_analyze as A
import v5_p001_process_run as RUN
import v5_p001_spec as S
import v5_process_oracle as O
from kimina_prover_rl.reward.proof_utils import extract_proof_from_text

from tinylean_rl.verifier.policy import Classified, VerifyOutcome

#: Canonical tiny theorem reused by the fixtures: the frame is ``pred`` itself
#: (no blank/import header), so frame positions are readable off the string.
FORMAL = "theorem demo : True := by"
PRED_FAIL = FORMAL + "\n  simp\n  exact bogus"
PRED_SUCCESS = FORMAL + "\n  simp\n  trivial"
SPAN_SIMP = (2, 2, 2, 6)
SPAN_EXACT = (3, 2, 3, 13)
SPAN_TRIVIAL = (3, 2, 3, 9)
SPAN_TRIVIAL_ALONE = (2, 2, 2, 9)  # a one-tactic proof, as in the cohort's solved pairs
NORMALIZED_FORMAL_CHARS = len(FORMAL)


# --------------------------------------------------------------------------------------
# fixture builders
# --------------------------------------------------------------------------------------


def response_for(pred: str) -> str:
    """A model response whose last lean4 block carries exactly ``pred``."""

    return "Let me prove this.\n\n```lean4\n" + pred + "\n```\n"


def tactic_node(name, span, *, pp="tac", synthetic=None):
    rng = {
        "start": {"line": span[0], "column": span[1]},
        "finish": {"line": span[2], "column": span[3]},
    }
    if synthetic is not None:
        rng["synthetic"] = synthetic
    return {
        "node": {
            "name": name,
            "stx": {"range": rng, "pp": pp},
            "goalsBefore": ["g"],
            "goalsAfter": ["g"],
        }
    }


def tactic(index, span, name="Lean.Parser.Tactic.trivial", pp="tac"):
    return O.Tactic(
        index=index,
        name=name,
        start_line=span[0],
        start_col=span[1],
        finish_line=span[2],
        finish_col=span[3],
        goals_before=None,
        goals_after=None,
        pp=pp,
    )


def error_message(line, column, end_line=None, end_column=None):
    entry = {"severity": "error", "pos": {"line": line, "column": column}, "data": "boom"}
    if end_line is not None:
        entry["endPos"] = {"line": end_line, "column": end_column}
    return entry


def item(tactics=(), messages=(), sorries=()):
    return {
        "response": {
            "infotree": list(tactics),
            "messages": list(messages),
            "sorries": list(sorries),
        }
    }


def char_lattice(text: str) -> dict:
    """One token per character: exercises the mapping logic without a tokenizer."""

    return {
        "ids": list(range(len(text))),
        "offsets": [(index, index + 1) for index in range(len(text))],
        "n_tokens": len(text),
        "roundtrip_ok": True,
    }


def call_derive(
    pred,
    response,
    formal,
    *,
    oracle_item=None,
    outcome=VerifyOutcome.LEAN_ERROR,
    lattice=None,
    meta=None,
    candidate_id="seed1:g00:c00",
    d1=S.D1_CANONICAL,
    d2=S.D2_CANONICAL,
):
    oracle_result = None
    if oracle_item is not None:
        oracle_result = {
            "classified": Classified(outcome=outcome, message="oracle message"),
            "item": oracle_item,
            "attempts": [1],
            "seconds": 0.25,
        }
    return O.derive_facts(
        candidate_id=candidate_id,
        pred=pred,
        response_text=response,
        formal_statement=formal,
        oracle_result=oracle_result,
        token_lattice=lattice,
        d1=d1,
        d2=d2,
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# frozen spec
# --------------------------------------------------------------------------------------


def test_spec_constants_are_the_preregistered_ones():
    assert S.EXPERIMENT_ID == "V5-P001"
    assert S.GROUP_N == 8
    assert (S.EXPECTED_GROUPS_TOTAL, S.EXPECTED_GROUPS_PRIMARY) == (720, 686)
    assert (S.EXPECTED_ALL_FAIL, S.EXPECTED_MIXED, S.EXPECTED_ALL_SUCCESS) == (603, 110, 7)
    assert S.PRED_SENTINELS == (
        "No proof found in the output.",
        "Theorem statement couldn't be parsed from statement.",
    )
    assert (S.D1_CANONICAL, S.D2_CANONICAL) == (-0.05, -0.10)
    assert set(S.SENSITIVITY_SETTINGS) == {"canonical_d1_d2", "stronger_gap", "equal_penalty"}
    assert S.SENSITIVITY_SETTINGS["stronger_gap"] == (-0.05, -0.50)
    assert S.SENSITIVITY_SETTINGS["equal_penalty"] == (-0.10, -0.10)
    assert S.TACTIC_NAME_PREFIX == "Lean.Parser.Tactic."
    assert S.TACTIC_WRAPPER_NAMES == {
        "Lean.Parser.Tactic.tacticSeq",
        "Lean.Parser.Tactic.tacticSeq1Indented",
        "Lean.Parser.Tactic.tacticSeqBracketed",
    }
    assert set(S.PROCESS_STATUSES) == {
        "SUCCESS",
        "PREFIX_BEARING_FAILURE",
        "FIRST_TACTIC_FAILURE",
        "PARSE_OR_SYNTAX_FAILURE",
        "FORMAT_NO_CODE",
        "PROCESS_ORACLE_INFRA",
    }
    assert (S.GATE_E1_MIN, S.GATE_E2_MIN) == (0.95, 0.90)
    assert (S.GATE_G1_RATE_MIN, S.GATE_G1_CI_LOWER_MIN) == (0.25, 0.15)
    assert (S.GATE_G2_PER_SEED_MIN, S.GATE_G3_RATE_MIN) == (0.20, 0.15)
    assert (S.GATE_G3_QUARTILES_MIN, S.GATE_QUARTILE_MIN_N) == (3, 5)
    assert S.CLASSIFICATIONS[0] == "PROCESS_SIGNAL_GO"
    assert S.BOOTSTRAP["cluster"] == "component_id"
    assert S.BOOTSTRAP["n_resamples"] == 10000 and S.BOOTSTRAP["method"] == "percentile"
    assert S.TOKENIZER_EXPECTED_VOCAB == 151643
    assert S.TOKENIZER_ADD_SPECIAL_TOKENS is False
    assert O.MAPPABLE_STATUSES == ("exact", "contained")


# --------------------------------------------------------------------------------------
# text layer: frame and generated tail
# --------------------------------------------------------------------------------------


def test_strip_header_stops_at_the_first_non_header_line():
    code = (
        "import Mathlib\n"
        "import Std\n"
        "\n"
        "-- a comment stops the stripping\n"
        "import Late\n"
        "\n"
        "theorem demo : True := by\n"
        "  trivial\n"
    )
    frame, prefix_chars = O.strip_header(code)
    assert prefix_chars == len("import Mathlib\nimport Std\n\n")
    assert frame == code[prefix_chars:]
    assert frame.startswith("-- a comment")
    assert "import Late" in frame and "theorem demo" in frame
    assert O.strip_header("theorem demo : True := by\n  trivial") == (
        "theorem demo : True := by\n  trivial",
        0,
    )


def test_frame_offset_is_one_based_lines_and_zero_based_columns():
    frame = "-- c\nimport Late\n  \ntheorem demo : True := by\n  trivial\n"
    assert frame[O.frame_offset(frame, 1, 0):].startswith("-- c")
    assert frame[O.frame_offset(frame, 4, 0):].startswith("theorem")
    assert frame[O.frame_offset(frame, 5, 0):].startswith("  trivial")
    assert O.frame_offset(frame, 4, 2) == O.frame_offset(frame, 4, 0) + 2
    assert frame[O.frame_offset(frame, 4, 2):].startswith("eorem")
    assert O.frame_offset(frame, 1, 0) == 0
    for line, column in ((0, 0), (9, 0), (1, 99), (1, -1)):
        with pytest.raises(O.PositionError):
            O.frame_offset(frame, line, column)


def test_locate_generated_tail_agrees_with_the_upstream_extraction():
    response = response_for(PRED_FAIL)
    out = O.locate_generated_tail(response, FORMAL)
    assert out["found"] is True
    assert out["pred_recomputed"] == extract_proof_from_text(response, FORMAL) == PRED_FAIL
    assert response[out["tail_offset_in_response"] :] == "\n  simp\n  exact bogus\n```\n"
    assert out["n_lean4_blocks"] == 1


def test_locate_generated_tail_skips_commented_and_incomplete_blocks():
    commented = response_for("-- " + FORMAL + "\n  simp")
    real = response_for(PRED_SUCCESS)
    response = "Try one.\n\n" + real + "\nAnother.\n\n" + commented
    out = O.locate_generated_tail(response, FORMAL)
    assert out["n_lean4_blocks"] == 2
    assert out["tail"] == "\n  simp\n  trivial"
    assert O.locate_generated_tail(response_for("theorem demo : True := sorry"), FORMAL)["found"] is False


# --------------------------------------------------------------------------------------
# tactic extraction and blame
# --------------------------------------------------------------------------------------


def test_extract_tactics_skips_wrappers_synthetics_and_exact_duplicates():
    tree = [
        tactic_node("Lean.Parser.Tactic.tacticSeq", (1, 0, 4, 0)),
        tactic_node("Lean.Parser.Tactic.simp", (2, 2, 2, 6), pp="simp"),
        tactic_node("Lean.Parser.Tactic.simp", (2, 2, 2, 6), pp="simp"),
        tactic_node("Lean.Parser.Tactic.rw", (3, 2, 3, 8), synthetic=True),
        tactic_node("Lean.Parser.Tactic.exact", (4, 2, 4, 13)),
        tactic_node("Foo.Bar.custom", (5, 2, 5, 4)),
        {"node": "not a dict"},
        {"node": {"name": "Lean.Parser.Tactic.apply"}},
    ]
    tactics, stats = O.extract_tactics(tree)
    assert [entry.name for entry in tactics] == [
        "Lean.Parser.Tactic.simp",
        "Lean.Parser.Tactic.exact",
    ]
    assert [entry.index for entry in tactics] == [0, 1]
    assert stats == {
        "nodes_total": 8,
        "tactic_named": 6,
        "wrappers_skipped": 1,
        "synthetic_skipped": 1,
        "malformed_skipped": 2,
        "duplicates_removed": 1,
        "tactics": 2,
    }


def test_blame_contains_prefers_the_innermost_tactic():
    tactics = [tactic(0, (2, 2, 4, 10)), tactic(1, (3, 4, 3, 12))]
    blamed, unmapped, kind = O.blame_index(tactics, [(3, 5, 3, 6)])
    assert (blamed, unmapped, kind) == (1, 0, "contains")


def test_blame_range_last_is_the_point_where_elaboration_stopped():
    # `unsolved goals` starts right after `by` and covers the whole tactic block.
    tactics = [tactic(0, (3, 2, 3, 5)), tactic(1, (4, 2, 4, 5))]
    blamed, unmapped, kind = O.blame_index(tactics, [(2, 1, 5, 0)])
    assert (blamed, unmapped, kind) == (1, 0, "range_last")
    # Without an end position the range cannot be attributed from its interior.
    blamed, unmapped, kind = O.blame_index(tactics, [(2, 1, None, None)])
    assert (blamed, unmapped, kind) == (None, 1, None)


def test_blame_returns_the_earliest_tactic_regardless_of_message_order():
    tactics = [tactic(0, (3, 4, 3, 12)), tactic(1, (4, 4, 4, 12))]
    blamed, unmapped, kind = O.blame_index(tactics, [(4, 5, 4, 6), (3, 5, 3, 6)])
    assert (blamed, unmapped, kind) == (0, 0, "contains")
    assert O.blame_index([], [(1, 0, None, None)]) == (None, 1, None)


# --------------------------------------------------------------------------------------
# candidate derivation: statuses, labels, credit
# --------------------------------------------------------------------------------------


def test_no_code_candidate_gets_no_labels_of_any_kind():
    record = call_derive(S.SENTINEL_NO_PROOF, "irrelevant", FORMAL)
    assert record["code"]["extractable"] is False
    assert record["code"]["sentinel"] == S.SENTINEL_NO_PROOF
    assert record["process_status"] == "FORMAT_NO_CODE"
    assert record["oracle"] is None and record["global_outcome"] is None
    assert record["tactics"] == [] and record["blamed_index"] is None
    assert record["token_credit"]["n_credit_tokens"] == 0
    assert O.structured_recoverable(record) is False and O.any_active(record) is False


def test_infrastructure_outcome_is_censored_not_failed():
    response = response_for(PRED_FAIL)
    record = call_derive(
        PRED_FAIL,
        response,
        FORMAL,
        oracle_item=item(
            [tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP), tactic_node("Lean.Parser.Tactic.exact", SPAN_EXACT)],
            [error_message(3, 2)],
        ),
        outcome=VerifyOutcome.VERIFIER_TIMEOUT,
        lattice=char_lattice(response),
    )
    assert record["process_status"] == "PROCESS_ORACLE_INFRA"
    assert record["oracle"]["infra"] is True
    assert record["global_outcome"] is None and record["failure_kind"] is None
    assert all(entry["label"] is None for entry in record["tactics"])
    assert record["token_credit"]["n_credit_tokens"] == 0
    assert record["token_credit"]["blamed_mappable"] is None
    assert O.structured_recoverable(record) is False


def test_success_credits_every_parsed_tactic_with_the_success_label():
    response = response_for(PRED_SUCCESS)
    record = call_derive(
        PRED_SUCCESS,
        response,
        FORMAL,
        oracle_item=item(
            [tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP), tactic_node("Lean.Parser.Tactic.trivial", SPAN_TRIVIAL)]
        ),
        outcome=VerifyOutcome.VERIFIED,
        lattice=char_lattice(response),
    )
    assert record["process_status"] == "SUCCESS"
    assert record["global_outcome"] == 1
    assert [entry["label"] for entry in record["tactics"]] == ["success", "success"]
    assert record["token_credit"]["n_credit_tokens"] == 2
    assert len(record["token_credit"]["success_token_positions"]) == 2
    assert O.structured_recoverable(record) is False  # a solved candidate needs no recovery


def test_prefix_bearing_failure_labels_d1_before_the_first_error_only():
    response = response_for(PRED_FAIL)
    record = call_derive(
        PRED_FAIL,
        response,
        FORMAL,
        oracle_item=item(
            [tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP), tactic_node("Lean.Parser.Tactic.exact", SPAN_EXACT)],
            [error_message(3, 2, 3, 13)],
        ),
        lattice=char_lattice(response),
    )
    assert record["process_status"] == "PREFIX_BEARING_FAILURE"
    assert record["blamed_index"] == 1 and record["blame_kind"] == "contains"
    assert record["failure_kind"] == "error"
    assert [entry["label"] for entry in record["tactics"]] == ["d1", "d2"]
    assert [entry["locally_verified"] for entry in record["tactics"]] == [True, False]
    assert record["token_credit"]["d1_token_positions"] and record["token_credit"]["d2_token_positions"]
    assert record["token_credit"]["blamed_mappable"] is True
    assert record["token_credit"]["n_credit_conflicts"] == 0
    assert O.verified_prefix_count(record) == 1
    assert O.any_active(record) is True
    # Owner §11: the blamed tactic is mappable and a verified tactic precedes it.
    assert O.structured_recoverable(record) is True


def test_first_tactic_failure_is_not_structured_recoverable():
    response = response_for(PRED_FAIL)
    record = call_derive(
        PRED_FAIL,
        response,
        FORMAL,
        oracle_item=item(
            [tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP), tactic_node("Lean.Parser.Tactic.exact", SPAN_EXACT)],
            [error_message(2, 2, 2, 6)],
        ),
        lattice=char_lattice(response),
    )
    assert record["process_status"] == "FIRST_TACTIC_FAILURE"
    assert record["blamed_index"] == 0
    assert [entry["label"] for entry in record["tactics"]] == ["d2", "d2"]
    # The trivial "every parsed tactic is immediately erroneous" case is excluded.
    assert O.structured_recoverable(record) is False
    assert O.any_active(record) is False


def test_sorry_only_failures_use_the_sorry_ranges_for_blame():
    response = response_for(PRED_FAIL)
    record = call_derive(
        PRED_FAIL,
        response,
        FORMAL,
        oracle_item=item(
            [tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP), tactic_node("Lean.Parser.Tactic.exact", SPAN_EXACT)],
            sorries=[{"pos": {"line": 3, "column": 2}, "endPos": {"line": 3, "column": 13}}],
        ),
        lattice=char_lattice(response),
    )
    assert record["process_status"] == "PREFIX_BEARING_FAILURE"
    assert record["blamed_index"] == 1 and record["blame_kind"] == "contains"
    assert record["failure_kind"] == "sorry"
    assert record["n_sorries"] == 1
    assert O.structured_recoverable(record) is True


def test_failure_without_any_parsed_tactic_is_a_parse_failure():
    response = response_for(PRED_FAIL)
    record = call_derive(
        PRED_FAIL,
        response,
        FORMAL,
        oracle_item=item([], [error_message(1, 0, 1, 5)]),
        lattice=char_lattice(response),
    )
    assert record["process_status"] == "PARSE_OR_SYNTAX_FAILURE"
    assert record["blamed_index"] is None and record["token_credit"]["n_credit_tokens"] == 0
    assert record["failure_kind"] == "error"


def test_conflicts_prefer_the_error_region_over_the_verified_prefix():
    response = response_for(PRED_FAIL)
    record = call_derive(
        PRED_FAIL,
        response,
        FORMAL,
        oracle_item=item(
            [tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP), tactic_node("Lean.Parser.Tactic.exact", SPAN_EXACT)],
            [error_message(3, 2, 3, 13)],
        ),
        lattice=char_lattice(response),
    )
    assert record["token_credit"]["n_credit_conflicts"] == 0
    records = record["tactics"]
    # Two tactics sharing one first token: the error region must win (frozen precedence).
    shared = [
        {"mapping": {"status": "exact", "token_index": 7, "token_id": 11, "span_in_response": True}, "label": "d1"},
        {"mapping": {"status": "exact", "token_index": 7, "token_id": 11, "span_in_response": True}, "label": "d2"},
    ]
    credit = O._token_credit(shared, infra=False, blamed=1)
    assert credit["n_credit_conflicts"] == 1
    assert credit["n_credit_tokens"] == 1
    assert credit["d2_token_positions"] == [[7, 11]] and credit["d1_token_positions"] == []
    kept = O._token_credit(list(reversed(shared)), infra=False, blamed=1)
    assert kept["d2_token_positions"] == [[7, 11]] and kept["d1_token_positions"] == []
    assert O._CREDIT_PRECEDENCE["d2"] > O._CREDIT_PRECEDENCE["d1"] > O._CREDIT_PRECEDENCE["success"]
    assert records[0]["mapping"]["status"] in O.MAPPABLE_STATUSES


# --------------------------------------------------------------------------------------
# mapping statuses
# --------------------------------------------------------------------------------------


def _map(tactic_entry, *, frame=PRED_FAIL, prefix_chars=0, response=None, lattice=None, **overrides):
    response = response_for(PRED_FAIL) if response is None else response
    kwargs = {
        "frame": frame,
        "prefix_chars": prefix_chars,
        "normalized_formal_chars": NORMALIZED_FORMAL_CHARS,
        "tail_offset": O.locate_generated_tail(response, FORMAL)["tail_offset_in_response"],
        "response_text": response,
        "token_lattice": lattice,
    }
    kwargs.update(overrides)
    return O._map_tactic(tactic_entry, **kwargs)


def test_mapping_status_exact_and_contained():
    entry = tactic(0, SPAN_SIMP)
    response = response_for(PRED_FAIL)
    offset = response.index("simp")
    exact = _map(entry, response=response, lattice=char_lattice(response))
    assert exact["status"] == "exact" and exact["span_in_response"] is True
    assert exact["response_offset"] == offset and exact["token_index"] == offset
    merged = _map(entry, response=response, lattice={"ids": [1], "offsets": [(0, len(response))]})
    assert merged["status"] == "contained" and merged["token_start"] == 0
    assert merged["response_offset"] == offset


def test_mapping_status_outside_response_response_only_and_unmapped():
    entry = tactic(0, SPAN_SIMP)
    response = response_for(PRED_FAIL)
    outside = _map(entry, response=response, normalized_formal_chars=len(PRED_FAIL))
    assert outside["status"] == "outside_response" and outside["response_offset"] is None
    response_only = _map(entry, response=response, lattice=None)
    assert response_only["status"] == "response_only"
    assert response_only["span_in_response"] is True
    far = [(0, 1), (len(response) - 1, len(response))]
    unmapped = _map(entry, response=response, lattice={"ids": [1, 2], "offsets": far})
    assert unmapped["status"] == "unmapped" and unmapped["token_index"] is None
    # A span that does not fit the frame carries no offset at all.
    broken = _map(tactic(0, (9, 0, 9, 4)), response=response, lattice=char_lattice(response))
    assert broken["status"] == "unmapped" and broken["pred_offset"] is None


def test_mapping_marks_spans_that_are_not_verbatim_in_the_response():
    entry = tactic(0, SPAN_SIMP)
    response = response_for(PRED_FAIL).replace("  simp", "  omega")
    mapped = _map(entry, response=response, lattice=char_lattice(response))
    assert mapped["span_in_response"] is False
    assert mapped["status"] in O.MAPPABLE_STATUSES
    credit = O._token_credit(
        [{"mapping": mapped, "label": "d1"}, {"mapping": mapped, "label": "d2"}],
        infra=False,
        blamed=1,
    )
    assert credit["n_credit_tokens"] == 0
    assert credit["n_span_not_in_response"] == 2 and credit["n_mapped"] == 0
    assert credit["blamed_mappable"] is False


def test_mapping_ambiguous_is_defensive(monkeypatch):
    entry = tactic(0, SPAN_SIMP)
    response = response_for(PRED_FAIL)
    monkeypatch.setattr(
        O.TokenMapper,
        "first_token_at",
        staticmethod(lambda lattice, offset: {"token_index": 0, "token_id": 1, "token_start": offset + 1, "token_end": offset + 5, "exact": False}),
    )
    assert _map(entry, response=response, lattice=char_lattice(response))["status"] == "ambiguous"


def test_mapping_uses_the_frame_after_a_stripped_header():
    # A historical pred can carry the dataset's leading imports inside the
    # normalized formal statement; the pinned server strips that prefix and every
    # reported position is relative to the frame that remains.
    formal = "import Mathlib\n\n" + FORMAL
    pred = formal + "\n  simp\n  exact bogus"
    frame, prefix_chars = O.strip_header(pred)
    assert prefix_chars == len("import Mathlib\n\n")
    assert frame == PRED_FAIL
    response = response_for(pred)
    locate = O.locate_generated_tail(response, formal)
    assert locate["pred_recomputed"] == pred
    entry = tactic(0, SPAN_SIMP)
    mapped = _map(
        entry,
        frame=frame,
        prefix_chars=prefix_chars,
        response=response,
        lattice=char_lattice(response),
        normalized_formal_chars=len(formal),
        tail_offset=locate["tail_offset_in_response"],
    )
    assert mapped["pred_offset"] == prefix_chars + O.frame_offset(frame, *SPAN_SIMP[:2])
    assert mapped["span_in_response"] is True
    assert mapped["status"] == "exact"


# --------------------------------------------------------------------------------------
# frozen tokenizer
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not (S.TOKENIZER_DIR / "tokenizer.json").exists(), reason="frozen tokenizer weights absent"
)
def test_frozen_tokenizer_identity_and_first_token_lookup():
    mapper = O.TokenMapper()
    assert mapper.vocab_size == S.TOKENIZER_EXPECTED_VOCAB
    assert mapper.identity["tokenizer_json_sha256"] == S.sha256_file(
        S.TOKENIZER_DIR / "tokenizer.json"
    )
    assert mapper.identity["add_special_tokens"] is False
    response = response_for(PRED_FAIL)
    lattice = mapper.lattice(response)
    assert lattice["roundtrip_ok"] is True
    assert lattice["n_tokens"] == len(lattice["ids"]) == len(lattice["offsets"])
    for needle in ("simp", "exact"):
        offset = response.index(needle)
        token = O.TokenMapper.first_token_at(lattice, offset)
        assert token is not None
        assert token["token_start"] <= offset < token["token_end"]
    assert O.TokenMapper.first_token_at(lattice, len(response)) is None


# --------------------------------------------------------------------------------------
# Phase-B runner: plan, projection, logs, guards
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not S.HISTORICAL_SURFACE.exists(), reason="frozen historical surface not built yet"
)
def test_build_plan_reproduces_the_frozen_processing_order():
    surface = RUN.load_surface()
    planned = RUN.build_plan(surface)
    primary_groups = [group for group in surface["groups"] if not group["contaminated"]]
    assert len(surface["groups"]) == S.EXPECTED_GROUPS_TOTAL
    assert len(primary_groups) == S.EXPECTED_GROUPS_PRIMARY
    labels = Counter(group["label"] for group in primary_groups)
    assert labels[S.GROUP_ALL_FAIL] == 575
    assert labels[S.GROUP_MIXED] == 104
    assert labels[S.GROUP_ALL_SUCCESS] == 7
    assert len(planned) == sum(len(group["candidates"]) for group in primary_groups) == 5488
    assert sum(1 for entry in planned if entry["extractable"]) == 2476
    assert sum(1 for entry in planned if not entry["extractable"]) == 3012
    assert all(not entry["infra"] for entry in planned)
    assert len({entry["candidate_id"] for entry in planned}) == len(planned)
    for entry in planned:
        info = next(g for g in primary_groups if g["group_key"] == entry["group_key"])
        assert entry["group_label"] == info["label"]
        assert entry["statement_id"] == info["statement_id"]
    order = S.sha256_text(S.canonical_json([entry["candidate_id"] for entry in planned]))
    assert order == S.sha256_text(
        S.canonical_json([entry["candidate_id"] for entry in RUN.build_plan(surface)])
    )
    assert order.startswith("d04283eeb133f94b")
    assert {entry["seed"] for entry in planned} == set(S.SEED_DIRS)


def test_record_hash_ignores_only_the_oracle_wall_clock():
    record = {
        "candidate_id": "seed1:g00:c00",
        "process_status": "SUCCESS",
        "oracle": {"outcome": "verified", "infra": False, "seconds": 12.5},
        "token_credit": {"n_credit_tokens": 2},
    }
    same = json.loads(json.dumps(record))
    same["oracle"]["seconds"] = 99.0
    assert RUN.record_hash(record) == RUN.record_hash(same)
    for field, value in (
        ("process_status", "PREFIX_BEARING_FAILURE"),
        ("token_credit", {"n_credit_tokens": 3}),
    ):
        changed = json.loads(json.dumps(record))
        changed[field] = value
        assert RUN.record_hash(changed) != RUN.record_hash(record)
    changed = json.loads(json.dumps(record))
    changed["oracle"]["outcome"] = "lean_error"
    assert RUN.record_hash(changed) != RUN.record_hash(record)
    assert "seconds" not in RUN.stable_projection(record)["oracle"]


def test_raw_name_round_trips_candidate_ids():
    assert RUN.raw_name("seed1:0001:g00:c00") == "seed1_0001_g00_c00.json"
    assert RUN.raw_name("plain") == "plain.json"


def test_labels_log_recovers_a_partial_tail_and_rejects_bad_records(tmp_path):
    path = tmp_path / "labels.jsonl"
    planned = {"seed1:g00:c00", "seed1:g00:c01"}
    log = RUN.LabelsLog(path)
    assert log.load(planned) == {}
    log.append({"candidate_id": "seed1:g00:c00", "process_status": "SUCCESS"})
    log.append({"candidate_id": "seed1:g00:c01", "process_status": "FORMAT_NO_CODE"})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"candidate_id": "seed1:g00:c02"')  # crash during an append
    resumed = RUN.LabelsLog(path)
    existing = resumed.load(planned)
    assert resumed.n_truncated_suffixes == 1
    assert set(existing) == planned
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert existing["seed1:g00:c00"] == RUN.record_hash(
        {"candidate_id": "seed1:g00:c00", "process_status": "SUCCESS"}
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"candidate_id": "seed1:g00:c00"}) + "\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        RUN.LabelsLog(path).load(planned)
    path.write_text(json.dumps({"candidate_id": "seed9:g99:c99"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unplanned"):
        RUN.LabelsLog(path).load(planned)


def test_assert_stamp_matches_fails_closed_on_any_edit():
    stamp = {"git_head": "a" * 40, "scripts": {"v5_p001_spec.py": "b" * 64}}
    RUN.assert_stamp_matches({"code_stamp": json.loads(json.dumps(stamp))}, stamp)
    edited = json.loads(json.dumps(stamp))
    edited["scripts"]["v5_p001_spec.py"] = "c" * 64
    with pytest.raises(RuntimeError, match="run stamp mismatch"):
        RUN.assert_stamp_matches({"code_stamp": edited}, stamp)


def test_the_analyzer_and_the_runner_share_the_frozen_stamp_scripts():
    assert "v5_p001_analyze.py" in RUN.STAMP_SCRIPTS
    assert set(RUN.STAMP_SCRIPTS) == {
        "v5_p001_spec.py",
        "v5_process_oracle.py",
        "v5_p001_reconstruct.py",
        "v5_p001_process_run.py",
        "v5_p001_analyze.py",
    }
    for name in RUN.STAMP_SCRIPTS:
        assert (ROOT / "scripts" / name).exists()


def test_the_canonical_run_directory_refuses_a_debug_limited_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PROCESS_RUN_DIR", tmp_path)
    monkeypatch.setattr(S, "PROCESS_LABELS", tmp_path / Path(S.PROCESS_LABELS).name)
    monkeypatch.setattr(S, "PROCESS_FREEZE", tmp_path / Path(S.PROCESS_FREEZE).name)
    monkeypatch.setattr(S, "PROCESS_VALIDATION", tmp_path / Path(S.PROCESS_VALIDATION).name)
    paths = RUN.RunPaths(tmp_path)
    stamp = {"git_head": "a" * 40}
    paths.run_meta.write_text(
        json.dumps({"code_stamp": stamp, "debug_limit": 3}) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="debug-limited"):
        RUN.stage_freeze(paths, planned=[], stamp=stamp)
    # A scratch directory accepts the same debug-limited freeze (that is what the
    # fly122 smoke run used); only the canonical directory is protected.
    scratch = RUN.RunPaths(tmp_path / "scratch")
    scratch.root.mkdir(parents=True, exist_ok=True)
    scratch.run_meta.write_text(
        json.dumps({"code_stamp": stamp, "debug_limit": 3}) + "\n", encoding="utf-8"
    )
    scratch.labels.write_text("", encoding="utf-8")
    assert RUN.stage_freeze(scratch, planned=[], stamp=stamp) == 0
    assert json.loads(scratch.freeze.read_text(encoding="utf-8"))["debug_limit"] == 3
    # A different stamp is refused before the debug rule is even reached.
    with pytest.raises(RuntimeError, match="run stamp mismatch"):
        RUN.stage_freeze(scratch, planned=[], stamp={"git_head": "b" * 40})


# --------------------------------------------------------------------------------------
# analyzer: guards, gates, classification, report
# --------------------------------------------------------------------------------------


def _synthetic_cohort():
    """24 evaluable all-fail groups plus mixed/success/contaminated-excluded groups.

    Every candidate goes through the production derivation path (``derive_facts``
    with the same meta keys the runner passes), so the analyzer is exercised on
    records it could actually receive.  One all-fail group also carries a sentinel
    and a censored candidate, so the censoring trichotomy enters the denominators
    exactly as it does in production.
    """

    surface_groups: list[dict] = []
    plan: list[dict] = []
    records: list[dict] = []

    def add(group, candidate_id, formal, pred, response, *, oracle_item=None, outcome=None, lattice=True):
        meta = {
            "group_key": group["group_key"],
            "seed": group["seed"],
            "group_label": group["label"],
            "component_id": group["component_id"],
            "statement_id": group["statement_id"],
        }
        records.append(
            call_derive(
                pred,
                response,
                formal,
                oracle_item=oracle_item,
                outcome=outcome or VerifyOutcome.LEAN_ERROR,
                lattice=char_lattice(response) if lattice else None,
                meta=meta,
                candidate_id=candidate_id,
            )
        )
        plan.append(_plan_entry(candidate_id, meta, pred, response))

    def group(seed, seed_index, index, label):
        return {
            "group_key": f"{seed}:g{index:02d}",
            "seed": seed,
            "component_id": f"comp-{seed_index}",
            "statement_id": f"stmt-{seed_index}-{index}",
            "label": label,
            "contaminated": False,
            "candidates": [],
        }

    fail_item = item(
        [
            tactic_node("Lean.Parser.Tactic.simp", SPAN_SIMP),
            tactic_node("Lean.Parser.Tactic.exact", SPAN_EXACT),
        ],
        [error_message(3, 2, 3, 13)],
    )
    solved_item = item([tactic_node("Lean.Parser.Tactic.trivial", SPAN_TRIVIAL_ALONE)])

    for seed_index, seed in enumerate(sorted(S.SEED_DIRS)):
        for group_index in range(8):
            entry = group(seed, seed_index, group_index, S.GROUP_ALL_FAIL)
            surface_groups.append(entry)
            formal = f"theorem t{seed_index}{group_index} : True := by"
            pred = formal + "\n  simp\n  exact bogus"
            response = response_for(pred)
            for slot in range(2):
                add(entry, f"{seed}:{group_index:02d}:c{slot:02d}", formal, pred, response, oracle_item=fail_item)
            if seed_index == 0 and group_index == 0:
                add(entry, f"{seed}:{group_index:02d}:c98", formal, S.SENTINEL_NO_PROOF, "no code", lattice=False)
                add(
                    entry,
                    f"{seed}:{group_index:02d}:c99",
                    formal,
                    pred,
                    response,
                    oracle_item=fail_item,
                    outcome=VerifyOutcome.VERIFIER_TIMEOUT,
                )
        mixed = group(seed, seed_index, 8, S.GROUP_MIXED)
        solved_formal = f"theorem m{seed_index} : True := by"
        add(
            mixed,
            f"{seed}:08:c00",
            solved_formal,
            solved_formal + "\n  trivial",
            response_for(solved_formal + "\n  trivial"),
            oracle_item=solved_item,
            outcome=VerifyOutcome.VERIFIED,
        )
        add(
            mixed,
            f"{seed}:08:c01",
            solved_formal,
            solved_formal + "\n  simp\n  exact bogus",
            response_for(solved_formal + "\n  simp\n  exact bogus"),
            oracle_item=fail_item,
        )
        surface_groups.append(mixed)
        if seed_index == 1:
            solved = group(seed, seed_index, 9, S.GROUP_ALL_SUCCESS)
            add(
                solved,
                f"{seed}:09:c00",
                solved_formal,
                solved_formal + "\n  trivial",
                response_for(solved_formal + "\n  trivial"),
                oracle_item=solved_item,
                outcome=VerifyOutcome.VERIFIED,
            )
            surface_groups.append(solved)
    surface = {
        "artifact": "v5_historical_surface",
        "groups": surface_groups,
        "verification": {
            "n_groups": len(surface_groups),
            "primary_groups": len(surface_groups),
            "labels": {},
            "primary_labels": {},
            "contaminated_groups": [],
        },
    }
    return surface, plan, records


def _plan_entry(candidate_id, meta, pred, response):
    return {
        "candidate_id": candidate_id,
        "group_key": meta["group_key"],
        "seed": meta["seed"],
        "group_label": meta["group_label"],
        "component_id": meta["component_id"],
        "statement_id": meta["statement_id"],
        "slot": 0,
        "file": "fixture.jsonl",
        "line": 1,
        "score": 0.0,
        "acc": 0.0,
        "infra": False,
        "pred_kind": "sentinel" if pred in S.PRED_SENTINELS else "code",
        "pred_sha256": S.sha256_text(pred),
        "response_sha256": S.sha256_text(response),
        "extractable": pred not in S.PRED_SENTINELS,
    }


STAMP = {"git_head": "f" * 40, "scripts": {"v5_p001_spec.py": "0" * 64}}


def _write_run_dir(
    root: Path,
    *,
    surface,
    plan,
    records,
    oracle_validation=None,
    freeze_passed=True,
    validation_passed=True,
    debug_limit=None,
    tamper_labels=False,
):
    root.mkdir(parents=True, exist_ok=True)
    paths = RUN.RunPaths(root)
    labels_text = "".join(S.canonical_json(record) + "\n" for record in records)
    paths.labels.write_text(labels_text, encoding="utf-8")
    paths.raw_manifest.write_text(
        json.dumps({"n_files": 0, "total_bytes": 0, "files": []}, indent=1) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "artifact": "v5_p001_process_freeze",
        "experiment": S.EXPERIMENT_ID,
        "code_stamp": STAMP,
        "debug_limit": debug_limit,
        "labels": {
            "path": str(paths.labels),
            "sha256": S.sha256_file(paths.labels),
            "n_records": len(records),
        },
        "raw": {"manifest_sha256": S.sha256_file(paths.raw_manifest), "n_files": 0},
        "infrastructure": {
            "n_infra_censored": 0,
            "n_submissions": 3,
            "n_recoveries": 0,
            "n_isolated_retries": 0,
        },
        "coverage": {"n_planned": len(plan), "n_processed": len(records), "n_missing": 0},
        "passed": freeze_passed,
    }
    paths.freeze.write_text(json.dumps(freeze, indent=1) + "\n", encoding="utf-8")
    paths.run_meta.write_text(
        json.dumps(
            {
                "code_stamp": STAMP,
                "debug_limit": debug_limit,
                "tokenizer_identity": {"dir": "models/weights/kimina_distill_0_6b"},
                "oracle": {"endpoint": "http://127.0.0.1:8020", "image_digest": "sha256:fixture"},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.validation.write_text(
        json.dumps(
            {
                "passed": validation_passed,
                "summary": {"n_records": len(records), "all_validated": validation_passed},
                "checks": {"n_records": len(records)},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    surface_path = root / "surface.json"
    surface_path.write_text(json.dumps(surface, indent=1) + "\n", encoding="utf-8")
    oracle_validation_path = root / "oracle_validation.json"
    oracle_validation_path.write_text(
        json.dumps(
            oracle_validation
            or {
                "summary": {"all_validated": True, "all_deterministic": True},
                "fixture_set_sha256": "deadbeef",
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    if tamper_labels:
        paths.labels.write_text(labels_text + "\n", encoding="utf-8")
    return paths, surface_path, oracle_validation_path


def _load(root, surface_path, oracle_validation_path, monkeypatch, plan):
    """Load a synthetic run directory with the frozen artifact paths redirected."""

    monkeypatch.setattr(S, "HISTORICAL_SURFACE", surface_path)
    monkeypatch.setattr(S, "ORACLE_VALIDATION", oracle_validation_path)
    monkeypatch.setattr(A, "code_stamp", lambda: STAMP)
    monkeypatch.setattr(A, "build_plan", lambda surface: plan)
    return A.load_inputs(root)


@pytest.fixture
def cohort(tmp_path):
    surface, plan, records = _synthetic_cohort()
    root = tmp_path / "run"
    _paths, surface_path, oracle_validation_path = _write_run_dir(
        root, surface=surface, plan=plan, records=records
    )
    return {
        "root": root,
        "surface": surface,
        "surface_path": surface_path,
        "oracle_validation_path": oracle_validation_path,
        "plan": plan,
        "records": records,
    }


def test_load_inputs_accepts_the_fixture_and_fails_closed_on_every_mutation(
    cohort, tmp_path, monkeypatch
):
    loaded = _load(
        cohort["root"],
        cohort["surface_path"],
        cohort["oracle_validation_path"],
        monkeypatch,
        cohort["plan"],
    )
    assert len(loaded["records"]) == len(cohort["plan"]) == len(cohort["records"])
    assert loaded["freeze"]["passed"] is True

    serial = [0]

    def build(**overrides):
        serial[0] += 1
        root = tmp_path / f"run-{serial[0]}"
        _write_run_dir(
            root,
            surface=cohort["surface"],
            plan=cohort["plan"],
            records=cohort["records"],
            **overrides,
        )
        return root

    with pytest.raises(A.AnalyzerError, match="freeze did not pass"):
        _load(build(freeze_passed=False), cohort["surface_path"], cohort["oracle_validation_path"], monkeypatch, cohort["plan"])
    with pytest.raises(A.AnalyzerError, match="structural validation did not pass"):
        _load(build(validation_passed=False), cohort["surface_path"], cohort["oracle_validation_path"], monkeypatch, cohort["plan"])
    with pytest.raises(A.AnalyzerError, match="labels file changed"):
        _load(build(tamper_labels=True), cohort["surface_path"], cohort["oracle_validation_path"], monkeypatch, cohort["plan"])
    with pytest.raises(A.AnalyzerError, match="debug limit"):
        _load(build(debug_limit=5), cohort["surface_path"], cohort["oracle_validation_path"], monkeypatch, cohort["plan"])
    root = build(oracle_validation={"summary": {"all_validated": True, "all_deterministic": False}})
    with pytest.raises(A.AnalyzerError, match="not deterministic"):
        _load(root, root / "surface.json", root / "oracle_validation.json", monkeypatch, cohort["plan"])
    # The plan and the label file must agree exactly, in both directions.
    root = build()
    with pytest.raises(A.AnalyzerError, match="records for"):
        _load(root, root / "surface.json", root / "oracle_validation.json", monkeypatch, cohort["plan"][:-1])
    root = build()
    (root / RUN.RunPaths(root).validation.name).unlink()
    with pytest.raises(A.AnalyzerError, match="structural validation artifact is missing"):
        _load(root, root / "surface.json", root / "oracle_validation.json", monkeypatch, cohort["plan"])


def test_candidate_class_trichotomy_and_denominators(cohort):
    records = cohort["records"]
    classes = {A.candidate_class(record) for record in records}
    assert classes == {"conclusive", "censored", "no_code"}
    e2 = A.compute_e2(records)
    assert e2["n_code"] == sum(
        1 for record in records if record["process_status"] != "FORMAT_NO_CODE"
    )
    assert e2["n_censored"] == 1
    assert e2["rate"] == (e2["n_code"] - 1) / e2["n_code"]
    assert e2["passed"] is True and e2["censored_share"] < 0.05
    e1 = A.compute_e1(records)
    assert e1["rate"] == 1.0 and e1["passed"] is True
    assert e1["n_outside_response"] == 0 and e1["n_span_not_in_response"] == 0


def test_full_report_reaches_the_frozen_go_classification(cohort, monkeypatch, capsys):
    inputs = _load(
        cohort["root"],
        cohort["surface_path"],
        cohort["oracle_validation_path"],
        monkeypatch,
        cohort["plan"],
    )
    report = A.build_report(inputs)
    gates = report["gates"]
    assert gates["E1"]["passed"] and gates["E2"]["passed"]
    assert gates["G1"]["point"] == 1.0 and gates["G1"]["ci_lower"] > S.GATE_G1_CI_LOWER_MIN
    assert gates["G1"]["passed"] is True
    assert all(entry["passed"] for entry in gates["G2"].values())
    assert gates["G3"]["n_passed"] == 4
    assert report["FINAL_CLASSIFICATION"]["classification"] == "PROCESS_SIGNAL_GO"
    assert report["FINAL_CLASSIFICATION"]["failed_gates"] == []
    assert report["FINAL_CLASSIFICATION"]["inconclusive_trigger"] is None
    primary = report["primary_recovery"]
    assert primary["n_primary_all_fail_groups"] == 24
    assert primary["n_evaluable_groups"] == 24 and primary["n_excluded_groups_all_censored"] == 0
    assert primary["n_recoverable_groups"] == 24 and primary["rate"] == 1.0
    assert primary["no_code_candidates_in_population"] == 1
    assert primary["censored_candidates_in_population"] == 1
    assert primary["candidate_level_secondary"] == {
        "n_code_all_fail_candidates": 49,
        "n_recoverable_candidates": 48,
        "rate": 48 / 49,
    }
    assert report["candidate_mechanism"]["n_recovering_candidates"] == 48
    assert report["candidate_mechanism"]["n_groups_with_two_or_more_verified_prefix"] == 0
    assert report["sensitivity"]["canonical_d1_d2"]["recovery_rate"] == 1.0
    assert report["sensitivity"]["stronger_gap"]["d2"] == -0.50
    assert report["sensitivity"]["canonical_d1_d2"]["labels_identical_to_canonical"] is True
    statuses = report["format_decomposition"]["process_status"]
    assert statuses == {
        "PREFIX_BEARING_FAILURE": 51,
        "SUCCESS": 4,
        "FORMAT_NO_CODE": 1,
        "PROCESS_ORACLE_INFRA": 1,
    }
    # d1/d2 credit over every conclusive record (49 failing + 3 mixed failures are
    # the same 51 candidates: one d1 and one d2 token each).
    assert report["token_credit"]["d1_tokens"]["n"] == 51
    assert report["token_credit"]["d2_tokens"]["n"] == 51
    assert report["token_credit"]["success_tokens"]["n"] == 4
    assert report["token_credit"]["blamed_mappable_rate"] == 1.0
    assert report["FINAL_CLASSIFICATION"]["gate_order"] == ["E1", "E2", "GO", "G1", "G2", "G3"]
    assert report["provenance"]["code_stamp"] == STAMP
    assert report["permitted_claim"].startswith("Offline process-label recoverability")
    assert report["compute"] == {
        "new_model_generation": 0,
        "training": 0,
        "oracle_submissions": 3,
        "oracle_censored": 0,
        "container_recoveries": 0,
    }
    assert report["SEALED_RESERVE_TOUCHED"] == 0
    assert report["if_GO"]["V5_R001_TRAINING_LAUNCHED"] == "NO"
    A.print_report(report)
    printed = capsys.readouterr().out
    assert "V5_P001_RESULT" in printed
    assert "PROCESS_SIGNAL_GO" in printed


def _classification_fixture():
    e1 = {"passed": True}
    e2 = {"passed": True, "censored_share": 0.01}
    g1 = {"passed": True, "point": 0.5, "ci_lower": 0.3}
    g2 = {"seed1": {"passed": True, "n_groups": 10}, "seed2": {"passed": True, "n_groups": 10}}
    g3 = [{"passed": True}] * 4
    all_fail = [{"seed": "seed1"}] * 10 + [{"seed": "seed2"}] * 10
    return e1, e2, g1, g2, g3, all_fail


def test_classification_order_follows_the_frozen_gate_sequence():
    e1, e2, g1, g2, g3, all_fail = _classification_fixture()

    def run(**overrides):
        kwargs = {"e1": e1, "e2": e2, "g1": g1, "g2": g2, "g3": g3, "population": [], "all_fail": all_fail}
        kwargs.update(overrides)
        return A.classify(**kwargs)["classification"]

    assert run() == "PROCESS_SIGNAL_GO"
    assert run(e1={"passed": False}) == "INCONCLUSIVE_PROCESS_ORACLE"
    assert run(e2={"passed": False, "censored_share": 0.01}) == "INCONCLUSIVE_PROCESS_ORACLE"
    assert run(e2={"passed": True, "censored_share": 0.06}) == "INCONCLUSIVE_PROCESS_ORACLE"
    assert (
        run(g2={"seed1": {"passed": True, "n_groups": 10}, "seed2": {"passed": True, "n_groups": 4}})
        == "INCONCLUSIVE_PROCESS_ORACLE"
    )
    assert run(g1={"passed": False, "point": 0.01, "ci_lower": 0.0}) == "NO_PROCESS_RECOVERY"
    assert (
        run(g2={"seed1": {"passed": True, "n_groups": 10}, "seed2": {"passed": False, "n_groups": 10}})
        == "PROCESS_SIGNAL_SEED_UNSTABLE"
    )
    assert run(g3=[{"passed": True}, {"passed": False}, {"passed": True}, {"passed": False}]) == (
        "PROCESS_SIGNAL_LENGTH_CONFOUNDED"
    )
    # A gate failure never hides a lower-priority gate failure: the reported list is complete.
    report = A.classify(
        e1=e1, e2=e2, g1={"passed": False, "point": 0.0, "ci_lower": 0.0}, g2=g2,
        g3=[{"passed": False}] * 4, population=[], all_fail=all_fail,
    )
    assert report["classification"] == "NO_PROCESS_RECOVERY"
    assert {"G1", "G3"} <= set(report["failed_gates"])


def test_all_fail_groups_excludes_fully_censored_groups():
    surface = {
        "groups": [
            {"group_key": "seed1:g00", "seed": "seed1", "component_id": "c", "statement_id": "s",
             "label": S.GROUP_ALL_FAIL, "contaminated": False},
            {"group_key": "seed1:g01", "seed": "seed1", "component_id": "c", "statement_id": "s",
             "label": S.GROUP_ALL_FAIL, "contaminated": False},
        ]
    }
    censored = {"candidate_id": "a", "group_key": "seed1:g00", "oracle": {"infra": True},
                "process_status": "PROCESS_ORACLE_INFRA"}
    no_code = {"candidate_id": "b", "group_key": "seed1:g01", "oracle": None,
               "process_status": "FORMAT_NO_CODE"}
    entries = A.all_fail_groups(A.group_records([censored, no_code]), surface)
    assert [entry["status"] for entry in entries] == ["excluded_all_censored", "evaluable"]
    assert entries[1]["n_no_code"] == 1 and entries[1]["recoverable"] is False
    assert A.rate([entry for entry in entries if entry["status"] == "evaluable"]) == 0.0
