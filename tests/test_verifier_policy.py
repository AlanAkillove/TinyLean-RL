"""Unit tests for the B0 verification policy (no Lean server required)."""

import httpx
import pytest

from tinylean_rl.verifier import policy
from tinylean_rl.verifier.policy import (
    VerificationSession,
    VerifierUnhealthyError,
    VerifyOutcome,
    classify_result_item,
    classify_transport_error,
)


def valid_item(custom_id: str) -> dict:
    return {"custom_id": custom_id, "response": {"messages": [], "sorries": []}}


def timeout_item(custom_id: str) -> dict:
    return {"custom_id": custom_id, "error": "Lean REPL command timed out in 30.0 seconds"}


def server_error_item(custom_id: str) -> dict:
    return {"custom_id": custom_id, "error": "JSON decode error"}


def http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://server/verify")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


@pytest.mark.parametrize(
    ("item", "expected", "redeclaration"),
    [
        (valid_item("a"), VerifyOutcome.VERIFIED, False),
        ({"response": {}}, VerifyOutcome.VERIFIED, False),
        ({"response": {"messages": [], "sorries": [{"pos": {}}]}}, VerifyOutcome.SORRY, False),
        (
            {"response": {"messages": [{"severity": "error", "data": "unknown identifier"}]}},
            VerifyOutcome.LEAN_ERROR,
            False,
        ),
        (
            {"response": {"messages": [{"severity": "error", "data": "'x' has already been declared"}]}},
            VerifyOutcome.LEAN_ERROR,
            True,
        ),
        (timeout_item("a"), VerifyOutcome.VERIFIER_TIMEOUT, False),
        ({"error": "connection reset by peer"}, VerifyOutcome.VERIFIER_SERVER_ERROR, False),
        (server_error_item("a"), VerifyOutcome.VERIFIER_SERVER_ERROR, False),
        ({"response": {"message": "REPL crashed"}}, VerifyOutcome.VERIFIER_SERVER_ERROR, False),
        ({"response": {"error": "timed out in header"}}, VerifyOutcome.VERIFIER_TIMEOUT, False),
        ({"status": "valid"}, VerifyOutcome.VERIFIED, False),
        ({"status": "timeout_error"}, VerifyOutcome.VERIFIER_TIMEOUT, False),
        ({"status": "server_error"}, VerifyOutcome.VERIFIER_SERVER_ERROR, False),
        ({"is_valid": True}, VerifyOutcome.VERIFIED, False),
        ({"is_valid": False}, VerifyOutcome.UNRESOLVED_INFRA_ERROR, False),
        ({"custom_id": "a"}, VerifyOutcome.UNRESOLVED_INFRA_ERROR, False),
        (None, VerifyOutcome.UNRESOLVED_INFRA_ERROR, False),
    ],
)
def test_classify_result_item_taxonomy(item, expected, redeclaration):
    classified = classify_result_item(item)
    assert classified.outcome is expected
    assert classified.redeclaration is redeclaration


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout("client gave up"), VerifyOutcome.VERIFIER_TIMEOUT),
        (httpx.ConnectTimeout("connect"), VerifyOutcome.VERIFIER_TIMEOUT),
        (http_status_error(500), VerifyOutcome.VERIFIER_SERVER_ERROR),
        (http_status_error(429), VerifyOutcome.VERIFIER_SERVER_ERROR),
        (httpx.ConnectError("refused"), VerifyOutcome.VERIFIER_SERVER_ERROR),
        (ValueError("not transport"), VerifyOutcome.UNRESOLVED_INFRA_ERROR),
    ],
)
def test_classify_transport_error(exc, expected):
    assert classify_transport_error(exc).outcome is expected


def test_outcome_sets():
    assert VerifyOutcome.LEAN_ERROR.is_candidate_failure
    assert VerifyOutcome.SORRY.is_candidate_failure
    assert not VerifyOutcome.VERIFIER_TIMEOUT.is_candidate_failure
    assert VerifyOutcome.VERIFIER_TIMEOUT.is_infrastructure
    assert VerifyOutcome.UNRESOLVED_INFRA_ERROR.is_infrastructure
    assert VerifyOutcome.VERIFIED.is_conclusive
    assert VerifyOutcome.SORRY.is_conclusive
    assert not VerifyOutcome.VERIFIER_SERVER_ERROR.is_conclusive


def test_session_parameter_validation():
    with pytest.raises(ValueError):
        VerificationSession(server_timeout=0)
    with pytest.raises(ValueError):
        VerificationSession(client_slack=0)
    with pytest.raises(ValueError):
        VerificationSession(batch_size=0)
    with pytest.raises(ValueError):
        VerificationSession(max_single_retries=-1)
    assert VerificationSession(server_timeout=30, client_slack=60).client_timeout == 90


def make_session(**overrides) -> VerificationSession:
    options = {
        "server_timeout": 30.0,
        "client_slack": 60.0,
        "batch_size": 4,
        "max_single_retries": 2,
        "canary_retries": 0,
        "canary_backoff_seconds": 0.0,
        "retry_backoff_seconds": 0.0,
    }
    options.update(overrides)
    return VerificationSession(**options)


def test_batch_happy_path_no_canary(monkeypatch):
    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        return {"results": [valid_item(cid) for cid in custom_ids]}

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    session = make_session()
    results = session.verify(["p1", "p2"], ["c1", "c2"])
    assert [r.outcome for r in results] == [VerifyOutcome.VERIFIED, VerifyOutcome.VERIFIED]
    assert all(event.kind != "canary" for event in session.events)


def test_batch_item_timeout_is_final_no_isolate(monkeypatch):
    canary_calls = []

    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        return {"results": [timeout_item(custom_ids[0]), valid_item(custom_ids[1])]}

    def fake_verify_code(proof, *, custom_id="", **kwargs):
        canary_calls.append(custom_id)
        return {"results": [valid_item(custom_id)]}

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    monkeypatch.setattr(policy, "verify_code", fake_verify_code)
    session = make_session()
    results = session.verify(["p1", "p2"], ["c1", "c2"])
    assert [r.outcome for r in results] == [VerifyOutcome.VERIFIER_TIMEOUT, VerifyOutcome.VERIFIED]
    # infra seen -> canary gate ran
    assert canary_calls == [policy.CANARY_CUSTOM_ID]
    # the timed-out candidate was NOT re-run in isolation
    assert all(event.kind != "single" for event in session.events)


def test_batch_transport_failure_degrades_to_single(monkeypatch):
    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        raise httpx.ReadTimeout("client timeout")

    def fake_verify_code(proof, *, custom_id="", **kwargs):
        if custom_id == policy.CANARY_CUSTOM_ID:
            return {"results": [valid_item(custom_id)]}
        return {"results": [valid_item(custom_id)]}

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    monkeypatch.setattr(policy, "verify_code", fake_verify_code)
    session = make_session()
    results = session.verify(["p1", "p2"], ["c1", "c2"])
    assert [r.outcome for r in results] == [VerifyOutcome.VERIFIED, VerifyOutcome.VERIFIED]
    kinds = [event.kind for event in session.events]
    assert "batch_transport" in kinds
    assert kinds.count("single") == 2


def test_single_retry_exhaustion_keeps_infra_class(monkeypatch):
    attempt_counter = {"n": 0}

    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        raise http_status_error(500)

    def fake_verify_code(proof, *, custom_id="", **kwargs):
        if custom_id == policy.CANARY_CUSTOM_ID:
            return {"results": [valid_item(custom_id)]}
        attempt_counter["n"] += 1
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    monkeypatch.setattr(policy, "verify_code", fake_verify_code)
    session = make_session(max_single_retries=2)
    results = session.verify(["p1"], ["c1"])
    assert results[0].outcome is VerifyOutcome.VERIFIER_SERVER_ERROR
    assert attempt_counter["n"] == 3  # attempts 0,1,2 then stop


def test_canary_failure_fails_closed(monkeypatch):
    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        return {"results": [server_error_item(custom_ids[0])]}

    def fake_verify_code(proof, *, custom_id="", **kwargs):
        raise httpx.ReadTimeout("canary timeout")

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    monkeypatch.setattr(policy, "verify_code", fake_verify_code)
    session = make_session()
    with pytest.raises(VerifierUnhealthyError):
        session.verify(["p1"], ["c1"])


def test_result_is_classified_and_redeclaration_retries_isolate(monkeypatch):
    """A redeclaration item is server-side trouble: isolate, then succeed."""

    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        return {
            "results": [
                {
                    "custom_id": custom_ids[0],
                    "response": {
                        "messages": [{"severity": "error", "data": "'foo' has already been declared"}]
                    },
                }
            ]
        }

    def fake_verify_code(proof, *, custom_id="", **kwargs):
        if custom_id == policy.CANARY_CUSTOM_ID:
            return {"results": [valid_item(custom_id)]}
        return {"results": [valid_item(custom_id)]}

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    monkeypatch.setattr(policy, "verify_code", fake_verify_code)
    session = make_session()
    results = session.verify(["p1"], ["c1"])
    # Isolation pass re-verified the candidate on a fresh request and it passed.
    assert results[0].outcome is VerifyOutcome.VERIFIED
    assert any(event.kind == "single" for event in session.events)


def test_redeclaration_persisting_becomes_server_error(monkeypatch):
    """A candidate that keeps hitting redeclaration is infra, not a failure."""

    redeclaration_item = {
        "custom_id": "ignored",
        "response": {"messages": [{"severity": "error", "data": "'foo' has already been declared"}]},
    }

    def fake_verify_codes(proofs, *, custom_ids=None, **kwargs):
        return {"results": [{**redeclaration_item, "custom_id": custom_ids[0]}]}

    def fake_verify_code(proof, *, custom_id="", **kwargs):
        if custom_id == policy.CANARY_CUSTOM_ID:
            return {"results": [valid_item(custom_id)]}
        return {"results": [{**redeclaration_item, "custom_id": custom_id}]}

    monkeypatch.setattr(policy, "verify_codes", fake_verify_codes)
    monkeypatch.setattr(policy, "verify_code", fake_verify_code)
    session = make_session(max_single_retries=1)
    results = session.verify(["p1"], ["c1"])
    assert results[0].outcome is VerifyOutcome.VERIFIER_SERVER_ERROR
    assert results[0].redeclaration is True
    # batch pass + isolation attempts (canary requests are not 'single' events)
    assert sum(event.kind == "single" for event in session.events) == 2
