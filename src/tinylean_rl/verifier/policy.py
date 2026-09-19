"""Verification policy for V2 Track B (B0).

Companion code for ``docs/v2/b0_verifier_reliability.md``. It turns the B0
failure taxonomy into an operational verification flow so that candidate
failures and infrastructure failures can never be conflated in the
budget-response dataset.

Frozen taxonomy (do not extend casually):

    verified | lean_error | sorry | verifier_timeout |
    verifier_server_error | verifier_unhealthy | unresolved_infra_error

Operational rules (verified on the scratch server, see the B0 doc):

- the server-side per-request ``timeout`` is always sent explicitly and is
  strictly smaller than the client-side timeout, so the server kills a
  pathological candidate (REPL destroy) *before* the client gives up; a
  client-side timeout after that point is a genuine infrastructure signal
  rather than a modelling outcome;
- every infrastructure event (timeout / 5xx / transport loss) triggers a
  canary probe (a known-valid trivial proof). A failing canary raises
  ``VerifierUnhealthyError``: callers must fail closed instead of writing
  mislabelled rows;
- batch verification degrades to per-candidate requests (bounded retries);
  candidates that still cannot be attributed come back as infrastructure
  outcomes and must be quarantined by the caller (never silently counted as
  model failures);
- ``redeclaration`` (a same-name declaration lingering in a reused Kimina
  REPL) is an environment-state problem, not a proof failure: it is retried
  in isolation and, if it persists, labelled ``verifier_server_error``;
- no retry loop is unbounded.

Canary caveat: probe only while no other verification is in flight. On a
saturated REPL pool the canary request queues behind busy REPLs (the pool
waits up to ``LEAN_SERVER_MAX_WAIT`` = 60 s and then raises 429), which
would be misread as unhealth. ``VerificationSession.verify`` therefore
gates *between* passes, never inside a pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from tinylean_rl.verifier.kimina import verify_code, verify_codes

#: Canary proof: trivial, must verify in a healthy warm server in seconds.
CANARY_PROOF = "import Mathlib\ntheorem tinylean_v2_canary : (1 : Nat) = 1 := by rfl\n"
CANARY_CUSTOM_ID = "tinylean-v2-canary"

#: Same marker as the V1 evaluation path (scripts/promptset_rollout_probe.py).
REDECLARATION_MARKER = "has already been declared"


class VerifyOutcome(str, Enum):
    """Frozen B0 verification outcome vocabulary."""

    VERIFIED = "verified"
    LEAN_ERROR = "lean_error"
    SORRY = "sorry"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_SERVER_ERROR = "verifier_server_error"
    VERIFIER_UNHEALTHY = "verifier_unhealthy"
    UNRESOLVED_INFRA_ERROR = "unresolved_infra_error"

    @property
    def is_infrastructure(self) -> bool:
        return self in {
            VerifyOutcome.VERIFIER_TIMEOUT,
            VerifyOutcome.VERIFIER_SERVER_ERROR,
            VerifyOutcome.VERIFIER_UNHEALTHY,
            VerifyOutcome.UNRESOLVED_INFRA_ERROR,
        }

    @property
    def is_candidate_failure(self) -> bool:
        return self in {VerifyOutcome.LEAN_ERROR, VerifyOutcome.SORRY}

    @property
    def is_conclusive(self) -> bool:
        """True when the outcome is a final dataset label."""
        return self is VerifyOutcome.VERIFIED or self.is_candidate_failure


class VerifierUnhealthyError(RuntimeError):
    """Raised when the canary probe fails; callers must stop (fail-close)."""


@dataclass(frozen=True)
class Classified:
    """One classified verification result."""

    outcome: VerifyOutcome
    message: str = ""
    redeclaration: bool = False


def classify_result_item(item: Any) -> Classified:
    """Classify one Kimina ``/verify`` result item (pure, no IO).

    Mirrors the official validity rule (no error-severity message and no
    ``sorry``), then splits genuine proof failures from verifier-side
    trouble using the B0 taxonomy.
    """

    if not isinstance(item, dict):
        return Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, "missing response item")

    error = item.get("error")
    if error:
        text = str(error)
        if "timed out" in text.lower():
            return Classified(VerifyOutcome.VERIFIER_TIMEOUT, text[:500])
        return Classified(VerifyOutcome.VERIFIER_SERVER_ERROR, text[:500])

    response = item.get("response")
    if not isinstance(response, dict):
        # Legacy / simplified shapes kept for robustness against older servers.
        status = str(item.get("status", item.get("result", ""))).lower()
        if status in {"valid", "verified", "success"}:
            return Classified(VerifyOutcome.VERIFIED)
        if status == "sorry":
            return Classified(VerifyOutcome.SORRY, "declaration uses 'sorry'")
        if status in {"lean_error", "error"}:
            return Classified(VerifyOutcome.LEAN_ERROR, status)
        if status in {"timeout_error", "timeout"}:
            return Classified(VerifyOutcome.VERIFIER_TIMEOUT, status)
        if status in {"server_error", "repl_error"}:
            return Classified(VerifyOutcome.VERIFIER_SERVER_ERROR, status)
        for key in ("is_valid", "valid", "verified", "success", "isSuccess"):
            if isinstance(item.get(key), bool):
                if item[key]:
                    return Classified(VerifyOutcome.VERIFIED)
                return Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, f"legacy flag {key}=false")
        return Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, "unknown response shape")

    response_error = response.get("error")
    if response_error:
        text = str(response_error)
        if "timed out" in text.lower():
            return Classified(VerifyOutcome.VERIFIER_TIMEOUT, text[:500])
        return Classified(VerifyOutcome.VERIFIER_SERVER_ERROR, text[:500])
    if "message" in response:  # legacy REPL-level error shape
        return Classified(VerifyOutcome.VERIFIER_SERVER_ERROR, str(response["message"])[:500])

    messages = response.get("messages") or []
    errors = [
        str(message.get("data", ""))
        for message in messages
        if isinstance(message, dict) and str(message.get("severity", "")).lower() == "error"
    ]
    if errors:
        redeclaration = any(REDECLARATION_MARKER in error for error in errors)
        return Classified(VerifyOutcome.LEAN_ERROR, errors[0][:500], redeclaration=redeclaration)
    sorries = response.get("sorries") or []
    if sorries:
        return Classified(VerifyOutcome.SORRY, "declaration uses 'sorry'")
    return Classified(VerifyOutcome.VERIFIED)


def classify_transport_error(exc: BaseException) -> Classified:
    """Classify a client-side transport exception (pure, no IO)."""

    if isinstance(exc, httpx.TimeoutException):
        return Classified(VerifyOutcome.VERIFIER_TIMEOUT, f"{type(exc).__name__}: {exc}"[:500])
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return Classified(
                VerifyOutcome.VERIFIER_SERVER_ERROR, f"HTTP 429 (no free REPL): {exc}"[:500]
            )
        if status >= 500:
            return Classified(VerifyOutcome.VERIFIER_SERVER_ERROR, f"HTTP {status}: {exc}"[:500])
        return Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, f"HTTP {status}: {exc}"[:500])
    if isinstance(exc, httpx.HTTPError):
        return Classified(VerifyOutcome.VERIFIER_SERVER_ERROR, f"{type(exc).__name__}: {exc}"[:500])
    return Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, f"{type(exc).__name__}: {exc}"[:500])


@dataclass(frozen=True)
class SessionEvent:
    """Audit trail entry for one verification event."""

    kind: str
    detail: str
    at: float


def _response_items(decoded: dict[str, Any]) -> list[dict[str, Any]]:
    value: Any = decoded
    if isinstance(decoded, dict):
        for key in ("results", "codes", "data", "responses"):
            if key in decoded:
                value = decoded[key]
                break
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


class VerificationSession:
    """Batch verification with canary gating and bounded degradation.

    Parameters
    ----------
    base_url, api_key:
        Forwarded to the Kimina client (``None`` = environment defaults).
    server_timeout:
        Explicit server-side per-request budget (seconds). Must be large
        enough for a legitimately slow verification and small enough that a
        pathological candidate cannot occupy a REPL for minutes.
    client_slack:
        Client-side timeout = ``server_timeout + client_slack``. Must stay
        positive so the server always terminates a request first.
    batch_size:
        Candidates per HTTP request. ``>1`` trades speed for blast radius:
        one REPL crash fails the whole sub-batch, which then degrades to
        per-candidate requests.
    max_single_retries:
        Per-candidate retry bound during the isolation pass.
    canary_timeout, canary_retries, canary_backoff_seconds:
        Canary probe budgets. The canary must only run while no other
        verification is in flight (see module docstring).
    retry_backoff_seconds:
        Sleep between per-candidate retries.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        server_timeout: float = 30.0,
        client_slack: float = 60.0,
        batch_size: int = 4,
        max_single_retries: int = 2,
        canary_timeout: float = 60.0,
        canary_retries: int = 1,
        canary_backoff_seconds: float = 5.0,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        if server_timeout <= 0:
            raise ValueError("server_timeout must be positive")
        if client_slack <= 0:
            raise ValueError("client_slack must be positive (client timeout must exceed server timeout)")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_single_retries < 0:
            raise ValueError("max_single_retries must be >= 0")
        self.base_url = base_url
        self.api_key = api_key
        self.server_timeout = float(server_timeout)
        self.client_slack = float(client_slack)
        self.batch_size = int(batch_size)
        self.max_single_retries = int(max_single_retries)
        self.canary_timeout = float(canary_timeout)
        self.canary_retries = int(canary_retries)
        self.canary_backoff_seconds = float(canary_backoff_seconds)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.events: list[SessionEvent] = []

    @property
    def client_timeout(self) -> float:
        return self.server_timeout + self.client_slack

    def _record(self, kind: str, detail: str) -> None:
        self.events.append(SessionEvent(kind=kind, detail=detail[:500], at=time.time()))

    def canary_probe(self) -> bool:
        """Probe the server with a known-valid trivial proof (bounded retries)."""

        for attempt in range(self.canary_retries + 1):
            try:
                decoded = verify_code(
                    CANARY_PROOF,
                    custom_id=CANARY_CUSTOM_ID,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.canary_timeout + self.client_slack,
                    server_timeout=int(self.canary_timeout),
                )
                items = _response_items(decoded)
                classified = classify_result_item(items[0] if items else None)
            except httpx.HTTPError as exc:
                classified = classify_transport_error(exc)
            self._record("canary", f"attempt={attempt} outcome={classified.outcome.value} {classified.message}")
            if classified.outcome is VerifyOutcome.VERIFIED:
                return True
            if attempt < self.canary_retries:
                time.sleep(self.canary_backoff_seconds)
        return False

    def require_healthy(self) -> None:
        """Fail-close gate: raise when the canary probe fails."""

        if not self.canary_probe():
            raise VerifierUnhealthyError(
                "canary probe failed; verification must stop (fail-close). "
                "Restart the Lean server, re-run the prewarm ladder, then resume."
            )

    def verify(self, proofs: list[str], custom_ids: list[str]) -> list[Classified]:
        """Verify candidates and return one ``Classified`` per candidate.

        Pass 1 (batched): conclusive results and server-confirmed timeouts
        are final; other infrastructure events leave the candidate
        unattributed and trigger a canary gate.

        Pass 2 (isolation): each unattributed candidate is verified alone
        with bounded retries; every infrastructure outcome triggers a
        canary gate. A failing canary raises ``VerifierUnhealthyError``.
        """

        if len(proofs) != len(custom_ids):
            raise ValueError("proofs and custom_ids must have equal length")
        if not proofs:
            return []

        results: list[Classified | None] = [None] * len(proofs)
        for start in range(0, len(proofs), self.batch_size):
            batch = list(range(start, min(start + self.batch_size, len(proofs))))
            if self._batch_pass(batch, proofs, custom_ids, results):
                self.require_healthy()

        pending = [index for index, result in enumerate(results) if result is None]
        for index in pending:
            classified = self._single_pass(proofs[index], custom_ids[index])
            results[index] = classified
            if classified.outcome.is_infrastructure:
                self.require_healthy()

        return [
            result if result is not None else Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, "unattributed")
            for result in results
        ]

    def _batch_pass(
        self,
        batch: list[int],
        proofs: list[str],
        custom_ids: list[str],
        results: list[Classified | None],
    ) -> bool:
        """One sub-batch request; returns True when infrastructure was seen."""

        ids = [custom_ids[index] for index in batch]
        try:
            decoded = verify_codes(
                [proofs[index] for index in batch],
                custom_ids=ids,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.client_timeout,
                server_timeout=int(self.server_timeout),
            )
        except httpx.HTTPError as exc:
            classified = classify_transport_error(exc)
            self._record(
                "batch_transport",
                f"n={len(batch)} outcome={classified.outcome.value} {classified.message}",
            )
            return True  # every candidate degrades to the isolation pass

        items = _response_items(decoded)
        by_id = {str(item.get("custom_id")): item for item in items}
        infrastructure_seen = False
        for index, custom_id in zip(batch, ids):
            item = by_id.get(custom_id)
            if item is None:
                self._record("batch_item", f"id={custom_id} outcome=missing (degrading)")
                infrastructure_seen = True
                continue
            classified = classify_result_item(item)
            self._record("batch_item", f"id={custom_id} outcome={classified.outcome.value} {classified.message}")
            if classified.outcome is VerifyOutcome.VERIFIER_TIMEOUT:
                # Server-confirmed timeout is final for this candidate: the
                # isolation pass would only burn the same server budget again.
                results[index] = classified
                infrastructure_seen = True
            elif classified.outcome.is_conclusive and not classified.redeclaration:
                results[index] = classified
            else:
                infrastructure_seen = True  # server-side trouble: isolate
        return infrastructure_seen

    def _single_pass(self, proof: str, custom_id: str) -> Classified:
        """Verify one candidate in isolation with bounded retries."""

        last = Classified(VerifyOutcome.UNRESOLVED_INFRA_ERROR, "no attempt recorded")
        for attempt in range(self.max_single_retries + 1):
            try:
                decoded = verify_code(
                    proof,
                    custom_id=f"{custom_id}-iso{attempt}",
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.client_timeout,
                    server_timeout=int(self.server_timeout),
                )
                items = _response_items(decoded)
                classified = classify_result_item(items[0] if items else None)
            except httpx.HTTPError as exc:
                classified = classify_transport_error(exc)
            self._record(
                "single",
                f"id={custom_id} attempt={attempt} outcome={classified.outcome.value} {classified.message}",
            )
            if classified.redeclaration:
                # Environment-state problem (not a proof failure): retry on a
                # fresh request; a reused REPL may hand the candidate an env
                # that already contains a same-name declaration.
                last = classified
            elif classified.outcome.is_conclusive or classified.outcome is VerifyOutcome.VERIFIER_TIMEOUT:
                return classified
            else:
                last = classified
            if attempt < self.max_single_retries:
                time.sleep(self.retry_backoff_seconds)
        if last.redeclaration:
            return Classified(
                VerifyOutcome.VERIFIER_SERVER_ERROR,
                f"redeclaration persisted after retries: {last.message}",
                redeclaration=True,
            )
        return last
