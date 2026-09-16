"""Minimal HTTP client for the Kimina Lean Server `/verify` endpoint."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

# Status vocabulary mirrors `kimina_client.models.SnippetStatus` for the
# Kimina Lean Server 2.0.0 `/verify` response schema.
LEAN_STATUSES = (
    "valid",
    "sorry",
    "lean_error",
    "repl_error",
    "timeout_error",
    "server_error",
    "unknown",
)


def result_lean_status(item: Any) -> str:
    """Classify one `/verify` result using Kimina 2.0.0 semantics.

    Matches `kimina_client`: a snippet is ``valid`` only when the REPL-level
    ``error`` is absent, the ``messages`` carry no ``error`` severity and the
    ``sorries`` list is empty. Batch responses wrap each snippet as
    ``{"custom_id": ..., "error": ..., "response": {...}}``.
    """

    if not isinstance(item, Mapping):
        return "unknown"

    error = item.get("error")
    if error:
        return "timeout_error" if "timed out" in str(error).lower() else "server_error"

    response = item.get("response")
    if response is None:
        # Legacy/simplified shapes kept for robustness against older servers.
        status = str(item.get("status", item.get("result", ""))).lower()
        if status in LEAN_STATUSES:
            return status
        for key in ("is_valid", "valid", "verified", "success", "isSuccess"):
            if isinstance(item.get(key), bool):
                return "valid" if item[key] else "unknown"
        return "unknown"

    if not isinstance(response, Mapping):
        return "unknown"
    if "message" in response:
        return "repl_error"
    messages = response.get("messages") or []
    if any(isinstance(message, Mapping) and message.get("severity") == "error" for message in messages):
        return "lean_error"
    sorries = response.get("sorries") or []
    if sorries:
        return "sorry"
    return "valid"


def result_is_valid(item: Any) -> bool:
    """True only when the snippet compiles with no error and no sorry."""

    return result_lean_status(item) == "valid"


def verify_code(
    proof: str,
    *,
    custom_id: str = "tinylean-p0-smoke",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Submit one proof candidate and return the decoded server response."""

    url = (base_url or os.getenv("LEAN_SERVER_API_URL", "http://127.0.0.1:8000")).rstrip("/")
    request_timeout = timeout or float(os.getenv("TINYLEAN_HTTP_TIMEOUT", "60"))
    headers = {"Content-Type": "application/json"}
    token = api_key or os.getenv("LEAN_SERVER_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "codes": [{"custom_id": custom_id, "proof": proof}],
        "infotree_type": "original",
    }
    # The Lean server is a local service; a system-level proxy (for example an
    # HTTP proxy registered on a Windows development machine) must never
    # intercept these requests.
    response = httpx.post(
        f"{url}/verify",
        json=payload,
        headers=headers,
        timeout=request_timeout,
        trust_env=False,
    )
    response.raise_for_status()
    decoded = response.json()
    if isinstance(decoded, dict):
        return decoded
    return {"results": decoded}


def verify_codes(
    proofs: list[str],
    *,
    custom_ids: list[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Submit a batch of proof candidates to reduce verifier HTTP overhead."""

    if custom_ids is not None and len(custom_ids) != len(proofs):
        raise ValueError("custom_ids and proofs must have equal length")
    url = (base_url or os.getenv("LEAN_SERVER_API_URL", "http://127.0.0.1:8000")).rstrip("/")
    request_timeout = timeout or float(os.getenv("TINYLEAN_HTTP_TIMEOUT", "60"))
    headers = {"Content-Type": "application/json"}
    token = api_key or os.getenv("LEAN_SERVER_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    ids = custom_ids or [f"tinylean-{index}" for index in range(len(proofs))]
    payload = {
        "codes": [{"custom_id": custom_id, "proof": proof} for custom_id, proof in zip(ids, proofs)],
        "infotree_type": "original",
    }
    # The Lean server is a local service; a system-level proxy (for example an
    # HTTP proxy registered on a Windows development machine) must never
    # intercept these requests.
    response = httpx.post(
        f"{url}/verify",
        json=payload,
        headers=headers,
        timeout=request_timeout,
        trust_env=False,
    )
    response.raise_for_status()
    decoded = response.json()
    if isinstance(decoded, dict):
        return decoded
    return {"results": decoded}
