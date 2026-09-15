"""Minimal HTTP client for the Kimina Lean Server `/verify` endpoint."""

from __future__ import annotations

import os
from typing import Any

import httpx


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
