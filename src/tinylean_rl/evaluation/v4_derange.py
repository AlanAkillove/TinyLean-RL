"""V4-P001 Arm D — deterministic diagnostic derangement (Amendment A, owner §2).

Arm D (``MISMATCHED_DIAGNOSTIC``) is byte-identical to Arm C except that ``diagnostic_i`` is replaced
by a real normalized Lean diagnostic belonging to a different theorem ``j != i``. The mapping cannot
exist before the primary cohort's diagnostics do, so only the *algorithm* is frozen up front; the
runner applies it inside the Stage-1/Stage-2 freeze boundary and stores the mapping and its hash
before any second-stage generation.

Frozen algorithm:

1. group the 128 primary failures by their frozen error category (frozen category order);
2. within a category of size >= 2, order the members by ``(token-length bucket, sha256(statement_id))``
   and shift the diagnostics cyclically by one position -- no theorem can receive its own diagnostic
   and every member is a donor exactly once;
3. a singleton category takes the deterministic fallback, in recipient ``(bucket, hash)`` order:
   among the theorems that are neither the recipient nor an already-used fallback donor, take the one
   minimising ``(|bucket difference|, sha256(statement_id))``. Donor reuse is possible only through
   this fallback and is reported; the recipient always gets exactly one donor.

Nothing here reads a model outcome, a verifier result or a second-stage label: the inputs are the
frozen statement ids, the frozen first-attempt error categories and the normalized diagnostic token
counts.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence

from tinylean_rl.evaluation.v4_taxonomy import ALL_CATEGORIES

DERANGEMENT_VERSION = "v4-derange-1"

#: Diagnostic token-length buckets; the normalizer caps every diagnostic at 512 tokens, so bucket 3
#: is the top. Chosen before the cohort exists and independent of any outcome.
BUCKET_EDGES: tuple[int, ...] = (64, 128, 256)


def token_bucket(tokens: int) -> int:
    if not isinstance(tokens, int) or isinstance(tokens, bool):
        raise TypeError(f"diagnostic token count must be an int, got {type(tokens)!r}")
    if tokens < 0:
        raise ValueError(f"diagnostic token count must be non-negative, got {tokens}")
    return bisect_right(BUCKET_EDGES, tokens)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sort_key(item: Mapping) -> tuple[int, str]:
    return (token_bucket(int(item["diagnostic_tokens"])), _sha256(str(item["statement_id"])))


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _percentiles(values: list[int]) -> dict:
    ordered = sorted(values)

    def q(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def _validate(items: Sequence[Mapping]) -> list[dict]:
    seen: set[str] = set()
    cleaned: list[dict] = []
    for item in items:
        statement_id = str(item.get("statement_id") or "")
        category = str(item.get("category") or "")
        if not statement_id:
            raise ValueError("every derangement item needs a statement_id")
        if statement_id in seen:
            raise ValueError(f"duplicate statement_id {statement_id!r} in the cohort")
        if category not in ALL_CATEGORIES:
            raise ValueError(f"unknown frozen category {category!r}")
        digest = str(item.get("diagnostic_sha256") or "")
        if not digest:
            raise ValueError(f"missing diagnostic_sha256 for {statement_id!r}")
        seen.add(statement_id)
        cleaned.append(
            {
                "statement_id": statement_id,
                "category": category,
                "diagnostic_sha256": digest,
                "diagnostic_tokens": int(item["diagnostic_tokens"]),
            }
        )
    if len(cleaned) < 2:
        raise ValueError("a derangement needs at least two theorems")
    return cleaned


def derange(items: Sequence[Mapping]) -> dict:
    """Apply the frozen rule and return the mapping plus the owner-required reporting."""

    cleaned = _validate(items)
    groups: dict[str, list[dict]] = {category: [] for category in ALL_CATEGORIES}
    for item in cleaned:
        groups[item["category"]].append(item)

    donors: dict[str, dict] = {}
    fallback_recipients: list[str] = []
    same_category = 0

    for category in ALL_CATEGORIES:
        members = sorted(groups[category], key=_sort_key)
        if len(members) >= 2:
            for index, recipient in enumerate(members):
                donors[recipient["statement_id"]] = members[(index + 1) % len(members)]

    singletons = sorted(
        [item for category in ALL_CATEGORIES for item in groups[category] if len(groups[category]) == 1],
        key=_sort_key,
    )
    used_fallback_donors: set[str] = set()
    for recipient in singletons:
        bucket = token_bucket(recipient["diagnostic_tokens"])
        candidates = [
            item
            for item in cleaned
            if item["statement_id"] != recipient["statement_id"]
            and item["statement_id"] not in used_fallback_donors
        ]
        donor = min(
            candidates,
            key=lambda item: (abs(token_bucket(item["diagnostic_tokens"]) - bucket),
                              _sha256(item["statement_id"])),
        )
        donors[recipient["statement_id"]] = donor
        used_fallback_donors.add(donor["statement_id"])
        fallback_recipients.append(recipient["statement_id"])

    if len(donors) != len(cleaned):
        raise AssertionError("derangement did not assign exactly one donor per theorem")

    rows = []
    for recipient in sorted(cleaned, key=lambda item: item["statement_id"]):
        donor = donors[recipient["statement_id"]]
        same = donor["category"] == recipient["category"]
        same_category += int(same)
        rows.append(
            {
                "recipient": recipient["statement_id"],
                "donor": donor["statement_id"],
                "recipient_category": recipient["category"],
                "donor_category": donor["category"],
                "recipient_diagnostic_sha256": recipient["diagnostic_sha256"],
                "donor_diagnostic_sha256": donor["diagnostic_sha256"],
                "recipient_tokens": recipient["diagnostic_tokens"],
                "donor_tokens": donor["diagnostic_tokens"],
                "token_difference": abs(donor["diagnostic_tokens"] - recipient["diagnostic_tokens"]),
                "same_category": same,
                "fallback": recipient["statement_id"] in set(fallback_recipients),
            }
        )

    donor_use: dict[str, int] = {}
    for row in rows:
        donor_use[row["donor"]] = donor_use.get(row["donor"], 0) + 1

    checks = {
        "no_self_diagnostic": all(row["recipient"] != row["donor"] for row in rows),
        "every_recipient_assigned_exactly_once": len(rows) == len(cleaned),
        "all_recipients_distinct": len({row["recipient"] for row in rows}) == len(cleaned),
    }
    failing = [name for name, ok in checks.items() if not ok]
    if failing:
        raise AssertionError(f"derangement invariants violated: {failing}")

    return {
        "version": DERANGEMENT_VERSION,
        "n": len(rows),
        "mapping": rows,
        "mapping_sha256": _sha256(_canonical({"version": DERANGEMENT_VERSION, "mapping": rows})),
        "same_error_category_match_rate": round(same_category / len(rows), 4),
        "diagnostic_token_length_difference": _percentiles([row["token_difference"] for row in rows]),
        "fallback_count": len(fallback_recipients),
        "fallback_recipients": sorted(fallback_recipients),
        "donor_reuse_count": sum(1 for count in donor_use.values() if count > 1),
        "checks": checks,
    }


__all__ = [
    "BUCKET_EDGES",
    "DERANGEMENT_VERSION",
    "derange",
    "token_bucket",
]
