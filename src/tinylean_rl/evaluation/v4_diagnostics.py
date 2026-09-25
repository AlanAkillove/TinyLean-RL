"""V4-P001 diagnostic normalization (owner §H).

The second-stage VERIFIER_REPAIR arm receives the *exact* Lean diagnostic, normalized so that only
infrastructure noise disappears. The rule is asymmetric on purpose:

    removed   temporary/container paths, request ids, UUIDs, timestamps, server wrapper lines
    kept      error message, goal state, local context, expected/actual type, identifier/tactic
              information, line/column

Nothing is added: no human-written explanation, no generated hint, no proposed fix, no paraphrase.
The functions are pure and versioned (`NORMALIZATION_VERSION`) so the freeze is checkable by hash.
``bound_diagnostic`` enforces the frozen 512-token budget and reports truncation instead of hiding
it; the audit records the historical truncation rate before launch.
"""

from __future__ import annotations

import re

NORMALIZATION_VERSION = "v4-diag-1"

#: Frozen budget for the normalized diagnostic, in tokenizer tokens (owner §H).
DIAGNOSTIC_TOKEN_BUDGET = 512

_TRUNCATION_MARKER = "\n[diagnostic truncated at the frozen 512-token budget]"

# Infrastructure noise. Every pattern is anchored to a shape that cannot be Lean information.
_TMP_PATH = re.compile(r"(?:/tmp|/var/tmp|/dev/shm|/workspace|/root|/home|/mnt|C:\\\\)[^\s\"'`)]*")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_REQUEST_ID = re.compile(r"\b(?:x-)?(?:request|req|trace|span)[-_ ]?id\b\s*[:=#]?\s*[A-Za-z0-9._-]{4,}", re.IGNORECASE)
_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_NOISE_PREFIX = r"(?:(?:<time>|<id>)\s*)*"
_WRAPPER_BODY = (
    r"(?:"
    r"#\s*(?:system|server|internal)\s+error\b.*"
    r"|(?:internal|server|unexpected|http)\s+error\b.*"
    r"|lean server (?:error|exception)\b.*"
    r"|repl (?:error|exception|crash(?:ed)?|restart(?:ed)?)\b.*"
    r"|traceback \(most recent call last\):?"
    r"|file \"[^\"]*\", line \d+.*"
    r"|(?:request|req|trace|span|correlation)[-_ ]?id\s*[:=#]?\s*<id>"
    r"|at [\w.$/<>]+\([\w.$/<>:]*\)"
    r"|http[s]?://\S+"
    r")"
)
_WRAPPER_LINE = re.compile(rf"^{_NOISE_PREFIX}{_WRAPPER_BODY}\s*$", re.IGNORECASE)
_SERVER_HINT = re.compile(r"\b(?:traceback|stacktrace|connection (?:refused|reset)|read timed out|"
                          r"http response|status code|server error 5\d\d|429 too many)\b", re.IGNORECASE)
_BLANK_RUN = re.compile(r"\n{3,}")


def first_lean_error_block(text: str) -> str:
    """The first ``# Error N:`` block of a multi-error string, or the whole text.

    Deterministic and lossless for a single-error message: only the concatenation of *later* errors
    is dropped, and only when the V1/V3 tool-feedback shape is present (``# Error 1:`` ...).
    """

    payload = (text or "").strip()
    if not payload:
        return ""
    if not payload.startswith("# Error"):
        return payload
    head, _, _ = payload.partition("# Error 2:")
    body = head.split("Error message:", 1)[-1]
    return body.strip()


def strip_infrastructure_noise(text: str) -> str:
    """Remove paths, ids, timestamps and server wrapper lines; keep every Lean word.

    Substitution runs before the wrapper test so that a timestamped or id-prefixed wrapper line is
    still recognised and dropped, rather than surviving as "<time> ... server error ...".
    """

    lines = []
    for raw_line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _TMP_PATH.sub("<path>", raw_line)
        line = _UUID.sub("<id>", line)
        line = _REQUEST_ID.sub("<id>", line)
        line = _TIMESTAMP.sub("<time>", line)
        if _WRAPPER_LINE.match(line.strip()):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


def normalize_diagnostic(raw: str) -> str:
    """Frozen normalization: first error block, infra noise stripped, blank runs collapsed."""

    block = first_lean_error_block(raw)
    cleaned = strip_infrastructure_noise(block)
    cleaned = _BLANK_RUN.sub("\n\n", cleaned)
    return cleaned.strip()


def _position_prefix(pos) -> str:
    """Lean's own position, kept because it is Lean information (line/column)."""

    if not isinstance(pos, dict):
        return ""
    line, column = pos.get("line"), pos.get("column")
    if isinstance(line, int) and isinstance(column, int):
        return f"line {line}, column {column}: "
    if isinstance(line, int):
        return f"line {line}: "
    return ""


def diagnostic_from_verify_item(item) -> tuple[str, str]:
    """Pull the exact Lean diagnostic out of one Kimina ``/verify`` result item.

    Returns ``(normalized_text, status)`` where status is ``"lean_error"`` when the item carries an
    error-severity message, ``"repl_error"`` when the response only carries a REPL message, and
    ``""`` when there is nothing diagnostic to show. The first error block is used, never a
    concatenation of unrelated errors, so the repair arm sees the same single failure the verifier
    is able to attribute.
    """

    from collections.abc import Mapping

    if not isinstance(item, Mapping):
        return "", ""
    response = item.get("response")
    if not isinstance(response, Mapping):
        return "", ""
    if "message" in response:
        return normalize_diagnostic(str(response.get("message") or "")), "repl_error"
    messages = response.get("messages") or []
    errors = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("severity") == "error"
    ]
    if not errors:
        return "", ""
    first = errors[0]
    detail = str(first.get("data") or "").strip()
    if not detail:
        return "", ""
    return normalize_diagnostic(f"{_position_prefix(first.get('pos'))}{detail}"), "lean_error"


def bound_diagnostic(text: str, tokenizer, budget: int = DIAGNOSTIC_TOKEN_BUDGET) -> tuple[str, bool]:
    """Truncate a normalized diagnostic to the frozen token budget.

    Returns ``(text, truncated)``. The tokenizer is the frozen model tokenizer, so the budget is
    measured in the same units the model sees; truncation is reported, never silent.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text, False
    marker_ids = tokenizer.encode(_TRUNCATION_MARKER, add_special_tokens=False)
    keep = max(1, budget - len(marker_ids))
    return tokenizer.decode(ids[:keep], skip_special_tokens=True).rstrip() + _TRUNCATION_MARKER, True


def is_diagnostic_noise(normalized: str) -> bool:
    """True when nothing Lean-specific survived normalization (pure infrastructure text)."""

    if not normalized.strip():
        return True
    return bool(_SERVER_HINT.search(normalized)) and len(_SERVER_HINT.sub("", normalized).split()) <= 2


__all__ = [
    "DIAGNOSTIC_TOKEN_BUDGET",
    "NORMALIZATION_VERSION",
    "bound_diagnostic",
    "diagnostic_from_verify_item",
    "first_lean_error_block",
    "is_diagnostic_noise",
    "normalize_diagnostic",
    "strip_infrastructure_noise",
]
