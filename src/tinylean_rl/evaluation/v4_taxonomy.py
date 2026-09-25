"""V4-P001 frozen error taxonomy.

Owner §C fixes the categories before any V4 outcome exists and requires that the primary cohort
contain only genuine Lean-level failures. The rules below are deterministic and priority-ordered:
each candidate is assigned exactly one category by the first rule that matches, and the audit
records how often more than one primary rule would have matched, so the "ambiguous fraction" is a
measured property of the corpus rather than an assumption.

Categories (frozen):

    elaboration_type_mismatch   primary
    unsolved_goals              primary
    tactic_failure              primary
    unknown_identifier          primary
    typeclass_synthesis         primary
    other_semantic_lean_failure primary
    syntax_parser               NOT primary (counted descriptively)
    format_no_code              NOT primary (counted descriptively)
    timeout_resource            NOT primary (resource exhaustion, not a wrong proof)
    infra                       NOT primary (verifier/infrastructure, never a candidate failure)
    verified                    success

`PRIMARY_CATEGORIES` is the ordered tuple the screening stage counts. "Primary eligible failure"
means: not verified, a real Lean-level diagnostic is available, and the category is in
`PRIMARY_CATEGORIES`.
"""

from __future__ import annotations

from collections.abc import Sequence

VERIFIED = "verified"
ELABORATION_TYPE_MISMATCH = "elaboration_type_mismatch"
UNSOLVED_GOALS = "unsolved_goals"
TACTIC_FAILURE = "tactic_failure"
UNKNOWN_IDENTIFIER = "unknown_identifier"
TYPECLASS_SYNTHESIS = "typeclass_synthesis"
OTHER_SEMANTIC = "other_semantic_lean_failure"
SYNTAX_PARSER = "syntax_parser"
FORMAT_NO_CODE = "format_no_code"
TIMEOUT_RESOURCE = "timeout_resource"
INFRA = "infra"

PRIMARY_CATEGORIES: tuple[str, ...] = (
    ELABORATION_TYPE_MISMATCH,
    UNSOLVED_GOALS,
    TACTIC_FAILURE,
    UNKNOWN_IDENTIFIER,
    TYPECLASS_SYNTHESIS,
    OTHER_SEMANTIC,
)

NON_PRIMARY_CATEGORIES: tuple[str, ...] = (SYNTAX_PARSER, FORMAT_NO_CODE, TIMEOUT_RESOURCE, INFRA)

ALL_CATEGORIES: tuple[str, ...] = (VERIFIED, *PRIMARY_CATEGORIES, *NON_PRIMARY_CATEGORIES)

INFRA_STATUSES = frozenset(
    {"verifier_timeout", "verifier_server_error", "unresolved_infra_error", "verifier_error"}
)

_RESOURCE_HINTS = (
    "maximum number of heartbeats",
    "deterministic) timeout at",
    "maxheartbeats",
    "deep recursion",
    "stack overflow",
    "out of memory",
    "interpreter is out of memory",
    "killed by signal",
)

_UNKNOWN_NAME_HINTS = (
    "unknown identifier",
    "unknown constant",
    "unknown tactic",
    "unknown namespace",
    "unknown free variable",
    "unknown module",
    "unresolved name",
    "unknown declaration",
    "unknown field",
    "unknown projection",
    "invalid field notation",
    "declare a new declaration",
)

_TYPECLASS_HINTS = (
    "failed to synthesize",
    "typeclass instance problem",
    "failed to infer instance",
    "synthesize instance",
    "type class",
    "instance problem is stuck",
    "cannot find a typeclass",
)

_UNSOLVED_HINTS = ("unsolved goals", "goals remaining", "unsolved goal")

_TACTIC_HINTS = (
    "failed to find a contradiction",
    "did not find instance of the pattern",
    "made no progress",
    "failed to unify",
    "has not been implemented",
    "no goals to be solved",
    "there are no goals",
    "linarith failed",
    "nlinarith failed",
    "omega could not prove",
    "simp made no progress",
    "simp failed",
    "rewrite failed",
    "tactic '",
    "tactic `",
    "tactic failed",
    "not a tactic",
    "failed to close goal",
    "could not prove goal",
    "tactic execution",
)

_ELABORATION_HINTS = (
    "type mismatch",
    "application type mismatch",
    "function expected",
    "but is expected to have type",
    "failed to show termination",
    "fail to show termination",
    "declaration uses 'sorry'",
    "unknown metavariable",
    "metavariable",
    "isdefeq",
    "defeq",
    "don't know how to synthesize",
    "ambiguous",
    "not a definition",
    "elaborat",
    "unexpected binder",
    "expected type",
    "inferred type",
    "invalid field notation",
)

_SYNTAX_HINTS = (
    "unexpected token",
    "unexpected end of input",
    "invalid syntax",
    "unterminated",
    "expected '",
    "expected \"",
    "expected ')'",
    "expected '{'",
    "expected identifier",
    "expected command",
    "parser",
    "unexpected identifier",
    "expected token",
)

_THINK_MARKER = "<think>"
_PROSE_STARTS = ("#", "*", "-", ">", "let ", "we ", "to ", "the ", "i ", "first", "here ", "this ")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def looks_like_non_code_extraction(extracted: str) -> bool:
    """True when the 'proof' text is really reasoning/prose rather than Lean code.

    The P0 extractor falls back to the whole completion when no fenced Lean block exists, so a
    truncated answer that never produced a code block reaches Lean as prose and is rejected with
    ``unexpected token '<'``. That is a formatting failure, never a semantic proof failure.
    """

    head = extracted.lstrip()
    if not head:
        return True
    if head.startswith(_THINK_MARKER):
        return True
    first = head.splitlines()[0].strip().lower()
    return first.startswith(_PROSE_STARTS)


def classify_first_attempt(
    *,
    verify_status: str,
    lean_message: str | None,
    truncated: bool,
    format_ok: bool,
    has_lean_block: bool,
    extracted: str = "",
) -> str:
    """Assign one frozen category to a first-attempt candidate.

    ``verify_status`` is the verifier outcome vocabulary of the project's B0 policy
    (``verified`` / ``lean_error`` / ``verifier_timeout`` / ``verifier_server_error`` /
    ``unresolved_infra_error``); anything in ``INFRA_STATUSES`` is infrastructure and is never a
    candidate failure.
    """

    if verify_status == VERIFIED:
        return VERIFIED
    if verify_status in INFRA_STATUSES:
        return INFRA
    if not format_ok or not has_lean_block:
        return FORMAT_NO_CODE
    # The prose check needs text to judge; when a corpus only records "a proof was extracted"
    # without storing it, the two flags above are the whole evidence and are trusted.
    if extracted.strip() and looks_like_non_code_extraction(extracted):
        return FORMAT_NO_CODE

    message = (lean_message or "").strip()
    low = message.lower()
    first = _first_line(low)

    # A resource failure is decided before any semantic rule: "maximum number of heartbeats" and
    # friends can appear inside an otherwise semantic-looking message.
    if any(hint in low for hint in _RESOURCE_HINTS):
        return TIMEOUT_RESOURCE

    # Name resolution before syntax: "unknown identifier 'k'" contains no parser marker, but a
    # parser message such as "unexpected token ':'; expected ')'" must not be captured by "expected".
    if any(hint in low for hint in _UNKNOWN_NAME_HINTS):
        return UNKNOWN_IDENTIFIER
    if any(hint in low for hint in _TYPECLASS_HINTS):
        return TYPECLASS_SYNTHESIS
    if any(hint in first for hint in _UNSOLVED_HINTS) or "unsolved goals" in low:
        return UNSOLVED_GOALS
    if any(hint in low for hint in _TACTIC_HINTS):
        return TACTIC_FAILURE
    if any(hint in low for hint in _ELABORATION_HINTS):
        return ELABORATION_TYPE_MISMATCH
    if any(hint in low for hint in _SYNTAX_HINTS):
        return SYNTAX_PARSER
    if not message:
        # A Lean rejection without any diagnostic cannot be screened into the primary cohort.
        return FORMAT_NO_CODE
    return OTHER_SEMANTIC


def matched_primary_rules(*, verify_status: str, lean_message: str | None) -> tuple[str, ...]:
    """Which primary rules the message text alone would have matched (ambiguity measurement)."""

    if verify_status == VERIFIED or verify_status in INFRA_STATUSES:
        return ()
    low = (lean_message or "").lower()
    if any(hint in low for hint in _RESOURCE_HINTS):
        return ()
    matches: list[str] = []
    if any(hint in low for hint in _UNKNOWN_NAME_HINTS):
        matches.append(UNKNOWN_IDENTIFIER)
    if any(hint in low for hint in _TYPECLASS_HINTS):
        matches.append(TYPECLASS_SYNTHESIS)
    if "unsolved goals" in low:
        matches.append(UNSOLVED_GOALS)
    if any(hint in low for hint in _TACTIC_HINTS):
        matches.append(TACTIC_FAILURE)
    if any(hint in low for hint in _ELABORATION_HINTS):
        matches.append(ELABORATION_TYPE_MISMATCH)
    return tuple(matches)


def category_counts(categories: Sequence[str]) -> dict[str, int]:
    counts = {name: 0 for name in ALL_CATEGORIES}
    for name in categories:
        if name not in counts:
            raise ValueError(f"unknown V4 category: {name!r}")
        counts[name] += 1
    return counts


def primary_eligible(category: str) -> bool:
    return category in PRIMARY_CATEGORIES


__all__ = [
    "ALL_CATEGORIES",
    "ELABORATION_TYPE_MISMATCH",
    "FORMAT_NO_CODE",
    "INFRA",
    "INFRA_STATUSES",
    "NON_PRIMARY_CATEGORIES",
    "OTHER_SEMANTIC",
    "PRIMARY_CATEGORIES",
    "SYNTAX_PARSER",
    "TACTIC_FAILURE",
    "TIMEOUT_RESOURCE",
    "TYPECLASS_SYNTHESIS",
    "UNKNOWN_IDENTIFIER",
    "UNSOLVED_GOALS",
    "VERIFIED",
    "category_counts",
    "classify_first_attempt",
    "looks_like_non_code_extraction",
    "matched_primary_rules",
    "primary_eligible",
]
