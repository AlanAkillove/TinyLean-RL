"""Conservative extraction of Lean proof candidates from model text."""

from __future__ import annotations

import re

_FENCED_LEAN = re.compile(r"```(?:lean4?|Lean4?)\s*\n(.*?)```", re.DOTALL)


def extract_proof(text: str) -> str:
    """Return the most useful Lean candidate found in *text*.

    The P0 extractor intentionally does not try to repair proofs. It only removes
    common markdown wrappers so that parser/verifier failures remain observable.
    """

    if not text or not text.strip():
        raise ValueError("model output is empty")

    fenced = _FENCED_LEAN.findall(text)
    candidate = fenced[-1] if fenced else text
    candidate = candidate.strip()

    # Stop at an unmatched/secondary markdown fence. This is common when a
    # small model starts explaining after a short proof continuation.
    if "```" in candidate:
        candidate = candidate.split("```", 1)[0].rstrip()

    # Some models return a complete theorem while others return `by ...`. Keep
    # the complete code when available. Only strip a leading conversational
    # wrapper when the proof introducer is unambiguously at the start; searching
    # for arbitrary `by` tokens would corrupt natural-language explanations.
    if candidate.startswith(("by ", "by\n")):
        candidate = candidate.strip()

    # Remove conversational suffixes after a fenced/standalone proof only when
    # they are clearly introduced as prose. This is deliberately conservative.
    candidate = re.sub(r"\n+(?:Explanation|解释|说明)\s*:\s*.*$", "", candidate, flags=re.DOTALL)
    return candidate.strip()
