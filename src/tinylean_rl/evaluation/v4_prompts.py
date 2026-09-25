"""V4-P001 prompt renderer for the three paired arms (owner §E, §F, §G).

Arm A - FRESH_RETRY: the original theorem prompt only. No failed proof, no diagnostic, no mention
        that a previous attempt exists.
Arm B - SELF_REVISION: the theorem prompt, the extracted failed proof as an assistant turn, and a
        generic correction request that says the attempt was rejected but says nothing about why.
Arm C - VERIFIER_REPAIR: byte-identical to Arm B except that the exact normalized Lean diagnostic is
        appended to the correction request. ``text_c == text_b + "\\n\\n" + diagnostic`` is enforced,
        so the wording of the two arms cannot drift apart.

Format (§F): the frozen Kimina/Qwen3 chat template is used as-is with multi-turn user/assistant
messages; no tool role is invented and no template variant is chosen per theorem. The rendered
prompts and their hashes are frozen in the prompt audit before the formal run.

The failed proof is the *extracted Lean code* (§G), never the raw completion with its reasoning
wrapper; the caller passes ``extract_proof`` output and the extraction hashes are recorded per
candidate in the raw row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

RENDERER_VERSION = "v4-prompt-1"

#: Frozen correction request. Arm B and Arm C share it exactly; in Arm C the diagnostic is appended.
CORRECTION_REQUEST = (
    "The previous proof attempt was rejected. Revise it into a correct and complete Lean proof of "
    "the same theorem."
)

ARMS = ("A_FRESH_RETRY", "B_SELF_REVISION", "C_VERIFIER_REPAIR")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Normalize a prompt (parquet rows arrive as numpy arrays of dicts) to plain dicts."""

    out: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError(f"prompt message is not a mapping: {type(message)!r}")
        role = str(message.get("role") or "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role {role!r} in the frozen theorem prompt")
        out.append({"role": role, "content": str(message.get("content") or "")})
    if not out or out[-1]["role"] != "user":
        raise ValueError("a first-attempt theorem prompt must end with a user turn")
    return out


def arm_messages(
    base_messages: Sequence[Mapping[str, str]],
    *,
    arm: str,
    failed_proof: str = "",
    diagnostic: str = "",
) -> list[dict[str, str]]:
    """Build the frozen message list for one arm. Pure and deterministic."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    messages = canonical_messages(base_messages)
    if arm == "A_FRESH_RETRY":
        if failed_proof or diagnostic:
            raise ValueError("Arm A takes neither a failed proof nor a diagnostic by construction")
        return messages
    if not failed_proof.strip():
        raise ValueError("Arms B and C require the extracted failed proof")
    if arm == "C_VERIFIER_REPAIR" and not diagnostic.strip():
        raise ValueError("Arm C requires the normalized Lean diagnostic")
    if arm == "B_SELF_REVISION" and diagnostic:
        raise ValueError("Arm B must not carry a diagnostic")
    revision = CORRECTION_REQUEST if arm == "B_SELF_REVISION" else f"{CORRECTION_REQUEST}\n\n{diagnostic.strip()}"
    return [
        *messages,
        {"role": "assistant", "content": failed_proof.strip()},
        {"role": "user", "content": revision},
    ]


def render_prompt(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    """Render with the frozen chat template, exactly as the first attempt was rendered."""

    return tokenizer.apply_chat_template(
        [dict(message) for message in messages], add_generation_prompt=True, tokenize=False
    )


def render_arm(tokenizer, base_messages, *, arm: str, failed_proof: str = "", diagnostic: str = "") -> str:
    return render_prompt(tokenizer, arm_messages(base_messages, arm=arm, failed_proof=failed_proof, diagnostic=diagnostic))


def prompt_sha256(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    return sha256_text(render_prompt(tokenizer, messages))


def arm_suffix_invariant(base_messages, *, failed_proof: str, diagnostic: str) -> bool:
    """The §E invariant, checkable at runtime: C is B plus the diagnostic and nothing else."""

    b = canonical_messages(base_messages) + [
        {"role": "assistant", "content": failed_proof.strip()},
        {"role": "user", "content": CORRECTION_REQUEST},
    ]
    c = canonical_messages(base_messages) + [
        {"role": "assistant", "content": failed_proof.strip()},
        {"role": "user", "content": f"{CORRECTION_REQUEST}\n\n{diagnostic.strip()}"},
    ]
    return c[-1]["content"] == b[-1]["content"] + "\n\n" + diagnostic.strip()


__all__ = [
    "ARMS",
    "CORRECTION_REQUEST",
    "RENDERER_VERSION",
    "arm_messages",
    "arm_suffix_invariant",
    "canonical_messages",
    "prompt_sha256",
    "render_arm",
    "render_prompt",
    "sha256_text",
]
