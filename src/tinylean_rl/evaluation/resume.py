"""Resume-state validation for chunked checkpoint evaluations.

The checkpoint evaluator (``scripts/p3c_checkpoint_eval.py``) writes a
``<output>.partial.json`` after every chunk. ``--resume`` may only continue from
a partial state that proves to belong to the exact same evaluation (checkpoint
identity, fixed-set bytes, sampling settings, seed schedule, chunking); anything
else must fail closed instead of silently mixing records from different runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ResumeError(RuntimeError):
    """Raised when a partial state cannot be trusted as a resume source."""


@dataclass(frozen=True)
class ResumeState:
    records: list[dict]
    processed_theorems: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_resume_state(partial_path: Path, expected_settings: dict[str, Any]) -> ResumeState:
    """Load and validate a partial evaluation state (fail-closed).

    Validations: settings equality (exact), record count == processed x samples,
    records cover theorem indices ``0..processed-1`` with exactly ``samples``
    records each.
    """

    try:
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResumeError(f"partial state not found: {partial_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResumeError(f"partial state is not valid JSON: {partial_path}") from exc

    records = payload.get("records")
    processed = payload.get("processed_theorems")
    settings = payload.get("settings")
    if not isinstance(records, list) or not isinstance(processed, int) or processed < 0:
        raise ResumeError("partial state lacks records/processed_theorems")
    if settings != expected_settings:
        raise ResumeError(
            "partial state was produced with different settings; refuse to resume "
            f"(partial={settings!r}, expected={expected_settings!r})"
        )
    samples = expected_settings["samples_per_theorem"]
    if len(records) != processed * samples:
        raise ResumeError(
            f"partial records={len(records)} inconsistent with processed={processed} x samples={samples}"
        )
    seen: dict[int, int] = {}
    for record in records:
        index = record.get("theorem_index")
        if not isinstance(index, int) or index < 0 or index >= processed:
            raise ResumeError(f"record outside the processed range: theorem_index={index!r}")
        seen[index] = seen.get(index, 0) + 1
    if len(seen) != processed or any(count != samples for count in seen.values()):
        raise ResumeError(f"partial state does not cover 0..{processed - 1} x {samples} cleanly")
    return ResumeState(records=records, processed_theorems=processed)


def chunk_start_offset(processed: int, chunk_size: int, total: int) -> int:
    """Resume offset in theorems for the chunk loop (fail-closed).

    The partial file is written at chunk ends, so ``processed`` must either be
    exactly ``total`` (everything computed, final artifact not yet written) or a
    multiple of ``chunk_size``.
    """

    if processed < 0 or processed > total:
        raise ResumeError(f"processed={processed} outside [0, {total}]")
    if processed == total:
        return processed
    if chunk_size < 1 or processed % chunk_size != 0:
        raise ResumeError(f"processed={processed} is not aligned to chunk_size={chunk_size}")
    return processed
