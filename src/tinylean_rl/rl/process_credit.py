"""Frozen R001 process-credit semantics (owner pre-launch directive §1-§4).

This module is the *scientific* half of the intervention: it turns the frozen process oracle's
per-tactic labels (success / d1 / d2, see ``scripts/v5_process_oracle.py``) into token positions
and assembles the DrGRPO-style advantage tensor

    A_outcome_i      = g_i - g_bar                       (g_bar over VALID candidates of the group)
    A_process_i,j    = phi_i,j - g_bar                   (same g_bar; no separate baseline)
    A_treatment_i,t  = A_outcome_i + lambda * sum_j 1[t == first_token(tactic_i,j)] * A_process_i,j

with the frozen constants ``phi = {success: +1, d1: -0.05, d2: -0.10}`` and the primary
``lambda = 1``. Collision precedence is the oracle's frozen rule
(``d2 > d1 > success``, exactly one writer per token position, never last-write-wins).

Everything here is dependency-light and CPU-testable; the online adapter that decodes the batch's
``responses`` ids and calls the frozen oracle for the tactic spans lives in the training entry
(``scripts/v5_r001_train.py``), because the frozen oracle must not be duplicated.

Deliberately NOT here (owner §1): any lambda tuning, phi centering, response-level summation of
the process term, normalization search, or a length penalty. Those are prohibitions.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

PHI_SUCCESS = 1.0
PHI_D1 = -0.05
PHI_D2 = -0.10
PHI = {"success": PHI_SUCCESS, "d1": PHI_D1, "d2": PHI_D2}

LAMBDA_PRIMARY = 1.0

#: Frozen collision precedence (``scripts/v5_process_oracle.py::_CREDIT_PRECEDENCE``): the error
#: region beats the verified prefix, which beats success. One writer per token position.
CREDIT_PRECEDENCE = {"d2": 3, "d1": 2, "success": 1}

#: Oracle mapping statuses that carry a token position at all.
MAPPABLE_STATUSES = ("exact", "contained")

GROUP_SKIPPED_INFRA = "GROUP_SKIPPED_INFRA"


@dataclass
class CreditStats:
    """Per-arm counters; every censor path is counted, nothing is silently shifted.

    Mapping-stage counters (``mappable_tactics`` .. ``conflicts``) are per *tactic* and are
    maintained by :func:`assign_positions`; tensor-stage counters (``credit_tokens``,
    ``dropped_by_mask``, ``out_of_range``, ``infra_censored``, ``group_skipped_infra``) are per
    *(row, token position)* and are maintained by :func:`compute_process_advantage`. ``no_code``
    is set by the adapter for sentinel / format-no-code candidates (owner §2: they carry no
    process-token contribution). ``credit_tokens`` counts only credit that was actually written
    onto the advantage tensor.
    """

    candidates: int = 0
    mappable_tactics: int = 0
    exact: int = 0
    contained: int = 0
    ambiguous: int = 0
    outside_response: int = 0
    span_not_in_response: int = 0
    unmapped: int = 0
    retokenization_mismatch: int = 0
    no_code: int = 0
    conflicts: int = 0
    credit_tokens: int = 0
    dropped_by_mask: int = 0
    out_of_range: int = 0
    infra_censored: int = 0
    group_skipped_infra: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "mappable_tactics": self.mappable_tactics,
            "exact": self.exact,
            "contained": self.contained,
            "ambiguous": self.ambiguous,
            "outside_response": self.outside_response,
            "span_not_in_response": self.span_not_in_response,
            "unmapped": self.unmapped,
            "retokenization_mismatch": self.retokenization_mismatch,
            "no_code": self.no_code,
            "conflicts": self.conflicts,
            "credit_tokens": self.credit_tokens,
            "dropped_by_mask": self.dropped_by_mask,
            "out_of_range": self.out_of_range,
            "infra_censored": self.infra_censored,
            "group_skipped_infra": self.group_skipped_infra,
        }


def tactic_labels(
    n_tactics: int,
    *,
    blamed_index: int | None,
    outcome_valid: bool,
    outcome: int,
    infra: bool,
) -> list[str | None]:
    """Frozen oracle labelling (``v5_process_oracle.py`` L605-612), by tactic index.

    ``infra`` candidates and candidates without a blame carry no label; a verified candidate's
    tactics are all ``success``; otherwise the tactics strictly before the blamed index are ``d1``
    and the blamed tactic and everything after are ``d2``. A candidate that fails on its first
    tactic therefore produces a single ``d2`` on tactic 0.
    """

    labels: list[str | None] = []
    for index in range(n_tactics):
        if infra or not outcome_valid:
            labels.append(None)
        elif outcome == 1:
            labels.append("success")
        elif blamed_index is None:
            labels.append(None)
        else:
            labels.append("d1" if index < blamed_index else "d2")
    return labels


def assign_positions(
    tactic_records: Sequence[Mapping[str, Any]], stats: CreditStats | None = None
) -> tuple[dict[int, str], CreditStats]:
    """Map labelled oracle tactics onto first-token positions with the frozen precedence.

    ``tactic_records`` is the per-tactic list the frozen oracle emits
    (``scripts/v5_process_oracle.py::derive_facts`` -> ``record["tactics"]``): each entry carries
    ``label`` (``None`` = no credit) and ``mapping`` with ``status`` / ``span_in_response`` /
    ``token_index``. This mirrors ``v5_process_oracle.py::_token_credit`` exactly, including that
    re-writing the *same* label at an occupied position is not a conflict; the parity test in
    ``tests/test_v5_r001_process_credit.py`` pins the two implementations against each other. The
    R001-only ``retokenization_mismatch`` status (produced by the online adapter, never by the
    offline oracle) is counted separately.

    Returns ``{token_index: label}`` plus counters. Exactly one writer per position: on a collision
    the higher-precedence label wins, a lower one never overwrites, and nothing accumulates.
    """

    stats = stats or CreditStats()
    assigned: dict[int, str] = {}
    for tactic in tactic_records:
        mapping = tactic.get("mapping") or {}
        status = str(mapping.get("status", "unmapped"))
        if status in MAPPABLE_STATUSES and mapping.get("span_in_response") is True:
            stats.mappable_tactics += 1
            stats.exact += int(status == "exact")
            stats.contained += int(status == "contained")
            label = tactic.get("label")
            if label is None:
                continue
            if label not in PHI:
                stats.unmapped += 1
                continue
            token_index = int(mapping["token_index"])
            previous = assigned.get(token_index)
            if previous is not None and previous != label:
                stats.conflicts += 1
                if CREDIT_PRECEDENCE[label] <= CREDIT_PRECEDENCE[previous]:
                    continue
            assigned[token_index] = label
        elif status in MAPPABLE_STATUSES:
            stats.span_not_in_response += 1
            stats.unmapped += 1
        elif status == "ambiguous":
            stats.ambiguous += 1
        elif status == "outside_response":
            stats.outside_response += 1
        elif status == "retokenization_mismatch":
            stats.retokenization_mismatch += 1
        elif status != "response_only":
            stats.unmapped += 1
    return assigned, stats


def credit_phi(positions: Mapping[int, str]) -> dict[int, float]:
    """Token position -> phi value under the frozen constants."""

    return {int(position): PHI[label] for position, label in positions.items() if label in PHI}


def _group_indices(index: Sequence[Hashable], shape: int) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = {}
    for row in range(shape):
        groups.setdefault(index[row], []).append(row)
    return groups


def compute_process_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Hashable],
    row_valid: torch.Tensor | Sequence[int],
    credit_positions: Sequence[Mapping[int, str]] | None = None,
    *,
    lambda_process: float = LAMBDA_PRIMARY,
    stats: CreditStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CreditStats]:
    """DrGRPO outcome advantage plus the frozen per-token process term.

    ``row_valid`` is 1 for candidates that are NOT infrastructure-censored. ``g_bar`` is the mean
    global outcome over the *valid* candidates of the group; a group with fewer than two valid
    candidates is skipped entirely (``GROUP_SKIPPED_INFRA``). Infra-censored rows receive a zero
    advantage and a zeroed mask slot, are counted, and never enter ``g_bar`` — identical in both
    arms. With ``lambda_process = 0`` and no infra rows the result is bit-for-bit
    ``compute_grpo_outcome_advantage(norm_adv_by_std_in_grpo=False)``.

    ``credit_positions[i]`` is row ``i``'s ``{token_position: label}`` map from
    :func:`assign_positions` (empty/``None`` = no process credit); the process term is added only
    where the response mask is 1, and each written addition is counted in ``stats.credit_tokens``.

    Returns ``(advantages, returns, row_keep, stats)`` where ``row_keep`` is a ``(bs,)`` float
    mask the caller multiplies into ``response_mask`` (0 = contribute nothing to any reduction).
    """

    stats = stats or CreditStats()
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must share a shape")
    bs, response_length = token_level_rewards.shape
    valid = torch.as_tensor(row_valid, dtype=torch.float32).reshape(bs)
    scores = token_level_rewards.sum(dim=-1)
    groups = _group_indices(index, bs)

    advantages = torch.zeros_like(token_level_rewards)
    row_keep = torch.zeros(bs)
    with torch.no_grad():
        for rows in groups.values():
            valid_rows = [row for row in rows if float(valid[row]) > 0.0]
            stats.infra_censored += len(rows) - len(valid_rows)
            if len(valid_rows) < 2:
                stats.group_skipped_infra += 1
                continue
            group_mean = torch.mean(torch.stack([scores[row] for row in valid_rows]))
            for row in valid_rows:
                advantages[row] = (scores[row] - group_mean) * response_mask[row]
                row_keep[row] = 1.0
            if lambda_process == 0.0:
                continue
            for row in valid_rows:
                for position, label in (credit_positions[row] if credit_positions else {}).items():
                    if position < 0 or position >= response_length:
                        stats.out_of_range += 1
                        continue
                    if float(response_mask[row, position]) == 0.0:
                        stats.dropped_by_mask += 1
                        continue
                    advantages[row, position] += lambda_process * (PHI[label] - group_mean)
                    stats.credit_tokens += 1
    stats.candidates += bs
    return advantages, advantages.clone(), row_keep, stats


@dataclass
class GroupReport:
    """Optional per-step bookkeeping for the diagnostics log."""

    groups: int = 0
    groups_used: int = 0
    groups_skipped_infra: int = 0
    rows: int = 0
    rows_infra: int = 0
    n_collisions: int = 0
    notes: list[str] = field(default_factory=list)


def summarize_groups(
    index: Sequence[Hashable], row_valid: torch.Tensor | Sequence[int]
) -> GroupReport:
    bs = len(index)
    valid = torch.as_tensor(row_valid, dtype=torch.float32).reshape(bs)
    report = GroupReport(rows=bs, rows_infra=int((valid <= 0).sum()))
    for rows in _group_indices(index, bs).values():
        report.groups += 1
        if sum(1 for row in rows if float(valid[row]) > 0) < 2:
            report.groups_skipped_infra += 1
        else:
            report.groups_used += 1
    return report


__all__ = [
    "CREDIT_PRECEDENCE",
    "GROUP_SKIPPED_INFRA",
    "LAMBDA_PRIMARY",
    "MAPPABLE_STATUSES",
    "PHI",
    "PHI_D1",
    "PHI_D2",
    "PHI_SUCCESS",
    "CreditStats",
    "GroupReport",
    "assign_positions",
    "compute_process_advantage",
    "credit_phi",
    "summarize_groups",
    "tactic_labels",
]
