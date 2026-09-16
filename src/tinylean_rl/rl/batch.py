"""Loading and tensor assembly for cached rollout batches.

Consumes the artifacts written by ``scripts/promptset_rollout_probe.py``:
``prompts.jsonl`` / ``rollouts.jsonl`` / ``rewards.jsonl`` / ``metadata.json``.
The tensors follow the VERL conventions that the pinned GRPO code expects:

- ``input_ids`` / ``attention_mask``: prompt + completion, right-padded;
- ``response_mask``: 1 on completion tokens, 0 on prompt and padding;
- ``token_level_rewards``: the outcome reward sits on the last valid response
  token (the VERL naive reward-manager convention, so
  ``token_level_rewards.sum(dim=-1)`` recovers the scalar reward);
- ``uids``: group key (here the ``statement_id``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cached_rollout_batch(batch_dir: Path | str) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Load a cached batch, validating that prompts/rollouts/rewards line up."""

    batch_dir = Path(batch_dir)
    prompts = load_jsonl(batch_dir / "prompts.jsonl")
    rollouts = load_jsonl(batch_dir / "rollouts.jsonl")
    rewards = load_jsonl(batch_dir / "rewards.jsonl")
    metadata = json.loads((batch_dir / "metadata.json").read_text(encoding="utf-8"))

    if not prompts:
        raise ValueError(f"{batch_dir}: prompts.jsonl is empty")
    if not rollouts:
        raise ValueError(f"{batch_dir}: rollouts.jsonl is empty")
    if len(rollouts) != len(rewards):
        raise ValueError(
            f"{batch_dir}: rollouts ({len(rollouts)}) and rewards ({len(rewards)}) lengths differ"
        )
    for index, (rollout, reward) in enumerate(zip(rollouts, rewards)):
        if rollout["candidate_id"] != reward["candidate_id"]:
            raise ValueError(
                f"{batch_dir}: candidate_id mismatch at row {index}: "
                f"{rollout['candidate_id']} != {reward['candidate_id']}"
            )
        if not 0 <= rollout["theorem_index"] < len(prompts):
            raise ValueError(
                f"{batch_dir}: theorem_index {rollout['theorem_index']} out of range at row {index}"
            )
        if not isinstance(reward.get("reward"), (int, float)):
            raise TypeError(f"{batch_dir}: non-numeric reward at row {index}")

    return prompts, rollouts, rewards, metadata


def build_training_batch(
    prompts: list[dict],
    rollouts: list[dict],
    rewards: list[dict],
    pad_token_id: int,
    max_response_length: int | None = None,
) -> dict:
    """Assemble VERL-style training tensors from a cached batch.

    ``max_response_length`` truncates each completion (e.g. to rehearse
    different sequence budgets); the reward always lands on the last surviving
    response token.  Returns a dict with ``input_ids``/``attention_mask``/
    ``response_mask``/``token_level_rewards`` tensors, plus ``uids`` and the
    per-row ``response_lengths`` lists.
    """

    rows: list[tuple[str, list[int], list[int], float]] = []
    for rollout, reward in zip(rollouts, rewards):
        prompt_ids = prompts[rollout["theorem_index"]]["prompt_ids"]
        completion_ids = list(rollout["completion_ids"])
        if max_response_length is not None:
            completion_ids = completion_ids[:max_response_length]
        rows.append((rollout["statement_id"], list(prompt_ids), completion_ids, float(reward["reward"])))

    bsz = len(rows)
    width = max(len(prompt_ids) + len(completion_ids) for _, prompt_ids, completion_ids, _ in rows)
    input_ids = torch.full((bsz, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, width), dtype=torch.long)
    response_mask = torch.zeros((bsz, width), dtype=torch.float32)
    token_level_rewards = torch.zeros((bsz, width), dtype=torch.float32)
    uids: list[str] = []
    response_lengths: list[int] = []

    for row_index, (uid, prompt_ids, completion_ids, reward) in enumerate(rows):
        uids.append(uid)
        response_lengths.append(len(completion_ids))
        prompt_end = len(prompt_ids)
        input_ids[row_index, :prompt_end] = torch.tensor(prompt_ids, dtype=torch.long)
        attention_mask[row_index, :prompt_end] = 1
        if completion_ids:
            response_end = prompt_end + len(completion_ids)
            input_ids[row_index, prompt_end:response_end] = torch.tensor(completion_ids, dtype=torch.long)
            attention_mask[row_index, prompt_end:response_end] = 1
            response_mask[row_index, prompt_end:response_end] = 1.0
            token_level_rewards[row_index, response_end - 1] = reward

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "token_level_rewards": token_level_rewards,
        "uids": uids,
        "response_lengths": response_lengths,
    }
