"""Loading VERL/FSDP checkpoints as inference-ready HuggingFace weights.

The P3-B checkpoints are written by VERL's FSDP worker as a single
``model_world_size_1_rank_0.pt`` full state dict.  Audited 2026-09-17: the
file is a plain ``OrderedDict`` of fp32 tensors whose 311 keys already match
the HuggingFace ``Qwen3ForCausalLM`` naming exactly (identical to the base
``model.safetensors`` key set), so the conversion is an *identity key
mapping* that these helpers validate instead of silently rewriting.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Representative tensors from different layers for compact diff reporting.
HIGHLIGHT_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.13.self_attn.v_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
    "model.layers.27.mlp.gate_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a VERL/FSDP model state dict (CPU tensors)."""

    return torch.load(Path(path), map_location="cpu", mmap=True)


def load_safetensors_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def verify_key_compat(state_dict: dict[str, torch.Tensor], reference_keys: set[str], label: str) -> None:
    """Fail loudly unless the key sets match exactly (identity key mapping)."""

    keys = set(state_dict.keys())
    if keys != set(reference_keys):
        missing = sorted(set(reference_keys) - keys)[:5]
        extra = sorted(keys - set(reference_keys))[:5]
        raise ValueError(f"{label}: key mismatch - missing {missing}, extra {extra}")


def export_hf_weights(state_dict: dict[str, torch.Tensor], output_path: str | Path) -> None:
    """Write the state dict as a HuggingFace-compatible safetensors file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: tensor.contiguous() for key, tensor in state_dict.items()}, str(output), metadata={"format": "pt"})


def tensor_diff_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """max/mean absolute difference and relative L2 norm for one tensor."""

    a = candidate.to(torch.float32)
    b = reference.to(torch.float32)
    diff = a - b
    denominator = b.norm().item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_l2": diff.norm().item() / denominator if denominator else float("inf"),
    }


def state_dict_diff_report(candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict:
    """Aggregate and highlight-tensor difference statistics vs a reference."""

    verify_key_compat(candidate, set(reference.keys()), "diff report")
    sum_sq_diff = 0.0
    sum_sq_ref = 0.0
    max_abs = 0.0
    changed_keys = 0
    for key, reference_tensor in reference.items():
        reference_f32 = reference_tensor.to(torch.float32)
        diff = candidate[key].to(torch.float32) - reference_f32
        tensor_max = diff.abs().max().item()
        max_abs = max(max_abs, tensor_max)
        if tensor_max > 0:
            changed_keys += 1
        sum_sq_diff += diff.norm().item() ** 2
        sum_sq_ref += reference_f32.norm().item() ** 2
    highlights = {
        key: tensor_diff_stats(candidate[key], reference[key])
        for key in HIGHLIGHT_KEYS
        if key in candidate
    }
    return {
        "keys": len(reference),
        "changed_keys": changed_keys,
        "aggregate": {
            "rel_l2": (sum_sq_diff**0.5) / (sum_sq_ref**0.5) if sum_sq_ref else float("inf"),
            "max_abs": max_abs,
        },
        "highlight_tensors": highlights,
    }


def compare_state_dicts_exact(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    """Bitwise identity check between two state dicts (export verification)."""

    verify_key_compat(a, set(b.keys()), "identity check")
    equal_keys = 0
    first_mismatch: str | None = None
    for key, b_tensor in b.items():
        if torch.equal(a[key], b_tensor):
            equal_keys += 1
        elif first_mismatch is None:
            first_mismatch = key
    return {
        "keys": len(b),
        "equal_keys": equal_keys,
        "identical": equal_keys == len(b),
        "first_mismatch": first_mismatch,
    }
