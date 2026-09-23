#!/usr/bin/env python3
"""V3-D001 §20-step8 — frozen theta0 hidden-state extraction (FORMAL: run on fly122).

Extracts the pre-registered Transformer-block last-token representations for every unique
Kimina prompt that appears in the reconstructed rollout groups, plus the exact theta0-tokenized
prompt length (B1's `prompt_token_count`, single deterministic tokenizer source).

Pre-registered (owner §8/§9, frozen before any test outcome):
  primary   block 18  (~2/3 depth)   last non-padding token  ->  R^1024
  robust    blocks 9, 27             last non-padding token  ->  R^1024
The backbone is FROZEN: no parameter is updated, no gradient is taken.

This script self-records host identity (owner §0) so the artifact is unambiguously the
fly122 / RTX 3080 formal measurement, and refuses to run if the loaded architecture does
not match the pre-registered block indices.

Outputs (gitignored, host-local; only hashes/provenance are committed):
  runs/v3_d001/theta0_reps.npz   : statement_ids, token_lens, reps[stmt, {9,18,27}, 1024] fp32
  runs/v3_d001/theta0_reps_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import SEED_DIRS, build_records

DATASET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
MODEL_PATH = "models/weights/kimina_distill_0_6b"
THETA0_WEIGHTS_SHA = "34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2"

# owner §8/§9: primary=18, robustness={9,27}; must equal the arithmetic layers of the loaded config
PREREG_LAYERS = [9, 18, 27]


def verify_host() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv"])
    return {
        "hostname": platform.node(),
        "hostname_I": run(["hostname", "-I"]),
        "nvidia_smi": gpu,
        "pwd": os.getcwd(),
        "git_remote": run(["git", "-C", str(ROOT), "remote", "-v"]),
        "git_revision": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_branch": run(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--outdir", default="runs/v3_d001")
    ap.add_argument("--formal", action="store_true",
                    help="declare this the FORMAL fly122/RTX 3080 extraction. Omit on any other host "
                         "(the artifact is then tagged NON-FORMAL and must not be used as the formal result). "
                         "The device + hashes recorded in the metadata are the objective check.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    host = verify_host()
    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    built = build_records(ROOT, DATASET, REGISTRY)
    prompts = built["prompts"]  # statement_id -> full Kimina prompt (byte-identical across copies)
    stmt_ids = sorted(prompts)  # deterministic order

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
    load_s = time.time() - t_load

    nl = model.config.num_hidden_layers
    derived = [nl // 3, (2 * nl) // 3, nl - 1]
    if derived != PREREG_LAYERS:
        print(f"ERROR: loaded config layers {derived} != pre-registered {PREREG_LAYERS}; "
              f"backbone mismatch — refuse to extract.", file=sys.stderr)
        return 2

    dev = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    peak_bytes = 0
    t0 = time.time()
    token_lens = np.zeros(len(stmt_ids), dtype=np.int64)
    reps = np.zeros((len(stmt_ids), len(PREREG_LAYERS), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i, sid in enumerate(stmt_ids):
            enc = tok(prompts[sid], add_special_tokens=False, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            token_lens[i] = int(ids.shape[1])
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            for l_idx, blk in enumerate(PREREG_LAYERS):
                reps[i, l_idx] = hs[blk + 1][0, -1, :].float().cpu().numpy()  # last (non-pad) token
            if torch.cuda.is_available():
                peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
    extract_s = time.time() - t0
    peak_gb = (peak_bytes / 1e9) if torch.cuda.is_available() else 0.0

    np.savez_compressed(out / "theta0_reps.npz",
                        statement_ids=np.array(stmt_ids, dtype=object),
                        token_lens=token_lens, reps=reps, layers=np.array(PREREG_LAYERS))

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(reps).tobytes())
    h.update(np.ascontiguousarray(token_lens).tobytes())
    reps_content_sha = h.hexdigest()

    meta = {
        "artifact_type": "v3_d001_theta0_reps",
        "reps_content_sha256": reps_content_sha,
        "formality": ("FORMAL (operator-declared; verify device below == RTX 3080)" if args.formal
                      else "NON-FORMAL (tooling check only; NOT the V3-D001 formal result)"),
        "operator_declared_formal": args.formal,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model_path": args.model,
        "theta0_weights_sha256_expected": THETA0_WEIGHTS_SHA,
        "architecture": {"num_hidden_layers": nl, "hidden_size": int(model.config.hidden_size),
                         "layer_norm_eps": float(model.config.rms_norm_eps) if hasattr(model.config, "rms_norm_eps") else None},
        "pre_registered_layers_0based": PREREG_LAYERS,
        "pooling": "last non-padding token (batch=1, no padding present)",
        "rep_dtype": "float32 (heads), backbone forward bf16",
        "n_unique_statements": len(stmt_ids),
        "token_lens_min_max": [int(token_lens.min()), int(token_lens.max())],
        "load_time_s": round(load_s, 2),
        "extract_time_s": round(extract_s, 2),
        "peak_vram_gb": round(peak_gb, 3),
        "seed_dirs": SEED_DIRS,
    }
    (out / "theta0_reps_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["formality", "reps_content_sha256", "host", "n_unique_statements",
                                           "token_lens_min_max", "load_time_s", "extract_time_s", "peak_vram_gb"]}, indent=2))
    print("wrote", out / "theta0_reps.npz", "and", out / "theta0_reps_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
