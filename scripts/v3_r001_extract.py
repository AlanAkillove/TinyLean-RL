#!/usr/bin/env python3
"""V3-R001 §6 / §17 step 7 — frozen-theta0 forward pass over the family-clean candidate pool.

This is NOT a rollout and NOT RL. The backbone is frozen, no gradient is taken, nothing is
generated and no verifier runs. It produces the inputs to the frozen D001 controller so that every
candidate theorem has a prospective q BEFORE a sample is drawn and long before any outcome exists
(owner §6: unlabelled theorems must still get a q; no "undefined q because it has no label").

The statement list is the pool artifact's `extraction_union` = the union of the candidates of ALL
five §5 pool readings. The owner has not yet chosen a reading, so one reading-agnostic pass covers
every option: no second GPU run, and no choice that could follow an outcome.

Recipe identity with the V3-D001 formal extraction is proven here rather than asserted:
  * the Kimina prompt of a never-rolled-out theorem is reconstructed as
    decode(apply_chat_template(messages), skip_special_tokens) and checked against the 612 rollout
    `input` strings V1 actually sent -- it must be byte-identical on every one of them;
  * block indices / dtype / last-token pooling / batch=1 are the same arithmetic as
    scripts/v3_d001_extract.py, and that script's pre-registered layer check is repeated;
  * the forward pass is re-run on a deterministic subset of D001 statements and compared
    BIT-FOR-BIT with the committed formal reps;
  * the checkpoint file hash must equal the frozen theta0 sha, else the run refuses.

Outputs (gitignored, host-local; only hashes and provenance get committed):
  runs/v3_r001/theta0_reps.npz        statement_ids, token_lens, reps[stmt, {9,18,27}, 1024]
  runs/v3_r001/theta0_reps_meta.json
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
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import _normalize, build_records

POOL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
RAW_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
MODEL_DIR = "models/weights/kimina_distill_0_6b"
MODEL_WEIGHTS = f"{MODEL_DIR}/model.safetensors"
THETA0_WEIGHTS_SHA = "34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2"

# same pre-registered surface as V3-D001 (owner §8/§9): primary 18, robustness 9 / 27
PREREG_LAYERS = [9, 18, 27]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_host() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "hostname": platform.node(),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv"]),
        "pwd": os.getcwd(),
        "git_revision": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_branch": run(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def prompt_of(tok, messages) -> str:
    """The exact string theta0 was fed for a theorem: chat template rendered, then special tokens
    dropped the same way verl's rollout dump records `input`."""
    ids = tok.apply_chat_template([dict(m) for m in messages], add_generation_prompt=True,
                                  tokenize=True)
    if isinstance(ids, dict):                      # older/newer transformers return encodings
        ids = ids["input_ids"]
    ids = list(ids)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tok.decode(ids, skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--outdir", default="runs/v3_r001")
    ap.add_argument("--check-n", type=int, default=8,
                    help="how many D001 statements to re-forward and compare bit-for-bit against the "
                         "committed formal reps (0 disables; only a recipe check, not a result)")
    ap.add_argument("--d001-reps", default="runs/v3_d001/theta0_reps.npz")
    ap.add_argument("--d001-meta", default="runs/v3_d001/theta0_reps_meta.json",
                    help="metadata of the reference extraction; its device + torch version decide "
                         "whether the recipe check demands bit-exactness or a bf16 cross-GPU tolerance")
    ap.add_argument("--formal", action="store_true",
                    help="declare this the FORMAL fly122/RTX 3080 pass; on any other host the artifact "
                         "is tagged NON-FORMAL and must not be used as the formal prediction source")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    host = verify_host()
    host["transformers"] = transformers.__version__
    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    weights_sha = (sha256_file(ROOT / MODEL_WEIGHTS) if (ROOT / MODEL_WEIGHTS).exists() else None)
    if weights_sha != THETA0_WEIGHTS_SHA:
        raise SystemExit(f"FATAL: theta0 checkpoint hash {weights_sha} != frozen {THETA0_WEIGHTS_SHA}")

    pool = json.loads((ROOT / args.pool).read_text())
    stmt_ids = sorted(pool["extraction_union"])
    by_reading = {r: sorted(v["candidate_statement_ids"]) for r, v in pool["pools_by_reading"].items()}
    for reading, ids in by_reading.items():
        missing = set(ids) - set(stmt_ids)
        if missing:
            raise SystemExit(f"FATAL: {reading} has {len(missing)} candidates outside extraction_union")

    messages_by_stmt: dict[str, list] = {}
    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    for sid, p in zip(df["statement_id"].astype(str), df["prompt"].tolist(), strict=True):
        messages_by_stmt.setdefault(sid, p.tolist() if hasattr(p, "tolist") else list(p))

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = {s: prompt_of(tok, messages_by_stmt[s]) for s in stmt_ids}
    load_s = time.time() - t_load

    # --- recipe identity, part 1: the reconstruction must reproduce what V1 actually sent --------
    built = build_records(ROOT, RAW_PARQUET, REGISTRY)
    historical = built["prompts"]
    n_exact = sum(1 for s, p in historical.items()
                  if s in messages_by_stmt and prompt_of(tok, messages_by_stmt[s]) == p)
    reconstruction = {
        "n_historical_prompts_compared": len(historical),
        "n_reproduced_byte_identically": n_exact,
        "pass": n_exact == len(historical),
    }
    if not reconstruction["pass"]:
        raise SystemExit(f"FATAL: prompt reconstruction differs from the rollout inputs: {reconstruction}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
    nl = model.config.num_hidden_layers
    derived = [nl // 3, (2 * nl) // 3, nl - 1]
    if derived != PREREG_LAYERS:
        raise SystemExit(f"FATAL: loaded config layers {derived} != pre-registered {PREREG_LAYERS}")

    check_ids: list[str] = []
    if args.check_n:
        check_ids = sorted(set(historical) - set(stmt_ids))[:args.check_n]

    ref_host: dict = {}
    if (ROOT / args.d001_meta).exists():
        ref_host = json.loads((ROOT / args.d001_meta).read_text()).get("host", {})
    same_engine = (ref_host.get("device_name") == host["device_name"]
                   and ref_host.get("torch") == host["torch"])

    dev = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    peak_bytes = 0
    all_ids = stmt_ids + check_ids
    token_lens = np.zeros(len(all_ids), dtype=np.int64)
    reps = np.zeros((len(all_ids), len(PREREG_LAYERS), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, sid in enumerate(all_ids):
            text = prompts[sid] if sid in prompts else historical[sid]
            if not _normalize(text):
                raise SystemExit(f"FATAL: empty prompt for {sid}")
            enc = tok(text, add_special_tokens=False, return_tensors="pt").to(dev)
            token_lens[i] = int(enc["input_ids"].shape[1])
            hs = model(input_ids=enc["input_ids"], output_hidden_states=True,
                       use_cache=False).hidden_states
            for l_idx, blk in enumerate(PREREG_LAYERS):
                reps[i, l_idx] = hs[blk + 1][0, -1, :].float().cpu().numpy()
            if torch.cuda.is_available():
                peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
    extract_s = time.time() - t0

    # --- recipe identity, part 2: re-forward D001 statements and compare to the formal reps -------
    # On the reference engine (same GPU, same torch) bf16 forwards are reproducible bit-for-bit, so
    # anything but exactness means the recipe changed and the run refuses. On DIFFERENT hardware the
    # reductions genuinely differ at bf16 level (measured: cosine 0.9997-0.9999, rel-L2 ~2e-2 on the
    # 3090 vs the 3080 reference), so a tight tolerance is the honest criterion -- and such a run can
    # never be declared formal.
    recipe_check: dict = {"performed": bool(check_ids), "n_statements": len(check_ids),
                          "reference": args.d001_reps,
                          "reference_device": ref_host.get("device_name"),
                          "reference_torch": ref_host.get("torch"),
                          "this_device": host["device_name"], "this_torch": host["torch"],
                          "same_engine": same_engine}
    if check_ids:
        with np.load(ROOT / args.d001_reps, allow_pickle=True) as z:
            ref_ids = [str(s) for s in z["statement_ids"]]
            ref_reps, ref_layers = z["reps"], [int(x) for x in z["layers"]]
            ref_lens = {s: int(t) for s, t in zip(ref_ids, z["token_lens"], strict=True)}
        if ref_layers != PREREG_LAYERS:
            raise SystemExit(f"FATAL: reference reps layers {ref_layers} != {PREREG_LAYERS}")
        bad_len = [s for i, s in enumerate(check_ids)
                   if token_lens[len(stmt_ids) + i] != ref_lens[s]]
        worst_cos, worst_rel, non_exact = 1.0, 0.0, []
        for i, s in enumerate(check_ids):
            new, ref = reps[len(stmt_ids) + i], ref_reps[ref_ids.index(s)]
            if not np.array_equal(new, ref):
                non_exact.append(s)
            for k in range(len(PREREG_LAYERS)):
                a, b = new[k].astype(np.float64), ref[k].astype(np.float64)
                worst_cos = min(worst_cos, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
                worst_rel = max(worst_rel, float(np.linalg.norm(a - b) / np.linalg.norm(b)))
        recipe_check |= {"n_token_lens_matching": len(check_ids) - len(bad_len),
                         "n_bit_exact": len(check_ids) - len(non_exact),
                         "worst_cosine": round(worst_cos, 8),
                         "worst_relative_l2_difference": round(worst_rel, 6),
                         "criterion": ("bit-exact (same engine)" if same_engine else
                                       "cosine >= 0.999 and rel-L2 <= 0.05 (different engine: bf16 "
                                       "reduction order differs, so bit-exactness is not achievable "
                                       "and this run cannot be formal)"),
                         "not_bit_exact": non_exact, "token_len_mismatches": bad_len,
                         "pass": bool(not bad_len and worst_cos >= 0.999
                                      and worst_rel <= 0.05
                                      and (not same_engine or not non_exact))}
        if not recipe_check["pass"]:
            raise SystemExit(f"FATAL: extraction recipe does not reproduce V3-D001: {recipe_check}")
        if args.formal and (not same_engine or non_exact):
            raise SystemExit(f"FATAL: --formal requires the reference engine AND bit-exact reps; "
                             f"same_engine={same_engine}, not_bit_exact={len(non_exact)}")
    else:
        recipe_check["pass"] = False
        recipe_check["why"] = ("recipe check disabled with --check-n 0, so this artifact cannot be "
                               "used as the formal prediction source")
        if args.formal:
            raise SystemExit("FATAL: --formal requires the recipe check (drop --check-n 0)")

    np.savez_compressed(out / "theta0_reps.npz",
                        statement_ids=np.array(stmt_ids, dtype=object),
                        token_lens=token_lens[:len(stmt_ids)], reps=reps[:len(stmt_ids)],
                        layers=np.array(PREREG_LAYERS))
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(reps[:len(stmt_ids)]).tobytes())
    h.update(np.ascontiguousarray(token_lens[:len(stmt_ids)]).tobytes())
    reps_content_sha = h.hexdigest()

    meta = {
        "artifact_type": "v3_r001_theta0_reps",
        "purpose": ("inputs to the frozen D001 controller for prospective pre-scoring; NOT rollout "
                    "output, NOT a label, NOT a gradient step"),
        "formality": ("FORMAL (operator-declared; verify device below == RTX 3080)" if args.formal
                      else "NON-FORMAL (tooling check only)"),
        "operator_declared_formal": args.formal,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model_path": args.model,
        "theta0_weights_sha256_verified": weights_sha,
        "theta0_weights_sha256_expected": THETA0_WEIGHTS_SHA,
        "architecture": {"num_hidden_layers": nl, "hidden_size": int(model.config.hidden_size)},
        "pre_registered_layers_0based": PREREG_LAYERS,
        "pooling": "last non-padding token (batch=1, no padding present)",
        "rep_dtype": "float32 (heads), backbone forward bf16",
        "statement_list": "pool artifact extraction_union (all five §5 readings, reading-agnostic)",
        "n_statements": len(stmt_ids),
        "n_statements_by_reading": {r: len(v) for r, v in by_reading.items()},
        "prompt_reconstruction_vs_rollout_inputs": reconstruction,
        "extraction_recipe_check": recipe_check,
        "token_lens_min_max": [int(token_lens[:len(stmt_ids)].min()),
                               int(token_lens[:len(stmt_ids)].max())],
        "reps_content_sha256": reps_content_sha,
        "load_time_s": round(load_s, 2),
        "extract_time_s": round(extract_s, 2),
        "peak_vram_gb": round(peak_bytes / 1e9, 3) if torch.cuda.is_available() else 0.0,
        "prohibitions_respected": [
            "no generation, no verifier, no rollout, no n=8 sampling",
            "no optimizer step, no gradient, backbone frozen",
            "no MLP / new layer / mean pooling / new representation surface (D001 surface reused)",
            "no label of any kind was read to produce these vectors",
        ],
    }
    (out / "theta0_reps_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["formality", "reps_content_sha256", "n_statements",
                                           "prompt_reconstruction_vs_rollout_inputs",
                                           "extraction_recipe_check", "token_lens_min_max",
                                           "load_time_s", "extract_time_s", "peak_vram_gb"]}, indent=2))
    print("wrote", out / "theta0_reps.npz", "and", out / "theta0_reps_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
