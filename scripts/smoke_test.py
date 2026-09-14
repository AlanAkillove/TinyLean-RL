#!/usr/bin/env python3
"""Run the P0 model → extraction → Kimina Lean Server smoke test."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.kimina import verify_code

PROMPT = """Complete the Lean 4 proof. Return only Lean code.

import Mathlib

example : 1 + 1 = 2 := by
"""

FORMAL_STATEMENT = "import Mathlib\n\nexample : 1 + 1 = 2 := "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    model_dir = args.model or os.getenv("TINYLEAN_MODEL_DIR")
    if not model_dir:
        model_dir = str(ROOT / "models" / "weights" / args.model_key)
    if not Path(model_dir).exists():
        print(f"[ERROR] model directory not found: {model_dir}", file=sys.stderr)
        print("Prepare it with scripts/download_models.py before running this test.", file=sys.stderr)
        return 2

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[ERROR] inference dependencies unavailable: {exc}", file=sys.stderr)
        print("Install with: uv sync --extra inference", file=sys.stderr)
        return 2

    print("[1/4] Loading model")
    load_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs.update({"dtype": torch.float16, "device_map": "auto"})
    elif args.offline:
        load_kwargs.update({"local_files_only": True})
    tokenizer_kwargs = {"trust_remote_code": True}
    if args.offline:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    print("[OK] Model loaded")

    print("[2/4] Building prompt")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print("[OK] Prompt ready")

    print("[3/4] Generating and extracting candidate proof")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    extracted = extract_proof(generated)
    if extracted.startswith(("import ", "theorem ", "lemma ", "example ")):
        proof = extracted
    elif extracted.startswith("by"):
        proof = FORMAL_STATEMENT + extracted
    else:
        # The prompt ends after `:= by`; causal generation normally returns
        # only the tactic body, so reconstruct a complete verifier input.
        proof = FORMAL_STATEMENT + "by\n" + extracted
    print("[OK] Proof candidate extracted")
    print(f"  candidate: {proof[:240]!r}")

    print("[4/4] Verifying with Kimina Lean Server")
    try:
        result = verify_code(proof)
    except Exception as exc:  # noqa: BLE001 - smoke test must expose endpoint errors
        print(f"[ERROR] Lean verification request failed: {exc}", file=sys.stderr)
        return 1
    print("[OK] Lean response received")
    print(result)
    print("\nTinyLean-RL smoke test completed (generation and verification chain is alive).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
