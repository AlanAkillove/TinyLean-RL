#!/usr/bin/env python
"""V2-B002 pilot runner: family-clean compute-response generation + verification.

Frozen protocol (experiments/manifests/v2/V2-B002.yaml):

- 512 family-clean pilot theorems (experiments/manifests/v2/v2_b002_pilot_set.json),
  K=1 trajectory each - a cheap compute-response screening, not a p_i(b) estimate;
- canonical HF single-candidate path with the exact B1 semantics that passed the
  budget-semantics audit: transformers fp16, batch_size=1, num_return_sequences=1,
  do_sample=True, temperature=1.0, top_p=1.0, set_seed(gen_seed), max_new_tokens=4096;
- gen_seed = 20260920 + rank * 8 (replicate 0; rank 1..512);
- the token trajectory is truncated at 512/1024/2048/3072/4096; every prefix is
  decoded, proof-extracted (tinylean_rl.inference.extract.extract_proof), assembled
  (promptset_rollout_probe.complete_verifier_code, the V1 rule that strips the
  `:= by sorry` placeholder) and verified independently through the B0
  VerificationSession (explicit server timeout < client timeout, canary
  fail-close gate) against the local Kimina Lean server;
- label_source = prefix_truncation (semantics established by V2-B001: PASS).

Raw output (gitignored): experiments/results/v2_b002_rollouts.jsonl - one record
per theorem; per-budget entries carry the assembled candidate sha256, outcome,
message and verification runtime. Resume-safe at theorem granularity. vLLM must
not be used (V2-B001: same-config reruns diverge).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from promptset_rollout_probe import build_prompt_text, complete_verifier_code

from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.policy import VerificationSession, VerifyOutcome

SEED_BASE = 20260920
SEED_GROUP_SIZE = 8
BUDGETS = (512, 1024, 2048, 3072, 4096)
MAX_NEW_TOKENS = 4096
DEFAULT_PILOT = "experiments/manifests/v2/v2_b002_pilot_set.json"
DEFAULT_MODEL = "models/weights/kimina_distill_0_6b"
DEFAULT_OUTPUT = "experiments/results/v2_b002_rollouts.jsonl"
DEFAULT_META = "experiments/results/v2_b002_run_meta.jsonl"
MODEL_REVISION_SOURCE = "experiments/manifests/v2/V2-E001.yaml"
RUNNER_STATUS_NO_PROOF = "no_extracted_proof"
RUNNER_STATUS_ASSEMBLE_FAILED = "assemble_failed"


def sha256_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def theorem_seed(rank: int) -> int:
    return SEED_BASE + rank * SEED_GROUP_SIZE


def load_pilot(path: Path) -> dict[str, Any]:
    pilot = json.loads(path.read_text(encoding="utf-8"))
    if pilot.get("artifact_type") != "v2_b002_pilot_set":
        raise ValueError(f"{path} is not a v2_b002_pilot_set artifact")
    if pilot.get("theorem_count") != len(pilot["theorems"]):
        raise ValueError("pilot theorem_count does not match len(theorems)")
    return pilot


def load_model_revision() -> str:
    manifest = yaml.safe_load((ROOT / MODEL_REVISION_SOURCE).read_text(encoding="utf-8"))
    entry = manifest["models"][0]
    if entry["path"] != DEFAULT_MODEL:
        raise ValueError("V2-E001 theta0 path does not match the runner model path")
    return str(entry["revision"]).strip()


def load_done_ranks(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            done.add(int(json.loads(line)["theorem_rank"]))
    return done


def append_record(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_prefix_candidate(tokenizer, formal_statement: str, token_ids, budget: int):
    """Decode a token prefix and assemble the Lean candidate.

    Returns (proof_or_None, runner_status_or_None). Runner statuses cover the
    candidate-side failures that never reach the verifier:
    ``no_extracted_proof`` (empty extraction) and ``assemble_failed``
    (``complete_verifier_code`` could not build a Lean source).
    """

    prefix_ids = token_ids[:budget]
    raw = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    try:
        extracted = extract_proof(raw)
    except ValueError:
        return None, RUNNER_STATUS_NO_PROOF
    if not extracted.strip():
        return None, RUNNER_STATUS_NO_PROOF
    proof = complete_verifier_code(formal_statement, extracted)
    if not proof:
        return None, RUNNER_STATUS_ASSEMBLE_FAILED
    return proof, None


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B002 pilot runner (generation + prefix verification).")
    parser.add_argument("--pilot", default=DEFAULT_PILOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--limit", type=int, default=0, help="smoke: only the first N pilot theorems")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--server-timeout", type=float, default=60.0)
    parser.add_argument("--client-slack", type=float, default=30.0)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    pilot = load_pilot(resolve(args.pilot))
    theorems = pilot["theorems"]
    if args.limit > 0:
        theorems = theorems[: args.limit]
    output_path = resolve(args.output)
    meta_path = resolve(args.meta)
    model_dir = resolve(args.model)
    model_revision = load_model_revision()

    print(f"[V2-B002] model: {model_dir}")
    print(f"[V2-B002] theorems: {len(theorems)} (pilot K=1, budgets {BUDGETS})")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[V2-B002] model loaded on {device}")

    prompts = {t["rank"]: build_prompt_text(tokenizer, t) for t in theorems}

    server_url = args.server_url or os.getenv("LEAN_SERVER_API_URL", "http://127.0.0.1:8000")
    session = None
    if not args.skip_verify:
        session = VerificationSession(
            base_url=args.server_url,
            server_timeout=args.server_timeout,
            client_slack=args.client_slack,
            batch_size=1,  # per-candidate calls: exact per-prefix runtimes, no batch blast radius
            max_single_retries=2,
            canary_timeout=60.0,
        )
        canary_ok = session.canary_probe()
        append_meta(
            meta_path,
            {
                "event": "run_start",
                "canary_ok": canary_ok,
                "server_url": server_url,
                "server_timeout": args.server_timeout,
                "client_timeout": session.client_timeout,
                "verification_batch_size": 1,
                "limit": args.limit,
                "theorems": len(theorems),
                "output": str(output_path),
                "model_revision": model_revision,
                "git_revision": git_revision(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not canary_ok:
            print("[V2-B002] canary probe failed before the run - refusing to start (fail-close)", file=sys.stderr)
            return 2
        print(f"[V2-B002] canary ok; server {server_url}, server_timeout {args.server_timeout}s")

    done = load_done_ranks(output_path)
    jobs = [t for t in theorems if t["rank"] not in done]
    print(f"[V2-B002] pending theorems: {len(jobs)} (already done: {len(done)})")

    budget_counts = {b: 0 for b in BUDGETS}
    progress = tqdm(jobs, desc="V2-B002 pilot", unit="theorem", dynamic_ncols=True)
    for theorem in progress:
        rank = theorem["rank"]
        seed = theorem_seed(rank)
        encoded = tokenizer(prompts[rank], return_tensors="pt").to(device)
        set_seed(seed)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_runtime = time.perf_counter() - started
        prompt_width = int(encoded["input_ids"].shape[-1])
        token_ids = output[0][prompt_width:].tolist()
        hit_eos = bool(token_ids) and token_ids[-1] == tokenizer.eos_token_id

        prefix_entries = []
        for budget in BUDGETS:
            proof, runner_status = build_prefix_candidate(
                tokenizer, theorem["formal_statement"], token_ids, budget
            )
            entry = {
                "budget": budget,
                "prefix_len": min(budget, len(token_ids)),
                "candidate_sha256": sha256_hex(proof) if proof else None,
                "verified": False,
                "verify_status": runner_status,
                "verify_message": None,
                "verification_runtime_seconds": None,
            }
            if proof is not None and session is not None:
                custom_id = f"b002-r{rank}-b{budget}"
                started_verify = time.perf_counter()
                classified = session.verify([proof], [custom_id])[0]
                entry["verification_runtime_seconds"] = round(time.perf_counter() - started_verify, 4)
                entry["verify_status"] = classified.outcome.value
                entry["verify_message"] = classified.message
                entry["verified"] = classified.outcome is VerifyOutcome.VERIFIED
            elif proof is not None:
                entry["verify_status"] = "not_checked"
            if entry["verified"]:
                budget_counts[budget] += 1
            prefix_entries.append(entry)

        record = {
            "experiment": "V2-B002",
            "track": "B",
            "stage": "pilot",
            "theorem_rank": rank,
            "statement_id": theorem["statement_id"],
            "component_id": theorem["component_id"],
            "component_size": theorem["component_size"],
            "statement_sha256": theorem["statement_sha256"],
            "natural_language_sha256": theorem["natural_language_sha256"],
            "model_path": args.model,
            "model_revision": model_revision,
            "prompt_revision": "promptset_rollout_probe.build_prompt_text (frozen B1-compatible)",
            "gen_seed": seed,
            "budget_requested": MAX_NEW_TOKENS,
            "token_ids": token_ids,
            "n_generated": len(token_ids),
            "hit_eos": hit_eos,
            "generation_runtime_seconds": round(generation_runtime, 4),
            "label_source": "prefix_truncation",
            "prefixes": prefix_entries,
            "verifier_config": None
            if session is None
            else {
                "server_url": server_url,
                "server_timeout": args.server_timeout,
                "client_timeout": session.client_timeout,
                "verification_batch_size": 1,
            },
            "git_revision": git_revision(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_record(output_path, record)
        pattern = "".join("1" if e["verified"] else "0" for e in prefix_entries)
        progress.set_postfix_str(f"r{rank} n={len(token_ids)} {pattern} {generation_runtime:.0f}s")

    print(f"[V2-B002] complete; output: {output_path}")
    print(f"[V2-B002] verified-so-far by budget: {budget_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
