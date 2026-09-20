#!/usr/bin/env python
"""V2-B003 runner: multi-seed compute-response generation + verification (K=8).

Frozen protocol (experiments/manifests/v2/V2-B003.yaml):

- 192 family-clean evaluation theorems (experiments/manifests/v2/
  v2_b003_eval_set.json; untouched reserve-pool components, one deterministic
  representative per component), K=8 independent 4096-token trajectories each;
- canonical HF single-candidate path with the exact B1 semantics: batch=1,
  num_return_sequences=1, do_sample=True, temperature=1.0, top_p=1.0,
  set_seed(gen_seed), max_new_tokens=4096; vLLM and batch>1 are excluded;
- gen_seed = 20261001 + rank * 8 + replicate (rank 1..192 in component_id
  order, replicate 0..7); replicates 0-3 and 4-7 are the fixed cross-fitting
  folds used by the analyzer (no effect on generation);
- every trajectory is truncated at 512/1024/2048/3072/4096; each prefix is
  decoded, proof-extracted (tinylean_rl.inference.extract.extract_proof),
  assembled (promptset_rollout_probe.complete_verifier_code) and verified
  through the B0 VerificationSession exactly as in V2-B002 (infra outcomes =>
  missing trials, never failures);
- label_source = prefix_truncation (B1 PASS semantics).

Raw output (gitignored): experiments/results/v2_b003_rollouts.jsonl - one
record per (theorem, replicate); resume-safe at (theorem_rank, replicate)
granularity; --shard-index/--shard-count run disjoint rank shards
(rank %% shard_count == shard_index) - the concurrency semantics verified
token-identical 8/8 in the B002 amendment.

Statement text is joined from the frozen Promptset parquet by
representative_statement_id; the frozen evaluation set artifact is never
rewritten here.
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

from promptset_rollout_probe import build_prompt_text
from v2_b002_generate import build_prefix_candidate

from tinylean_rl.verifier.policy import VerificationSession, VerifyOutcome

SEED_BASE = 20261001
REPLICATES = 8
BUDGETS = (512, 1024, 2048, 3072, 4096)
MAX_NEW_TOKENS = 4096
DEFAULT_SET = "experiments/manifests/v2/v2_b003_eval_set.json"
DEFAULT_PROMPTSET = (
    "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
)
DEFAULT_MODEL = "models/weights/kimina_distill_0_6b"
DEFAULT_OUTPUT = "experiments/results/v2_b003_rollouts.jsonl"
DEFAULT_META = "experiments/results/v2_b003_run_meta.jsonl"
MODEL_REVISION_SOURCE = "experiments/manifests/v2/V2-E001.yaml"


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


def trajectory_seed(rank: int, replicate: int) -> int:
    return SEED_BASE + rank * REPLICATES + replicate


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_type") != "v2_b003_eval_set":
        raise ValueError(f"{path} is not a v2_b003_eval_set artifact")
    components = data["evaluation_components"]
    if len(components) != data["counts"]["evaluation"]:
        raise ValueError("evaluation_components does not match counts.evaluation")
    return components


def load_statements(parquet_path: Path, statement_ids: list[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    columns = ["statement_id", "formal_statement", "informal_problem", "name"]
    table = pq.read_table(parquet_path, columns=columns)
    wanted = set(statement_ids)
    found: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        sid = row["statement_id"]
        if sid in wanted and sid not in found:
            found[sid] = row
    missing = wanted - found.keys()
    if missing:
        preview = sorted(missing)[:3]
        raise ValueError(f"statements missing from parquet ({len(missing)}): {preview}...")
    return found


def build_theorems(
    components: list[dict[str, Any]], statements: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    theorems = []
    for index, comp in enumerate(components):
        sid = comp["representative_statement_id"]
        row = statements[sid]
        formal = row["formal_statement"]
        informal = row["informal_problem"] or ""
        theorems.append(
            {
                "rank": index + 1,
                "component_id": comp["component_id"],
                "component_size": comp["size"],
                "statement_id": sid,
                "name": row["name"],
                "formal_statement": formal,
                "statement_sha256": sha256_hex(formal),
                "natural_language": informal,
                "natural_language_sha256": sha256_hex(informal),
            }
        )
    return theorems


def load_model_revision() -> str:
    manifest = yaml.safe_load((ROOT / MODEL_REVISION_SOURCE).read_text(encoding="utf-8"))
    entry = manifest["models"][0]
    if entry["path"] != DEFAULT_MODEL:
        raise ValueError("V2-E001 theta0 path does not match the runner model path")
    return str(entry["revision"]).strip()


def load_done_keys(output_path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done.add((int(record["theorem_rank"]), int(record["replicate"])))
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V2-B003 runner (K=8 multi-seed generation + prefix verification)."
    )
    parser.add_argument("--set", dest="eval_set", default=DEFAULT_SET)
    parser.add_argument("--promptset", default=DEFAULT_PROMPTSET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--limit", type=int, default=0, help="smoke: only the first N theorems")
    parser.add_argument("--shard-index", type=int, default=0, help="this shard's index")
    parser.add_argument("--shard-count", type=int, default=1, help="total shard count")
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

    components = load_eval_set(resolve(args.eval_set))
    if args.limit > 0:
        components = components[: args.limit]
    statement_ids = [c["representative_statement_id"] for c in components]
    statements = load_statements(resolve(args.promptset), statement_ids)
    theorems = build_theorems(components, statements)
    output_path = resolve(args.output)
    meta_path = resolve(args.meta)
    model_dir = resolve(args.model)
    model_revision = load_model_revision()

    print(f"[V2-B003] model: {model_dir}")
    print(f"[V2-B003] evaluation set: {len(theorems)} theorems x K={REPLICATES} (budgets {BUDGETS})")
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
    print(f"[V2-B003] model loaded on {device}")

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
                "experiment": "V2-B003",
                "canary_ok": canary_ok,
                "server_url": server_url,
                "server_timeout": args.server_timeout,
                "client_timeout": session.client_timeout,
                "verification_batch_size": 1,
                "limit": args.limit,
                "theorems": len(theorems),
                "replicates": REPLICATES,
                "folds": {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]},
                "output": str(output_path),
                "model_revision": model_revision,
                "git_revision": git_revision(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not canary_ok:
            print("[V2-B003] canary probe failed before the run - refusing to start (fail-close)", file=sys.stderr)
            return 2
        print(f"[V2-B003] canary ok; server {server_url}, server_timeout {args.server_timeout}s")

    if args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            print("[V2-B003] bad shard index/count - refusing (fail-close)", file=sys.stderr)
            return 2
        theorems = [t for t in theorems if t["rank"] % args.shard_count == args.shard_index]
        print(f"[V2-B003] shard {args.shard_index}/{args.shard_count}: {len(theorems)} theorems")

    done = load_done_keys(output_path)
    jobs = [
        (theorem, replicate)
        for theorem in theorems
        for replicate in range(REPLICATES)
        if (theorem["rank"], replicate) not in done
    ]
    print(f"[V2-B003] pending trajectories: {len(jobs)} (already done: {len(done)})")

    budget_counts = {b: 0 for b in BUDGETS}
    progress = tqdm(jobs, desc="V2-B003 multi-seed", unit="trajectory", dynamic_ncols=True)
    current_rank = None
    encoded = None
    for theorem, replicate in progress:
        rank = theorem["rank"]
        if rank != current_rank:
            encoded = tokenizer(prompts[rank], return_tensors="pt").to(device)
            current_rank = rank
        seed = trajectory_seed(rank, replicate)
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
                custom_id = f"b003-r{rank}-k{replicate}-b{budget}"
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
            "experiment": "V2-B003",
            "track": "B",
            "stage": "multi_seed",
            "theorem_rank": rank,
            "replicate": replicate,
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
        progress.set_postfix_str(f"r{rank} k{replicate} n={len(token_ids)} {pattern}")

    print(f"[V2-B003] complete; output: {output_path}")
    print(f"[V2-B003] verified-so-far by budget: {budget_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
