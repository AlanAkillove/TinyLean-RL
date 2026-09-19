#!/usr/bin/env python
"""V2-B002 cross-process + concurrency equivalence check.

Question: does generating an ALREADY RECORDED theorem in (a) a fresh process and
(b) N concurrent processes reproduce the recorded token trajectory token-for-token?

If yes, this establishes three things at once:
- cross-process determinism (a mid-run pause/restart is harmless);
- multi-process sharding (each process batch=1) is a semantics-preserving
  throughput option for the remaining B002 theorems (manifest amendment);
- the live measured throughput gain of sharding.

Method: take the first N completed rollouts (default 4), launch one worker
subprocess per record (fully concurrent). Each worker reloads the model,
rebuilds the prompt from the pilot set, sets the recorded gen_seed, generates
with the frozen parameters (max_new_tokens=4096, do_sample, T=1.0, top_p=1.0)
and compares token-for-token against the recorded token_ids.

The systemd unit MUST be stopped before running this (exclusive GPU).

Artifact: experiments/results/v2_b002_concurrency_check.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from promptset_rollout_probe import build_prompt_text

MAX_NEW_TOKENS = 4096
DEFAULT_ROLLOUTS = "experiments/results/v2_b002_rollouts.jsonl"
DEFAULT_PILOT = "experiments/manifests/v2/v2_b002_pilot_set.json"
DEFAULT_MODEL = "models/weights/kimina_distill_0_6b"
DEFAULT_OUTPUT = "experiments/results/v2_b002_concurrency_check.json"
SERIAL_REFERENCE_SECONDS = 151.0


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def worker(rank: int, rollouts: Path, pilot_path: Path, model_path: Path) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    records = [r for r in read_records(rollouts) if r["theorem_rank"] == rank]
    if not records:
        print("WORKER_RESULT " + json.dumps({"rank": rank, "match": False, "error": "no record"}))
        return 2
    record = records[0]

    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    theorem = next(t for t in pilot["theorems"] if t["rank"] == rank)

    wall_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    prompt = build_prompt_text(tokenizer, theorem)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    set_seed(int(record["gen_seed"]))
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
    generation_seconds = time.perf_counter() - started
    prompt_width = int(encoded["input_ids"].shape[-1])
    new_ids = output[0][prompt_width:].tolist()
    recorded_ids = [int(x) for x in record["token_ids"]]

    match = new_ids == recorded_ids
    first_mismatch: int | None = None
    if not match:
        span = min(len(new_ids), len(recorded_ids))
        first_mismatch = next((i for i in range(span) if new_ids[i] != recorded_ids[i]), span)

    result = {
        "rank": rank,
        "gen_seed": int(record["gen_seed"]),
        "match": match,
        "n_new": len(new_ids),
        "n_recorded": len(recorded_ids),
        "first_mismatch": first_mismatch,
        "generation_seconds": round(generation_seconds, 2),
        "wall_seconds": round(time.perf_counter() - wall_start, 2),
    }
    print("WORKER_RESULT " + json.dumps(result))
    return 0 if match else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B002 concurrency equivalence check.")
    parser.add_argument("--worker-rank", type=int, default=0, help="internal: run one worker")
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--pilot", default=DEFAULT_PILOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--records", type=int, default=4, help="how many completed records to re-generate")
    parser.add_argument("--ranks", default="", help="explicit comma-separated ranks (overrides --records)")
    args = parser.parse_args()

    rollouts = resolve(args.rollouts)
    pilot_path = resolve(args.pilot)
    model_path = resolve(args.model)

    if args.worker_rank > 0:
        return worker(args.worker_rank, rollouts, pilot_path, model_path)

    recs = read_records(rollouts)
    if args.ranks.strip():
        wanted = {int(x) for x in args.ranks.split(",") if x.strip()}
        picked = [r for r in recs if int(r["theorem_rank"]) in wanted]
    else:
        picked = recs[: args.records]
    ranks = [int(r["theorem_rank"]) for r in picked]
    print(f"[check] re-generating {len(ranks)} recorded trajectories concurrently: ranks {ranks}")

    start = time.perf_counter()
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-rank",
                str(rank),
                "--rollouts",
                str(rollouts),
                "--pilot",
                str(pilot_path),
                "--model",
                str(model_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in ranks
    ]
    results: list[dict[str, Any]] = []
    for rank, proc in zip(ranks, processes):
        try:
            out, _ = proc.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            results.append({"rank": rank, "match": False, "error": "worker timeout"})
            continue
        lines = [ln for ln in out.splitlines() if ln.startswith("WORKER_RESULT ")]
        if lines:
            results.append(json.loads(lines[-1][len("WORKER_RESULT ") :]))
        else:
            results.append({"rank": rank, "match": False, "error": "no result", "tail": out[-400:]})
    wall_total = round(time.perf_counter() - start, 2)

    all_match = bool(results) and all(r.get("match") for r in results)
    gain = round((len(ranks) / wall_total) / (1.0 / SERIAL_REFERENCE_SECONDS), 2) if wall_total else None
    report = {
        "artifact_type": "v2_b002_concurrency_check",
        "purpose": "cross-process + concurrency equivalence evidence for a B002 amendment",
        "n_workers": len(ranks),
        "ranks": ranks,
        "results": results,
        "all_match": all_match,
        "wall_seconds_total": wall_total,
        "serial_reference_seconds_per_theorem": SERIAL_REFERENCE_SECONDS,
        "throughput_gain_estimate": gain,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if all_match else "FAIL",
    }
    for r in results:
        print("  ", json.dumps(r))
    print(
        json.dumps(
            {k: report[k] for k in ("all_match", "wall_seconds_total", "throughput_gain_estimate", "verdict")},
            indent=2,
        )
    )
    output_path = resolve(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[check] artifact: {output_path}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
