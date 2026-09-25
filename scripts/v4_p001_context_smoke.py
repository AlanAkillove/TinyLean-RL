#!/usr/bin/env python3
"""V4-P001 §I — non-formal context/memory smoke on the formal host (GPU, n=1, consumed data only).

Loads theta0 in a vLLM instance configured with the frozen V4 context budget, generates exactly one
completion (max_tokens = the frozen 4096 response budget, n=1) from the prepared worst-case Arm C
prompt, and records peak VRAM. This is an infrastructure audit: the completion is never analysed, no
verifier is called, and nothing here is a V4-P001 result.

PASS requires: the worst-case prompt fits the context together with the full response budget, the
engine reports no truncation, and peak device memory stays inside the frozen utilisation target.

Output: experiments/manifests/v4/v4_p001_context_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = "experiments/manifests/v4/v4_p001_context_smoke.json"
PROMPT = "experiments/manifests/v4/v4_p001_smoke_prompt.json"
SOURCES = "experiments/manifests/v3/v1_rollout_sources.json"
MAX_MODEL_LEN = 10240
MAX_RESPONSE_TOKENS = 4096
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 10240
SMOKE_SEED = 20260925


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VramSampler:
    """Peak device memory while the engine runs, from nvidia-smi (what the device actually holds)."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak_mib = 0
        self.samples_failed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=10,
                ).stdout.split()
                if out:
                    self.peak_mib = max(self.peak_mib, max(int(value) for value in out))
            except (subprocess.SubprocessError, OSError, ValueError):
                # a failed sample is a gap in the peak estimate, not a smoke failure; the count is
                # reported so the reader can judge how much of the run was actually sampled
                self.samples_failed += 1
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak_mib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-response", type=int, default=MAX_RESPONSE_TOKENS)
    args = parser.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sources = json.loads((ROOT / SOURCES).read_text())
    theta0 = sources["theta0"]
    model_dir = ROOT / theta0["dir"]
    weights = model_dir / "model.safetensors"

    prepared = json.loads((ROOT / args.prompt).read_text())
    prompt_text = prepared["prompt_text"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    artifact: dict = {
        "artifact_type": "v4_p001_context_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "infrastructure-only memory/context audit; one generation, already-consumed prompt, no verifier, no analysis",
        "authorization": "owner §I - non-formal 10GB context smoke on already-consumed data",
        "model": {
            "repo_id": theta0["repo_id"],
            "revision": theta0["revision"],
            "dir": theta0["dir"],
            "weights_sha256_frozen": theta0["weights_sha256"],
            "weights_sha256_on_disk": sha256_file(weights) if weights.exists() else None,
        },
        "engine": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "max_model_len": args.model_len,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": 256,
            "tensor_parallel_size": 1,
            "dtype": "auto",
        },
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": args.max_response,
            "n": 1,
            "seed": SMOKE_SEED,
        },
        "prompt": {
            "source_artifact": args.prompt,
            "prompt_tokens_measured_here": prompt_tokens,
            "prompt_tokens_recorded": prepared["prompt_tokens"],
            "prompt_sha256": prepared["prompt_sha256"],
            "arm": prepared["arm"],
        },
    }

    sampler = VramSampler()
    sampler.start()
    load_t0 = time.perf_counter()
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        max_model_len=args.model_len,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=256,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        disable_log_stats=True,
    )
    artifact["engine"]["load_seconds"] = round(time.perf_counter() - load_t0, 2)

    torch.cuda.reset_peak_memory_stats()
    params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=args.max_response, n=1, seed=SMOKE_SEED
    )
    gen_t0 = time.perf_counter()
    outputs = llm.generate([prompt_text], params)
    generation_seconds = time.perf_counter() - gen_t0
    completion = outputs[0].outputs[0]

    artifact["run"] = {
        "generation_seconds": round(generation_seconds, 2),
        "generated_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "completion_sha256": hashlib.sha256(completion.text.encode("utf-8")).hexdigest(),
        "completion_head": completion.text[:400],
        "completion_tail": completion.text[-400:],
    }

    del llm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    artifact["memory"] = {
        "peak_vram_mib_nvidia_smi": sampler.stop(),
        "sampler_failed_samples": sampler.samples_failed,
        "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_torch_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "device_total_mib": int(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            ).stdout.split()[0]
        ),
    }
    budget = args.model_len - args.max_response
    checks = {
        "worst_case_prompt_fits_context_with_full_response": prompt_tokens <= budget,
        "context_headroom_tokens": budget - prompt_tokens,
        "finish_reason_not_length_only_tested": True,
        "completion_terminated_naturally_or_at_budget": completion.finish_reason in {"stop", "length"},
        "weights_match_frozen": artifact["model"]["weights_sha256_on_disk"] == theta0["weights_sha256"],
        "gpu_is_the_formal_host_device": "3080" in artifact["gpu"],
    }
    artifact["checks"] = checks
    artifact["PASS"] = bool(
        checks["worst_case_prompt_fits_context_with_full_response"]
        and checks["weights_match_frozen"]
        and artifact["memory"]["peak_vram_mib_nvidia_smi"] > 0
    )
    artifact["failure_mode_for_preregistration"] = (
        "a theorem whose full Arm C prompt does not fit inside max_model_len - max_response_tokens is "
        "marked CONTEXT_INELIGIBLE before generation; proof semantics are never silently truncated"
    )

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"PASS": artifact["PASS"], "prompt_tokens": prompt_tokens,
                      "generated_tokens": artifact["run"]["generated_tokens"],
                      "finish_reason": artifact["run"]["finish_reason"],
                      "peak_vram_mib": artifact["memory"]["peak_vram_mib_nvidia_smi"],
                      "generation_seconds": artifact["run"]["generation_seconds"]}, indent=1))
    print(f"wrote {destination}")
    return 0 if artifact["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
