#!/usr/bin/env python3
"""Warm-up + latency ladder for the local Kimina Lean Server (P2.5 W6).

Context: the official reward function checks candidates with
``max_workers=40`` (``kimina_prover_rl/reward/reward.py``), while the W4 audit
(``docs/p3_config_audit.md`` S6) records that a *cold* local server rebuilds
its REPL pool slowly and can drop concurrent requests with 500s.  This script

1. fires a few sequential warm-up requests (the first pays the cold-start
   cost), then
2. measures a small serial-vs-concurrent latency ladder,
3. derives the concurrency policy for the reward path (smallest failure-free
   concurrency within 90% of peak throughput).

Output: ``experiments/results/lean_server_warmup.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Windows consoles default to GBK, which cannot print Lean/arrow symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
from promptset_rollout_probe import analyze_item, response_items

from tinylean_rl.verifier.kimina import verify_codes

WARMUP_CODE = "import Mathlib\ntheorem tinylean_warmup : (1 : Nat) = 1 := by rfl\n"


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def one_request(url: str, api_key: str | None, timeout: float, tag: str) -> dict:
    """Submit one warm-up proof candidate and time it end to end."""

    custom_id = f"tinylean-warmup-{tag}"
    started = time.perf_counter()
    try:
        decoded = verify_codes(
            [WARMUP_CODE],
            custom_ids=[custom_id],
            base_url=url,
            api_key=api_key,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        items = response_items(decoded)
        item = None
        for candidate in items:
            if str(candidate.get("custom_id")) == custom_id:
                item = candidate
                break
        if item is None and items:
            item = items[0]
        analysis = analyze_item(item)
        return {
            "ok": True,
            "seconds": round(elapsed, 3),
            "verified": bool(analysis["verified"]),
            "status": analysis["status"],
        }
    except httpx.HTTPError as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "seconds": round(elapsed, 3),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize(results: list[dict], wall_seconds: float) -> dict:
    ok_results = [result for result in results if result["ok"]]
    latencies = sorted(result["seconds"] for result in ok_results)
    return {
        "requests": len(results),
        "ok": len(ok_results),
        "failed": len(results) - len(ok_results),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(len(ok_results) / wall_seconds, 3) if wall_seconds > 0 else None,
        "latency_seconds": {
            "min": latencies[0] if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": latencies[-1] if latencies else None,
        },
        "verified_count": sum(1 for result in ok_results if result.get("verified")),
        "errors": [result.get("error") for result in results if not result["ok"]][:5],
    }


def measure_serial(url: str, api_key: str | None, timeout: float, count: int) -> dict:
    results = []
    started = time.perf_counter()
    for index in range(count):
        results.append(one_request(url, api_key, timeout, f"serial-{index}-{uuid.uuid4().hex[:6]}"))
    return summarize(results, time.perf_counter() - started)


def measure_concurrent(url: str, api_key: str | None, timeout: float, count: int, workers: int) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(one_request, url, api_key, timeout, f"c{workers}-{index}-{uuid.uuid4().hex[:6]}")
            for index in range(count)
        ]
        results = [future.result() for future in futures]
    return summarize(results, time.perf_counter() - started)


def recommend(levels: list[dict]) -> dict:
    """Smallest failure-free concurrency within 90% of the peak throughput."""

    clean = [level for level in levels if level["failed"] == 0 and level["ok"] > 0]
    if not clean:
        return {
            "recommended_concurrency": 1,
            "policy": "every concurrent level recorded failures; keep the reward path serial",
        }
    peak = max(clean, key=lambda level: level["throughput_rps"])
    for level in sorted(clean, key=lambda level: level["concurrency"]):
        if level["throughput_rps"] >= 0.9 * peak["throughput_rps"]:
            return {
                "recommended_concurrency": level["concurrency"],
                "peak_throughput_concurrency": peak["concurrency"],
                "policy": (
                    "smallest failure-free concurrency within 90% of peak throughput "
                    f"({peak['throughput_rps']} rps at c={peak['concurrency']})"
                ),
            }
    return {"recommended_concurrency": 1, "policy": "fallback to serial"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Lean server base URL; defaults to LEAN_SERVER_API_URL.")
    parser.add_argument("--api-key", help="Defaults to LEAN_SERVER_API_KEY.")
    parser.add_argument("--output", default="experiments/results/lean_server_warmup.json")
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measure-requests", type=int, default=8)
    parser.add_argument("--concurrency", default="1,2,4", help="Comma-separated concurrency ladder.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--first-timeout", type=float, default=900.0)
    args = parser.parse_args()

    url = (args.url or os.getenv("LEAN_SERVER_API_URL", "http://127.0.0.1:8000")).rstrip("/")
    api_key = args.api_key or os.getenv("LEAN_SERVER_API_KEY")
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concurrency_levels = sorted({int(value) for value in args.concurrency.split(",") if value.strip()})
    if not concurrency_levels:
        print("[ERROR] --concurrency must include at least one level", file=sys.stderr)
        return 2

    print(f"[1/3] Warm-up: {args.warmup_requests} sequential requests against {url}")
    warmup_seconds: list[float] = []
    warmup_results: list[dict] = []
    for index in range(args.warmup_requests):
        timeout = args.first_timeout if index == 0 else args.timeout
        result = one_request(url, api_key, timeout, f"warmup-{index}-{uuid.uuid4().hex[:6]}")
        warmup_results.append(result)
        warmup_seconds.append(result["seconds"])
        state = "ok" if result["ok"] else f"FAILED ({result.get('error')})"
        print(f"  warm-up [{index}] {result['seconds']:.2f}s {state}")

    print(f"[2/3] Latency ladder: levels={concurrency_levels}, {args.measure_requests} requests each")
    ladder: list[dict] = []
    for workers in concurrency_levels:
        if workers == 1:
            summary = measure_serial(url, api_key, args.timeout, args.measure_requests)
        else:
            summary = measure_concurrent(url, api_key, args.timeout, args.measure_requests, workers)
        summary["concurrency"] = workers
        ladder.append(summary)
        print(
            f"  c={workers}: ok={summary['ok']}/{summary['requests']} "
            f"wall={summary['wall_seconds']}s throughput={summary['throughput_rps']}rps "
            f"median={summary['latency_seconds']['median']}s"
        )

    recommendation = recommend(ladder)
    summary_json = {
        "artifact_type": "lean_server_warmup",
        "url": url,
        "warmup_code": WARMUP_CODE,
        "warmup": {
            "requests": len(warmup_results),
            "seconds_each": warmup_seconds,
            "cold_start_seconds": warmup_seconds[0] if warmup_seconds else None,
            "all_ok": all(result["ok"] for result in warmup_results),
        },
        "ladder": ladder,
        "recommendation": recommendation,
        "policy_notes": (
            "The official reward path checks with max_workers=40 on a warm server pool. "
            "Locally the same server must first be warmed up (see cold_start_seconds) and the "
            "reward path should not exceed the recommended_concurrency measured here."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[3/3] Recommendation")
    print(json.dumps(recommendation, indent=2))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
