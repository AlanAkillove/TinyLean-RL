#!/usr/bin/env python
"""V2-B002 infra adjudication: clean-environment re-verification.

26 theorems carry >=1 infra cell (verifier_timeout / verifier_server_error,
mostly "redeclaration persisted" from a polluted Kimina REPL pool). ALL 5
prefix candidates of each of these theorems (130 cells) are re-verified in
CLEAN environments. Every candidate is rebuilt from the recorded token_ids
(truncate -> decode -> extract -> assemble); its sha256 must equal the raw
candidate_sha256. The raw rollouts file is never modified.

Clean-environment strategy (the server exposes no per-request REPL control):
cells are processed COLUMN-WISE, one budget level at a time. Inside one column
each theorem contributes exactly one candidate and all top-level names are
distinct, so a freshly restarted server stays clean for the entire column.
The server is restarted (docker restart + /health + canary) before each
column. Unexpected redeclarations inside a column trigger that candidate's
restart+retry (bounded). Timeouts are retried without restart (bounded).

Output artifact: experiments/results/v2_b002_adjudication.json
(per-cell raw -> adjudicated status, corrected theorem patterns, and
pessimistic/optimistic sensitivity for unresolved cells).
"""

from __future__ import annotations

import argparse
import hashlib
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

import httpx
from v2_b002_generate import BUDGETS, build_prefix_candidate, sha256_hex

from tinylean_rl.verifier.policy import VerificationSession, VerifyOutcome

DEFAULT_ROLLOUTS = "experiments/results/v2_b002_rollouts.jsonl"
DEFAULT_PILOT = "experiments/manifests/v2/v2_b002_pilot_set.json"
DEFAULT_OUTPUT = "experiments/results/v2_b002_adjudication.json"
DEFAULT_PARTIAL = "experiments/results/v2_b002_adjudication.partial.jsonl"
SERVER_CONTAINER = "tinylean-rl-lean-server"
SERVER_URL = "http://127.0.0.1:8000"

INFRA_STATUSES = ("verifier_timeout", "verifier_server_error")
FAILED_STATUSES = ("lean_error", "no_extracted_proof", "assemble_failed")
INFRA_OUTCOMES = (
    VerifyOutcome.VERIFIER_TIMEOUT,
    VerifyOutcome.VERIFIER_SERVER_ERROR,
    VerifyOutcome.UNRESOLVED_INFRA_ERROR,
)
MAX_RESTART_RETRIES = 3
MAX_TIMEOUT_RETRIES = 2


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def restart_server(timeout_seconds: float = 150.0) -> bool:
    subprocess.run(["docker", "restart", SERVER_CONTAINER], check=True, capture_output=True)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{SERVER_URL}/health", timeout=5.0, trust_env=False)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def canary_ready(session: VerificationSession, attempts: int = 6) -> bool:
    for _ in range(attempts):
        if session.canary_probe():
            return True
        time.sleep(5)
    return False


def pattern_of_statuses(statuses: list[str]) -> str:
    bits = []
    for status in statuses:
        if status == "verified":
            bits.append("1")
        elif status in FAILED_STATUSES:
            bits.append("0")
        else:
            bits.append("?")
    return "".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B002 infra adjudication.")
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--pilot", default=DEFAULT_PILOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", default=DEFAULT_PARTIAL)
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--dry-run", action="store_true", help="rebuild + hash check only")
    parser.add_argument("--no-restart", action="store_true", help="skip docker restarts (debug)")
    args = parser.parse_args()

    rollouts_path = resolve(args.rollouts)
    pilot_path = resolve(args.pilot)
    partial_path = resolve(args.partial)
    output_path = resolve(args.output)

    records = read_jsonl(rollouts_path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    theorems = {t["rank"]: t for t in pilot["theorems"]}

    targets = [r for r in records if any(p["verify_status"] in INFRA_STATUSES for p in r["prefixes"])]
    target_ranks = sorted(r["theorem_rank"] for r in targets)
    names = [theorems[rk]["name"] for rk in target_ranks]
    if len(set(names)) != len(names):
        print("[ERROR] duplicate top-level names among targets - column strategy unsafe", file=sys.stderr)
        return 2
    print(f"[adjudication] {len(targets)} theorems, {len(targets) * len(BUDGETS)} cells: ranks {target_ranks}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/weights/kimina_distill_0_6b"), trust_remote_code=True, local_files_only=True
    )

    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for record in targets:
        rank = record["theorem_rank"]
        theorem = theorems[rank]
        raw_by_budget = {p["budget"]: p for p in record["prefixes"]}
        for budget in BUDGETS:
            raw = raw_by_budget[budget]
            proof, _ = build_prefix_candidate(tokenizer, theorem["formal_statement"], record["token_ids"], budget)
            rebuilt_sha = sha256_hex(proof) if proof else None
            cell = {
                "rank": rank,
                "budget": budget,
                "raw_status": raw["verify_status"],
                "raw_message": raw["verify_message"],
                "candidate_sha256_raw": raw["candidate_sha256"],
                "candidate_sha256_rebuilt": rebuilt_sha,
                "rebuilt_match": rebuilt_sha == raw["candidate_sha256"],
                "adjudicated_status": None,
                "adjudicated_message": None,
                "adjudicated_runtime_seconds": None,
                "clean_attempts": 0,
            }
            cells[(rank, budget)] = cell
            if proof is None:
                cell["adjudicated_status"] = cell["raw_status"]  # no candidate to re-verify
            elif not cell["rebuilt_match"]:
                print(f"[ERROR] rank {rank} budget {budget}: rebuilt sha mismatch", file=sys.stderr)

    built = sum(1 for c in cells.values() if c["candidate_sha256_rebuilt"])
    matched = sum(1 for c in cells.values() if c["rebuilt_match"])
    print(f"[adjudication] rebuilt candidates: {built}; sha256 match: {matched}")
    if matched != built:
        print("[ERROR] some candidates failed the hash check - inspect before verifying", file=sys.stderr)
    if args.dry_run:
        print("[dry-run] rebuild-only; nothing verified")
        return 0

    done: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in read_jsonl(partial_path):
        done[(cell["rank"], cell["budget"])] = cell

    session = VerificationSession(
        base_url=args.server_url,
        server_timeout=60.0,
        client_slack=30.0,
        batch_size=1,
        max_single_retries=2,
        canary_timeout=60.0,
    )

    for budget in BUDGETS:
        column = [cells[(rk, budget)] for rk in target_ranks if (rk, budget) not in done]
        column = [c for c in column if c["candidate_sha256_rebuilt"] and not c.get("adjudicated_status")]
        if not column:
            continue
        if not args.no_restart:
            print(f"[adjudication] restarting server before budget {budget} column ({len(column)} cells)")
            if not restart_server():
                print("[ERROR] server did not become healthy after restart", file=sys.stderr)
                return 2
            if not canary_ready(session):
                print("[ERROR] canary failed after restart", file=sys.stderr)
                return 2
        for cell in column:
            rank, budget_now = cell["rank"], cell["budget"]
            theorem = theorems[rank]
            record = next(r for r in targets if r["theorem_rank"] == rank)
            proof, _ = build_prefix_candidate(tokenizer, theorem["formal_statement"], record["token_ids"], budget_now)
            assert proof is not None
            attempts = 0
            while True:
                attempts += 1
                cell["clean_attempts"] = attempts
                try:
                    classified = session.verify([proof], [f"adj-r{rank}-b{budget_now}-a{attempts}"])[0]
                    outcome = classified.outcome
                    message = classified.message
                except Exception as exc:  # noqa: BLE001 - VerifierUnhealthyError and transport failures
                    outcome, message = VerifyOutcome.UNRESOLVED_INFRA_ERROR, f"{type(exc).__name__}: {exc}"
                is_redeclaration = outcome is VerifyOutcome.VERIFIER_SERVER_ERROR and "redeclaration" in (
                    message or ""
                ).lower()
                if is_redeclaration:
                    if attempts >= MAX_RESTART_RETRIES:
                        outcome, message = VerifyOutcome.UNRESOLVED_INFRA_ERROR, f"unresolved: {message}"
                        break
                    print(f"  rank {rank} b{budget_now}: redeclaration on attempt {attempts}; restart+retry")
                    if not args.no_restart:
                        restart_server()
                        canary_ready(session)
                    continue
                if outcome in (VerifyOutcome.VERIFIER_TIMEOUT,):
                    if attempts >= MAX_TIMEOUT_RETRIES:
                        outcome, message = VerifyOutcome.UNRESOLVED_INFRA_ERROR, f"unresolved: {message}"
                    else:
                        continue
                if outcome is VerifyOutcome.UNRESOLVED_INFRA_ERROR and attempts < MAX_RESTART_RETRIES and not args.no_restart:
                    restart_server()
                    canary_ready(session)
                    continue
                break
            cell["adjudicated_status"] = outcome.value
            cell["adjudicated_message"] = (message or "")[:500]
            cell["adjudicated_runtime_seconds"] = round(getattr(classified, "runtime_seconds", 0.0) or 0.0, 4)
            line = json.dumps(cell, ensure_ascii=False)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(f"  rank {rank} b{budget_now}: raw={cell['raw_status']} -> adj={cell['adjudicated_status']} (attempts {attempts})")

    all_cells = {**cells}
    for key, cell in done.items():
        all_cells[key] = cell
    for key, cell in all_cells.items():
        if cell["adjudicated_status"] is None:
            cell["adjudicated_status"] = "missing"

    # corrected patterns + sensitivity
    theorem_patterns = {}
    for rank in target_ranks:
        raw_statuses = [cells[(rank, b)]["raw_status"] for b in BUDGETS]
        adj_statuses = [all_cells[(rank, b)]["adjudicated_status"] for b in BUDGETS]
        theorem_patterns[rank] = {
            "raw_pattern": pattern_of_statuses(raw_statuses),
            "adjudicated_pattern": pattern_of_statuses(adj_statuses),
            "unresolved_cells": [b for b in BUDGETS if all_cells[(rank, b)]["adjudicated_status"] == "missing"],
        }

    unresolved = sum(1 for c in all_cells.values() if c["adjudicated_status"] == "missing")
    changed = sum(
        1
        for c in all_cells.values()
        if c["adjudicated_status"] != "missing" and c["adjudicated_status"] != c["raw_status"]
    )
    result = {
        "artifact_type": "v2_b002_infra_adjudication",
        "raw_rollouts": {"path": str(rollouts_path), "sha256": hashlib.sha256(rollouts_path.read_bytes()).hexdigest()},
        "target_ranks": target_ranks,
        "cells": [all_cells[(rk, b)] for rk in target_ranks for b in BUDGETS],
        "theorem_patterns": {str(k): v for k, v in theorem_patterns.items()},
        "summary": {
            "cells_total": len(all_cells),
            "rebuilt_hash_match": matched,
            "changed_cells": changed,
            "unresolved_cells": unresolved,
            "raw_status_counts": {
                s: sum(1 for c in all_cells.values() if c["raw_status"] == s)
                for s in {c["raw_status"] for c in all_cells.values()}
            },
            "adjudicated_status_counts": {
                s: sum(1 for c in all_cells.values() if c["adjudicated_status"] == s)
                for s in {c["adjudicated_status"] for c in all_cells.values()}
            },
        },
        "sensitivity": {
            "note": "unresolved cells are NEVER counted as failures; pessimistic = treat as not solved, optimistic = treat as solved",
            "pessimistic_solved_targets": sum(
                1 for p in theorem_patterns.values() if "1" in p["adjudicated_pattern"]
            ),
            "optimistic_solved_targets": sum(
                1
                for p in theorem_patterns.values()
                if "1" in p["adjudicated_pattern"].replace("?", "1")
            ),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "script": "scripts/v2_b002_infra_adjudication.py",
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[adjudication] changed {changed} cells; unresolved {unresolved}; artifact {output_path}")
    for rank in target_ranks:
        tp = theorem_patterns[rank]
        if tp["raw_pattern"] != tp["adjudicated_pattern"]:
            print(f"  rank {rank}: {tp['raw_pattern']} -> {tp['adjudicated_pattern']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
