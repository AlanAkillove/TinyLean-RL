#!/usr/bin/env python3
"""Compile generated Lean candidates against a local Mathlib checkout.

This is a native fallback for environments without Kimina Lean Server. It is
deliberately separate from the official verifier path: a candidate is counted
as verified only when Lean exits successfully on the complete theorem source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.inference.extract import extract_proof


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model_key: str
    theorem_index: int
    sample: int
    name: str
    formal_statement: str
    raw_output: str


def _candidate_source(formal_statement: str, proof_text: str) -> str | None:
    """Build a complete Lean source from a full theorem or a proof body."""

    proof = proof_text.strip()
    if not proof:
        return None
    first_line = proof.splitlines()[0].strip()
    if first_line.startswith(("import ", "set_option ", "open ", "theorem ")):
        return proof + "\n"
    marker = ":= by"
    if marker not in formal_statement:
        return None
    prefix = formal_statement.rsplit(marker, 1)[0]
    if proof.startswith(("by ", "by\n")):
        return prefix + ":= " + proof + "\n"
    return prefix + ":= by\n" + proof + "\n"


def _load_formal_statements(dataset_path: Path) -> dict[str, str]:
    if dataset_path.is_dir():
        files = sorted(dataset_path.rglob("*.parquet"))
        if len(files) != 1:
            raise ValueError(f"expected one parquet file under {dataset_path}, found {len(files)}")
        dataset_path = files[0]
    rows = parquet.read_table(dataset_path).to_pylist()
    return {row["name"]: row["formal_statement"] for row in rows}


def _load_candidates(artifacts: list[Path], formal: dict[str, str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for artifact in artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        model_key = payload.get("model_key", artifact.stem)
        for record in payload.get("records", []):
            if not record.get("has_lean4_code_block"):
                continue
            name = record["name"]
            if name not in formal:
                continue
            candidates.append(
                Candidate(
                    candidate_id=f"{model_key}-{record['theorem_index']}-{record['sample']}",
                    model_key=model_key,
                    theorem_index=record["theorem_index"],
                    sample=record["sample"],
                    name=name,
                    formal_statement=formal[name],
                    raw_output=record["raw_output"],
                )
            )
    return candidates


def _find_elan() -> str:
    found = shutil.which("elan")
    if found:
        return found
    fallback = Path.home() / ".elan" / "bin" / "elan.exe"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("elan executable not found; install Elan or add it to PATH")


def _compile_one(candidate: Candidate, mathlib_root: Path, source_root: Path, timeout: int, elan: str) -> dict:
    source = _candidate_source(candidate.formal_statement, extract_proof(candidate.raw_output))
    result = {
        "candidate_id": candidate.candidate_id,
        "model_key": candidate.model_key,
        "theorem_index": candidate.theorem_index,
        "sample": candidate.sample,
        "name": candidate.name,
        "verified": False,
    }
    if source is None:
        result["status"] = "unbuildable_candidate"
        return result

    source_path = source_root / f"{candidate.candidate_id}.lean"
    source_path.write_text(source, encoding="utf-8")
    result["source_path"] = str(source_path.resolve())
    command = [elan, "run", "leanprover/lean4:v4.34.0", "lake", "env", "lean", str(source_path)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=mathlib_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=os.environ.copy(),
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        result["verified"] = completed.returncode == 0
        result["status"] = "verified" if result["verified"] else "lean_error"
        result["returncode"] = completed.returncode
        if output:
            result["diagnostic"] = output[-4000:]
    except subprocess.TimeoutExpired as exc:
        result["status"] = "timeout"
        result["diagnostic"] = str(exc)
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True, help="generation JSON; may be repeated")
    parser.add_argument("--dataset", default="data/raw/minif2f_hf")
    parser.add_argument("--mathlib-root", default="data/raw/mathlib4")
    parser.add_argument("--output", default="experiments/results/native_verify_long_budget.json")
    parser.add_argument("--source-root", default="experiments/results/native_verify_sources")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else ROOT / value

    artifacts = [resolve(path) for path in args.artifact]
    dataset_path = resolve(args.dataset)
    mathlib_root = resolve(args.mathlib_root)
    source_root = resolve(args.source_root)
    output_path = resolve(args.output)
    if not mathlib_root.exists():
        print(f"[ERROR] Mathlib root not found: {mathlib_root}", file=sys.stderr)
        return 2
    formal = _load_formal_statements(dataset_path)
    candidates = _load_candidates(artifacts, formal)
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    source_root.mkdir(parents=True, exist_ok=True)
    elan = _find_elan()

    print(f"[1/2] Compiling {len(candidates)} candidates with {args.workers} workers")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_compile_one, candidate, mathlib_root, source_root, args.timeout, elan): candidate
            for candidate in candidates
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 8 == 0 or index == len(futures):
                print(f"  completed {index}/{len(futures)}")
    results.sort(key=lambda item: item["candidate_id"])
    summary = {
        "artifact_type": "native_lean_verification",
        "warning": "Native Mathlib compilation is a fallback; official scores require Kimina Lean Server parity.",
        "mathlib_root": str(mathlib_root.resolve()),
        "mathlib_commit": subprocess.run(
            ["git", "-C", str(mathlib_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "candidate_count": len(results),
        "verified_count": sum(item["verified"] for item in results),
        "failed_count": sum(not item["verified"] for item in results),
        "status_counts": {
            status: sum(item.get("status") == status for item in results)
            for status in sorted({item.get("status") for item in results})
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[2/2] Verification saved")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"Output: {output_path}")
    return 0 if summary["candidate_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
