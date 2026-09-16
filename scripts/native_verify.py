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
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LEAN_TOOLCHAIN = "leanprover/lean4:v4.34.0-rc2"
LEAN_TOOLCHAIN_DIR = "leanprover--lean4---v4.34.0-rc2"
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


def _find_lean_environment(mathlib_root: Path) -> tuple[list[str], dict[str, str]]:
    """Return a direct Lean command and the equivalent of ``lake env``.

    Calling ``elan run ... lake env lean`` for every candidate leaves the
    spawned Lake/Lean children alive when a Windows timeout fires.  The local
    rc2 toolchain is already materialized in ``.tools/elan``; use its compiler
    directly and construct the library search path that Lake would provide.
    """

    configured_lean = os.environ.get("TINYLEAN_LEAN_BIN")
    if configured_lean:
        lean_executable = Path(configured_lean)
        toolchain_bin = lean_executable.parent
        toolchain_root = toolchain_bin.parent
    else:
        toolchain_root = ROOT / ".tools" / "elan" / "toolchains" / LEAN_TOOLCHAIN_DIR
        toolchain_bin = toolchain_root / "bin"
        lean_executable = toolchain_bin / ("lean.exe" if os.name == "nt" else "lean")
    if not lean_executable.exists():
        raise FileNotFoundError(
            f"Lean toolchain {LEAN_TOOLCHAIN} not found at {lean_executable}; "
            "install the pinned toolchain or set TINYLEAN_LEAN_BIN"
        )

    package_root = mathlib_root / ".lake" / "packages"
    package_dirs = sorted((path for path in package_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    lean_path: list[Path] = []
    source_path: list[Path] = []
    binary_path: list[Path] = []
    library_path: list[Path] = []
    for package in package_dirs:
        package_build = package / ".lake" / "build"
        lean_path.append(package_build / "lib" / "lean")
        source_path.extend([package, package])
        binary_path.append(package_build / "bin")
        library_path.append(package_build / "lib")
    project_build = mathlib_root / ".lake" / "build"
    lean_path.append(project_build / "lib" / "lean")
    source_path.extend([mathlib_root, mathlib_root])
    binary_path.append(project_build / "bin")
    library_path.append(project_build / "lib")
    lean_path.append(toolchain_root / "lib" / "lean")
    source_path.append(toolchain_root / "src" / "lean" / "lake")

    environment = os.environ.copy()
    environment["LEAN_PATH"] = os.pathsep.join(str(path) for path in lean_path if path.exists())
    environment["LEAN_SRC_PATH"] = os.pathsep.join(str(path) for path in source_path if path.exists())
    environment["PATH"] = os.pathsep.join(
        [str(path) for path in [*binary_path, *library_path, toolchain_bin] if path.exists()]
        + [environment.get("PATH", "")]
    )
    environment["LEAN_SYSROOT"] = str(toolchain_root)
    return [str(lean_executable)], environment


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a compiler and all descendants after a timeout."""

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, 9)
    except ProcessLookupError:
        pass


def _strip_imports(source: str) -> str:
    """Remove per-candidate imports before placing a source in one batch."""

    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("import "))


def _candidate_namespace(candidate_id: str) -> str:
    return "candidate_" + re.sub(r"[^A-Za-z0-9_]", "_", candidate_id)


def _build_batch_source(
    candidates: list[Candidate], source_root: Path
) -> tuple[Path, list[dict], list[dict]]:
    """Create one Lean file so Mathlib is imported only once on Windows."""

    lines = ["import Mathlib", "import Aesop", ""]
    entries: list[dict] = []
    results: list[dict] = []
    for candidate in candidates:
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
            results.append(result)
            continue

        individual_path = source_root / f"{candidate.candidate_id}.lean"
        individual_path.write_text(source, encoding="utf-8")
        result["source_path"] = str(individual_path.resolve())
        namespace = _candidate_namespace(candidate.candidate_id)
        lines.append(f"-- TINYLEAN_CANDIDATE {candidate.candidate_id}")
        lines.append(f"namespace {namespace}")
        start_line = len(lines) + 1
        body_lines = _strip_imports(source).splitlines()
        lines.extend(body_lines)
        end_line = len(lines)
        lines.append(f"end {namespace}")
        lines.append("")
        entries.append(
            {
                "candidate": candidate,
                "result": result,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    batch_path = source_root / "native_verify_batch.lean"
    batch_path.write_text("\n".join(lines), encoding="utf-8")
    return batch_path, entries, results


def _compile_batch(
    candidates: list[Candidate],
    mathlib_root: Path,
    source_root: Path,
    timeout: int,
    command: list[str],
    environment: dict[str, str],
) -> tuple[list[dict], float]:
    """Compile all candidates in one process and attribute diagnostics by line."""

    batch_path, entries, results = _build_batch_source(candidates, source_root)
    if not entries:
        return results, 0.0

    started = time.perf_counter()
    process = subprocess.Popen(
        [*command, str(batch_path)],
        cwd=mathlib_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _terminate_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        output = (str(exc) + "\n" + stdout + "\n" + stderr).strip()
    else:
        output = (stdout + "\n" + stderr).strip()
    elapsed = round(time.perf_counter() - started, 3)

    diagnostics_by_line: dict[int, list[str]] = {}
    for line in output.splitlines():
        match = re.search(r":(\d+):\d+: error:", line)
        if match:
            diagnostics_by_line.setdefault(int(match.group(1)), []).append(line)

    for entry in entries:
        result = entry["result"]
        result["batch_source_path"] = str(batch_path.resolve())
        result["elapsed_seconds"] = elapsed
        if timed_out:
            result["status"] = "timeout"
            result["diagnostic"] = output[-4000:]
            results.append(result)
            continue
        candidate_diagnostics = [
            diagnostic
            for line_number, diagnostics in diagnostics_by_line.items()
            if entry["start_line"] <= line_number <= entry["end_line"]
            for diagnostic in diagnostics
        ]
        if candidate_diagnostics:
            result["status"] = "lean_error"
            result["diagnostic"] = "\n".join(candidate_diagnostics)[-4000:]
        elif process.returncode == 0:
            result["verified"] = True
            result["status"] = "verified"
        else:
            result["status"] = "batch_error"
            result["diagnostic"] = output[-4000:]
        result["returncode"] = process.returncode
        results.append(result)
    return results, elapsed


def _compile_one(
    candidate: Candidate,
    mathlib_root: Path,
    source_root: Path,
    timeout: int,
    command: list[str],
    environment: dict[str, str],
) -> dict:
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
    full_command = [*command, str(source_path)]
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            full_command,
            cwd=mathlib_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        output = (stdout + "\n" + stderr).strip()
        result["verified"] = process.returncode == 0
        result["status"] = "verified" if result["verified"] else "lean_error"
        result["returncode"] = process.returncode
        if output:
            result["diagnostic"] = output[-4000:]
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            _terminate_process_tree(process)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
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
    parser.add_argument(
        "--mode",
        choices=("batch", "per-candidate"),
        default="batch",
        help="Batch imports Mathlib once (recommended); per-candidate isolates compiler runs.",
    )
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
    try:
        lean_command, lean_environment = _find_lean_environment(mathlib_root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"[1/2] Compiling {len(candidates)} candidates in {args.mode} mode")
    if args.mode == "batch":
        results, verification_seconds = _compile_batch(
            candidates,
            mathlib_root,
            source_root,
            args.timeout,
            lean_command,
            lean_environment,
        )
        print(f"  completed {len(results)}/{len(candidates)}")
    else:
        results = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    _compile_one,
                    candidate,
                    mathlib_root,
                    source_root,
                    args.timeout,
                    lean_command,
                    lean_environment,
                ): candidate
                for candidate in candidates
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="candidates", unit="candidate"):
                result = future.result()
                results.append(result)
        verification_seconds = round(time.perf_counter() - started, 3)
    results.sort(key=lambda item: item["candidate_id"])
    summary = {
        "artifact_type": "native_lean_verification",
        "warning": "Native Mathlib compilation is a fallback; official scores require Kimina Lean Server parity.",
        "lean_toolchain": LEAN_TOOLCHAIN,
        "verification_mode": args.mode,
        "verification_seconds": verification_seconds,
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
