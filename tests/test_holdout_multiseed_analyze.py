"""Tests for the E023 multi-seed holdout analyzer input plumbing.

These tests drive the analysis script via subprocess with synthetic
artifacts, covering: explicit input-path mapping, the statement-id set
guard, and the samples-per-theorem consistency guard.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "holdout_multiseed_analyze.py"


def _artifact(
    counts_by_theorem: dict[int, int],
    *,
    samples_per_theorem: int = 2,
    statement_prefix: str = "stmt",
) -> dict:
    records = [
        {
            "theorem_index": index,
            "sample_index": sample_index,
            "statement_id": f"{statement_prefix}-{index:03d}",
            "verified": sample_index < verified_count,
        }
        for index, verified_count in counts_by_theorem.items()
        for sample_index in range(samples_per_theorem)
    ]
    return {"settings": {"samples_per_theorem": samples_per_theorem}, "records": records}


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(tmp_path: Path, paths: dict[str, Path]) -> tuple[subprocess.CompletedProcess, Path]:
    output = tmp_path / "analysis.json"
    command = [sys.executable, str(SCRIPT), "--output", str(output)]
    for label, path in paths.items():
        command += [f"--{label}", str(path)]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False), output


def test_explicit_paths_are_used_and_reported(tmp_path: Path) -> None:
    paths = {
        "theta0": _write(tmp_path, "base.json", _artifact({0: 0, 1: 1, 2: 0})),
        "seed1": _write(tmp_path, "seed1.json", _artifact({0: 1, 1: 1, 2: 0})),
        "seed2": _write(tmp_path, "seed2.json", _artifact({0: 0, 1: 1, 2: 0})),
        "seed3": _write(tmp_path, "seed3.json", _artifact({0: 0, 1: 1, 2: 1})),
    }
    process, output = _run(tmp_path, paths)
    assert process.returncode == 0, process.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["sources"] == {label: str(path) for label, path in paths.items()}
    assert artifact["theta0_verified"] == 1
    assert artifact["per_seed"]["seed1"]["verified_total"] == 2
    # seed1 flips theorem 0 from 0/2 to 2/2: deltas = [0.5, 0.0, 0.0].
    assert artifact["per_seed"]["seed1"]["bootstrap"]["mean_delta"] == pytest.approx(1 / 6)


def test_statement_id_set_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = {
        "theta0": _write(tmp_path, "base.json", _artifact({0: 0, 1: 1})),
        "seed1": _write(tmp_path, "seed1.json", _artifact({0: 0, 1: 1})),
        "seed2": _write(tmp_path, "seed2.json", _artifact({0: 0, 1: 1}, statement_prefix="other")),
        "seed3": _write(tmp_path, "seed3.json", _artifact({0: 0, 1: 1})),
    }
    process, _ = _run(tmp_path, paths)
    assert process.returncode != 0
    assert "different statement set" in process.stderr


def test_samples_per_theorem_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = {
        "theta0": _write(tmp_path, "base.json", _artifact({0: 0, 1: 1})),
        "seed1": _write(tmp_path, "seed1.json", _artifact({0: 0, 1: 1})),
        "seed2": _write(tmp_path, "seed2.json", _artifact({0: 0, 1: 1})),
        "seed3": _write(tmp_path, "seed3.json", _artifact({0: 0, 1: 1}, samples_per_theorem=3)),
    }
    process, _ = _run(tmp_path, paths)
    assert process.returncode != 0
    assert "samples_per_theorem differs" in process.stderr
