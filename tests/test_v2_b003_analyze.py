"""Tests for the V2-B003 analyzer (multi-seed p_i(b) + Expected Oracle gate)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_b003_analyze as an


def make_y(rows: list[list[str]]) -> np.ndarray:
    """rows[i][r] = 5-char pattern ('1' verified, '0' fail, '?' missing)."""
    y = np.full((len(rows), 8, len(an.BUDGETS)), an.MISSING, dtype=np.int8)
    for i, reps in enumerate(rows):
        for r, pattern in enumerate(reps):
            for b_index, char in enumerate(pattern):
                if char == "1":
                    y[i, r, b_index] = 1
                elif char == "0":
                    y[i, r, b_index] = 0
    return y


def make_meta(n: int) -> list[dict]:
    return [
        {"rank": i + 1, "statement_id": f"s{i}", "component_id": f"fc-{i}", "component_size": 1}
        for i in range(n)
    ]


def write_rollouts(path: Path, n_theorems: int, infra_cell: tuple[int, int] | None = None) -> None:
    """Minimal raw-file fixture: n theorems x 8 replicates, pattern 00001."""
    with path.open("w", encoding="utf-8") as handle:
        for rank in range(1, n_theorems + 1):
            for replicate in range(8):
                prefixes = []
                for b_index, budget in enumerate(an.BUDGETS):
                    verified = b_index == len(an.BUDGETS) - 1
                    status = "verified" if verified else "lean_error"
                    if infra_cell is not None and (rank, replicate) == infra_cell and b_index == 0:
                        status = "verifier_timeout"
                        verified = False
                    prefixes.append(
                        {
                            "budget": budget,
                            "prefix_len": budget,
                            "candidate_sha256": "0" * 64,
                            "verified": verified,
                            "verify_status": status,
                            "verify_message": None,
                            "verification_runtime_seconds": 1.0,
                        }
                    )
                record = {
                    "experiment": "V2-B003",
                    "stage": "multi_seed",
                    "theorem_rank": rank,
                    "replicate": replicate,
                    "statement_id": f"s{rank}",
                    "component_id": f"fc-{rank}",
                    "component_size": 1,
                    "prefixes": prefixes,
                }
                handle.write(json.dumps(record) + "\n")


class TestLoadAndCounts:
    def test_load_rollouts(self, tmp_path):
        path = tmp_path / "rollouts.jsonl"
        write_rollouts(path, n_theorems=2, infra_cell=(1, 0))
        y, _, meta = an.load_rollouts(path)
        assert y.shape == (2, 8, 5)
        assert len(meta) == 2
        # rank 1 replicate 0 budget 512 is infra -> missing
        assert y[0, 0, 0] == an.MISSING
        # rank 1 replicate 0 budget 4096 verified -> 1
        assert y[0, 0, 4] == 1
        # rank 2 replicate 7 budget 512 lean_error -> 0
        assert y[1, 7, 0] == 0

    def test_counts_ignore_infra(self, tmp_path):
        path = tmp_path / "rollouts.jsonl"
        write_rollouts(path, n_theorems=1, infra_cell=(1, 0))
        y, _, _ = an.load_rollouts(path)
        k, n = an.counts(y)
        # budget 512: 7 valid (1 infra), 0 successes
        assert k[0, 0] == 0
        assert n[0, 0] == 7
        # budget 4096: 8 valid, 8 successes
        assert k[0, 4] == 8
        assert n[0, 4] == 8


class TestDpAllocate:
    def test_floor_cap_forces_512(self):
        p = np.full((6, 5), 0.5)
        value, alloc = an.dp_allocate(p, cap_units=6, allow_skip=False)
        assert list(alloc) == [512] * 6
        assert value == sum(0.5 for _ in range(6))

    def test_full_cap_forces_4096(self):
        # strictly increasing p: 4096 is the unique optimum
        p = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), (6, 1))
        _, alloc = an.dp_allocate(p, cap_units=48, allow_skip=False)
        assert list(alloc) == [4096] * 6

    def test_mid_cap_concentrates_on_gainers(self):
        # theorems 0-2 gain at 4096, theorems 3-5 do not
        p = np.zeros((6, 5))
        p[:3, 4] = 1.0
        _, alloc = an.dp_allocate(p, cap_units=24, allow_skip=False)
        assert all(b >= 2048 for b in alloc[:3]) or sum(alloc[:3]) > sum(alloc[3:])
        units = [b // 512 for b in alloc]
        assert sum(units) <= 24
        assert all(b >= 512 for b in alloc)

    def test_cap_never_exceeded(self):
        p = np.random.default_rng(0).random((10, 5))
        for cap in (10, 20, 40, 80):
            _, alloc = an.dp_allocate(p, cap_units=cap, allow_skip=False)
            assert sum(b // 512 for b in alloc) <= cap
            assert all(b in an.BUDGETS for b in alloc)

    def test_skip_allowed(self):
        p = np.zeros((4, 5))
        _, alloc_no_skip = an.dp_allocate(p, cap_units=4, allow_skip=False)
        assert all(b >= 512 for b in alloc_no_skip)
        _, alloc_skip = an.dp_allocate(p, cap_units=8, allow_skip=True)
        assert list(alloc_skip) == [0, 0, 0, 0]


class TestCrossFitted:
    def test_symmetric_folds(self):
        rows = [["00001"] * 8 for _ in range(4)]
        y = make_y(rows)
        subset = np.arange(4)
        result = an.cross_fitted(y, subset, cap_units=4 * 8, allow_skip=False)
        # solution: all 4096 -> every fold eval value 1.0 (counts: 4 theorems)
        assert abs(result["value"] - 4.0) < 1e-9
        assert len(result["folds"]) == 2

    def test_swap_directions_present(self):
        rows = [["00001"] * 8 for _ in range(2)] + [["00000"] * 8 for _ in range(2)]
        y = make_y(rows)
        result = an.cross_fitted(y, np.arange(4), cap_units=4 * 8, allow_skip=False)
        directions = {f["direction"] for f in result["folds"]}
        assert directions == {"A_to_B", "B_to_A"}
        # expected solved = 2 successes out of 4 theorems (counts)
        assert abs(result["value"] - 2.0) < 1e-9


class TestUniformAndSavings:
    def test_uniform_table(self):
        rows = [["11111"] * 8 for _ in range(3)]
        y = make_y(rows)
        table = an.uniform_table(y, np.arange(3), allow_skip=False)
        assert abs(table["512"]["expected_solved"] - 3.0) < 1e-9
        assert table["512"]["alloc_cap_tokens"] == 3 * 512

    def test_savings_reference(self):
        table = {
            "512": {"expected_solved": 0.0},
            "1024": {"expected_solved": 2.0},
            "2048": {"expected_solved": 3.0},
            "3072": {"expected_solved": 3.0},
            "4096": {"expected_solved": 4.0},
        }
        result = an.savings_for(oracle_value=2.5, oracle_cap_tokens=2048, uniform=table, n_sub=4)
        assert result["uniform_budget_reference"] == 2048
        assert abs(result["allocated_cap_savings_pct"] - (100.0 * (1 - 2048 / (4 * 2048)))) < 1e-6
        none_result = an.savings_for(oracle_value=99.0, oracle_cap_tokens=1, uniform=table, n_sub=4)
        assert none_result["uniform_budget_reference"] is None


class TestBootstrap:
    def test_determinism(self):
        rng = np.random.default_rng(7)
        d_oracle = rng.random(50)
        d_uniform = rng.random(50)
        first = an.bootstrap_uncertainty(d_oracle, d_uniform)
        second = an.bootstrap_uncertainty(d_oracle, d_uniform)
        assert first == second

    def test_center_matches_mean(self):
        d_oracle = np.full(20, 0.5)
        d_uniform = np.full(20, 0.25)
        result = an.bootstrap_uncertainty(d_oracle, d_uniform)
        assert abs(result["abs_gain_mean"] - 0.25) < 1e-9
        assert abs(result["rel_gain_mean"] - 1.0) < 1e-9


class TestComputeResponse:
    def test_monotonic_counts_and_correlations(self):
        rows = [["00001"] * 8, ["11111"] * 8, ["00000"] * 8]
        y = make_y(rows)
        result = an.compute_response(y, np.arange(3))
        counts = result["monotonic_counts"]
        assert counts["non_decreasing"] >= 2  # 00001 and 11111 are non-decreasing
        assert counts["non_monotonic"] + counts["non_decreasing"] + counts["non_increasing"] == 3
        assert len(result["marginal_gains"]) == 4
        assert result["correlations"]["pearson_difficulty_sensitivity"] is not None


class TestAnalyzeEndToEnd:
    def test_small_run(self, tmp_path):
        path = tmp_path / "rollouts.jsonl"
        write_rollouts(path, n_theorems=6)
        y, _, meta = an.load_rollouts(path)
        analysis = an.analyze(y, meta, allow_skip=False)
        assert analysis["config"]["primary_set_size"] == 6
        assert analysis["coverage"]["n_low_coverage_cells"] == 0
        assert len(analysis["budget_points"]) == 5
        point = analysis["budget_points"]["4096"]
        # at N*4096 the oracle can solve everything and equal the uniform value
        assert abs(point["cross_fitted"]["value"] - 6.0) < 1e-9
        assert point["cross_fitted_gain"]["relative"] is not None
        # at the 512 floor the allocation is forced to 512 for everyone
        point512 = analysis["budget_points"]["512"]
        assert point512["cross_fitted"]["alloc_cap_tokens_mean"] == 6 * 512

    def test_low_coverage_gate(self, tmp_path):
        path = tmp_path / "rollouts.jsonl"
        write_rollouts(path, n_theorems=2)
        # reopen and degrade one cell below the gate by hand: 3 infra on (rank 1, budget 512)
        lines = path.read_text().splitlines()
        rewritten = []
        for line in lines:
            record = json.loads(line)
            if record["theorem_rank"] == 1 and record["replicate"] < 3:
                record["prefixes"][0]["verify_status"] = "verifier_server_error"
                record["prefixes"][0]["verified"] = False
            rewritten.append(json.dumps(record))
        path.write_text("\n".join(rewritten) + "\n")
        y, _, meta = an.load_rollouts(path)
        analysis = an.analyze(y, meta, allow_skip=False)
        # rank 1 budget 512 has n = 5 < 6 -> low coverage, excluded from the primary set
        assert analysis["coverage"]["n_low_coverage_cells"] == 1
        assert analysis["config"]["primary_set_size"] == 1
