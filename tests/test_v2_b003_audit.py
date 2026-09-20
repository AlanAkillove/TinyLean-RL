"""Tests for the V2-B003 post-run audit helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_b003_analyze as an
import v2_b003_audit as audit


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


class TestPrimarySubset:
    def test_excludes_low_coverage_theorem(self):
        rows = [["00000"] * 8 for _ in range(3)]
        # theorem 2 has only 5 valid trials at budget 512 (3 missing)
        rows[1][0] = "?0000"
        rows[1][1] = "?0000"
        rows[1][2] = "?0000"
        y = make_y(rows)
        subset = audit.primary_subset(y)
        assert list(subset) == [0, 2]


class TestFoldCoverage:
    def test_clean_data_is_coverage_ok(self):
        rows = [["00001"] * 8 for _ in range(10)]
        y = make_y(rows)
        subset = audit.primary_subset(y)
        result = audit.fold_coverage(y, subset)
        assert result["gate_coverage_status"] == "coverage-ok"
        for name in ("fold_A", "fold_B"):
            stat = result["per_fold"][name]
            assert stat["cells_below_3"] == 0
            assert stat["hist_valid_trials"]["4"] == stat["cells"]

    def test_widespread_missingness_flags_inconclusive(self):
        rows = [["00001"] * 8 for _ in range(20)]
        # every theorem loses 2 trials at budget 512 in fold A (reps 0,1)
        for i in range(20):
            rows[i][0] = "?0001"
            rows[i][1] = "?0001"
        y = make_y(rows)
        subset = audit.primary_subset(y)  # n=6 at 512 still >= gate
        result = audit.fold_coverage(y, subset)
        stat_a = result["per_fold"]["fold_A"]
        assert stat_a["cells_below_3"] == 20  # one cell per theorem
        assert abs(stat_a["share_below_3"] - 20 / (20 * 5)) < 1e-12
        assert result["gate_coverage_status"] == "coverage-limited/inconclusive"
        assert result["per_fold"]["fold_B"]["cells_below_3"] == 0


class TestFullProcedureBootstrap:
    def test_determinism(self):
        rows = [["00001"] * 8 for _ in range(6)] + [["00000"] * 8 for _ in range(6)]
        y = make_y(rows)
        subset = audit.primary_subset(y)
        first = audit.full_procedure_bootstrap(y, subset, 2048, False, n_boot=8, seed=7)
        second = audit.full_procedure_bootstrap(y, subset, 2048, False, n_boot=8, seed=7)
        first.pop("wall_seconds")
        second.pop("wall_seconds")
        assert first == second

    def test_zero_success_gives_zero_gain(self):
        rows = [["00000"] * 8 for _ in range(6)]
        y = make_y(rows)
        subset = audit.primary_subset(y)
        result = audit.full_procedure_bootstrap(y, subset, 2048, False, n_boot=6, seed=1)
        assert abs(result["abs_gain_mean"]) < 1e-12
        assert result["rel_gain_mean"] is None  # uniform rate is 0 everywhere

    def test_all_success_gives_zero_gain(self):
        rows = [["11111"] * 8 for _ in range(6)]
        y = make_y(rows)
        subset = audit.primary_subset(y)
        result = audit.full_procedure_bootstrap(y, subset, 4096, False, n_boot=6, seed=1)
        assert abs(result["abs_gain_mean"]) < 1e-12
        assert abs(result["rel_gain_mean"]) < 1e-12


class TestHeadlineTable:
    def test_rows_and_fields(self):
        fake_point = {
            "uniform_expected_solved": 1.0,
            "cross_fitted": {"value": 2.0},
            "cross_fitted_gain": {"absolute": 1.0, "relative": 1.0},
            "bootstrap": {"rel_gain_ci95": [0.5, 1.5], "abs_gain_ci95": [0.5, 1.5]},
            "savings_cross_fitted": {"allocated_cap_savings_pct": 50.0},
            "plug_in": {"expected_solved": 3.0},
        }
        analysis = {
            "config": {"primary_set_size": 192},
            "budget_points": {str(tokens): fake_point for tokens in an.BUDGETS},
        }
        rows = audit.headline_table(analysis)
        assert len(rows) == 5
        assert rows[0]["uniform_budget"] == 512
        assert rows[-1]["uniform_budget"] == 4096
        for row in rows:
            assert set(row) >= {
                "uniform_expected_solved",
                "cf_oracle_expected_solved",
                "absolute_gain",
                "relative_gain",
                "n_theorems",
            }
