"""Tests for the V2-B002 analysis helpers (pure functions)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v2_b002_analyze import (
    analyze_monotonicity,
    analyze_oracle,
    analyze_patterns,
    first_success,
    is_monotonic,
    pattern_of,
)


def make_record(rank: int, pattern: str) -> dict:
    prefixes = []
    for index, char in enumerate(pattern):
        budget = (512, 1024, 2048, 3072, 4096)[index]
        prefixes.append(
            {
                "budget": budget,
                "verified": char == "1",
                "verify_status": "verified" if char == "1" else "lean_error",
                "verify_message": None,
                "verification_runtime_seconds": 1.0,
            }
        )
    return {
        "theorem_rank": rank,
        "statement_id": f"s{rank}",
        "component_id": f"fc-{rank}",
        "prefixes": prefixes,
        "n_generated": 4096,
        "hit_eos": False,
        "generation_runtime_seconds": 100.0,
    }


class TestPatternHelpers:
    def test_pattern_of(self):
        assert pattern_of(make_record(1, "10110")) == "10110"

    def test_first_success(self):
        assert first_success("00111") == 2048
        assert first_success("00000") is None

    def test_is_monotonic(self):
        assert is_monotonic("00111") is True
        assert is_monotonic("00000") is True
        assert is_monotonic("11111") is True
        assert is_monotonic("11011") is False
        assert is_monotonic("01110") is False


class TestPatternAnalysis:
    def test_categories(self):
        records = [
            make_record(1, "11111"),  # easy + always
            make_record(2, "00111"),  # compute-sensitive
            make_record(3, "00000"),  # hopeless + never
            make_record(4, "00001"),  # late success
        ]
        result = analyze_patterns(records)
        cats = result["categories"]
        assert cats["easy_success_at_512"] == 1
        assert cats["compute_sensitive"] == 2  # ranks 2 and 4
        assert cats["hopeless_at_4096"] == 1
        assert cats["always_success"] == 1
        assert cats["never_success"] == 1
        assert cats["late_success_ge_3072"] == 1


class TestMonotonicity:
    def test_lists_non_monotonic_cases(self):
        records = [make_record(1, "00111"), make_record(2, "11011")]
        result = analyze_monotonicity(records)
        assert result["monotonic"] == 1
        assert result["non_monotonic"] == 1
        assert result["non_monotonic_cases"][0]["theorem_rank"] == 2
        assert result["non_monotonic_cases"][0]["pattern"] == "11011"


class TestOracle:
    def test_uniform_and_oracle(self):
        records = [
            make_record(1, "11111"),  # cost 512
            make_record(2, "00111"),  # cost 2048
            make_record(3, "00000"),  # never
        ]
        result = analyze_oracle(records)
        assert result["uniform_solved"]["512"] == 1
        assert result["uniform_solved"]["1024"] == 1
        assert result["uniform_solved"]["2048"] == 2
        assert result["uniform_solved"]["4096"] == 2
        assert result["min_success_cost_distribution"]["512"] == 1
        assert result["min_success_cost_distribution"]["2048"] == 1
        assert result["min_success_cost_distribution"]["never"] == 1

        points = {point["budget"]: point for point in result["uniform_points"]}
        # n=3; uniform-4096 cap = 12288; oracle solves all solvable (2) with cap >= 2560
        assert points[4096]["allocated_cap"] == 3 * 4096
        assert points[4096]["oracle_solved_at_same_cap"] == 2
        assert points[4096]["oracle_gain_solved"] == 0
        # uniform-1024: solved 1; oracle at the same cap 3072 can solve 2 (512 + 2048)
        assert points[1024]["allocated_cap"] == 3 * 1024
        assert points[1024]["uniform_solved"] == 1
        assert points[1024]["oracle_solved_at_same_cap"] == 2
        assert points[1024]["oracle_gain_solved"] == 1

        savings = result["savings"]
        assert savings["uniform_4096_solved"] == 2
        assert savings["oracle_cap_for_same_solved"] == 2560

    def test_frontier_breakpoints_are_cumulative(self):
        records = [make_record(1, "10000"), make_record(2, "00100")]
        result = analyze_oracle(records)
        assert result["oracle_frontier_breakpoints"] == [
            {"cap": 0, "solved": 0},
            {"cap": 512, "solved": 1},
            {"cap": 2560, "solved": 2},
        ]
