"""V4-P001 — frozen taxonomy, diagnostics, prompts, seeds and statistics."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation import v4_derange, v4_schedule
from tinylean_rl.evaluation import v4_diagnostics as diag
from tinylean_rl.evaluation import v4_prompts as prompts
from tinylean_rl.evaluation import v4_seeds as seeds
from tinylean_rl.evaluation import v4_stats as stats
from tinylean_rl.evaluation.v4_taxonomy import (
    ELABORATION_TYPE_MISMATCH,
    FORMAT_NO_CODE,
    INFRA,
    OTHER_SEMANTIC,
    PRIMARY_CATEGORIES,
    SYNTAX_PARSER,
    TACTIC_FAILURE,
    TIMEOUT_RESOURCE,
    TYPECLASS_SYNTHESIS,
    UNKNOWN_IDENTIFIER,
    UNSOLVED_GOALS,
    VERIFIED,
    category_counts,
    classify_first_attempt,
    matched_primary_rules,
    primary_eligible,
)


def classify(message: str, **overrides) -> str:
    kwargs = {
        "verify_status": "lean_error",
        "lean_message": message,
        "truncated": False,
        "format_ok": True,
        "has_lean_block": True,
        "extracted": "theorem t : True := by\n  trivial",
    }
    kwargs.update(overrides)
    return classify_first_attempt(**kwargs)


# --- taxonomy ------------------------------------------------------------------------------------

def test_verified_wins_over_everything() -> None:
    assert classify("unsolved goals", verify_status="verified") == VERIFIED


@pytest.mark.parametrize(
    "status", ["verifier_timeout", "verifier_server_error", "unresolved_infra_error", "verifier_error"]
)
def test_infra_is_never_a_proof_failure(status: str) -> None:
    assert classify("unsolved goals", verify_status=status) == INFRA


def test_format_gate_precedes_semantic_rules() -> None:
    assert classify("unsolved goals", format_ok=False) == FORMAT_NO_CODE
    assert classify("unsolved goals", has_lean_block=False) == FORMAT_NO_CODE
    assert classify("unsolved goals", extracted="<think> let me think about this") == FORMAT_NO_CODE


def test_empty_message_is_not_screenable() -> None:
    # A Lean rejection with no attributable diagnostic cannot satisfy §C's "real Lean diagnostic"
    # precondition, so the frozen taxonomy routes it out of the primary cohort.
    assert classify("") == FORMAT_NO_CODE
    assert not primary_eligible(FORMAT_NO_CODE)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("maximum number of heartbeats (200000) has been exceeded", TIMEOUT_RESOURCE),
        ("deterministic) timeout at 'whnf'", TIMEOUT_RESOURCE),
        ("unknown identifier 'foo'", UNKNOWN_IDENTIFIER),
        ("unknown tactic 'nope'", UNKNOWN_IDENTIFIER),
        ("failed to synthesize instance OfNat Nat 3", TYPECLASS_SYNTHESIS),
        ("unsolved goals\ncase h\n⊢ False", UNSOLVED_GOALS),
        ("linarith failed to find a contradiction", TACTIC_FAILURE),
        ("tactic 'simp' failed", TACTIC_FAILURE),
        ("type mismatch\n  x\nhas type Nat\nbut is expected to have type Int", ELABORATION_TYPE_MISMATCH),
        ("unexpected token ';'; expected ')'", SYNTAX_PARSER),
        ("unexpected end of input", SYNTAX_PARSER),
        ("tauto failed to solve some goals.", OTHER_SEMANTIC),
    ],
)
def test_priority_ordered_classification(message: str, expected: str) -> None:
    assert classify(message) == expected


def test_primary_categories_are_exactly_the_semantic_ones() -> None:
    assert set(PRIMARY_CATEGORIES) == {
        ELABORATION_TYPE_MISMATCH, UNSOLVED_GOALS, TACTIC_FAILURE,
        UNKNOWN_IDENTIFIER, TYPECLASS_SYNTHESIS, OTHER_SEMANTIC,
    }
    assert not primary_eligible(SYNTAX_PARSER)
    assert not primary_eligible(FORMAT_NO_CODE)
    assert not primary_eligible(TIMEOUT_RESOURCE)
    assert not primary_eligible(INFRA)
    assert not primary_eligible(VERIFIED)


def test_category_counts_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        category_counts(["not_a_category"])
    assert category_counts([VERIFIED, OTHER_SEMANTIC])[VERIFIED] == 1


def test_ambiguity_measurement_reports_overlaps() -> None:
    assert matched_primary_rules(
        verify_status="lean_error", lean_message="unsolved goals ... failed to synthesize instance"
    ) == (TYPECLASS_SYNTHESIS, UNSOLVED_GOALS)
    assert matched_primary_rules(verify_status="lean_error", lean_message="unsolved goals") == (UNSOLVED_GOALS,)
    assert matched_primary_rules(verify_status="verified", lean_message="unsolved goals") == ()


# --- diagnostics ---------------------------------------------------------------------------------

def test_first_error_block_keeps_only_the_first_error() -> None:
    text = "# Error 1:\nError message: unsolved goals\ncase h\n⊢ False\n\n# Error 2:\nother"
    assert diag.first_lean_error_block(text) == "unsolved goals\ncase h\n⊢ False"


def test_normalization_removes_infra_noise_and_keeps_lean_information() -> None:
    raw = (
        "Traceback (most recent call last):\n"
        "  File \"/tmp/kimina-abc123/server.py\", line 42, in run\n"
        "request_id: 7c1f0a55-4f66-4f4e-9d3a-2f7a2d3f9b11\n"
        "2026-09-25T08:39:12Z lean server error: repl crashed\n"
        "unsolved goals\ncase h\nx y : Nat\n⊢ x + y = y + x\n"
    )
    normalized = diag.normalize_diagnostic(raw)
    assert "unsolved goals" in normalized
    assert "⊢ x + y = y + x" in normalized
    assert "x y : Nat" in normalized
    assert "/tmp/kimina" not in normalized
    assert "7c1f0a55" not in normalized
    assert "2026-09-25T08:39:12Z" not in normalized
    for gone in ("Traceback", "line 42", "repl crashed"):
        assert gone not in normalized


def test_normalization_is_deterministic_and_idempotent() -> None:
    raw = "# Error 1:\nError message: type mismatch\n  x\nhas type Nat\n"
    once = diag.normalize_diagnostic(raw)
    assert once == diag.normalize_diagnostic(raw)
    assert once == diag.normalize_diagnostic(once)


def test_normalization_version_is_frozen() -> None:
    assert diag.NORMALIZATION_VERSION == "v4-diag-1"
    assert diag.DIAGNOSTIC_TOKEN_BUDGET == 512


class _WordTokenizer:
    """Deterministic stand-in: one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_diagnostic_budget_truncates_and_reports() -> None:
    tokenizer = _WordTokenizer()
    text = " ".join(f"w{i}" for i in range(600))
    bounded, truncated = diag.bound_diagnostic(text, tokenizer, budget=50)
    assert truncated is True
    assert len(tokenizer.encode(bounded)) <= 50
    assert "truncated" in bounded

    short, truncated_short = diag.bound_diagnostic("unsolved goals", tokenizer, budget=50)
    assert short == "unsolved goals"
    assert truncated_short is False


def test_diagnostic_from_verify_item_uses_first_error_with_position() -> None:
    item = {
        "custom_id": "x",
        "response": {
            "messages": [
                {"severity": "info", "data": "ignored"},
                {"severity": "error", "data": "type mismatch", "pos": {"line": 3, "column": 7}},
                {"severity": "error", "data": "second error"},
            ],
            "sorries": [],
        },
    }
    text, status = diag.diagnostic_from_verify_item(item)
    assert status == "lean_error"
    assert text.startswith("line 3, column 7: ")
    assert "type mismatch" in text
    assert "second error" not in text


def test_diagnostic_from_verify_item_handles_repl_error_and_empty() -> None:
    assert diag.diagnostic_from_verify_item({"response": {"message": "boom"}}) == ("boom", "repl_error")
    assert diag.diagnostic_from_verify_item({"response": {"messages": [], "sorries": []}}) == ("", "")
    assert diag.diagnostic_from_verify_item({"error": "timed out"}) == ("", "")


# --- prompts -------------------------------------------------------------------------------------

BASE = [
    {"role": "system", "content": "You are an expert."},
    {"role": "user", "content": "Prove: 1 + 1 = 2"},
]
PROOF = "theorem t : 1 + 1 = 2 := by\n  norm_num"
DIAGNOSTIC = "unsolved goals\ncase h\n⊢ 1 + 1 = 2"


def test_arm_a_is_the_original_prompt_untouched() -> None:
    messages = prompts.arm_messages(BASE, arm="A_FRESH_RETRY")
    assert messages == prompts.canonical_messages(BASE)
    with pytest.raises(ValueError):
        prompts.arm_messages(BASE, arm="A_FRESH_RETRY", failed_proof=PROOF)


def test_arm_b_and_c_differ_only_by_the_diagnostic() -> None:
    b = prompts.arm_messages(BASE, arm="B_SELF_REVISION", failed_proof=PROOF)
    c = prompts.arm_messages(BASE, arm="C_VERIFIER_REPAIR", failed_proof=PROOF, diagnostic=DIAGNOSTIC)
    assert len(b) == len(c) == len(BASE) + 2
    assert b[2] == {"role": "assistant", "content": PROOF}
    assert b[3]["content"] == prompts.CORRECTION_REQUEST
    assert c[3]["content"] == f"{prompts.CORRECTION_REQUEST}\n\n{DIAGNOSTIC}"
    assert prompts.arm_suffix_invariant(BASE, failed_proof=PROOF, diagnostic=DIAGNOSTIC)


def test_arm_b_rejects_a_diagnostic_and_arms_b_c_require_a_proof() -> None:
    with pytest.raises(ValueError):
        prompts.arm_messages(BASE, arm="B_SELF_REVISION", failed_proof=PROOF, diagnostic=DIAGNOSTIC)
    with pytest.raises(ValueError):
        prompts.arm_messages(BASE, arm="B_SELF_REVISION")
    with pytest.raises(ValueError):
        prompts.arm_messages(BASE, arm="C_VERIFIER_REPAIR", failed_proof=PROOF)


def test_arm_b_correction_text_is_generic_and_frozen() -> None:
    b = prompts.arm_messages(BASE, arm="B_SELF_REVISION", failed_proof=PROOF)
    revision = b[-1]["content"]
    assert revision == prompts.CORRECTION_REQUEST
    for leak in ("verifier", "diagnostic", "error message", "unsolved", "reported", "line "):
        assert leak not in revision.lower()


def test_unknown_arm_and_bad_prompt_shape_are_rejected() -> None:
    with pytest.raises(ValueError):
        prompts.arm_messages(BASE, arm="D_IMAGINED")
    with pytest.raises(ValueError):
        prompts.canonical_messages([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}])


def test_renderer_uses_the_frozen_template_without_a_tool_role() -> None:
    class _Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    text = prompts.render_arm(_Tokenizer(), BASE, arm="B_SELF_REVISION", failed_proof=PROOF)
    assert "tool" not in text
    assert text.endswith("|assistant:")
    assert prompts.RENDERER_VERSION == "v4-prompt-1"


# --- seeds ---------------------------------------------------------------------------------------

def test_seed_streams_are_frozen_arithmetic_and_shared_across_arms() -> None:
    assert seeds.V4_BASE_SEED == 20260925
    assert seeds.first_stage_seeds(3) == [20260925, 20260926, 20260927]
    assert seeds.second_stage_seeds(3) == [21260925, 21260926, 21260927]
    assert len(set(seeds.first_stage_seeds(seeds.MAX_SCREENING))) == seeds.MAX_SCREENING
    assert len(set(seeds.second_stage_seeds())) == seeds.N_PRIMARY
    assert not set(seeds.first_stage_seeds(seeds.MAX_SCREENING)) & set(seeds.second_stage_seeds())


@pytest.mark.parametrize("rank", [0, -1, 129, 641])
def test_seed_ranks_are_range_checked(rank: int) -> None:
    with pytest.raises(ValueError):
        seeds.second_stage_seed(rank)
    if rank <= 0 or rank > 640:
        with pytest.raises(ValueError):
            seeds.first_stage_seed(rank)


# --- statistics ----------------------------------------------------------------------------------

def test_mcnemar_exact_matches_the_binomial_tail() -> None:
    assert stats.mcnemar_exact_greater(0, 0) == 1.0
    assert stats.mcnemar_exact_greater(0, 7) == pytest.approx(1 / 128)
    assert stats.mcnemar_exact_greater(1, 9) == pytest.approx(11 / 1024)
    assert stats.mcnemar_exact_greater(3, 11) > stats.mcnemar_exact_greater(3, 20)


def test_paired_gate_counts_directions_correctly() -> None:
    a = [False, True, False, False, True, False]
    c = [True, False, True, False, True, False]
    gate = stats.paired_gate(a, c)
    assert gate.n_favor == 2        # a fail -> c pass
    assert gate.n_against == 1      # a pass -> c fail
    assert gate.delta == pytest.approx(1 / 6)
    assert 0.0 < gate.mcnemar_p < 1.0


def test_paired_gate_is_symmetric_in_its_inputs() -> None:
    a = [False] * 20 + [True] * 5
    c = [True] * 20 + [False] * 5
    forward = stats.paired_gate(a, c)
    backward = stats.paired_gate(c, a)
    assert forward.delta == pytest.approx(-backward.delta)
    assert forward.n_favor == backward.n_against


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    diffs = [1.0] * 30 + [0.0] * 70
    lower, upper = stats.paired_bootstrap_ci(diffs)
    assert lower <= 0.3 <= upper
    assert (lower, upper) == stats.paired_bootstrap_ci(diffs)
    assert lower > 0.0 and upper < 0.6


BALANCED_RATES = {"A_FRESH_RETRY": 0.02, "B_SELF_REVISION": 0.02, "C_VERIFIER_REPAIR": 0.02,
                  "D_MISMATCHED_DIAGNOSTIC": 0.03}


def test_outcome_taxonomy_follows_the_frozen_order() -> None:
    passing = stats.Gate(delta=0.10, mcnemar_p=0.01, ci_lower=0.02, n_favor=12, n_against=2)
    failing = stats.Gate(delta=0.02, mcnemar_p=0.30, ci_lower=-0.03, n_favor=4, n_against=3)
    assert stats.decide_outcome(
        120, BALANCED_RATES, passing, passing) is stats.Outcome.A_VERIFIER_SPECIFIC_GO
    assert stats.decide_outcome(
        120, BALANCED_RATES, passing, failing) is stats.Outcome.B_SELF_REVISION_ONLY
    assert stats.decide_outcome(
        120, BALANCED_RATES, failing, passing) is stats.Outcome.C_NO_REPAIR_GAIN
    assert stats.decide_outcome(
        102, BALANCED_RATES, passing, passing) is stats.Outcome.D_INCONCLUSIVE_BY_DATA
    # the data guard is checked before any gate
    assert stats.decide_outcome(
        0, BALANCED_RATES, passing, passing) is stats.Outcome.D_INCONCLUSIVE_BY_DATA


def test_verifier_specific_go_needs_both_deltas_above_threshold() -> None:
    strong = stats.Gate(delta=0.10, mcnemar_p=0.01, ci_lower=0.02, n_favor=12, n_against=2)
    weak_cb = stats.Gate(delta=0.049, mcnemar_p=0.01, ci_lower=0.001, n_favor=10, n_against=4)
    assert stats.decide_outcome(
        110, BALANCED_RATES, strong, weak_cb) is stats.Outcome.B_SELF_REVISION_ONLY
    tight_ca = stats.Gate(delta=0.079, mcnemar_p=0.01, ci_lower=0.001, n_favor=10, n_against=1)
    assert stats.decide_outcome(
        110, BALANCED_RATES, tight_ca, strong) is stats.Outcome.C_NO_REPAIR_GAIN


def test_differential_censoring_guard_precedes_the_gates() -> None:
    passed = stats.Gate(delta=0.10, mcnemar_p=0.01, ci_lower=0.02, n_favor=12, n_against=2)
    unbalanced = {"A_FRESH_RETRY": 0.01, "B_SELF_REVISION": 0.02, "C_VERIFIER_REPAIR": 0.03,
                  "D_MISMATCHED_DIAGNOSTIC": 0.09}
    assert stats.censoring_range(unbalanced) == pytest.approx(0.08)
    assert stats.decide_outcome(
        128, unbalanced, passed, passed
    ) is stats.Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING
    # the range is inclusive of the frozen ceiling: 0.05 exactly is not a guard failure
    assert stats.decide_outcome(
        128, {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.05}, passed, passed
    ) is stats.Outcome.A_VERIFIER_SPECIFIC_GO


def test_mechanism_label_is_separate_from_the_classification() -> None:
    go = stats.Outcome.A_VERIFIER_SPECIFIC_GO
    specific = stats.Gate(delta=0.09, mcnemar_p=0.01, ci_lower=0.03, n_favor=11, n_against=2)
    nonspecific = stats.Gate(delta=0.00, mcnemar_p=0.60, ci_lower=-0.04, n_favor=6, n_against=6)
    assert stats.mechanism_label(go, specific) is stats.Mechanism.DIAGNOSTIC_SPECIFIC
    assert stats.mechanism_label(go, nonspecific) is stats.Mechanism.DIAGNOSTIC_NONSPECIFIC
    # zero effect with a positive lower bound cannot be "specific" -- only C > D directionally
    borderline = stats.Gate(delta=0.0, mcnemar_p=1.0, ci_lower=0.0, n_favor=0, n_against=0)
    assert stats.mechanism_label(go, borderline) is stats.Mechanism.DIAGNOSTIC_NONSPECIFIC
    for outcome in (stats.Outcome.D_INCONCLUSIVE_BY_DATA,
                    stats.Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING):
        assert stats.mechanism_label(outcome, specific) is stats.Mechanism.NOT_EVALUABLE


def test_infra_as_failure_sensitivity_never_redefines_the_endpoint() -> None:
    a = [True, False, None, True]
    c = [True, True, True, None]
    gate, completed = stats.paired_gate_infra_as_failure(a, c)
    assert completed == 2                      # the reading had to complete two censored arms
    assert gate.n_favor == 2                   # None -> False, so both censored pairs went to C
    primary = stats.paired_gate([x for x in a if x is not None], [y for y in c if y is not None])
    assert primary.n_favor == 1 and primary.n_against == 0


def test_min_discordant_helper_matches_the_exact_tail() -> None:
    # P(X>=5 | n=5) = 1/32 = 0.03125 <= 0.05, and with one opposing pair 7 of 8 is the first
    # combination that clears alpha; both are also reported in the frozen power artifact.
    assert stats.min_discordant_for_p(0) == 5
    assert stats.min_discordant_for_p(1) == 7
    assert stats.min_discordant_for_p(0, alpha=0.01) == 7


sys.path.insert(0, str(ROOT / "scripts"))

from v4_p001_verifier_plan import (
    CEILING_ROUNDING,
    V4_MAX_ARMS_PER_THEOREM,
    V4_WORST_CASE_VERIFICATIONS,
    poisson_cdf,
    poisson_upper_quantile,
    read_executions,
)


def test_poisson_helpers_match_hand_computable_values() -> None:
    assert poisson_cdf(0.0, 0) == pytest.approx(1.0)
    # P(X <= 1 | lambda = 1) = 2 / e
    assert poisson_cdf(1.0, 1) == pytest.approx(2.0 / 2.718281828459045, rel=1e-12)
    # P(X <= 3 | lambda = 3) = 13 / e^3
    assert poisson_cdf(3.0, 3) == pytest.approx(13.0 / 20.085536923187668, rel=1e-12)
    assert poisson_cdf(2.0, 0) == pytest.approx(0.1353352832366127, rel=1e-12)


def test_poisson_upper_quantile_is_the_smallest_k_crossing_the_level() -> None:
    for lam in (1.0, 5.0, 47.0, 136.0):
        k = poisson_upper_quantile(lam, 0.999)
        assert poisson_cdf(lam, k) >= 0.999
        assert poisson_cdf(lam, k - 1) < 0.999


def test_frozen_ceiling_clears_the_worst_observed_density_projection() -> None:
    executions = read_executions()
    assert len(executions) == 3  # deduplicated by run_id: two aborted, one complete
    candidates = sum(e["candidates_generated"] for e in executions)
    recoveries = sum(e["recoveries_attempted"] for e in executions)
    assert (candidates, recoveries) == (1024, 47)
    worst = max(executions, key=lambda e: e["recoveries_attempted"] / e["candidates_generated"])
    bound = poisson_upper_quantile(worst["recoveries_attempted"] / worst["candidates_generated"] * V4_WORST_CASE_VERIFICATIONS, 0.999)
    assert bound == 173
    assert CEILING_ROUNDING > bound
    # a per-theorem cap below the arm count would censor a theorem that is merely unlucky
    assert V4_MAX_ARMS_PER_THEOREM == 3
    assert V4_MAX_ARMS_PER_THEOREM < CEILING_ROUNDING < V4_WORST_CASE_VERIFICATIONS



# --- frozen design artifacts (Amendment A §16) ---------------------------------------------------

V4_MANIFESTS = ROOT / "experiments" / "manifests" / "v4"
V3_MANIFESTS = ROOT / "experiments" / "manifests" / "v3"
PROMPTSET_PARQUET = (ROOT / "data" / "processed" / "p3_promptset" / "prompt_sets" / "AI-MO"
                     / "Kimina-Prover-Promptset" / "train.parquet")
MODEL_DIR = ROOT / "models" / "weights" / "kimina_distill_0_6b"


def v4_manifest(name: str) -> dict:
    return json.loads((V4_MANIFESTS / name).read_text(encoding="utf-8"))


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_the_frozen_arm_schedule_is_position_and_predecessor_balanced() -> None:
    artifact = v4_manifest("v4_p001_arm_schedule.json")
    rows = v4_schedule.schedule()
    assert rows == artifact["schedule"]
    assert v4_schedule.schedule_hash() == artifact["schedule_hash"]
    assert artifact["schedule_canonical_sha256_cross_check"] == artifact["schedule_hash"]
    assert all(artifact["checks"].values())
    positions = Counter((arm, index) for row in rows
                        for index, arm in enumerate(row["order"], start=1))
    assert len(positions) == 16 and set(positions.values()) == {32}
    predecessors = Counter((first, second) for row in rows
                           for first, second in zip(row["order"], row["order"][1:]))
    assert len(predecessors) == 12
    assert set(predecessors.values()) <= {30, 31, 32, 33}
    assert artifact["balance"] == v4_schedule.balance_report()
    # the execution order is a function of the formal rank alone: there is no second parameter
    # through which a source, an error category or an outcome could reach it
    assert tuple(inspect.signature(v4_schedule.arm_order).parameters) == ("formal_rank",)
    assert [v4_schedule.arm_order(rank) for rank in range(1, 129)] == [
        row["order"] for row in artifact["schedule"]]


def derangement_fixture(multi: int = 3) -> list[dict]:
    """Two multi-member categories plus one singleton per remaining class, by construction.

    Token counts put the multi-member categories in bucket 1 (64 < tokens <= 128) and the two
    singletons in buckets 2 (130) and 3 (300), so the fallback's nearest-bucket preference is
    hand-computable.
    """
    items = []
    for category, base_tokens, step in (("unsolved_goals", 70, 10), ("tactic_failure", 70, 20)):
        for index in range(multi):
            statement_id = f"{category}-{index}"
            items.append({
                "statement_id": statement_id, "category": category,
                "diagnostic_sha256": hashlib.sha256(statement_id.encode("utf-8")).hexdigest(),
                "diagnostic_tokens": base_tokens + step * index,
            })
    for statement_id, category, tokens in (
            ("singleton-typeclass", "typeclass_synthesis", 130),
            ("singleton-other", "other_semantic_lean_failure", 300)):
        items.append({
            "statement_id": statement_id, "category": category,
            "diagnostic_sha256": hashlib.sha256(statement_id.encode("utf-8")).hexdigest(),
            "diagnostic_tokens": tokens,
        })
    return items


def test_the_derangement_never_hands_a_theorem_its_own_diagnostic() -> None:
    cohort = derangement_fixture(multi=32)
    result = v4_derange.derange(cohort)
    assert result["n"] == len(cohort) == 66
    assert all(row["recipient"] != row["donor"] for row in result["mapping"])
    assert len({row["recipient"] for row in result["mapping"]}) == len(cohort)
    assert result["checks"] == {"no_self_diagnostic": True,
                                "every_recipient_assigned_exactly_once": True,
                                "all_recipients_distinct": True}
    by_id = {item["statement_id"]: item for item in cohort}
    # a donor always contributes its own real diagnostic, never a copy of the recipient's
    assert all(row["donor_diagnostic_sha256"] == by_id[row["donor"]]["diagnostic_sha256"]
               for row in result["mapping"])
    # the 64 multi-member theorems keep their category; the two singletons cannot
    assert result["same_error_category_match_rate"] == round(64 / 66, 4)
    assert result["diagnostic_token_length_difference"]["n"] == 66


def test_the_derangement_is_deterministic_and_independent_of_input_order() -> None:
    items = derangement_fixture(multi=8)
    first = v4_derange.derange(items)
    second = v4_derange.derange(list(reversed(items)))
    assert first == second
    assert first["mapping_sha256"] == second["mapping_sha256"]
    assert first["mapping_sha256"] == hashlib.sha256(canonical_json(
        {"version": v4_derange.DERANGEMENT_VERSION, "mapping": first["mapping"]}).encode(
            "utf-8")).hexdigest()


def test_a_singleton_category_falls_back_to_the_nearest_token_length_bucket() -> None:
    items = derangement_fixture()
    result = v4_derange.derange(items)
    assert result["n"] == 8
    assert result["fallback_count"] == 2
    assert sorted(result["fallback_recipients"]) == ["singleton-other", "singleton-typeclass"]
    rows = {row["recipient"]: row for row in result["mapping"]}
    assert rows["singleton-other"]["fallback"] and rows["singleton-typeclass"]["fallback"]
    # the bucket-3 singleton must take the bucket-2 one: distance 1 beats the multi members' 2, and
    # no tie-break is needed to see it
    assert v4_derange.token_bucket(300) == 3 and v4_derange.token_bucket(130) == 2
    assert v4_derange.token_bucket(120) == 1
    assert rows["singleton-other"]["donor"] == "singleton-typeclass"
    assert rows["singleton-other"]["token_difference"] == 170
    # the bucket-2 singleton prefers distance 1 too (a bucket-1 member or the bucket-3 singleton)
    assert abs(v4_derange.token_bucket(rows["singleton-typeclass"]["donor_tokens"])
               - v4_derange.token_bucket(130)) == 1
    # every multi-member theorem is served by a same-category donor, cyclically shifted
    for row in result["mapping"]:
        if row["recipient_category"] in {"unsolved_goals", "tactic_failure"}:
            assert row["same_category"] is True and row["fallback"] is False
    assert result["same_error_category_match_rate"] == round(6 / 8, 4)
    # the mapping is a permutation: donor reuse happens only through the fallback and is reported
    donors = Counter(row["donor"] for row in result["mapping"])
    assert result["donor_reuse_count"] == sum(1 for count in donors.values() if count > 1)


def test_the_second_stage_seed_stream_is_a_function_of_the_rank_alone() -> None:
    artifact = v4_manifest("v4_p001_seeds.json")
    assert len(artifact["checks"]) == 10 and all(artifact["checks"].values())
    assert tuple(inspect.signature(seeds.second_stage_seed).parameters) == ("formal_rank",)
    paired = [seeds.second_stage_seed(rank) for rank in range(1, artifact["n_primary"] + 1)]
    assert artifact["paired_seeds_by_rank"] == paired
    assert len(set(paired)) == artifact["n_primary"]
    assert artifact["paired_seed_hash"] == hashlib.sha256(
        canonical_json({"n": artifact["n_primary"], "seeds": paired}).encode("utf-8")).hexdigest()
    # the screening stream has its own recipe and 640 entries; the two streams cannot collide
    screening = [row["seed"] for row in artifact["screening_schedule"]]
    assert screening == [seeds.first_stage_seed(rank)
                         for rank in range(1, artifact["max_screening"] + 1)]
    assert artifact["screening_seed_hash"] == hashlib.sha256(
        canonical_json(screening).encode("utf-8")).hexdigest()
    assert len(set(screening)) == artifact["max_screening"]
    assert max(screening) < min(paired)
    assert artifact["formulas"]["second_stage"].startswith("V4_BASE_SEED + SECOND_STAGE_OFFSET")


def test_arm_a_reproduces_the_canonical_prompt_on_every_audited_example() -> None:
    artifact = v4_manifest("v4_p001_prompt_provenance.json")
    assert all(artifact["checks"].values())
    assert artifact["renderer_version"] == prompts.RENDERER_VERSION
    reproduction = artifact["arm_a_canonical_reproduction"]
    assert reproduction["target"] == "100% on every audited historical example"
    archive = reproduction["v3_r001_archive"]
    assert archive["n_theorems"] == 128 and set(archive["checks"].values()) == {128}
    dumps = reproduction["v1_rollout_dumps"]
    assert dumps["n_distinct_inputs"] == 612
    assert dumps["n_matched_to_a_pinned_theorem"] == dumps["n_byte_identical_to_arm_a"] == 612
    holdout = reproduction["e023_holdout"]
    assert holdout["n_records"] == holdout["n_prompt_tokens_matched"] == 512
    assert holdout["n_statement_not_in_parquet"] == 0
    assert reproduction["pool_size"] == 1371
    assert reproduction["pool_members_with_arm_a_ids_equal_to_canonical"] == 1371


def test_arm_a_renders_the_pinned_promptset_the_way_the_chat_template_does() -> None:
    """The live check behind the provenance artifact, over a deterministic sample of the pool."""
    if not PROMPTSET_PARQUET.exists() or not MODEL_DIR.exists():
        pytest.skip("the pinned promptset parquet / tokenizer are not present on this node")
    import pandas as pd
    from transformers import AutoTokenizer

    frame = pd.read_parquet(PROMPTSET_PARQUET, columns=["statement_id", "prompt"])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True,
                                              local_files_only=True)
    step = max(1, len(frame) // 24)
    for row in frame.iloc[::step].head(24).itertuples(index=False):
        raw = row.prompt.tolist() if hasattr(row.prompt, "tolist") else list(row.prompt)
        messages = prompts.canonical_messages(raw)
        rendered = prompts.render_prompt(tokenizer, messages)
        canonical_ids = tokenizer.apply_chat_template([dict(m) for m in messages],
                                                      add_generation_prompt=True, tokenize=True)
        if isinstance(canonical_ids, dict):
            canonical_ids = canonical_ids["input_ids"]
        canonical_ids = list(canonical_ids)
        if canonical_ids and isinstance(canonical_ids[0], list):
            canonical_ids = canonical_ids[0]
        assert tokenizer(rendered, add_special_tokens=False)["input_ids"] == canonical_ids, \
            str(row.statement_id)


def test_the_sealed_final_holdout_reserve_is_untouched_by_the_pool() -> None:
    artifact = v4_manifest("v4_p001_pool.json")
    # the "touched" entries are counts and must all be zero; every other check is a boolean
    assert all(value is True for name, value in artifact["checks"].items()
               if not name.endswith("_touched"))
    assert {value for name, value in artifact["checks"].items()
            if name.endswith("_touched")} == {0}
    for name in ("sealed_components_touched", "sealed_statements_touched",
                 "v3_formal_sample_components_touched", "v3_formal_sample_statements_touched"):
        assert artifact["checks"][name] == 0
    reserve = json.loads((V3_MANIFESTS / "v3_final_holdout_reserve.json").read_text(
        encoding="utf-8"))
    assert reserve["status"] == "SEALED" and reserve["n_components"] == 93
    components = reserve["components"]
    assert len(components) == reserve["n_components"]
    sealed_components = {member["component_id"] for member in components}
    sealed_statements = {member["statement_id"] for member in components}
    assert len(sealed_components) == len(sealed_statements) == 93
    members = artifact["members"]
    assert not (sealed_components & {member["component_id"] for member in members})
    assert not (sealed_statements & {member["statement_id"] for member in members})


def test_the_pool_order_is_the_frozen_outcome_free_hash_permutation() -> None:
    artifact = v4_manifest("v4_p001_pool.json")
    members = artifact["members"]
    assert [member["screening_rank"] for member in members] == list(range(1, len(members) + 1))
    assert [member["tier"] for member in members] == sorted(member["tier"] for member in members)
    class_rank = {name: index
                  for index, name in enumerate(artifact["tier_rule"]["tier2_classes_in_order"])}

    def key(member: dict) -> tuple:
        classes = [name for name in member["classes"] if name != "v1_used"]
        block = (0 if member["tier"] == "tier1" else 1, 1 + class_rank[classes[0]] if classes else 0)
        return (*block, member["screening_order_key"])

    for member in members:
        recipe = hashlib.sha256(
            f"{artifact['base_seed']}|{member['statement_id']}".encode()).hexdigest()
        assert member["screening_order_key"] == recipe
    keys = [key(member) for member in members]
    assert keys == sorted(keys)
    assert len({member["screening_order_key"] for member in members}) == len(members)
