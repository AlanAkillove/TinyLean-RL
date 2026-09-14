import pytest

from tinylean_rl.evaluation.metrics import group_rates, pass_at_k


def test_pass_at_k_extremes_and_monotonicity():
    assert pass_at_k(0, 8, 1) == 0.0
    assert pass_at_k(8, 8, 8) == 1.0
    assert pass_at_k(1, 8, 8) == 1.0
    assert pass_at_k(2, 8, 1) < pass_at_k(2, 8, 4)


def test_group_rates():
    assert group_rates([0, 2, 4, 0], 4) == {"all_zero": 0.5, "mixed": 0.25, "all_one": 0.25}


def test_pass_at_k_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        pass_at_k(9, 8, 1)
    with pytest.raises(ValueError):
        pass_at_k(1, 8, 0)

