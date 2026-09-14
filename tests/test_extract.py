from tinylean_rl.inference.extract import extract_proof


def test_extracts_lean_fence():
    assert extract_proof("Here is the proof:\n```lean4\nby norm_num\n```") == "by norm_num"


def test_keeps_complete_theorem():
    source = "example : 1 + 1 = 2 := by\n  norm_num"
    assert extract_proof(source) == source


def test_stops_at_extra_markdown_and_keeps_tactic():
    output = "norm_num\n\n```\nExplanation: this is extra text."
    assert extract_proof(output) == "norm_num"


def test_plain_markdown_fence_is_not_selected_as_lean_code():
    output = "  rfl\n\n```\nThis is an explanation.\n```"
    assert extract_proof(output) == "rfl"
