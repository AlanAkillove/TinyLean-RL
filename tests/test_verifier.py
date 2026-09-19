import pytest

from tinylean_rl.verifier import kimina


class DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return [{"custom_id": "a", "status": "valid"}]


def test_verify_codes_builds_kimina_payload(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return DummyResponse()

    monkeypatch.setattr(kimina.httpx, "post", fake_post)
    result = kimina.verify_codes(["proof"], custom_ids=["a"], base_url="http://server")
    assert result["results"][0]["status"] == "valid"
    assert captured["url"] == "http://server/verify"
    assert captured["json"]["codes"] == [{"custom_id": "a", "proof": "proof"}]


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"error": None, "response": {"messages": [], "sorries": []}}, "valid"),
        ({"response": {}}, "valid"),
        ({"response": {"messages": [{"severity": "error", "data": "unknown identifier"}]}}, "lean_error"),
        ({"response": {"messages": [], "sorries": [{"pos": {"line": 1, "column": 1}}]}}, "sorry"),
        ({"error": "Timed out while executing the snippet"}, "timeout_error"),
        ({"error": "connection reset by peer"}, "server_error"),
        ({"response": {"message": "REPL crashed"}}, "repl_error"),
        ({"status": "valid"}, "valid"),
        ({"is_valid": True}, "valid"),
        ({"is_valid": False}, "unknown"),
        ({"custom_id": "a"}, "unknown"),
        (None, "unknown"),
    ],
)
def test_result_lean_status(item, expected):
    assert kimina.result_lean_status(item) == expected


def test_result_is_valid_matches_official_semantics():
    assert kimina.result_is_valid({"response": {"messages": [], "sorries": []}})
    assert not kimina.result_is_valid({"response": {"messages": [], "sorries": [{"pos": {}}]}})
    assert not kimina.result_is_valid({"response": {"messages": [{"severity": "error"}]}})
    assert not kimina.result_is_valid({"error": "Connection error."})
    assert not kimina.result_is_valid({})



def test_verify_codes_sends_explicit_server_timeout(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return DummyResponse()

    monkeypatch.setattr(kimina.httpx, "post", fake_post)
    kimina.verify_codes(["proof"], custom_ids=["a"], base_url="http://server", server_timeout=30)
    assert captured["json"]["timeout"] == 30


def test_timeout_field_omitted_by_default_and_set_when_requested(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return DummyResponse()

    monkeypatch.setattr(kimina.httpx, "post", fake_post)
    kimina.verify_codes(["proof"], custom_ids=["a"], base_url="http://server")
    assert "timeout" not in captured["json"]
    kimina.verify_code("proof", custom_id="a", base_url="http://server", server_timeout=45)
    assert captured["json"]["timeout"] == 45
