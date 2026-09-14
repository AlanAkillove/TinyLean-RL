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

