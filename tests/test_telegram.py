from signal_telegram import send


def test_send_caps_messages(monkeypatch):
    sent = []
    monkeypatch.setattr("urllib.request.urlopen", lambda url, body, timeout: sent.append(body) or _Resp())
    monkeypatch.setattr("time.sleep", lambda s: None)
    ok, failed = send([f"m{i}" for i in range(40)], "TOKEN", "1")
    assert (ok, failed) == (30, 0)
    assert "%E9%82%84%E6%9C%89+11" in sent[-1].decode()  # 「還有 11」


class _Resp:
    def read(self):
        return b"{}"


def test_send_retries_after_telegram_rate_limit(monkeypatch):
    import io
    import urllib.error

    calls = []

    def fake(url, body, timeout):
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {},
                                         io.BytesIO(b'{"ok":false,"parameters":{"retry_after":3}}'))
        return _Resp()

    waits = []
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", waits.append)
    assert send(["m1", "m2"], "TOKEN", "1") == (2, 0)
    assert len(calls) == 3 and 4.0 in waits


def test_send_does_not_retry_bad_requests(monkeypatch):
    import io
    import urllib.error

    calls = []

    def fake(url, body, timeout):
        calls.append(body)
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert send(["m1"], "TOKEN", "1") == (0, 1)
    assert len(calls) == 1


def test_send_does_not_resend_after_timeout(monkeypatch):
    calls = []

    def fake(url, body, timeout):
        calls.append(body)
        raise TimeoutError("timed out")  # 訊息可能已送達，只是沒等到回應

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert send(["m1"], "TOKEN", "1") == (1, 0)
    assert len(calls) == 1

