"""帳戶推播（user data stream）：用假的 websocket 和假時鐘，不連網、不真的睡。"""

import json
import sys

import pandas as pd
import pytest

from grid_trading.trader import stream as smod
from grid_trading.trader.binance import BinanceError, BinanceFutures
from grid_trading.trader.bot import Config, LiveBroker, Notifier, PaperBroker, Trader
from grid_trading.trader.stream import UserStream, url_templates

KEY = "SECRETLISTENKEY0123456789abcdef"
ORDER = json.dumps({"e": "ORDER_TRADE_UPDATE", "o": {"s": "XUSDT"}})
ALGO = json.dumps({"e": "ALGO_UPDATE"})
LITE = json.dumps({"e": "TRADE_LITE"})
EXPIRED = json.dumps({"e": "listenKeyExpired"})


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeClient:
    def __init__(self, keys=(KEY,)):
        self.keys = list(keys)
        self.posts = 0
        self.puts = 0
        self.put_error = None
        self.post_error = None

    def new_listen_key(self):
        self.posts += 1
        if self.post_error:
            raise self.post_error
        return self.keys[min(self.posts - 1, len(self.keys) - 1)]

    def keepalive_listen_key(self):
        self.puts += 1
        if self.put_error:
            raise self.put_error
        return None


class FakeWS:
    """script 裡：字串 = 收到的訊息，None = recv 逾時、"" = pong（兩者時鐘都走 step 秒），例外 = 丟出去。用完就當作斷線。"""

    def __init__(self, clock, script, step=10.0):
        self.clock, self.script, self.step = clock, list(script), step
        self.closed = self.pings = 0

    def recv(self):
        if not self.script:
            raise ConnectionError("script 用完")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if item is None or item == "":
            self.clock.t += self.step
        return item

    def ping(self):
        self.pings += 1

    def close(self):
        self.closed += 1


def make(templates=("u0/{listenKey}", "u1/{listenKey}"), scripts=(), client=None, fail=()):
    """fail：第幾次連線（從 0 起）直接連不上。"""
    clock = Clock()
    logs, urls, sockets = [], [], []
    scripts = list(scripts)

    def connect(url):
        n = len(urls)
        urls.append(url)
        if n in fail:
            raise ConnectionRefusedError("refused " + url)
        ws = FakeWS(clock, scripts.pop(0) if scripts else [])
        sockets.append(ws)
        return ws

    st = UserStream(client or FakeClient(), list(templates), connect=connect, clock=clock,
                    sleep=lambda s: setattr(clock, "t", clock.t + s), log=logs.append)
    return st, clock, logs, urls, sockets


def run_cycles(st, n):
    """跑 run() 直到 n 個連線週期結束（用假的 sleep 計數後停止）。"""
    count = [0]
    orig = st._cycle

    def cycle():
        count[0] += 1
        out = orig()
        if count[0] >= n:
            st._stop.set()
        return out
    st._cycle = cycle
    st.run()


# ---------- 事件、證明、健康 ----------
def test_event_sets_wake_and_proves_template():
    st, clock, logs, urls, _ = make(scripts=[[None, ORDER, None]])
    seen = []
    orig = st._handle
    st._handle = lambda m, i: (orig(m, i), seen.append((st.wake.is_set(), st.state)))[0]
    assert st.state == "down" and not st.healthy
    st._cycle()
    assert seen == [(True, "healthy")]
    assert st.last_event is not None and 0 in st.proven
    assert st.state == "down" and not st.healthy   # 斷線後就不健康，即使證明過
    assert urls == [f"u0/{KEY}"]


def test_connected_without_proving_event_is_not_healthy():
    states = []
    st, clock, logs, urls, _ = make(scripts=[[None, LITE, json.dumps({"result": None, "id": 1}), None]])
    orig = st._handle
    st._handle = lambda m, i: (orig(m, i), states.append(st.state))[0]
    st._cycle()
    assert states == ["connected", "connected"]   # TRADE_LITE 叫醒但不算證明；訂閱回應不算事件
    assert st.wake.is_set() and not st.proven


def test_combined_stream_payload_is_unwrapped():
    st, *_ = make(scripts=[[json.dumps({"stream": "x", "data": {"e": "ACCOUNT_UPDATE"}})]])
    st._cycle()
    assert st.wake.is_set() and st.proven == {0}


# ---------- 範本輪替 ----------
def test_failed_connect_rotates_template_with_backoff():
    st, clock, logs, urls, _ = make(fail={0, 1, 2})
    sleeps = []
    st._sleep = sleeps.append
    run_cycles(st, 4)
    assert urls == [f"u0/{KEY}", f"u1/{KEY}", f"u0/{KEY}", f"u1/{KEY}"]
    assert sleeps[:3] == [1.0, 2.0, 4.0]
    assert all(s <= smod.BACKOFF_MAX for s in sleeps)


def test_quick_close_rotates_long_connection_does_not():
    quick = [None, ConnectionError("closed")]          # 10 秒就斷
    long_ = [""] * 7 + [ConnectionError("closed")]   # 70 秒後才斷
    st, clock, logs, urls, _ = make(scripts=[quick, long_, long_])
    run_cycles(st, 3)
    assert [u[:2] for u in urls] == ["u0", "u1", "u1"]


def test_proven_template_survives_short_outage_and_unproven_returns_to_proven():
    st, clock, logs, urls, _ = make(scripts=[[ORDER, ConnectionError("x")]], fail={1})
    run_cycles(st, 2)
    assert st.index == 0   # 證明過的範本失敗一次不換
    # 連在沒證明過的範本、但另一個證明過：一段時間後換回去
    st2, clock2, _, urls2, _ = make(scripts=[[""] * 100])
    st2.proven = {0}
    st2.index = 1
    assert st2._cycle() == "planned" and st2.index == 0


def test_backoff_capped():
    st, clock, logs, urls, _ = make(fail=set(range(20)))
    sleeps = []
    st._sleep = sleeps.append
    run_cycles(st, 12)
    assert max(sleeps) == smod.BACKOFF_MAX


# ---------- listenKey ----------
def test_listen_key_expired_gets_new_key():
    client = FakeClient(keys=(KEY, KEY + "2"))
    st, clock, logs, urls, _ = make(scripts=[[EXPIRED], []], client=client)
    run_cycles(st, 2)
    assert client.posts == 2 and urls[1] == f"u0/{KEY}2"
    assert not st.wake.is_set()   # listenKeyExpired 不叫醒主迴圈


def test_keepalive_every_30_minutes():
    client = FakeClient()
    st, clock, logs, urls, _ = make(scripts=[[""] * 400], client=client)   # 4000 秒
    st._cycle()
    assert client.posts == 1 and client.puts == 2   # 1800 秒、3600 秒各一次


def test_keepalive_failure_retries_then_unknown_key_reconnects():
    client = FakeClient()
    client.put_error = RuntimeError("斷網")
    st, clock, logs, urls, _ = make(scripts=[[""] * 186], client=client)  # 1860 秒
    st._cycle()
    assert client.puts == 2   # 1800 秒失敗，60 秒後再試，連線沒斷
    client.put_error = BinanceError(400, -1125, "This listenKey does not exist.")
    st2, *_ = make(scripts=[[""] * 200], client=client)
    assert st2._cycle() == "planned" and st2._key is None


def test_reconnect_before_24h():
    st, clock, *_ = make(scripts=[[""] * 9000], client=FakeClient())
    st._sleep = lambda s: None
    clock_start = clock.t
    assert st._cycle() == "planned"
    assert clock.t - clock_start == pytest.approx(smod.MAX_AGE, abs=20)


def test_silent_connection_is_dropped():
    st, clock, *_ = make(scripts=[[None] * 100])
    st._handle = lambda m, i: None
    assert st._cycle() == "dropped" and "沒有收到" in st.last_error


def test_key_never_logged():
    st, clock, logs, urls, _ = make(fail={0}, scripts=[[json.dumps({"code": 1, "msg": KEY}), ORDER,
                                                         ConnectionError("bad " + KEY)]])
    run_cycles(st, 2)
    assert logs and all(KEY not in line for line in logs)
    assert KEY not in st.last_error and any("***" in line for line in logs)


def test_listen_key_rest_respects_ban_and_uses_own_session(monkeypatch):
    calls = []

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"listenKey": KEY}

    class Sess:
        def __init__(self, name):
            self.name = name

        def request(self, method, url, headers=None, **k):
            calls.append((self.name, method, url, headers))
            return Resp()

    cli = BinanceFutures(key="k", secret="s", session=Sess("main"), stream_session=Sess("stream"))
    assert cli.new_listen_key() == KEY
    cli.keepalive_listen_key()
    assert [(c[0], c[1]) for c in calls] == [("stream", "POST"), ("stream", "PUT")]
    assert calls[0][3] == {"X-MBX-APIKEY": "k"} and "signature" not in calls[0][2]
    import time
    cli.banned_until = time.time() + 600
    with pytest.raises(BinanceError):
        cli.new_listen_key()
    assert len(calls) == 2   # 被封期間沒送


def test_run_never_raises_when_client_breaks():
    client = FakeClient()
    client.post_error = BinanceError(418, -1003, "banned")
    st, clock, logs, urls, _ = make(client=client)
    run_cycles(st, 3)
    assert urls == [] and st.state == "down" and len(logs) == 1   # 同樣的原因只印一次


def test_missing_websocket_module_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "websocket", None)
    logs = []
    st = UserStream(FakeClient(), ["u/{listenKey}"], log=logs.append)
    assert st.start() is False
    assert st.state == "off" and not st.healthy and len(logs) == 1 and "websocket-client" in logs[0]


def test_url_templates_env_override():
    assert url_templates(False, env="")[0].startswith("wss://fstream.binance.com/private/ws?listenKey={listenKey}")
    assert url_templates(True, env="")[1] == "wss://fstream.binancefuture.com/ws/{listenKey}"
    logs = []
    assert url_templates(False, env=" a/{listenKey} , bad ,b/{listenKey}", log=logs.append) == \
        ["a/{listenKey}", "b/{listenKey}"]
    assert len(logs) == 1


# ---------- 主迴圈 ----------
class StubStream:
    def __init__(self, healthy=False, state=None):
        import threading
        self.wake = threading.Event()
        self.enabled = True
        self._healthy = healthy
        self._state = state
        self.last_event = None
        self.last_error = "斷線"

    @property
    def healthy(self):
        return self._healthy

    @property
    def state(self):
        return self._state or ("healthy" if self._healthy else "down")

    def start(self):
        return True

    def stop(self):
        pass


class FakeVenue:
    banned_until = 0.0

    def open_conditionals(self):
        return []

    def positions(self):
        return {}


def loop_trader(tmp_path, stream, t0):
    sink = []
    cfg = Config(mode="testnet", state_dir=tmp_path, coins=[], scan_delay=15.0)
    v = FakeVenue()
    tr = Trader(cfg, v, LiveBroker(v, v, cfg), Notifier("t", sink=sink), {}, stream=stream)
    clock = Clock(t0)
    waits = []

    def wait(ev, t):
        waits.append(t)
        if ev.is_set():
            return True
        clock.t += t
        return False
    tr._clock, tr._wait = clock, wait
    tr._sleep = lambda s: setattr(clock, "t", clock.t + s)
    return tr, clock, sink, waits


H2 = 7200.0
BASE = 1_800_000_000.0 - 1_800_000_000.0 % H2 + 1000.0   # 2H 收盤後 1000 秒


def test_idle_uses_short_poll_when_unhealthy_and_long_when_healthy(tmp_path):
    for healthy, expect in ((False, 60.0), (True, 300.0)):
        tr, clock, *_ = loop_trader(tmp_path, StubStream(healthy), BASE)
        assert tr.idle(60.0) == "poll" and clock.t - BASE == pytest.approx(expect)
    tr, clock, *_ = loop_trader(tmp_path, None, BASE)   # 沒有推播（紙上、--no-stream）照舊
    assert tr.idle(60.0) == "poll" and clock.t - BASE == pytest.approx(60.0)


def test_idle_ends_at_next_scan_time(tmp_path):
    t0 = BASE - 1000 + H2 - 100      # 下一根收盤前 100 秒；掃描在收盤後 15 秒
    tr, clock, *_ = loop_trader(tmp_path, StubStream(True), t0)
    tr.idle(60.0)
    assert clock.t == pytest.approx(t0 + 115)


def test_idle_switches_to_short_poll_when_stream_drops(tmp_path):
    s = StubStream(True)
    tr, clock, _, waits = loop_trader(tmp_path, s, BASE)
    orig = tr._wait

    def wait(ev, t):
        if clock.t - BASE >= 100:
            s._healthy = False
        return orig(ev, t)
    tr._wait = wait
    tr.idle(60.0)
    assert clock.t - BASE == pytest.approx(100, abs=tr_check())


def tr_check():
    from grid_trading.trader.bot import HEALTH_CHECK
    return HEALTH_CHECK


def test_wake_ends_wait_early_with_debounce_and_storm_guard(tmp_path):
    s = StubStream(True)
    tr, clock, _, waits = loop_trader(tmp_path, s, BASE)
    tr._last_pass = BASE
    s.wake.set()
    assert tr.idle(60.0) == "event"
    assert clock.t - BASE == pytest.approx(3.0)     # 離上一輪不到 3 秒：等滿 3 秒
    assert not s.wake.is_set()
    tr._last_pass = clock.t - 100
    start = clock.t
    s.wake.set()
    tr.idle(60.0)
    assert clock.t - start == pytest.approx(1.0)     # 一般情況只等 1 秒合併事件


def test_heartbeat_and_notifications_follow_stream_state(tmp_path):
    s = StubStream(False, state="connected")
    tr, clock, sink, _ = loop_trader(tmp_path, s, BASE)
    now = pd.Timestamp(BASE, unit="s", tz="UTC")
    tr.run_once(now)
    hb = json.loads((tmp_path / "testnet.heartbeat.json").read_text())
    assert hb["stream"] == "connected" and not any("即時推送" in m for m in sink)
    s._healthy, s._state = True, None
    for _ in range(3):
        tr.run_once(now)
    assert sum("即時推送已啟用" in m for m in sink) == 1
    assert json.loads((tmp_path / "testnet.heartbeat.json").read_text())["stream"] == "healthy"
    # 斷線又馬上連回來（重連迴圈）：不推
    s._healthy = False
    clock.t += 120
    tr.run_once(now)
    s._healthy = True
    clock.t += 60
    tr.run_once(now)
    assert not any("警報" in m for m in sink)
    # 斷超過 10 分鐘：推一次；恢復推一次
    s._healthy = False
    for dt in (0, 300, 700, 900):
        clock.t += dt
        tr.run_once(now)
    alerts = [m for m in sink if "警報" in m]
    assert len(alerts) == 1 and "即時推送" in alerts[0]
    s._healthy = True
    tr.run_once(now)
    assert sum("已恢復" in m for m in sink) == 1


def test_paper_trader_has_no_stream(tmp_path):
    cfg = Config(mode="paper", state_dir=tmp_path, coins=[])
    tr = Trader(cfg, None, PaperBroker(None, cfg), Notifier("p", sink=[]), {})
    assert tr.stream_state() == "off"


def test_run_starts_and_stops_stream(tmp_path):
    s = StubStream(True)
    calls = []
    s.start, s.stop = lambda: calls.append("start"), lambda: calls.append("stop")
    tr, clock, *_ = loop_trader(tmp_path, s, BASE)

    def boom(poll):
        raise KeyboardInterrupt
    tr.idle = boom
    with pytest.raises(KeyboardInterrupt):
        tr.run()
    assert calls == ["start", "stop"]
    calls.clear()
    tr.run(once=True)   # 單次模式不開推播
    assert calls == []


def test_real_thread_wakes_main_loop_and_stops():
    import threading
    gate = threading.Event()

    class BlockingWS:
        def __init__(self):
            self.sent = False

        def recv(self):
            if not self.sent:
                self.sent = True
                return ORDER
            gate.wait(0.05)
            return None

        def ping(self):
            pass

        def close(self):
            gate.set()

    st = UserStream(FakeClient(), ["u/{listenKey}"], connect=lambda url: BlockingWS(), log=lambda m: None)
    assert st.start()
    assert st.wake.wait(2.0) and st.healthy
    st.stop()
    st._thread.join(2.0)
    assert not st._thread.is_alive() and not st.healthy


def test_conn_adapter_maps_frames():
    import websocket

    class WS:
        def __init__(self, frames):
            self.frames = frames

        def recv_data(self, control_frame=False):
            f = self.frames.pop(0)
            if isinstance(f, Exception):
                raise f
            return f
    A = websocket.ABNF
    c = smod._Conn(WS([websocket.WebSocketTimeoutException("t"), (A.OPCODE_PING, b""),
                       (A.OPCODE_TEXT, ORDER.encode()), (A.OPCODE_CLOSE, (1008).to_bytes(2, "big"))]), websocket)
    assert c.recv() is None and c.recv() == "" and c.recv() == ORDER
    with pytest.raises(ConnectionError, match="1008"):
        c.recv()
