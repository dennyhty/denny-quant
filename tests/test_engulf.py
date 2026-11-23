"""日線吞沒策略 engulf_1d：訊號規則、限價進場的追蹤（tracker）、網站紀錄的狀態。"""

import numpy as np
import pandas as pd
import pytest

from grid_trading import indicators as ind
from grid_trading.signals import get, ledger
from grid_trading.signals.strategy import LONG, SHORT, Signal
from grid_trading.signals.tracker import track

DAY = pd.Timedelta(days=1)


def synth(seed, n=700):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.03, n) + 0.02 * np.sin(np.arange(n) / 9)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    t = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": lo, "close": c, "volume": 1.0})


def _idx(df, sig):
    return int((sig.time - df["time"].iloc[0]) / DAY) - 1  # 訊號時間 = 訊號 K 線的收盤時間


def test_matches_research_signals_orders_and_exits():
    """和 research/threads_strategies.py 的 engulf_signals + build_orders(half, atr, 3) + tp1 出場逐筆一致。"""
    T = pytest.importorskip("research.threads_strategies")
    strat = get("engulf_1d")
    total = 0
    for seed in range(8):
        df = synth(seed)
        d = df.copy()
        d["atr"], d["e8"], d["e20"] = ind.atr(d), ind.ema(d["close"], 8), ind.ema(d["close"], 20)
        sigs = T.engulf_signals(d, True, 5)
        orders = {o[5]: o for o in T.build_orders(d, sigs, "half", "atr", 3)}
        res = {int(x.sig): x for x in T.run(d, list(orders.values()), "1D").itertuples()}
        ours = strat.scan(df, "XUSDT", "1D")
        assert [_idx(df, s) for s in ours] == [i for i, _ in sigs]
        total += len(ours)
        for s in ours:
            i = _idx(df, s)
            tr = track(s, df, 0.0005)
            if i not in orders:  # 3 根內沒碰到掛單價
                assert tr.status in ("EXPIRED", "PENDING")
                continue
            j0, _, px, sp, _, _ = orders[i]
            assert s.price == pytest.approx(px) and s.sl == pytest.approx(sp)
            assert tr.filled_at == df["time"].iloc[j0 - 1]  # 成交那根
            if i in res:
                exit_bar = int((tr.events[-1].time - df["time"].iloc[0]) / DAY)
                assert exit_bar == int(res[i].j_tp1)
    assert total > 30  # 資料要真的有訊號，比對才有意義


def _frame(rows, start="2026-01-01"):
    """rows: (open, high, low, close)。"""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    df["time"] = pd.date_range(start, periods=len(df), freq="1D", tz="UTC")
    return df


def test_signal_rules_short_and_long_and_atr():
    strat = get("engulf_1d")
    n = 60
    # 先漲後跌：EMA8 剛下穿 EMA20，最後一根是看跌吞沒
    close = list(np.linspace(100, 140, 40)) + list(np.linspace(139, 131, n - 41))
    rows = [(c - 0.5, c + 1, c - 1, c) for c in close]
    rows = [(a, b, c_, d) for a, b, c_, d in rows]
    prev_o, prev_c = rows[-1][0], 131.0
    rows[-1] = (prev_o, prev_o + 1, prev_o - 1, prev_c)            # 前一根收紅
    rows.append((prev_c + 0.01, prev_c + 1, prev_c - 6, prev_o - 5))  # 開在前收附近、收在前一根開盤下方：吞沒
    df = _frame(rows)
    d = strat.prepare(df)
    assert d["e8"].iloc[-1] < d["e20"].iloc[-1]
    sigs = strat.scan(df, "XUSDT", "1D", start=len(df) - 1)
    assert len(sigs) == 1 and sigs[0].side == SHORT
    s = sigs[0]
    o, c = rows[-1][0], rows[-1][3]
    px = (o + c) / 2
    atr = float(d["atr"].iloc[-1])
    assert s.price == pytest.approx(px) and s.sl == pytest.approx(px + 1.5 * atr)
    assert s.tps == [pytest.approx(px - 1.5 * atr)]            # 1R
    assert s.exit["portions"] == [1.0] and s.exit["limit"] and s.exit["fill_bars"] == 3 and s.exit["max_bars"] == 60
    assert s.time == df["time"].iloc[-1] + DAY and s.strategy == "engulf_1d"
    # 鏡像做多
    mirror = _frame([(300 - o_, 300 - l_, 300 - h_, 300 - c_) for o_, h_, l_, c_ in rows])
    ms = strat.scan(mirror, "XUSDT", "1D", start=len(mirror) - 1)
    assert len(ms) == 1 and ms[0].side == LONG
    assert ms[0].price == pytest.approx(300 - px)
    # 太早的 K 線（i < 30）不出訊號；EMA 排列不對也不出
    assert strat.scan(df.iloc[:25], "XUSDT", "1D") == []


def test_no_signal_when_cross_is_older_than_five_bars():
    strat = get("engulf_1d")
    n = 70
    close = list(np.linspace(100, 140, 40)) + list(np.linspace(139, 100, n - 40))  # 下穿很久了
    rows = [(c + 0.5, c + 1, c - 1, c) for c in close]
    prev_c = close[-1]
    rows.append((prev_c + 0.01, prev_c + 1, prev_c - 3, prev_c - 2.6))
    rows[-2] = (prev_c - 1, prev_c + 1, prev_c - 2, prev_c)  # 前一根收紅
    df = _frame(rows)
    d = strat.prepare(df)
    assert (d["e8"].iloc[-6:-1] < d["e20"].iloc[-6:-1]).all()
    assert strat.scan(df, "XUSDT", "1D", start=len(df) - 1) == []


def _sig(side=SHORT, px=100.0, risk=6.0):
    d = 1 if side == LONG else -1
    return Signal("XUSDT", "1D", side, pd.Timestamp("2026-01-01", tz="UTC"), px, px - d * risk, [px + d * risk],
                  "engulf_1d", {}, {"portions": [1.0], "limit": True, "fill_bars": 3, "max_bars": 60,
                                    "bar_ms": 86_400_000})


def test_tracker_pending_until_touched_then_tp_without_breakeven_event():
    s = _sig()
    df = _frame([(95, 98, 94, 96), (96, 99, 95, 98), (98, 101, 97, 100), (99, 100, 93, 95)])
    # 第 3 根碰到 100：成交；第 4 根（成交後第一根）低點 93 < 94：1R 止盈，全平
    tr = track(s, df, 0.0005)
    assert [e.kind for e in tr.events] == ["FILLED", "TP1"] and tr.status == "CLOSED"
    assert tr.remaining == pytest.approx(0.0) and tr.tps_hit == 1 and tr.realized_r == pytest.approx(1 - 0.0005 * (100 + 94) / 6)
    assert track(s, df.iloc[:2], 0.0005).status == "PENDING"
    assert track(s, df.iloc[:3], 0.0005).status == "OPEN"


def test_tracker_expires_after_three_bars_without_fill():
    s = _sig()
    df = _frame([(95, 98, 94, 96)] * 3 + [(96, 99, 95, 98)])
    tr = track(s, df.iloc[:3], 0.0005)
    assert tr.status == "PENDING"            # 第 3 根還沒收完前不算過期
    tr = track(s, df, 0.0005)
    assert tr.status == "EXPIRED" and tr.events[-1].kind == "EXPIRED" and tr.realized_r == 0
    assert tr.events[-1].time == df["time"].iloc[3]
    # 第 4 根才碰到掛單價：不算成交（只有 3 根）
    late = _frame([(95, 98, 94, 96)] * 3 + [(96, 101, 95, 100)])
    assert track(s, late, 0.0005).status == "EXPIRED"


def test_tracker_stop_on_fill_bar_counts_and_tp_on_fill_bar_does_not():
    s = _sig()
    stop_bar = _frame([(99, 107, 98, 100), (100, 101, 99, 100)])      # 成交那根高點 107 >= 止損 106
    tr = track(s, stop_bar, 0.0005)
    assert [e.kind for e in tr.events] == ["FILLED", "SL"] and tr.status == "CLOSED" and tr.realized_r < -1
    tp_bar = _frame([(99, 101, 90, 95), (95, 99, 96, 97)])            # 成交那根低點到過 TP，但不算（研究也不算）
    tr = track(s, tp_bar, 0.0005)
    assert tr.status == "OPEN" and [e.kind for e in tr.events] == ["FILLED"]


def test_tracker_time_exit_after_60_closed_bars_only():
    s = _sig()
    rows = [(99, 101, 99, 100)] + [(100, 100.5, 99.5, 100.2)] * 60
    df = _frame(rows)                       # 第 1 根成交，之後 60 根都沒碰到止損、止盈
    tr = track(s, df, 0.0005)
    assert tr.status == "OPEN"              # 第 60 根是最後一根（可能還沒收完）：先不平
    tr = track(s, _frame(rows + [(100, 100.5, 99.5, 100.3)]), 0.0005)
    assert tr.status == "CLOSED" and tr.events[-1].kind == "TIMEOUT" and tr.bars == 60
    assert tr.events[-1].price == pytest.approx(100.2)


def test_ledger_and_summary_treat_expired_as_neither_trade_nor_open():
    s = _sig()
    df = _frame([(95, 98, 94, 96)] * 4)
    sig = {**s.to_dict(), "id": "XUSDT|1D|engulf_1d|1", "market": "crypto", "track": track(s, df).to_dict()}
    led = ledger.update([], [sig], lambda *_: df)
    assert led[0]["status"] == "EXPIRED" and led[0]["r"] is None
    g = ledger.summary(led)["groups"]["engulf_1d|crypto|1D"]["all"]
    assert g["trades"] == 0 and g["open"] == 0 and g["expired"] == 1
    # 已經結束的不會被再次追蹤改掉
    again = ledger.update(led, [{**sig, "track": track(s, df.iloc[:1]).to_dict()}], lambda *_: df)
    assert again[0]["status"] == "EXPIRED"


def test_engine_marks_new_signal_with_strategy_label():
    from grid_trading.signals.engine import Engine
    import time
    from pathlib import Path
    eng = Engine([get("engulf_1d")], ["XUSDT"], ["1D"], state_dir=Path("/nonexistent-unused"))
    eng._prices_at = time.time() + 1e6   # 不要真的去抓行情
    st = eng.state()
    assert st["strategies"]["engulf_1d"]["short"] == "日線吞沒" and st["strategies"]["engulf_1d"]["intervals"] == ["1D"]


def test_engine_notifies_pending_fill_and_expiry_with_strategy_label(tmp_path):
    from grid_trading.signals.engine import Engine
    eng = Engine([get("engulf_1d")], ["XUSDT"], ["1D"], state_dir=tmp_path)
    eng.notifications = [{"id": "seed", "signal_id": "seed", "kind": "NEW", "time": 0}]   # 不是第一次執行
    s = _sig()
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    s = Signal(s.symbol, s.interval, s.side, pd.Timestamp(now_ms, unit="ms", tz="UTC"), s.price, s.sl, s.tps,
               s.strategy, {}, s.exit)
    pending = Engine._entry(s, track(s, _frame([(95, 98, 94, 96)], start=s.time)))
    assert pending["track"]["status"] == "PENDING"
    eng._notify({pending["id"]: pending})
    new = [n for n in eng.notifications if n["kind"] == "NEW" and n["signal_id"] == pending["id"]]
    assert len(new) == 1 and "日線吞沒" in new[0]["text"] and "掛單" in new[0]["text"] and not new[0]["observe"]
    # 之後成交
    filled = Engine._entry(s, track(s, _frame([(95, 98, 94, 96), (96, 101, 95, 100)], start=s.time)))
    eng.signals = {pending["id"]: pending}
    eng._notify({filled["id"]: filled})
    assert any(n["kind"] == "FILLED" and "掛單成交" in n["text"] for n in eng.notifications)
