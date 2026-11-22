"""research/public_strategies.py 指標與訊號的測試：合成資料，驗證數值與不偷看未來。"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")
from research import public_strategies as ps  # noqa: E402


def frame(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 0.3 * np.sin(np.arange(n) / 40))
    spread = np.abs(rng.normal(0, 0.004, n)) + 0.002
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * (1 + spread), "low": c * (1 - spread), "close": c,
                         "volume": 1000.0})


SIGS = {"macd": lambda d: ps.macd_ema200_signal(d["close"].to_numpy()), "ut": ps.atr_trail_signal,
        "chan": ps.close_atr_stop_signal, "wt": ps.chan_osc_signal}


@pytest.mark.parametrize("name", list(SIGS))
def test_no_lookahead(name):
    d = frame()
    full = SIGS[name](d)
    for cut in (800, 1100, 1400):
        part = SIGS[name](d.iloc[:cut].reset_index(drop=True))
        np.testing.assert_array_equal(part[:cut - 1], full[:cut - 1])   # 最後一根以前完全一樣
        assert part[cut - 1] == full[cut - 1]


def test_signals_fire_and_are_sparse():
    d = frame()
    for name, f in SIGS.items():
        s = f(d)
        assert set(np.unique(s)) <= {-1, 0, 1}
        assert (s != 0).sum() > 0, name


def test_atr_trail_cross_matches_definition():
    d = frame()
    src = d["close"].to_numpy()
    loss = ps.ind.atr(d, 10).to_numpy()
    stop = ps._atr_trail_stop(src, loss)
    s = ps.atr_trail_signal(d)
    k = np.flatnonzero(s == 1)[0]
    assert src[k] > stop[k] and src[k - 1] <= stop[k - 1]


def test_macd_conditions():
    d = frame()
    s = ps.macd_ema200_signal(d["close"].to_numpy())
    m = ps.ind.macd(d["close"])
    e = ps.ind.ema(d["close"], 200).to_numpy()
    for k in np.flatnonzero(s == 1):
        assert d["close"].iloc[k] > e[k] and m["macd"].iloc[k] < 0 and m["signal"].iloc[k] < 0
    for k in np.flatnonzero(s == -1):
        assert d["close"].iloc[k] < e[k] and m["macd"].iloc[k] > 0


def test_chan_osc_thresholds():
    d = frame()
    wt1, wt2 = ps.chan_osc_lines(d)
    s = ps.chan_osc_signal(d)
    assert all(wt1[k] < -53 for k in np.flatnonzero(s == 1))
    assert all(wt1[k] > 53 for k in np.flatnonzero(s == -1))


def test_psar_numba_matches_reference():
    d = frame()
    ref = ps.ind.psar(d).to_numpy()
    got = ps._psar(d["high"].to_numpy(), d["low"].to_numpy(), 0.02, 0.02, 0.2)
    np.testing.assert_allclose(got[1:], ref[1:])


def test_trend_filter_no_lookahead():
    d = frame()
    idx = np.arange(800, 1400, 7)
    side = np.where(np.arange(len(idx)) % 2 == 0, 1, -1)
    full = ps.trend_filter(d, side, idx)
    cut = 1000
    keep = idx < cut
    part = ps.trend_filter(d.iloc[:cut].reset_index(drop=True), side[keep], idx[keep])
    np.testing.assert_array_equal(part, full[keep])
