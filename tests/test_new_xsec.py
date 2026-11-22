"""research/new_xsec.py 分數函式的測試：合成面板，驗證數值與「不偷看未來」。"""
import numpy as np
import pandas as pd

import research.candidates_xsec as cx
from research import new_xsec as nx


def make_panel(n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    syms = ["BTCUSDT", "AAAUSDT", "BBBUSDT"]
    days = pd.date_range("2022-01-01", periods=n_days, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, 3)), axis=0))
    z = np.zeros_like(close)
    fund = rng.normal(0.0003, 0.0002, (n_days, 3))
    return cx.Panel(days, syms, close, close.copy(), z + 1e8, z + 0.02, z + 1.0, np.full((n_days, 3), 99), fund)


def test_values():
    p = make_panel()
    a = 40
    np.testing.assert_allclose(nx.new_scores(p, a, "fund7"), -p.fund[a - 6:a + 1].sum(axis=0))
    np.testing.assert_allclose(nx.new_scores(p, a, "rev7"), -(p.close[a] / p.close[a - 7] - 1))
    r = np.log(p.close[a] / p.close[a - 7])
    np.testing.assert_allclose(nx.new_scores(p, a, "resrev7"), -(r - p.beta[a] * r[0]))
    ret = p.close[a - 29:a + 1] / p.close[a - 30:a] - 1
    np.testing.assert_allclose(nx.new_scores(p, a, "max30"), -ret.max(axis=0))


def test_no_lookahead():
    p = make_panel()
    a = 40
    before = {s: nx.new_scores(p, a, s) for s in nx.SIGNALS}
    p2 = make_panel()
    p2.close[a + 1:] *= 5          # 竄改第 a 天以後的資料
    p2.fund[a + 1:] += 1.0
    p2.beta[a + 1:] = 9.0
    for s in nx.SIGNALS:
        np.testing.assert_allclose(nx.new_scores(p2, a, s), before[s])


def test_warmup_nan_and_missing_funding():
    p = make_panel()
    assert np.isnan(nx.new_scores(p, 3, "rev7")).all()
    assert np.isnan(nx.new_scores(p, 20, "max30")).all()
    p.fund[:, 1] = 0.0
    s = nx.new_scores(p, 40, "fund7")
    assert np.isnan(s[1]) and np.isfinite(s[0])


def test_patch_restores():
    old = cx.scores
    with nx.patched_scores():
        assert cx.scores is not old
        assert np.isfinite(cx.scores(make_panel(), 40, "mom14")).all()
    assert cx.scores is old
