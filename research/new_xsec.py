"""新策略搜尋 N1：橫斷面新分數（資金費率、短期反轉、特質反轉、MAX 效應），沿用 candidates_xsec 的週組合引擎。

經濟理由（誰把錢輸給你）：
- fund7：過去 7 天資金費率加總越高 = 多頭越擁擠、做多的人在付錢；做空這類幣、做多費率最負的幣，
  同時收到資金費率（carry），輸家是擁擠的槓桿多頭。
- rev7：短期（1 週）橫斷面反轉，散戶追漲殺跌後的過度反應回吐，流動性提供者賺價差。
- resrev7：扣掉 BTC 影響（beta x BTC 週報酬）之後的特質報酬反轉，只反轉幣自己的消息面過度反應。
- max30：過去 30 天單日最大漲幅（樂透型）越大，後續越差（Bali 等的 MAX 效應），輸家是追樂透幣的散戶。

規則和 candidates_xsec 完全相同（週日收盤再平衡、波動縮放、BTC beta 中性、成本含滑價與資金費率），
只是換分數。分數只用第 a 天（含）以前的資料。不修改 candidates_xsec.py：用暫時替換它的 scores 來擴充。

網格（預先登記，共 8 組）：分數 ∈ {fund7, rev7, resrev7, max30} x q ∈ {0.2, 0.3}。
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager

import numpy as np

import research.candidates_xsec as cx

FAMILY = "new_xsec_factors"
SIGNALS = ("fund7", "rev7", "resrev7", "max30")


def configs() -> dict[str, dict]:
    out = {}
    for sig, q in itertools.product(SIGNALS, (0.2, 0.3)):
        out[f"{sig}|q{q:g}"] = {"signal": sig, "q": q}
    return out


def neighbors(name: str) -> list[str]:
    """鄰近設定：q 換另一格；rev7 與 resrev7 互為鄰近。"""
    p = configs()[name]
    q2 = 0.3 if p["q"] == 0.2 else 0.2
    out = [f"{p['signal']}|q{q2:g}"]
    if p["signal"] in ("rev7", "resrev7"):
        other = "resrev7" if p["signal"] == "rev7" else "rev7"
        out.append(f"{other}|q{p['q']:g}")
    return out


def new_scores(p: cx.Panel, a: int, signal: str) -> np.ndarray:
    """第 a 天收盤時每個幣的分數（越大越想做多）。只用第 a 天（含）以前的資料。"""
    n = len(p.syms)
    nan = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if signal == "fund7":
            if a < 6:
                return nan
            seg = p.fund[a - 6:a + 1]                      # 第 a-6..a 天的資金費率加總（第 a 天的結算在 a 日收盤前都已發生）
            s = -seg.sum(axis=0)
            return np.where(np.abs(seg).sum(axis=0) > 0, s, np.nan)   # 完全沒有費率資料的幣不給分數
        if signal == "rev7":
            if a < 7:
                return nan
            return -(p.close[a] / p.close[a - 7] - 1)
        if signal == "resrev7":
            if a < 7:
                return nan
            b = p.syms.index(cx.HEDGE)
            r = np.log(p.close[a] / p.close[a - 7])
            rb = np.log(p.close[a, b] / p.close[a - 7, b])
            return -(r - p.beta[a] * rb)
        if signal == "max30":
            if a < 30:
                return nan
            c = p.close[a - 30:a + 1]
            ret = c[1:] / c[:-1] - 1                        # 第 a-29..a 天的日報酬
            ok = np.isfinite(ret).sum(axis=0)
            m = np.where(ok >= 20, np.nanmax(np.where(np.isfinite(ret), ret, -np.inf), axis=0), np.nan)
            return -m
    raise ValueError(signal)


@contextmanager
def patched_scores():
    """暫時把 candidates_xsec.scores 換成可處理新分數的版本。"""
    old = cx.scores

    def f(p, a, signal):
        return new_scores(p, a, signal) if signal in SIGNALS else old(p, a, signal)
    cx.scores = f
    try:
        yield
    finally:
        cx.scores = old


def run(p: cx.Panel, pool: set, cfg: dict, rng=None, **kw):
    with patched_scores():
        return cx.run(p, pool, cfg, rng=rng, **kw)
