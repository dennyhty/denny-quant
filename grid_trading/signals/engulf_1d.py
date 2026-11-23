"""第二個實盤策略：日線 EMA8/20 剛交叉 + 吞沒 K 棒，掛限價單在吞沒 K 實體一半。

規則和 research/threads_strategies.py 的 engulf_signals(df, cross_recent=True, window=5)
+ build_orders(df, sigs, "half", "atr", wait=3) + 出場 "tp1" 一致（研究組合 engulf_1D_cross5_half_atr|tp1|all）：

- 日線（幣安 1d K 線，UTC 00:00 開盤），EMA8、EMA20、ATR(14) 都用日線收盤價算。
- 做空：EMA8 < EMA20（收盤後），而且前 5 根（i-5 .. i-1）至少有一根 EMA8 >= EMA20（剛向下交叉）；
  剛收盤的 K 棒是看跌吞沒（前一根收紅、這根收黑、開盤 >= 前收 × 0.9995、收盤 < 前一根開盤）。做多反過來。
- 進場：限價單掛在吞沒 K 的實體中點（開盤 + 收盤）/ 2，之後 3 根日線內（訊號收盤後 3 天）沒成交就取消，不追市價。
- 止損：從掛單價算 1.5 × ATR（ATR 是訊號 K 棒的）；止盈：只有一個，1R、平 100%，沒有保本、沒有追蹤。
- 成交後超過 60 根日線還沒出場就用收盤價平倉。
- 沒有 30 天成交額門檻（交易程式的 --min-adv 只用在維加斯，這個策略另用 --engulf-min-adv，預設 0）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .strategy import LONG, SHORT, Signal, Strategy, _bar_length, register

WINDOW = 5          # 剛交叉：前幾根內
FILL_BARS = 3       # 掛單有效幾根日線
MAX_BARS = 60       # 成交後最多持有幾根日線
SL_ATR = 1.5
TP_R = 1.0


@register
class Engulf1D(Strategy):
    name = "engulf_1d"
    title = "日線 EMA8/20 吞沒"
    short = "日線吞沒"
    description = ("日線 EMA8 剛和 EMA20 交叉（5 根內）後，出現順著新方向的吞沒 K 棒時，限價掛在吞沒 K 實體一半，"
                   "3 天內沒成交就取消（不追市價）。止損從掛單價算 1.5 倍 ATR；只有 1R 一個止盈、平 100%，沒有保本和追蹤；"
                   "成交後 60 根日線還沒出場就收盤平倉。同一個幣同時只做一張，和維加斯互不影響；沒有成交額門檻。")
    chart_emas = {"EMA8": 8, "EMA20": 20}
    min_bars = 30
    intervals = ("1D",)
    markets = ("crypto",)
    one_per_coin = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["e8"], df["e20"] = ind.ema(df["close"], 8), ind.ema(df["close"], 20)
        df["atr"] = ind.atr(df)
        return df

    def check(self, df: pd.DataFrame, i: int):
        """回傳 (side, 止損, [1R 止盈], info)。info["entry"] 是掛單價（吞沒 K 實體中點）。"""
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        e8, e20, atr = df["e8"].to_numpy(), df["e20"].to_numpy(), df["atr"].to_numpy()
        if i < self.min_bars or np.isnan(atr[i]) or atr[i] <= 0:
            return None
        prev = e8[i - WINDOW:i] - e20[i - WINDOW:i]
        if e8[i] < e20[i] and (prev >= 0).any() and \
                c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] * 0.9995 and c[i] < o[i - 1]:
            side, d = SHORT, -1
            reason = "日線 EMA8 剛向下穿過 EMA20（5 根內），出現看跌吞沒；掛單在吞沒 K 實體一半"
        elif e8[i] > e20[i] and (prev <= 0).any() and \
                c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] * 1.0005 and c[i] > o[i - 1]:
            side, d = LONG, 1
            reason = "日線 EMA8 剛向上穿過 EMA20（5 根內），出現看漲吞沒；掛單在吞沒 K 實體一半"
        else:
            return None
        px = float((o[i] + c[i]) / 2)
        sl = px - d * SL_ATR * float(atr[i])
        tp = px + d * TP_R * abs(px - sl)
        r = df.iloc[i]
        info = {"reason": reason, "entry": px, "atr": float(atr[i]), "ema8": float(e8[i]), "ema20": float(e20[i]),
                "details": [["EMA8 / EMA20", f"{e8[i]:.6g} / {e20[i]:.6g}"],
                            ["吞沒 K 開 / 收", f"{o[i]:.6g} / {c[i]:.6g}"],
                            ["掛單價（實體一半）", f"{px:.6g}"],
                            ["掛單有效", f"訊號收盤後 {FILL_BARS} 天，沒成交就取消"]],
                "candle": {"open": float(r.open), "high": float(r.high), "low": float(r.low),
                           "close": float(r.close), "time": int(r.time.timestamp() * 1000)}}
        return side, sl, [tp], info

    def scan(self, df: pd.DataFrame, symbol: str, interval: str, start: int | None = None) -> list[Signal]:
        """和基底的 scan 不同：訊號價 = 掛單價（吞沒 K 實體一半），不是收盤價；出場方式寫在 exit。"""
        df = self.prepare(df.reset_index(drop=True))
        bar = _bar_length(df)
        out = []
        for i in range(max(start or 0, self.min_bars), len(df)):
            hit = self.check(df, i)
            if hit is None:
                continue
            side, sl, tps, info = hit
            exit_ = {"portions": [1.0], "limit": True, "fill_bars": FILL_BARS, "max_bars": MAX_BARS,
                     "bar_ms": int(bar.total_seconds() * 1000), "close": float(df["close"].iloc[i])}
            out.append(Signal(symbol, interval, side, df["time"].iloc[i] + bar, info["entry"], float(sl),
                              [float(t) for t in tps], self.name, info, exit_))
        return out
