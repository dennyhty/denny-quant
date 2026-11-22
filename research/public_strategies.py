"""網路上常見的公開策略：事先登記、同一套成本與關卡，看有沒有比實盤維加斯好。2026-11-22。

變體（每個各跑 2H、4H；指標用常見預設）：
  macd_ema200   經典「MACD + 200 EMA」：收盤在 EMA200 之上、MACD(12,26,9) 線在 0 軸下方由下往上穿過訊號線 → 做多；反之做空
  vegas_ma200   實盤維加斯訊號（vegas_params 的 base）且收盤在同週期 SMA200 同側
  vegas_ema200  同上，改用 EMA200
  atr_trail     ATR 追蹤線（係數 1、ATR 10）：收盤上穿追蹤線做多、下穿做空
  close_atr_stop 收盤價 ATR 止損線翻轉（ATR 22、倍數 3、用收盤價的最高/最低）：由空翻多做多、反之做空
  chan_osc      通道指數振盪（n1=10、n2=21，wt2 為 wt1 的 4 期均線）：wt1 在 -53 以下上穿 wt2 做多、在 53 以上下穿做空
  vegas_trend_only 實盤維加斯訊號且 HMA(55) 同向（多單上升、空單下降）且拋物線 SAR(0.02,0.02,0.2) 在收盤同側（多單在下方）
出場兩種：live（實盤：止損 1.5 ATR、TP 1/3/5R 各 1/4、保本、追蹤 3 ATR、無持倉上限）、
fix15（全倉止損 1.5 ATR、單一 TP 1.5R、最多 240 根）。共 7 x 2 x 2 = 28 組。
另外記錄參照帳本 vegas_base（同一批 K 棒、同一套模擬的實盤維加斯）和「濾網拿掉的單」，不算新試驗。
所有訊號只用到訊號 K 棒收盤；進場在該收盤、出場從下一根檢查；從第 700 根 K 棒開始（和 vegas_params 一樣）。
執行：python -m research.public_strategies run [--log]  →  output/ps_trades.csv.gz 與報表
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numba import njit

import research.entry_research as er
import research.vegas_params as vp
from grid_trading import indicators as ind
from grid_trading.costs import apply_costs
from grid_trading.pit_universe import coin_of, fetch_symbol, membership
from research.exit_unify import per_interval
from research.pit_backtest import KLINE, PLAN, START, snapshot_symbols

WARMUP = 700
FAMILY = "public_strats"
VARIANTS = ("macd_ema200", "vegas_ma200", "vegas_ema200", "atr_trail", "close_atr_stop", "chan_osc", "vegas_trend_only")
REFS = ("vegas_base", "removed_ma200", "removed_ema200", "removed_trend")     # 參照帳本，不算新試驗
EXITS = {"live": (PLAN, 10 ** 7), "fix15": (((1.5,), (1.0,), 0.0), 240)}
INTERVALS = ("2H", "4H")
BPD = {"2H": 12, "4H": 6}


# ---------------------------------------------------------------- 指標（只用當根與以前的資料）

def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).mean().to_numpy()


def _cross_up(a, b):
    pa, pb = np.r_[np.nan, a[:-1]], np.r_[np.nan, b[:-1]]
    return (a > b) & (pa <= pb)


def _cross_dn(a, b):
    return _cross_up(b, a)


def macd_ema200_signal(close: np.ndarray) -> np.ndarray:
    """+1 / -1 / 0：收盤在 EMA200 同側，MACD 線在 0 軸同側穿過訊號線。"""
    c = pd.Series(close)
    m = ind.macd(c)
    line, sig = m["macd"].to_numpy(), m["signal"].to_numpy()
    e200 = ind.ema(c, 200).to_numpy()
    lg = (close > e200) & _cross_up(line, sig) & (line < 0) & (sig < 0)
    sh = (close < e200) & _cross_dn(line, sig) & (line > 0) & (sig > 0)
    return np.where(lg, 1, np.where(sh, -1, 0))


@njit(cache=True)
def _atr_trail_stop(src, loss):
    n = len(src)
    out = np.zeros(n)
    for k in range(1, n):
        if not (np.isfinite(loss[k]) and np.isfinite(src[k])):
            out[k] = 0.0
            continue
        p = out[k - 1]
        if src[k] > p and src[k - 1] > p:
            out[k] = max(p, src[k] - loss[k])
        elif src[k] < p and src[k - 1] < p:
            out[k] = min(p, src[k] + loss[k])
        elif src[k] > p:
            out[k] = src[k] - loss[k]
        else:
            out[k] = src[k] + loss[k]
    return out


def atr_trail_signal(df: pd.DataFrame, key: float = 1.0, n: int = 10) -> np.ndarray:
    """ATR 追蹤線：+1 收盤上穿追蹤線、-1 下穿。"""
    src = df["close"].to_numpy(float)
    loss = key * ind.atr(df, n).to_numpy(float)
    stop = _atr_trail_stop(src, loss)
    buy = (src > stop) & _cross_up(src, stop)
    sell = (src < stop) & _cross_dn(src, stop)
    return np.where(buy, 1, np.where(sell, -1, 0))


@njit(cache=True)
def _close_atr_stop_dir(close, hi_c, lo_c, atr, mult):
    n = len(close)
    d = np.ones(n, np.int64)
    ls = np.full(n, np.nan)
    ss = np.full(n, np.nan)
    for k in range(n):
        if not (np.isfinite(hi_c[k]) and np.isfinite(atr[k])):
            continue
        l, s = hi_c[k] - mult * atr[k], lo_c[k] + mult * atr[k]
        lp = ls[k - 1] if k > 0 and np.isfinite(ls[k - 1]) else l
        sp = ss[k - 1] if k > 0 and np.isfinite(ss[k - 1]) else s
        if k > 0 and close[k - 1] > lp:
            l = max(l, lp)
        if k > 0 and close[k - 1] < sp:
            s = min(s, sp)
        ls[k], ss[k] = l, s
        d[k] = d[k - 1] if k > 0 else 1
        if close[k] > sp:
            d[k] = 1
        elif close[k] < lp:
            d[k] = -1
    return d


def close_atr_stop_signal(df: pd.DataFrame, n: int = 22, mult: float = 3.0) -> np.ndarray:
    """收盤價 ATR 止損線：方向由 -1 翻成 +1 = 做多、由 +1 翻成 -1 = 做空。"""
    c = df["close"]
    atr = ind.atr(df, n).to_numpy(float)
    d = _close_atr_stop_dir(c.to_numpy(float), c.rolling(n).max().to_numpy(), c.rolling(n).min().to_numpy(), atr, mult)
    pd_ = np.r_[1, d[:-1]]
    return np.where((d == 1) & (pd_ == -1), 1, np.where((d == -1) & (pd_ == 1), -1, 0))


def chan_osc_lines(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> tuple[np.ndarray, np.ndarray]:
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ind.ema(ap, n1)
    d = ind.ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ind.ema(ci, n2)
    return wt1.to_numpy(), wt1.rolling(4).mean().to_numpy()


def chan_osc_signal(df: pd.DataFrame, ob: float = 53.0) -> np.ndarray:
    wt1, wt2 = chan_osc_lines(df)
    lg = _cross_up(wt1, wt2) & (wt1 < -ob)
    sh = _cross_dn(wt1, wt2) & (wt1 > ob)
    return np.where(lg, 1, np.where(sh, -1, 0))


@njit(cache=True)
def _psar(high, low, start, step, maximum):
    """拋物線 SAR（和 grid_trading.indicators.psar 同算法，numba 版）。"""
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    up = high[1] >= high[0]
    sar = low[0] if up else high[0]
    ep = high[0] if up else low[0]
    af = start
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:
                up, sar, ep, af = False, ep, low[i], start
            elif high[i] > ep:
                ep, af = high[i], min(af + step, maximum)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                up, sar, ep, af = True, ep, high[i], start
            elif low[i] < ep:
                ep, af = low[i], min(af + step, maximum)
        out[i] = sar
    return out


def trend_filter(df: pd.DataFrame, direction: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """vegas_trend_only 濾網：HMA(55) 在訊號 K 棒比前一根高（多）／低（空），且 SAR 在收盤下方（多）／上方（空）。"""
    hma = ind.hma(df["close"], 55).to_numpy(float)
    sar = _psar(df["high"].to_numpy(float), df["low"].to_numpy(float), 0.02, 0.02, 0.2)
    cl = df["close"].to_numpy(float)
    ok = np.isfinite(hma[idx]) & np.isfinite(sar[idx]) & np.isfinite(hma[idx - 1])
    up = (hma[idx] > hma[idx - 1]) & (sar[idx] < cl[idx])
    dn = (hma[idx] < hma[idx - 1]) & (sar[idx] > cl[idx])
    return ok & np.where(direction > 0, up, dn)


# ---------------------------------------------------------------- 訊號

def all_signals(df: pd.DataFrame, interval: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """變體 -> (訊號 K 棒位置, 方向)。包含參照帳本。"""
    cl = df["close"].to_numpy(float)
    n = len(cl)
    atr = ind.atr(df).to_numpy(float)
    ok = (np.arange(n) >= WARMUP) & (np.arange(n) + 1 < n) & (atr > 0)
    out = {}

    def put(name, sig):
        idx = np.flatnonzero((sig != 0) & ok)
        out[name] = (idx, sig[idx].astype(int))

    put("macd_ema200", macd_ema200_signal(cl))
    put("atr_trail", atr_trail_signal(df))
    put("close_atr_stop", close_atr_stop_signal(df))
    put("chan_osc", chan_osc_signal(df))
    bidx, bds = vp.signals(df, interval)["base"]
    s200, e200 = sma(cl, 200), ind.ema(df["close"], 200).to_numpy()
    for tag, ma in (("ma200", s200), ("ema200", e200)):
        good = ((bds > 0) & (cl[bidx] > ma[bidx])) | ((bds < 0) & (cl[bidx] < ma[bidx]))
        out[f"vegas_{tag}"] = (bidx[good], bds[good])
        out[f"removed_{tag}"] = (bidx[~good], bds[~good])
    good = trend_filter(df, bds, bidx)
    out["vegas_trend_only"] = (bidx[good], bds[good])
    out["removed_trend"] = (bidx[~good], bds[~good])
    out["vegas_base"] = (bidx, bds)
    return out


def _arrays(df: pd.DataFrame) -> dict:
    return {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")} | {"atr": ind.atr(df).to_numpy(float)}


def _sim(arr, idx, ds, interval, exit_name, fee=0.0005):
    plan, mb = EXITS[exit_name]
    er.PLANS = {interval: plan}
    er.MAX_BARS = mb
    return er.simulate(arr, idx, ds, interval, fee, 0.0)


def _frame(sym, interval, t, adv, variant, exit_name, idx, ds, s, extra=None):
    d = pd.DataFrame({
        "variant": variant, "exit": exit_name, "symbol": sym, "coin": coin_of(sym), "interval": interval, "i": idx,
        "side": ds, "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
        "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "risk_pct": s["risk_pct"], "adv_usdt": adv[idx]})
    return d if extra is None else d.assign(**extra)


def _load(sym, interval):
    for k in range(6):
        try:
            df = fetch_symbol(sym, KLINE[interval])
            break
        except Exception:
            if k == 5:
                raise
            time.sleep(5 * (k + 1))
    return df.reset_index(drop=True)


def _work(args):
    sym, interval = args
    df = _load(sym, interval)
    if len(df) < WARMUP + 50:
        return None
    arr = _arrays(df)
    t = pd.to_datetime(df["time"], utc=True)
    adv = ((df["volume"] * df["close"]).rolling(30 * BPD[interval], min_periods=BPD[interval]).sum() / 30).to_numpy()
    parts = []
    for v, (idx, ds) in all_signals(df, interval).items():
        if not len(idx):
            continue
        for ex in EXITS:
            parts.append(_frame(sym, interval, t, adv, v, ex, idx, ds, _sim(arr, idx, ds, interval, ex)))
    return pd.concat(parts) if parts else None


def _rand_work(args):
    """隨機進場基準：每筆策略交易換成同幣、同年、同方向、在幣池內的隨機一根 K 棒進場，出場相同。"""
    sym, interval, jobs, months, draws, seed = args
    df = _load(sym, interval)
    if len(df) < WARMUP + 50:
        return None
    arr = _arrays(df)
    t = pd.to_datetime(df["time"], utc=True)
    n = len(df)
    adv = ((df["volume"] * df["close"]).rolling(30 * BPD[interval], min_periods=BPD[interval]).sum() / 30).to_numpy()
    ent = t.shift(-1)       # 進場時間 = 下一根開盤
    ok = (np.arange(n) >= WARMUP) & (np.arange(n) + 1 < n) & (arr["atr"] > 0) & np.array(
        [m in months for m in ent.dt.strftime("%Y-%m").fillna("")])
    years = ent.dt.year.fillna(0).to_numpy()
    rng = np.random.default_rng(seed + (hash(sym) % 10_000))
    parts = []
    for v, (ex, yrs, sides) in jobs.items():
        for k in range(draws):
            idx, ds = [], []
            for y, s in zip(yrs, sides):
                cand = np.flatnonzero(ok & (years == y))
                if len(cand):
                    idx.append(int(rng.choice(cand)))
                    ds.append(int(s))
            if not idx:
                continue
            idx, ds = np.array(idx), np.array(ds)
            o = np.argsort(idx, kind="stable")
            idx, ds = idx[o], ds[o]
            parts.append(_frame(sym, interval, t, adv, v, ex, idx, ds, _sim(arr, idx, ds, interval, ex), {"draw": k}))
    return pd.concat(parts) if parts else None


def membership_table() -> tuple[pd.DataFrame, set]:
    mem = membership(n=100)
    mem["month"] = mem["month"].astype(str)
    return mem, set(zip(mem["symbol"], mem["month"]))


def universe(mem) -> list[str]:
    return sorted(set(mem.loc[mem["month"] >= "2020-01", "symbol"]) | snapshot_symbols())


def run_signals(workers: int = 4) -> pd.DataFrame:
    mem, _ = membership_table()
    jobs = [(s, iv) for iv in INTERVALS for s in universe(mem)]
    print(f"{len(jobs)} 個工作", file=sys.stderr)
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(_work, jobs, chunksize=4)):
            if got is not None:
                parts.append(got)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr, flush=True)
    tr = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    return tr[tr["time"] >= START]


def to_ledger(tr: pd.DataFrame, pit: set, funding: dict | None = None) -> pd.DataFrame:
    """篩幣池 → 每個 (變體、週期、出場) 同一個幣同時只一張 → 滑價與資金費率（r_net）。"""
    tr = tr[[(s, m) in pit for s, m in zip(tr["symbol"], tr["time"].dt.strftime("%Y-%m"))]]
    keys = [c for c in ("variant", "exit", "interval", "draw") if c in tr.columns]
    parts = [per_interval(g) for _, g in tr.groupby(keys, sort=False)]
    out = pd.concat(parts, ignore_index=True)
    return apply_costs(out, funding=funding)
