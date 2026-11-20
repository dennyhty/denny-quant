"""實盤維加斯換到其他週期（15m、1H、12H、1D、2D、3D、1W；2H、4H 當對照）的誠實版回測（只做分析，不改實盤規則）。2026-11-20。

python -m grid_trading.pit_universe            # 先建每月成交額表
python -m research.vegas_timeframes fetch      # 下載各週期合約 K 線（12H 以上再補合約上市前的現貨 K 線，只當指標暖機）
python -m research.vegas_timeframes run        → output/tf_trades.csv.gz、output/tf_random.csv.gz
python -m research.vegas_timeframes report     → output/tf_summary.csv、tf_yearly.csv、tf_gate.json

規則全部照實盤（2026-11-20）：止損 1.5 ATR = 1R；TP 1/3/5R 各平 1/4，TP1 後保本、TP2 後移到 TP1，最後 1/4 追蹤 3 ATR；
沒有持倉根數上限（PR #73 拿掉了 240 根）；ADX ≥ 24 不做只用在 2H（其他週期沒測過這條，不套）。
幣池：每月「上個月成交額前 100 名」的幣安 USDT 永續（含後來下市的，沒有倖存者偏差），2020 年起；
成本：每邊手續費 0.05% + 依日均成交額分級的滑價 + 實際資金費率（grid_trading/costs.py）。
同一週期同一個合約同時只一張。1M 不測：EMA169 要 169 個月（14 年）暖機，沒有任何幣有這麼長的資料。
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import research.entry_research as er
from grid_trading.binance_history import SPOT_CACHE_DIR, clean
from grid_trading.costs import apply_costs
from grid_trading.pit_universe import coin_of, fetch_symbol, membership
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from research.exit_unify import per_interval

OUT = Path("output")
PLAN = ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0)
START = pd.Timestamp("2020-01-01", tz="UTC")
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
RISK = 0.003
# 週期 → (幣安 K 線, 重取樣天數, 每天幾根)
TF = {"15m": ("15m", None, 96), "1H": ("1h", None, 24), "2H": ("2h", None, 12), "4H": ("4h", None, 6),
      "12H": ("12h", None, 2), "1D": ("1d", None, 1), "2D": ("1d", 2, 0.5), "3D": ("1d", 3, 1 / 3),
      "1W": ("1d", 7, 1 / 7)}
# 幣安的 3d、1w 月檔只到 2024-01，所以 2D、3D、1W 都由日線合成：2D、3D 從 1970-01-01 起算，1W 從週一開盤
EXTEND = {"12H", "1D", "2D", "3D", "1W"}  # 用現貨補暖機的週期
NO_LIMIT = 10 ** 9


def pool_symbols() -> list[str]:
    mem = membership(n=100)
    return sorted(set(mem.loc[mem["month"].astype(str) >= "2020-01", "symbol"]))


# ---------------------------------------------------------------- 資料

def _spot_base(sym: str) -> tuple[str, float]:
    c = coin_of(sym)
    for p, scale in (("1000000", 1e6), ("1000", 1e3)):
        if c.startswith(p) and c[len(p):].isalpha():
            return c[len(p):], scale
    return c, 1.0


def spot_before(sym: str, kline: str, before_ms: int) -> pd.DataFrame:
    """合約第一根 K 線之前的現貨 K 線（time 毫秒）。只拿來讓 EMA169 暖機，不在現貨期間進場。"""
    from grid_trading.spot_history import _keys, _read_zip

    path = SPOT_CACHE_DIR / f"sym_{sym}_{kline}_pre.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    base, scale = _spot_base(sym)
    session = requests.Session()
    first = pd.Timestamp(before_ms, unit="ms").to_period("M")
    keys = [k for k in _keys(f"data/spot/monthly/klines/{base}USDT/{kline}/", session)
            if pd.Period(k[-11:-4], "M") <= first]
    parts = [_read_zip(k, session) for k in keys]  # 下載失敗就丟例外、不寫快取，下次再試
    df = pd.concat(parts) if parts else pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    if len(df):
        df[["open", "high", "low", "close"]] *= scale
        df["volume"] /= scale
        df = df[df["time"] < before_ms]
    SPOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def resample(df: pd.DataFrame, days: int) -> pd.DataFrame:
    origin = pd.Timestamp("1970-01-05", tz="UTC") if days == 7 else pd.Timestamp(0, tz="UTC")  # 1970-01-05 是週一
    g = ((df["time"] - origin) // pd.Timedelta(days=days)).to_numpy()
    out = df.groupby(g).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                            close=("close", "last"), volume=("volume", "sum"), n=("close", "size"))
    out["time"] = origin + pd.to_timedelta(out.index * days, unit="D")
    # 最後一根沒走完的不要（中間缺日的照用）
    if len(out) and out["n"].iloc[-1] < days:
        out = out.iloc[:-1]
    return out[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def frame(sym: str, tf: str) -> tuple[pd.DataFrame, int]:
    """回傳 (K 線, 合約第一根的位置)。合約前面接現貨的週期，進場只算合約期間。"""
    kline, days, _ = TF[tf]
    fut = fetch_symbol(sym, kline)
    if fut.empty:
        return fut, 0
    first_fut = fut["time"].iloc[0]
    df = fut
    if tf in EXTEND:
        ms = int(first_fut.value // 1_000_000)
        pre = spot_before(sym, kline, ms)
        if len(pre):
            raw = fut.assign(time=(fut["time"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1))
            df = clean(pd.concat([pre[raw.columns], raw]))
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    if days:
        df = resample(df, days)
    df = df.reset_index(drop=True)
    return df, int(df["time"].searchsorted(first_fut))


def _fetch(args):
    sym, tf = args
    try:
        frame(sym, tf)
        return None
    except Exception as e:  # 下載失敗下次再試
        return f"{sym} {tf}: {e}"


# ---------------------------------------------------------------- 訊號與模擬

def _sim(p: pd.DataFrame, idx: np.ndarray, ds: np.ndarray, tf: str, fee: float) -> dict:
    arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
    er.PLANS = {tf: PLAN}
    er.MAX_BARS = NO_LIMIT
    return er.simulate(arr, idx, ds, tf, fee, 0.0)


def _frame_out(sym, tf, p, idx, ds, s, adv):
    t = pd.to_datetime(p["time"], utc=True)
    return pd.DataFrame({"symbol": sym, "coin": coin_of(sym), "interval": tf, "i": idx, "side": ds,
                         "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                         "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "risk_pct": s["risk_pct"],
                         "adv_usdt": adv[idx]})


def _work(args):
    sym, tf, fee, seed = args
    df, first = frame(sym, tf)
    strat = VegasTrend()
    if len(df) < strat.min_bars + 5:
        return None
    p = strat.prepare(df).reset_index(drop=True)
    n = len(p)
    lo = max(strat.min_bars, first)
    idx, ds = [], []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < lo or i + 1 >= n:
            continue
        hit = strat.check(p, int(i))
        if hit is None or not strat.accept(p, int(i), tf):
            continue
        idx.append(int(i))
        ds.append(1 if hit[0] == LONG else -1)
    bpd = TF[tf][2]
    # 30 天平均日成交額（滑價分級用），至少 2 根
    w = max(2, int(round(30 * bpd)))
    adv = ((p["volume"] * p["close"]).rolling(w, min_periods=2).mean() * bpd).to_numpy()
    out = []
    if idx:
        idx, ds = np.array(idx), np.array(ds)
        out.append(_frame_out(sym, tf, p, idx, ds, _sim(p, idx, ds, tf, fee), adv))
    # 隨機基準：同一個幣、同樣筆數，在通道方向成立的 K 棒隨機進場，方向跟通道（測 MACD/RSI 時機有沒有用）
    rnd = None
    if len(idx):
        c, e12 = p["close"].to_numpy(float), p["ema12"].to_numpy(float)
        e144, e169 = p["ema144"].to_numpy(float), p["ema169"].to_numpy(float)
        top, bot = np.maximum(e144, e169), np.minimum(e144, e169)
        trend = np.where((c > top) & (e12 > top) & (e144 > e169), 1, np.where((c < bot) & (e12 < bot) & (e144 < e169), -1, 0))
        ok = (np.arange(n) >= lo) & (np.arange(n) + 1 < n) & (trend != 0) & (p["atr"].to_numpy(float) > 0)
        pool = np.flatnonzero(ok)
        if len(pool):
            rng = np.random.default_rng(seed)
            ridx = np.sort(rng.choice(pool, size=min(len(pool), 3 * len(idx)), replace=len(pool) < 3 * len(idx)))
            rds = trend[ridx]
            rnd = _frame_out(sym, tf, p, ridx, rds, _sim(p, ridx, rds, tf, fee), adv)
    return (out[0] if out else None), rnd


def run(fee: float, workers: int, tfs: list[str]) -> None:
    syms = pool_symbols()
    mem = membership(n=100)
    pit = set(zip(mem["symbol"], mem["month"].astype(str)))
    jobs = [(s, tf, fee, k) for tf in tfs for k, s in enumerate(syms)]
    sig, rnd = [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(_work, jobs, chunksize=2)):
            if got is not None:
                if got[0] is not None:
                    sig.append(got[0])
                if got[1] is not None:
                    rnd.append(got[1])
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr)
    for name, parts in (("tf_trades", sig), ("tf_random", rnd)):
        tr = pd.concat(parts, ignore_index=True)
        for c in ("time", "exit_t"):
            tr[c] = pd.to_datetime(tr[c], utc=True)
        tr = tr[tr["time"] >= START]
        tr["in_pit"] = [(s, m) in pit for s, m in zip(tr["symbol"], tr["time"].dt.strftime("%Y-%m"))]
        tr = tr[tr["in_pit"]]
        # 同一週期同一個合約同時一張
        tr = pd.concat([per_interval(g.rename(columns={"coin": "_c", "symbol": "coin"}))
                        .rename(columns={"coin": "symbol", "_c": "coin"}) for _, g in tr.groupby("interval")],
                       ignore_index=True)
        tr = apply_costs(tr)
        path = OUT / f"{name}.csv.gz"
        if path.exists() and set(tfs) != set(TF):
            old = pd.read_csv(path)
            for c in ("time", "exit_t"):
                old[c] = pd.to_datetime(old[c], utc=True)
            tr = pd.concat([old[~old["interval"].isin(tfs)], tr], ignore_index=True)
        tr.to_csv(path, index=False)
        print(f"{name}: {len(tr)} 筆", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "report"])
    ap.add_argument("--tf", nargs="*", default=list(TF))
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.cmd == "fetch":
        syms = pool_symbols()
        jobs = [(s, tf) for tf in a.tf for s in syms]
        print(f"{len(syms)} 個合約 × {len(a.tf)} 個週期", file=sys.stderr)
        with ProcessPoolExecutor(a.workers) as ex:
            for k, err in enumerate(ex.map(_fetch, jobs)):
                if err:
                    print("  失敗", err, file=sys.stderr)
                if k % 100 == 0:
                    print(f"  {k}/{len(jobs)}", file=sys.stderr)
    elif a.cmd == "run":
        run(a.fee, a.workers, a.tf)
    else:
        from research.vegas_timeframes_report import report
        report()


if __name__ == "__main__":
    main()
