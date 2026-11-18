"""幣安美股永續合約能不能納入維加斯實盤：用股票本身的價格（長歷史）和幣安合約自己的價格（短歷史）回測。

只做研究，不改實盤和網站。

python -m research.binance_stock_perps universe   # 列出幣安目前的美股/ETF 永續（用價格和 Yahoo 對上才算）→ output/stock_perps_universe.csv
python -m research.binance_stock_perps fetch      # 抓 Yahoo 正股 1 小時（730 天）、日線（上市以來）；幣安合約 2h/4h K 線和資金費率
python -m research.binance_stock_perps run        # 回測 → output/stock_perps_*.csv

和 2026-10-18 那次（research/rwa_underlying.py，派網代幣）不一樣的地方：
- 標的換成「幣安」現在上架的美股/ETF 永續（172 檔，派網那次 62 檔），槓桿/反向 ETF、波動率 ETF 不算。
- 照 2026-10-20 之後的實盤規則：2H、4H 各自可以同幣同方向各一張（不再兩個週期只做一張）。
- 加上成本：滑價依幣安合約 30 日平均成交額分級（grid_trading/costs.py）、資金費率（合約期間用實際費率，
  正股期間用該合約的平均費率），手續費每邊 0.05%。
- 隨機進場對照也算群集標準誤，直接比「策略 − 隨機」，做多做空分開。
- 幣安合約本身的 2H/4H 是 24 小時切（和實盤交易程式看到的一樣），另外分開看訊號在美股開市/休市時段。

正股資料：Yahoo Finance。1 小時最多回查 730 天，所以正股 2H/4H 從 2024-11 起（扣暖機後訊號約 2025-03 起）；
日線從上市開始。正股只用正規時段（美東 9:30-16:00），每個交易日內從開盤切 2H/4H。
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import research.rwa_underlying as ru
from grid_trading.costs import _ns, fetch_funding, slippage_bps
from grid_trading.pit_universe import fetch_symbol
from research.exit_unify import per_interval
from research.stop_research import LIVE_ATR, MIN_STOP_PCT, cluster_key, cluster_stats
from research.exit_unify import account

OUT = Path("output")
CACHE = Path("data/yahoo_us")
UA = {"User-Agent": "Mozilla/5.0"}
LISTED_AFTER = "2025-12-01"  # 幣安美股永續 2025-12 之後才陸續上架
# 槓桿、反向、波動率 ETF：每天重設槓桿，長期會耗損，不適合拿趨勢策略比較
LEVERAGED = {"INTW", "KORU", "MSTU", "MUU", "MVLL", "NVDL", "RAM", "SKDD", "SKUU", "SNXX", "SOXL", "SOXS",
             "SQQQ", "TBT", "TMF", "TQQQ", "TSLL", "TZA", "UVXY"}
# 代號和美股撞名但其實不是那檔股票（CL 是原油合約，不是高露潔）
NOT_STOCK = {"CL"}
YAHOO = {"BRKB": "BRK-B"}
MIN_ADV = 50e6  # 實盤交易程式 --min-adv：30 日平均日成交額低於 5000 萬 U 的不做
ADV: pd.Series = pd.Series(dtype=float)


# ---------------------------------------------------------------- 標的清單

def _listing_dates() -> list[tuple[str, str | None]]:
    import re
    from grid_trading.pit_universe import LISTING, UM_KLINES, _list_prefixes
    s = requests.Session()
    names = sorted(p[len(UM_KLINES):].strip("/") for p in _list_prefixes(UM_KLINES, s))
    names = [n for n in names if n.endswith("USDT") and "_" not in n and n.isascii()]

    def first(n):
        r = s.get(LISTING, params={"prefix": f"data/futures/um/daily/klines/{n}/1d/", "max-keys": "1"}, timeout=30)
        k = re.findall(r"<Key>[^<]*-(\d{4}-\d{2}-\d{2})\.zip", r.text)
        return n, (k[0] if k else None)

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(first, names))


def _yahoo_quote(tk: str, day: str):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}", params={"interval": "1d", "range": "2mo"},
                     headers=UA, timeout=30)
    try:
        j = r.json()["chart"]["result"][0]
        s = pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s").date).dropna()
        s = s[[str(i) <= day for i in s.index]]
        m = j["meta"]
        return float(s.iloc[-1]), m.get("instrumentType"), m.get("exchangeName"), m.get("longName") or m.get("shortName")
    except Exception:  # noqa: BLE001
        return None, None, None, None


def universe() -> pd.DataFrame:
    """2025-12 之後上架、且最後收盤價和 Yahoo 同名美股/ETF 差不到 15% 的幣安 USDT 永續。"""
    new = [(n, d) for n, d in _listing_dates() if d and d >= LISTED_AFTER]
    rows = []
    for n, listed in new:
        try:
            df = fetch_symbol(n, "1d")
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        day = df["time"].iloc[-1].date().isoformat()
        base = n[:-4]
        tk = YAHOO.get(base, base)
        yc, typ, exch, name = _yahoo_quote(tk, day)
        time.sleep(0.3)
        rows.append({"symbol": n, "base": base, "ticker": tk, "listed": listed, "bclose": float(df["close"].iloc[-1]),
                     "yclose": yc, "type": typ, "exchange": exch, "name": name,
                     "adv_usdt": float((df["close"] * df["volume"]).tail(30).mean())})
    u = pd.DataFrame(rows)
    u["ratio"] = u["bclose"] / u["yclose"]
    us_exch = {"NMS", "NYQ", "NGM", "NCM", "PCX", "ASE", "BTS", "NasdaqGS", "NYSE", "NasdaqGM", "NYSEArca"}
    u["match"] = u["ratio"].between(0.85, 1.15) & u["exchange"].isin(us_exch) & ~u["base"].isin(NOT_STOCK)
    u["leveraged"] = u["base"].isin(LEVERAGED)
    OUT.mkdir(exist_ok=True)
    u.to_csv(OUT / "stock_perps_universe.csv", index=False)
    return u


def load_universe() -> pd.DataFrame:
    u = pd.read_csv(OUT / "stock_perps_universe.csv")
    return u[u["match"] & ~u["leveraged"]].reset_index(drop=True)


# ---------------------------------------------------------------- 抓資料

def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    u = load_universe()
    for tk in u["ticker"]:
        for name, params in (("1h", {"interval": "1h", "range": "730d", "includePrePost": "false"}),
                             ("1d", {"interval": "1d", "period1": 0, "period2": int(time.time())})):
            path = CACHE / f"{tk}_{name}.csv.gz"
            if path.exists():
                continue
            df = ru._get(tk, params)
            df.to_csv(path, index=False)
            print(tk, name, len(df), df["time"].min() if len(df) else "-", file=sys.stderr)
            time.sleep(0.6)

    def contract(sym):
        for iv in ("2h", "4h"):
            fetch_symbol(sym, iv)
        fetch_funding(sym)
        return sym

    with ThreadPoolExecutor(6) as ex:
        for s in ex.map(contract, u["symbol"]):
            print("binance", s, file=sys.stderr)


def yahoo(tk: str, name: str) -> pd.DataFrame:
    path = CACHE / f"{tk}_{name}.csv.gz"
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def stock_frame(tk: str, iv: str) -> pd.DataFrame:
    if iv == "1D":
        return yahoo(tk, "1d")
    h = yahoo(tk, "1h")
    return ru.session_bars(h, iv) if len(h) else h


def contract_frame(sym: str, iv: str) -> pd.DataFrame:
    df = fetch_symbol(sym, iv.lower())
    return df


# ---------------------------------------------------------------- 成本

def add_costs(tr: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """r_net = r（已含手續費、跳空）＋ 滑價（進出兩邊）＋ 資金費率。

    資金費率：交易時間落在合約有費率資料的期間就用實際費率，否則用該合約整段的平均每小時費率乘持有小時。
    全部假設整張抱到出場（實際 TP 會分批平掉，所以資金費率算多了，是保守估計）。
    """
    tr = tr.copy()
    meta = u.set_index("base")
    tr["risk_pct"] = np.maximum(LIVE_ATR * tr["atr_pct"], MIN_STOP_PCT)
    bps = tr["coin"].map(meta["adv_usdt"]).map(slippage_bps).to_numpy(float)
    tr["slip_r"] = -(bps / 1e4) * 2 / tr["risk_pct"]
    fr = np.zeros(len(tr))
    for coin, idx in tr.groupby("coin").indices.items():
        f = fetch_funding(meta.loc[coin, "symbol"])
        if f.empty:
            continue
        ft = _ns(f["time"]) // 1_000_000_000 * 1_000_000_000
        cum = np.concatenate([[0.0], np.cumsum(f["rate"].to_numpy(float))])
        hours = (f["time"].iloc[-1] - f["time"].iloc[0]).total_seconds() / 3600
        per_h = f["rate"].sum() / hours if hours > 0 else 0.0
        sub = tr.iloc[idx]
        a, b = _ns(sub["time"]), _ns(sub["exit_t"])
        inside = a >= ft[0]
        paid = np.where(inside, cum[np.searchsorted(ft, b, "right")] - cum[np.searchsorted(ft, a, "right")],
                        per_h * (b - a) / 3.6e12)
        fr[idx] = -(sub["side"].to_numpy() * paid) / sub["risk_pct"].to_numpy()
    tr["funding_r"] = fr
    tr["r_gross"] = tr["r"]
    tr["r"] = tr["r_gross"] + tr["slip_r"] + tr["funding_r"]
    return tr


# ---------------------------------------------------------------- 回測

def build(u: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in u.iterrows():
        for iv in ("4H", "2H", "1D"):
            got = ru.signals(row["base"], stock_frame(row["ticker"], iv), iv, "正股")
            if got is not None:
                parts.append(got)
        for iv in ("4H", "2H"):
            df = contract_frame(row["symbol"], iv)
            if len(df):
                got = ru.signals(row["base"], df, iv, "幣安合約")
                if got is not None:
                    parts.append(got)
    allt = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    return allt


def live_book(tr: pd.DataFrame) -> pd.DataFrame:
    """實盤規則：每個週期同幣同時一張；2H 和 4H 各自獨立（同方向可以各一張）。"""
    parts = [per_interval(tr[tr["interval"] == iv]) for iv in tr["interval"].unique()]
    return pd.concat(parts, ignore_index=True) if parts else tr


def stats(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"trades": 0}
    a, b = x["time"].min(), x["time"].max() + pd.Timedelta(days=1)
    acc = account(x, a, b)
    m_g, se_g, *_ = cluster_stats(x["r_gross"].to_numpy(), cluster_key(x))
    acc["exp_r_gross"] = m_g
    acc["t"] = acc["exp_r"] / acc["se_cluster"] if acc["se_cluster"] else np.nan
    for nm, s in (("long", 1), ("short", -1)):
        y = x[x["side"] == s]
        if len(y) >= 2:
            m, se, *_ = cluster_stats(y["r"].to_numpy(), cluster_key(y))
        else:
            m, se = (y["r"].mean() if len(y) else np.nan), np.nan
        acc[f"n_{nm}"], acc[f"{nm}_r"], acc[f"{nm}_se"] = len(y), m, se
    acc["cost_r"] = (x["slip_r"] + x["funding_r"]).mean()
    acc["start"] = a.date().isoformat()
    return acc


def random_trades(u: pd.DataFrame, allt: pd.DataFrame, n_per: int = 150, seed: int = 0) -> pd.DataFrame:
    """同一批標的、同樣出場與成本，隨機 K 棒進場（做多、做空各一份），只抽和策略同一段期間的 K 棒。"""
    rng = np.random.default_rng(seed)
    parts = []
    for _, row in u.iterrows():
        for src, iv, df in ([("正股", iv, stock_frame(row["ticker"], iv)) for iv in ("4H", "2H", "1D")]
                            + [("幣安合約", iv, contract_frame(row["symbol"], iv)) for iv in ("4H", "2H")]):
            if len(df) < 260:
                continue
            p = ru.VegasTrend().prepare(df.reset_index(drop=True)).reset_index(drop=True)
            arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
            t = pd.to_datetime(p["time"], utc=True)
            ok = np.flatnonzero((np.arange(len(p)) >= 200) & (np.arange(len(p)) < len(p) - 2) & np.isfinite(arr["atr"]))
            if len(ok) == 0:
                continue
            for d in (1, -1):
                idx = np.sort(rng.choice(ok, min(n_per, len(ok)), replace=False))
                ds = np.full(len(idx), d)
                r = ru.gap_sim(arr, idx, ds)
                ru.er.PLANS = {iv: ru.PLAN}
                s = ru.er.simulate(arr, idx, ds, iv, ru.FEE, 0.0)
                parts.append(pd.DataFrame({"source": src, "coin": row["base"], "interval": iv, "side": d, "r": r,
                                           "time": t.iloc[idx + 1].to_numpy(),
                                           "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                                           "atr_pct": arr["atr"][idx] / arr["close"][idx]}))
    rt = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        rt[c] = pd.to_datetime(rt[c], utc=True)
    return rt


def run() -> None:
    u = load_universe()
    allt = add_costs(build(u), u)
    allt.to_csv(OUT / "stock_perps_trades.csv.gz", index=False)
    rt = add_costs(random_trades(u, allt), u)
    rt.to_csv(OUT / "stock_perps_random.csv.gz", index=False)
    summarize(allt, rt)


def summarize(allt: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    global ADV
    ADV = load_universe().set_index("base")["adv_usdt"]
    rows = []
    for src, ivs, label in (("正股", ("4H", "2H"), "4H+2H"), ("正股", ("4H",), "4H"), ("正股", ("2H",), "2H"),
                            ("正股", ("1D",), "1D"), ("幣安合約", ("4H", "2H"), "4H+2H"),
                            ("幣安合約", ("4H",), "4H"), ("幣安合約", ("2H",), "2H")):
        tr = live_book(allt[(allt["source"] == src) & allt["interval"].isin(ivs)])
        r0 = rt[(rt["source"] == src) & rt["interval"].isin(ivs)]
        if tr.empty:
            continue
        r0 = r0[r0["time"] >= tr["time"].min()]
        base = {"source": src, "interval": label}
        rows.append({**base, "slice": "全部", **stats(tr)})
        rr = stats(r0)
        rows.append({**base, "slice": "隨機進場", **{k: rr.get(k) for k in ("trades", "exp_r", "se_cluster", "n_long",
                                                                          "long_r", "long_se", "n_short", "short_r",
                                                                          "short_se", "start")}})
        for y in sorted(tr["time"].dt.year.unique()):
            rows.append({**base, "slice": str(y), **stats(tr[tr["time"].dt.year == y])})
        if src == "正股" and label == "1D":
            for a, b, nm in ((None, 2010, "-2009"), (2010, 2018, "2010-17"), (2018, 2024, "2018-23"), (2024, None, "2024+")):
                yy = tr["time"].dt.year
                m = (yy >= (a or 0)) & (yy < (b or 9999))
                rows.append({**base, "slice": nm, **stats(tr[m])})
        liquid = tr["coin"].map(ADV) >= MIN_ADV
        rows.append({**base, "slice": "只看成交額 >= 5000 萬 U（實盤門檻）", **stats(tr[liquid])})
        if src == "幣安合約":
            cl = tr["closed"].astype(bool)
            rows.append({**base, "slice": "訊號在美股開市時段", **stats(tr[~cl])})
            rows.append({**base, "slice": "訊號在美股休市時段", **stats(tr[cl])})
        if src == "正股" and label != "1D":
            # 同一段期間（幣安合約上架後）正股的結果，和合約直接比
            c0 = allt[allt["source"] == "幣安合約"].groupby("coin")["time"].min()
            sub = tr[tr["coin"].isin(c0.index)]
            sub = sub[sub["time"] >= sub["coin"].map(c0)]
            rows.append({**base, "slice": "合約上架後（同標的）", **stats(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "stock_perps_summary.csv", index=False)
    cols = ["source", "interval", "slice", "start", "trades", "win_rate", "exp_r_gross", "exp_r", "se_cluster", "t",
            "n_long", "long_r", "long_se", "n_short", "short_r", "short_se", "cost_r", "sum_r_per_year", "max_dd_pct"]
    with pd.option_context("display.width", 300, "display.max_rows", 400):
        print(out[[c for c in cols if c in out]].round(3).to_string())
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "universe":
        u = universe()
        print(u[u["match"]].to_string())
    elif cmd == "fetch":
        fetch()
    elif cmd == "run":
        run()
    elif cmd == "summary":
        allt = pd.read_csv(OUT / "stock_perps_trades.csv.gz")
        rt = pd.read_csv(OUT / "stock_perps_random.csv.gz")
        for d in (allt, rt):
            for c in ("time", "exit_t"):
                d[c] = pd.to_datetime(d[c], utc=True)
        summarize(allt, rt)


if __name__ == "__main__":
    main()
