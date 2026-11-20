"""擴大幣種範圍：實盤維加斯（2H+4H）在不同市值名次、不同成交額的幣上表現如何（只做研究，不改實盤）。

    python -m grid_trading.pit_universe          # 每月成交額表（給全部合約清單）
    python -m research.universe_buckets fetch     # CoinMarketCap 每月市值排名快照 → data/cmc_hist/
    python -m research.universe_buckets trades    # 全部幣安 USDT 永續逐筆交易 → output/ub_trades.csv.gz
    python -m research.universe_buckets report    # 分組帳本、年度、驗證關卡 → output/ub_*.csv

- 幣池全部用幣安 USDT 永續合約（含已下架），不經派網過濾。
- 市值名次是時點的：交易所在月份用「上個月最後一天」的 CoinMarketCap 排名（扣掉穩定幣後重排）。
  幣安代號去掉 1000 / 1000000 / 1M 前綴後對 CMC 代號；同代號多個幣取名次最前的。
- 成交額是時點的：訊號 K 棒當下的近 30 天日均成交額（和交易程式 --min-adv 同一個算法）。
- 每一組各自當一本帳（同週期同幣同時一張，2H、4H 合併），就像實盤只交易那一組。
- 規則同實盤：1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR、止損 1.5 ATR，
  沒有 240 根持倉上限（PR #73）。成本：手續費每邊 0.05%、資金費率、依成交額分級的滑價（grid_trading/costs.py），
  另做一組「成交額 < 5,000 萬的幣滑價加倍」的壓力測試。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import research.entry_research as er
import research.pit_backtest as pb
from grid_trading.costs import apply_costs
from grid_trading.pit_universe import monthly_volume
from research.stop_research import cluster_key, cluster_stats
from research.validation_gate import GateConfig, cluster_t_values, evaluate

CMC_HIST = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
CMC_DIR = Path("data/cmc_hist")
OUT = Path("output")
START = pd.Timestamp("2020-01-01", tz="UTC")


# ---------------------------------------------------------------- 市值排名

def fetch_cmc(first: str = "2020-01", last: str | None = None) -> None:
    last = last or str(pd.Timestamp.now().to_period("M"))
    s = requests.Session()
    CMC_DIR.mkdir(parents=True, exist_ok=True)
    for m in pd.period_range(first, last, freq="M"):
        p = CMC_DIR / f"{m}.json"
        if p.exists():
            continue
        d = (m.start_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for _ in range(4):
            try:
                data = s.get(CMC_HIST, params={"date": d, "start": 1, "limit": 500, "convertId": 2781}, timeout=60).json()["data"]
                rows = [{"rank": c["cmcRank"], "symbol": c["symbol"], "slug": c["slug"],
                         "stable": bool({"stablecoin", "fiat-stablecoin"} & set(c.get("tags") or []))} for c in data]
                p.write_text(json.dumps(rows))
                break
            except Exception as e:  # noqa: BLE001
                print(m, e, file=sys.stderr)
                time.sleep(3)
        time.sleep(0.5)


def cmc_ranks() -> pd.DataFrame:
    """month, cmc_symbol, rank（扣掉穩定幣後的名次，1 起算）。"""
    rows = []
    for p in sorted(CMC_DIR.glob("*.json")):
        d = pd.DataFrame(json.loads(p.read_text()))
        d = d[~d["stable"]].sort_values("rank").drop_duplicates("symbol")
        d["rank"] = np.arange(1, len(d) + 1)
        d["month"] = p.stem
        rows.append(d[["month", "symbol", "rank"]])
    return pd.concat(rows, ignore_index=True)


def base_coin(symbol: str) -> list[str]:
    c = symbol[:-4] if symbol.endswith("USDT") else symbol
    out = [c]
    for pre in ("1000000", "1000", "1M"):
        if c.startswith(pre) and len(c) > len(pre):
            out.append(c[len(pre):])
    return out


def attach_rank(tr: pd.DataFrame) -> pd.DataFrame:
    rk = cmc_ranks()
    look = {(m, s): r for m, s, r in zip(rk["month"], rk["symbol"], rk["rank"])}
    months = tr["time"].dt.strftime("%Y-%m")
    ranks = []
    for sym, m in zip(tr["symbol"], months):
        r = np.nan
        for c in base_coin(sym):
            if (m, c) in look:
                r = look[(m, c)]
                break
        ranks.append(r)
    tr = tr.copy()
    tr["mcap_rank"] = ranks
    return tr


# ---------------------------------------------------------------- 交易

def _work(args):
    er.MAX_BARS = 10**9  # 沒有持倉根數上限（實盤 PR #73 起）
    for k in range(6):  # data.binance.vision 偶爾斷線，重試
        try:
            return pb._work(args)
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    print(f"放棄 {args[0]} {args[1]}", file=sys.stderr)
    return None


def trades(fee: float, workers: int) -> pd.DataFrame:
    table = monthly_volume()
    syms = sorted(table.loc[table["month"] >= "2019-12", "symbol"].unique())
    print(f"{len(syms)} 個合約", file=sys.stderr)
    jobs = [(s, iv, fee) for iv in ("4H", "2H") for s in syms]
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(_work, jobs, chunksize=4)):
            if got is not None:
                parts.append(got)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr)
    tr = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    tr = tr[tr["time"] >= START]
    tr["in_snapshot"] = tr["symbol"].isin(pb.snapshot_symbols())
    tr = apply_costs(tr)
    return attach_rank(tr)


# ---------------------------------------------------------------- 分組與報告

def buckets(tr: pd.DataFrame) -> dict[str, np.ndarray]:
    rank, adv = tr["mcap_rank"].to_numpy(), tr["adv_usdt"].to_numpy()
    top = (rank <= 100)
    liq = adv >= 5e7
    return {
        "現行實盤：固定名單 ∩ 成交額≥5000萬": tr["in_snapshot"].to_numpy() & liq,
        "新基準：幣安當月市值前100 ∩ ≥5000萬": top & liq,
        "市值前100（不設成交額門檻）": top,
        "市值 101-200": (rank > 100) & (rank <= 200),
        "市值 201-300": (rank > 200) & (rank <= 300),
        "市值 101-200 ∩ ≥5000萬": (rank > 100) & (rank <= 200) & liq,
        "市值 201-300 ∩ ≥5000萬": (rank > 200) & (rank <= 300) & liq,
        "成交額 ≥5000萬（不看市值）": liq,
        "成交額 1000萬-5000萬": (adv >= 1e7) & (adv < 5e7),
        "成交額 500萬-1000萬": (adv >= 5e6) & (adv < 1e7),
        "市值前100 ∩ 1000萬-5000萬": top & (adv >= 1e7) & (adv < 5e7),
        "市值前300 ∩ ≥5000萬": (rank <= 300) & liq,
    }


def stats(b: pd.DataFrame, rcol: str) -> dict:
    r = b[rcol].to_numpy()
    if len(r) < 2:
        return {"trades": len(r)}
    mean, se, *_ = cluster_stats(r, cluster_key(b))
    ts = cluster_t_values(b, rcol)
    day = b.groupby(b["exit_t"].dt.floor("D"))[rcol].sum() * pb.RISK
    idx = pd.date_range(b["time"].min().floor("D"), b["exit_t"].max().floor("D"), freq="D", tz="UTC")
    day = day.reindex(idx, fill_value=0.0)
    eq = (1 + day).cumprod()
    yrs = (idx[-1] - idx[0]).days / 365.25
    return {"trades": len(r), "coins": b["symbol"].nunique(), "per_month": len(r) / (yrs * 12), "win": (r > 0).mean(),
            "mean_r": mean, "t_day": mean / se if se else np.nan, "t_strict": min(ts.values()),
            "sharpe": day.mean() / day.std() * np.sqrt(365) if day.std() else np.nan,
            "simple_pct_yr": r.sum() * pb.RISK * 100 / yrs, "max_dd_pct": ((eq / eq.cummax()) - 1).min() * 100,
            "long_r": r[b["side"].to_numpy() > 0].mean(), "short_r": r[b["side"].to_numpy() < 0].mean(),
            "slip_r": b["slip_r"].mean(), "is_r": b.loc[b["time"] < "2024-01-01", rcol].mean(),
            "oos_r": b.loc[b["time"] >= "2024-01-01", rcol].mean()}


def report() -> None:
    tr = pd.read_csv(OUT / "ub_trades.csv.gz", parse_dates=["time", "exit_t"])
    tr["r_stress"] = tr["r_net"] + np.where(tr["adv_usdt"] < 5e7, tr["slip_r"], 0.0)  # 小幣滑價加倍
    print(f"有市值名次的交易比例 {tr['mcap_rank'].notna().mean():.1%}", file=sys.stderr)
    rows, yrows, gates = [], [], []
    cfg = GateConfig()
    for name, mask in buckets(tr).items():
        b = pb.book(tr.assign(_m=mask), "_m")
        for rcol, label in (("r", "未扣成本"), ("r_net", "扣成本"), ("r_stress", "扣成本＋小幣滑價加倍")):
            rows.append({"bucket": name, "costs": label, **stats(b, rcol)})
        for y, g in b.groupby(b["exit_t"].dt.year):
            yrows.append({"bucket": name, "year": y, "trades": len(g), "mean_r_net": g["r_net"].mean(),
                          "sum_r_net": g["r_net"].sum()})
        res = evaluate(b, cfg=cfg, rcol="r_net")
        gates.append({"bucket": name, "verdict": res["verdict"],
                      **{k: c["status"] for k, c in res["checks"].items()}})
    s = pd.DataFrame(rows)
    y = pd.DataFrame(yrows).pivot(index="bucket", columns="year", values="mean_r_net")
    g = pd.DataFrame(gates)
    s.to_csv(OUT / "ub_summary.csv", index=False)
    y.to_csv(OUT / "ub_yearly.csv")
    g.to_csv(OUT / "ub_gate.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 200,
                           "display.max_colwidth", 40):
        print(s.round(3).to_string())
        print(y.round(3).to_string())
        print(g.to_string())
        print(f"gate required t = {cfg.n_trials} trials")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "trades", "report"])
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.step == "fetch":
        fetch_cmc()
    elif a.step == "trades":
        trades(a.fee, a.workers).to_csv(OUT / "ub_trades.csv.gz", index=False)
    else:
        report()


if __name__ == "__main__":
    main()
