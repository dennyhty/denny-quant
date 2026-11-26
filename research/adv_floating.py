"""成交額門檻：固定 vs 浮動（只做研究，不改實盤）。

    python -m research.universe_buckets trades    # 先產生 output/ub_trades.csv.gz（全部幣安 USDT 永續逐筆交易，時點幣池）
    python -m research.adv_floating               # 比較各種門檻 → output/adv_floating_*.csv

幣池固定用「幣安當月市值前 100」（和實盤一樣），只比較成交額門檻怎麼定：
- 固定：不設、10M、20M、30M（現行）、40M、50M
- 相對排名：上個月全部幣安 USDT 永續的成交額排名在前 N 名才做（N = 30..100），量縮時門檻自動降低
- 隨大盤縮放：門檻 = 基準 × 大盤量比，大盤量比 = 上個月全部永續成交額 ÷ 再前 12 個月的中位數（不到 6 個月 = 1）
  基準取 30M 和 50M 各一組；大盤量比限制在 0.4 ~ 2.5 倍

全部只用訊號當下已知的資訊（上個月的月成交額、訊號當下近 30 天日均成交額）。成本和滑價同 universe_buckets。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import research.pit_backtest as pb
from grid_trading.pit_universe import monthly_volume
from research.universe_buckets import OUT, stats
from research.validation_gate import GateConfig, evaluate


def market_context(tr: pd.DataFrame) -> pd.DataFrame:
    """每筆交易：上個月的大盤量比、自己上個月在全部永續中的成交額名次。"""
    m = monthly_volume()
    m = m[m["days"] >= 20].copy()  # 不完整的月份不算
    m["period"] = pd.PeriodIndex(m["month"], freq="M")
    m["vol_rank"] = m.groupby("period")["quote_volume"].rank(ascending=False, method="first")
    tot = m.groupby("period")["quote_volume"].sum().sort_index()
    base = tot.rolling(12, min_periods=6).median().shift(1)  # 不含自己那個月
    ratio = (tot / base).clip(0.4, 2.5)
    ctx = pd.DataFrame({"mkt_total": tot, "mkt_ratio": ratio}).reset_index()
    tr = tr.copy()
    tr["period"] = tr["time"].dt.tz_localize(None).dt.to_period("M")
    prev = tr["period"] - 1
    tr["prev"] = prev
    tr = tr.merge(ctx.rename(columns={"period": "prev"}), on="prev", how="left")
    tr = tr.merge(m[["symbol", "period", "vol_rank"]].rename(columns={"period": "prev"}), on=["symbol", "prev"], how="left")
    tr["mkt_ratio"] = tr["mkt_ratio"].fillna(1.0)
    tr["vol_rank"] = tr["vol_rank"].fillna(9999)
    return tr


def configs(tr: pd.DataFrame) -> dict[str, np.ndarray]:
    adv, rank = tr["adv_usdt"].to_numpy(), tr["vol_rank"].to_numpy()
    ratio = tr["mkt_ratio"].to_numpy()
    top = (tr["mcap_rank"] <= 100).to_numpy()
    out = {"不設門檻": top}
    for f in (10, 20, 30, 40, 50):
        out[f"固定 {f}M"] = top & (adv >= f * 1e6)
    for n in (30, 40, 50, 60, 70, 80, 100):
        out[f"相對排名 前{n}名"] = top & (rank <= n)
    for base in (30, 50):
        out[f"隨大盤縮放 基準{base}M"] = top & (adv >= base * 1e6 * ratio)
    out["隨大盤縮放 基準20M"] = top & (adv >= 20e6 * ratio)
    return out


def main() -> None:
    tr = pd.read_csv(OUT / "ub_trades.csv.gz", parse_dates=["time", "exit_t"])
    tr = market_context(tr)
    quiet = tr["mkt_ratio"] < 0.8
    print(f"交易 {len(tr)} 筆；量縮期（大盤量比<0.8）佔 {quiet.mean():.1%}", file=sys.stderr)
    rows, yrows, qrows, gates = [], [], [], []
    cfg = GateConfig()
    for name, mask in configs(tr).items():
        b = pb.book(tr.assign(_m=mask), "_m")
        rows.append({"config": name, **stats(b, "r_net")})
        for y, g in b.groupby(b["exit_t"].dt.year):
            yrows.append({"config": name, "year": y, "trades": len(g), "mean_r": g["r_net"].mean()})
        q, n = b[b["mkt_ratio"] < 0.8], b[b["mkt_ratio"] >= 0.8]
        qrows.append({"config": name, "quiet_trades": len(q), "quiet_mean_r": q["r_net"].mean(),
                      "normal_trades": len(n), "normal_mean_r": n["r_net"].mean(),
                      "quiet_per_month_share": len(q) / max(len(b), 1)})
        res = evaluate(b, cfg=cfg, rcol="r_net")
        gates.append({"config": name, "verdict": res["verdict"]})
    s = pd.DataFrame(rows)
    y = pd.DataFrame(yrows).pivot(index="config", columns="year", values="mean_r").reindex(s["config"])
    q = pd.DataFrame(qrows)
    g = pd.DataFrame(gates)
    s.to_csv(OUT / "adv_floating_summary.csv", index=False)
    y.to_csv(OUT / "adv_floating_yearly.csv")
    q.to_csv(OUT / "adv_floating_quiet.csv", index=False)
    cols = ["config", "trades", "per_month", "mean_r", "t_day", "win", "sharpe", "max_dd_pct", "simple_pct_yr", "oos_r"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 30):
        print(s[cols].round(3).to_string(index=False))
        print(y.round(3).to_string())
        print(q.round(3).to_string(index=False))
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
