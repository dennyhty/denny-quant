"""vegas_timeframes 的統計：每個週期的期望值、群集 t、bootstrap、逐年、和 2H+4H 的相關性、加進組合後的變化、驗證關卡。"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from research.validation_gate import GateConfig, cluster_t_values, evaluate
from research.vegas_improve import cl_stats
from research.vegas_timeframes import OUT, RISK, SPLIT, TF

BASE = ("2H", "4H")
N_BOOT = 2000


def _load(name: str) -> pd.DataFrame:
    tr = pd.read_csv(OUT / f"{name}.csv.gz")
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    return tr


def day_side(tr: pd.DataFrame) -> np.ndarray:
    return (tr["time"].dt.floor("D").astype(str) + tr["side"].astype(str)).to_numpy()


def block_boot(tr: pd.DataFrame, r: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """以月為區塊的 bootstrap：平均 R 的 5% 分位數、平均 > 0 的比例。"""
    m = tr["time"].dt.strftime("%Y-%m").to_numpy()
    g = pd.DataFrame({"m": m, "r": r}).groupby("m")["r"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    means = s[pick].sum(1) / c[pick].sum(1)
    return float(np.quantile(means, 0.05)), float((means > 0).mean())


def daily_pnl(tr: pd.DataFrame, rcol: str = "r_net") -> pd.Series:
    return pd.Series(tr[rcol].to_numpy() * RISK, index=tr["exit_t"].dt.floor("D").to_numpy()).groupby(level=0).sum()


def account(tr: pd.DataFrame, start, end) -> dict:
    d = daily_pnl(tr).reindex(pd.date_range(start, end, freq="D"), fill_value=0.0)
    eq = (1 + d).cumprod()
    years = len(d) / 365.25
    return {"simple_per_year_pct": d.sum() / years * 100, "sharpe": d.mean() / d.std() * np.sqrt(365) if d.std() else np.nan,
            "max_dd_pct": (eq / eq.cummax() - 1).min() * 100}


def capped(tr: pd.DataFrame, k: int = 3) -> float:
    """每天同方向最多算前 k 筆（訊號擠在同一天時不重複押）。"""
    t = tr.sort_values("time")
    keep = t.groupby(day_side(t)).cumcount() < k
    return float(t.loc[keep.to_numpy(), "r_net"].mean())


def summary_row(tf: str, tr: pd.DataFrame, base_daily: pd.Series) -> dict:
    r = tr["r_net"].to_numpy()
    m, se = cl_stats(r, day_side(tr))
    ts = cluster_t_values(tr.assign(r_net=r), "r_net") if len(tr) > 10 else {}
    q5, ppos = block_boot(tr, r) if len(tr) > 10 else (np.nan, np.nan)
    is_ = (tr["time"] < SPLIT).to_numpy()
    yr = tr.groupby(tr["exit_t"].dt.year)["r_net"].mean()
    d = daily_pnl(tr)
    idx = base_daily.index.union(d.index)
    corr = (d.reindex(idx, fill_value=0).corr(base_daily.reindex(idx, fill_value=0)) if len(d) > 10 else np.nan)
    hold_days = (tr["exit_t"] - tr["time"]).dt.total_seconds().median() / 86400
    return {"tf": tf, "trades": len(tr), "coins": tr["symbol"].nunique(), "first": tr["time"].min().date(),
            "per_month": len(tr) / max(1, (tr["time"].max() - tr["time"].min()).days / 30.4),
            "win": (r > 0).mean(), "r_gross": tr["r"].mean(), "cost_r": (tr["r_net"] - tr["r"]).mean(),
            "r_net": m, "se": se, "t": m / se if se else np.nan, "t_strict": min(ts.values()) if ts else np.nan,
            "r_cap3": capped(tr), "boot_q5": q5, "boot_pos": ppos,
            "r_2020_23": r[is_].mean() if is_.any() else np.nan, "n_2020_23": int(is_.sum()),
            "r_2024_": r[~is_].mean() if (~is_).any() else np.nan, "n_2024_": int((~is_).sum()),
            "years_pos": f"{(yr > 0).sum()}/{len(yr)}", "long_r": r[tr["side"].to_numpy() > 0].mean(),
            "short_r": r[tr["side"].to_numpy() < 0].mean(), "stop_pct": tr["risk_pct"].median() * 100,
            "hold_days": hold_days, "corr_2h4h": corr}


def report() -> None:
    tr = _load("tf_trades")
    rnd = _load("tf_random")
    base = tr[tr["interval"].isin(BASE)]
    base_daily = daily_pnl(base)
    start, end = tr["time"].min().floor("D"), tr["exit_t"].max().floor("D")
    rows, yrs, gate, combo = [], [], {}, []
    combo.append({"book": "2H+4H（現在）", **account(base, start, end)})
    cfg = GateConfig()
    for tf in TF:
        g = tr[tr["interval"] == tf]
        if g.empty:
            continue
        rows.append(summary_row(tf, g, base_daily))
        for y, gy in g.groupby(g["exit_t"].dt.year):
            yrs.append({"tf": tf, "year": y, "trades": len(gy), "r_net": gy["r_net"].mean()})
        if tf not in BASE:
            combo.append({"book": f"2H+4H+{tf}", **account(pd.concat([base, g]), start, end)})
        rg = rnd[rnd["interval"] == tf]
        res = evaluate(g, baseline=None if tf in BASE else base, random_baseline=rg if len(rg) else None, cfg=cfg)
        gate[tf] = {"verdict": res["verdict"], "info": res["info"],
                    "checks": {k: {kk: c[kk] for kk in ("label", "value", "threshold", "status")} for k, c in res["checks"].items()}}
        rr = rg["r_net"].to_numpy()
        rows[-1]["random_r"] = rr.mean() if len(rr) else np.nan
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "tf_summary.csv", index=False)
    y = pd.DataFrame(yrs).pivot(index="tf", columns="year", values="r_net").reindex([t for t in TF if t in set(tr["interval"])])
    yn = pd.DataFrame(yrs).pivot(index="tf", columns="year", values="trades").reindex(y.index)
    y.to_csv(OUT / "tf_yearly.csv")
    yn.to_csv(OUT / "tf_yearly_n.csv")
    c = pd.DataFrame(combo)
    c.to_csv(OUT / "tf_combo.csv", index=False)
    (OUT / "tf_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1, default=str))
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(s.round(3).to_string())
        print(y.round(3).to_string())
        print(yn.to_string())
        print(c.round(2).to_string())
        for tf, gres in gate.items():
            print(tf, gres["verdict"])
            print("   ", {k: (round(v["value"], 3) if isinstance(v["value"], float) else v["value"], v["status"])
                         for k, v in gres["checks"].items()})
    print(f"n_trials={cfg.n_trials}", file=sys.stderr)
