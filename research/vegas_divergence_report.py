"""research.vegas_divergence 的比較表（python -m research.vegas_divergence report）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments import log_experiment, required_t, trials
from research.pit_backtest import book
from research.stop_research import cluster_key, cluster_stats
from research.validation_gate import GateConfig, cluster_t_values, evaluate
from research.vegas_divergence import CONDS, MAIN_W, OUT, SPLIT, WINDOWS

RISK = 0.003
T_NEED = 3.0  # 18 組 Bonferroni（雙尾 5%）


def daily(tr: pd.DataFrame, days: pd.DatetimeIndex, w: np.ndarray | None = None) -> pd.Series:
    w = np.ones(len(tr)) if w is None else w
    d = pd.Series(tr["r_net"].to_numpy() * w * RISK, index=tr["exit_t"].dt.floor("D")).groupby(level=0).sum()
    return d.reindex(days, fill_value=0.0)


def perf(d: pd.Series) -> dict:
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    return {"yr_pct": d.sum() / yrs * 100, "sharpe": d.mean() / d.std() * np.sqrt(365),
            "dd_pct": (eq / eq.cummax() - 1).min() * 100}


def diff(tr: pd.DataFrame, flag: np.ndarray) -> tuple[float, float]:
    """有條件 − 沒條件的每筆 R 差和 t（同日同向一群）：對 flag 去中心後的迴歸係數。"""
    r = tr["r_net"].to_numpy()
    if flag.sum() < 5 or (~flag).sum() < 5:
        return np.nan, np.nan
    m1, m0 = r[flag].mean(), r[~flag].mean()
    resid = r - np.where(flag, m1, m0)
    x = flag - flag.mean()
    g = pd.Series(x * resid).groupby(cluster_key(tr)).sum().to_numpy()
    se = np.sqrt((g ** 2).sum()) / (x ** 2).sum()
    return m1 - m0, (m1 - m0) / se


def report() -> None:
    tr = pd.read_csv(OUT / "vd_trades.csv.gz", parse_dates=["time", "exit_t"])
    tr["all"] = True
    base = book(tr, "all")
    days = pd.date_range(base["time"].min().floor("D"), base["exit_t"].max().floor("D"), freq="D", tz="UTC")
    d0 = daily(base, days)
    rows, yrows, gates = [], [], []
    p0 = perf(d0)
    ts0 = cluster_t_values(base, "r_net")
    rows.append({"cond": "base", "w": "", "use": "實盤維加斯", "trades": len(base), "flag_share": 1.0,
                 "exp_r": base["r_net"].mean(), "win": (base["r_net"] > 0).mean(), "t_min": min(ts0.values()),
                 "is_r": base.loc[base["time"] < SPLIT, "r_net"].mean(),
                 "oos_r": base.loc[base["time"] >= SPLIT, "r_net"].mean(), **p0})
    for y, g in base.groupby(base["exit_t"].dt.year):
        yrows.append({"cond": "base", "year": y, "trades": len(g), "exp_r": g["r_net"].mean()})
    verdicts = {}
    for cond in CONDS:
        for w in WINDOWS:
            col = f"{cond}_w{w}"
            fl = base[col].astype(bool).to_numpy()
            is_ = (base["time"] < SPLIT).to_numpy()
            dif, t = diff(base, fl)
            dif_is, t_is = diff(base[is_], fl[is_])
            dif_oos, t_oos = diff(base[~is_], fl[~is_])
            by_iv = {}
            for iv in ("4H", "2H"):
                m = (base["interval"] == iv).to_numpy()
                by_iv[f"diff_{iv}"], by_iv[f"t_{iv}"] = diff(base[m], fl[m])
            common = {"cond": cond, "w": w, "flag_share": fl.mean(), "r_flag": base.loc[fl, "r_net"].mean(),
                      "r_noflag": base.loc[~fl, "r_net"].mean(), "diff": dif, "t_diff": t, "diff_is": dif_is,
                      "t_is": t_is, "diff_oos": dif_oos, "t_oos": t_oos, **by_iv}
            # filter：只做有條件的單（重新跑帳本，被濾掉的單不佔位子）
            fb = book(tr.assign(_m=tr[col].astype(bool)), "_m")
            df_ = daily(fb, days)
            ts = cluster_t_values(fb, "r_net") if len(fb) > 30 else {"x": np.nan}
            pf = perf(df_)
            rows.append({**common, "use": "filter", "trades": len(fb), "exp_r": fb["r_net"].mean(),
                         "win": (fb["r_net"] > 0).mean(), "t_min": min(ts.values()),
                         "is_r": fb.loc[fb["time"] < SPLIT, "r_net"].mean(),
                         "oos_r": fb.loc[fb["time"] >= SPLIT, "r_net"].mean(), "corr_base": df_.corr(d0), **pf})
            # size2x：基準帳本，有條件的單風險加倍
            ds_ = daily(base, days, np.where(fl, 2.0, 1.0))
            ps = perf(ds_)
            rows.append({**common, "use": "size2x", "trades": len(base), "exp_r": np.nan, "win": np.nan,
                         **ps, "corr_base": ds_.corr(d0)})
            if w == MAIN_W:
                ok = (t >= T_NEED) and dif_is > 0 and dif_oos > 0
                verdicts[(cond, "filter")] = "pass" if ok and pf["sharpe"] > p0["sharpe"] else "fail"
                verdicts[(cond, "size2x")] = "pass" if ok and ps["sharpe"] > p0["sharpe"] else "fail"
                for y, g in fb.groupby(fb["exit_t"].dt.year):
                    yrows.append({"cond": cond, "year": y, "trades": len(g), "exp_r": g["r_net"].mean()})
                if len(fb) >= 30:
                    nb = [book(tr.assign(_m=tr[f"{cond}_w{x}"].astype(bool)), "_m") for x in WINDOWS if x != w]
                    res = evaluate(fb, baseline=base, cfg=GateConfig(), rcol="r_net", neighbors=nb)
                    gates.append({"cond": cond, "verdict": res["verdict"],
                                  **{k: c["status"] for k, c in res["checks"].items()}})
    s = pd.DataFrame(rows)
    y = pd.DataFrame(yrows).pivot(index="cond", columns="year", values="exp_r")
    g = pd.DataFrame(gates)
    s.to_csv(OUT / "vd_summary.csv", index=False)
    y.to_csv(OUT / "vd_yearly.csv")
    g.to_csv(OUT / "vd_gate.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 40, "display.max_rows", 200):
        cols = ["cond", "w", "use", "trades", "flag_share", "r_flag", "r_noflag", "diff", "t_diff", "diff_is",
                "diff_oos", "t_4H", "t_2H", "exp_r", "oos_r", "yr_pct", "sharpe", "dd_pct"]
        print(s[cols].round(3).to_string())
        print(y.round(3).to_string())
        print(g.to_string())
    print("verdicts", verdicts)
    n_tr = trials(count_configs=True)
    print(f"ledger trials before {n_tr}, +36 → required t {required_t(n_tr + 36):.2f}")
    return s, verdicts


def log(s: pd.DataFrame, verdicts: dict) -> None:
    for (cond, use), v in verdicts.items():
        r = s[(s["cond"] == cond) & (s["w"] == MAIN_W) & (s["use"] == use)].iloc[0]
        log_experiment(family="vegas_divergence", hypothesis=f"維加斯共振（{use}）：{cond}",
                       params={"cond": cond, "use": use, "w": MAIN_W}, data_source="binance", period="2020-2026",
                       trades=int(r["trades"]), exp_r=round(float(r["exp_r"]), 3) if use == "filter" else "",
                       t_stat=round(float(r["t_diff"]), 2), sharpe=round(float(r["sharpe"]), 2), verdict=v,
                       report="signals/report_vegas_divergence_20261122.md",
                       notes="configs=1; t_stat = flagged minus unflagged R; W 6/24 neighbors in separate row")
    log_experiment(family="vegas_divergence", hypothesis="維加斯共振：W=6、24 參數鄰居（只當高原檢查）",
                   params={"windows": [6, 24], "conds": list(CONDS), "uses": ["filter", "size2x"]},
                   data_source="binance", period="2020-2026", verdict="info",
                   report="signals/report_vegas_divergence_20261122.md", notes="configs=18")
