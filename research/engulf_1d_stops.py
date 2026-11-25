"""日線吞沒：停損倍數、到期天數 4 組（補登記 2，reports/signals/prereg_engulf_1d_20261122.md）。

python -m research.engulf_1d_stops   → output/cv_ledgers/engulf_1d_stops.pkl、/mnt/project-files/signals/engulf_opt_stops_20261122.csv

基準 LIVE = 實盤規則（時點前 100、不設成交額門檻、剛交叉 5 根＋吞沒、掛單實體一半 3 天、停損 1.5 ATR、1R 全出、60 根到期）。
S10 / S20 = 停損 1.0 / 2.0 ATR；T30 / T90 = 到期 30 / 90 根。其他全部照 research.engulf_1d_opt。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research.cv_export_old_patterns as cx
import research.threads_strategies as ts
from research.engulf_1d_eval import OUT, RISK, TAG, _cfg
from research.experiments import log_experiment, trials
from research.time_cv import CVConfig, FlipRule, evaluate_family, ledger_stats, summary_row
from research.validation_gate import GateConfig, evaluate

LED = cx.OUT / "engulf_1d_stops.pkl"
VARIANTS = {"LIVE": (1.5, 60), "S10": (1.0, 60), "S20": (2.0, 60), "T30": (1.5, 30), "T90": (1.5, 90)}
NEW = ["S10", "S20", "T30", "T90"]


def export() -> pd.DataFrame:
    ts.EXITS = {"tp1": ((1.0,), (1.0,), 0.0)}
    mem, pit = cx.pit_index()
    parts = []
    for n, sym in enumerate(cx.universe_symbols(mem)):
        df = cx._frame(sym, "1D")
        if df is None:
            continue
        sigs = ts.engulf_signals(df, True)
        for name, (k, bars) in VARIANTS.items():
            d2 = df.copy()
            d2["atr"] = df["atr"] * (k / 1.5)       # build_orders 的 atr 停損 = 1.5 x atr；tp1 沒有追蹤，atr 只用在停損
            ts.MAX_BARS["1D"] = bars
            r = ts.run(d2, ts.build_orders(d2, sigs, "half", "atr", 3), "1D")
            if r.empty:
                continue
            r = r.copy()
            r["entry_bar"], r["busy_i"] = r["start"], r["start"] - 1
            parts.append(cx._ledger_from_orders(r, sym, "1D", ["tp1"], lambda tt, ex, s=name: s, True))
        if n % 50 == 0:
            cx.log(f"stops {n}")
    ts.MAX_BARS["1D"] = 60
    tr = pd.concat(parts, ignore_index=True)
    tr = tr[(tr["time"] >= cx.START) & cx.in_pit(pit, tr["symbol"], tr["time"])]
    tr = cx.one_at_a_time(tr, ["config", "symbol"])
    led = cx.finish(tr)
    led.to_pickle(LED)
    return led


def evaluate_all(led: pd.DataFrame) -> pd.DataFrame:
    from research.cv_all import external_live
    for c in ("time", "exit_t"):
        led[c] = pd.to_datetime(led[c], utc=True)
    vegas = external_live().copy()
    for c in ("time", "exit_t"):
        vegas[c] = pd.to_datetime(vegas[c], utc=True)
    if trials(count_configs=True) < 1939:
        log_experiment(family="engulf_1d_opt", hypothesis="日線吞沒停損 1/2 ATR、到期 30/90 根（補登記 2）", params={"configs": NEW},
                       data_source="binance um 1h->1D, PIT top100", period="2020-2026", verdict="info",
                       report="signals/report_engulf_1d_opt_20261122.md", notes="configs=4")
    n_trials = trials(count_configs=True)
    rows = []
    for c in ["LIVE", *NEW]:
        g = _cfg(led, c)
        y = g["time"].dt.year
        res = evaluate(g, baseline=vegas, cfg=GateConfig(n_trials=n_trials, risk=RISK))
        ch = res["checks"]
        row = {"config": c, "trades": len(g), "mean_r": g["r_net"].mean(), "win": (g["r_net"] > 0).mean(),
               "r_2020_23": g[y < 2024]["r_net"].mean(), "r_2024p": g[y >= 2024]["r_net"].mean(),
               "t_strict": ch["cluster_t"]["value"], "t_required": ch["cluster_t"]["threshold"], "dsr": ch["deflated_sharpe"]["value"],
               "med_stop_pct": g["risk_pct"].median() * 100, "passed": res["passed"]}
        if c != "LIVE":
            pair = led[led["config"].isin(["LIVE", c])]
            fam = evaluate_family(pair, CVConfig(), FlipRule(), baseline="LIVE")
            vb = fam["vs_baseline"] or {}
            row.update({"wf_years_better": vb.get("wf_years_better_frac"), "cpcv_better": vb.get("cpcv_frac_better"),
                        "flip": fam["flip"]})
            log_experiment(family="engulf_1d_opt", hypothesis=f"validation_gate {c}", params={"config": c, "n_trials": n_trials},
                           data_source="binance um 1h->1D, PIT top100", period="2020-2026", trades=len(g),
                           exp_r=round(row["mean_r"], 4), t_stat=round(row["t_strict"], 3),
                           verdict="pass" if res["passed"] else "fail", report="signals/report_engulf_1d_opt_20261122.md",
                           notes=f"{res['verdict']}; configs=0")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"engulf_opt_stops_{TAG}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string())
    return out


if __name__ == "__main__":
    evaluate_all(export())
