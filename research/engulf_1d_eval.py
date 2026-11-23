"""評估 research/engulf_1d_opt.py 的帳本（事先登記見 reports/signals/prereg_engulf_1d_20261122.md）。

python -m research.engulf_1d_opt evaluate

需要 output/cv_ledgers/engulf_1d_opt.pkl 和 output/pit_trades.csv.gz（實盤維加斯時點帳本，python -m research.pit_backtest）。
輸出：/mnt/project-files/signals/engulf_opt_{gate,yearly,cv,combo}_20261122.csv、output/engulf_opt_20261122.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.cv_all import external_live
from research.engulf_1d_opt import LED, REGISTERED
from research.experiments import log_experiment, trials
from research.time_cv import CVConfig, FlipRule, evaluate_family, ledger_stats, summary_row
from research.validation_gate import GateConfig, daily_returns, evaluate

TAG = "20261122"
OUT = Path("/mnt/project-files/signals")
RISK = 0.003
REPORT = "signals/report_engulf_1d_opt_20261122.md"
PLATEAU = ["P_cross3", "P_cross10", "P_half1", "P_half5"]


def _cfg(led, c):
    return led[led["config"] == c].reset_index(drop=True)


def yearly(led: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in led.groupby("config"):
        y = g["time"].dt.year
        for yr, gg in g.groupby(y):
            rows.append({"config": c, "year": int(yr), "trades": len(gg), "mean_r": gg["r_net"].mean(),
                         "win": (gg["r_net"] > 0).mean()})
        for lab, gg in (("2020-23", g[y < 2024]), ("2024+", g[y >= 2024]), ("all", g)):
            rows.append({"config": c, "year": lab, "trades": len(gg), "mean_r": gg["r_net"].mean(),
                         "win": (gg["r_net"] > 0).mean(), **{k: v for k, v in ledger_stats(gg).items() if k != "trades"}})
    return pd.DataFrame(rows)


def overlap(a: pd.DataFrame, v: pd.DataFrame) -> float:
    """a 的每筆進場時，同幣同方向是否已有維加斯單開著。"""
    hit = 0
    vv = {k: g for k, g in v.groupby(["symbol", "side"])}
    for (s, d), g in a.groupby(["symbol", "side"]):
        b = vv.get((s, d))
        if b is None:
            continue
        t = g["time"].to_numpy()[:, None]
        hit += int(((b["time"].to_numpy()[None, :] <= t) & (b["exit_t"].to_numpy()[None, :] > t)).any(axis=1).sum())
    return hit / max(len(a), 1)


def _port(d: pd.Series) -> dict:
    eq = d.cumsum()
    dd = (eq - eq.cummax()).min()
    yrs = len(d) / 365
    return {"ann_simple": d.sum() / yrs, "sharpe": d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan,
            "max_dd_simple": dd}


def combo(led: pd.DataFrame, vegas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start in ("2020-01-01", "2024-01-01"):
        lo = pd.Timestamp(start, tz="UTC")
        hi = max(vegas["exit_t"].max(), led["exit_t"].max())
        vd = daily_returns(vegas[vegas["time"] >= lo], "r_net", RISK, lo, hi)
        rows.append({"start": start, "book": "vegas", **_port(vd)})
        for c in ["BASE", *REGISTERED]:
            g = _cfg(led, c)
            g = g[g["time"] >= lo]
            ed = daily_returns(g, "r_net", RISK, lo, hi)
            rows.append({"start": start, "book": c, **_port(ed), "corr_vegas": float(np.corrcoef(vd, ed)[0, 1])})
            rows.append({"start": start, "book": f"vegas+{c}", **_port(vd + ed)})
    return pd.DataFrame(rows)


def evaluate_all() -> None:
    led = pd.read_pickle(LED)
    led["time"] = pd.to_datetime(led["time"], utc=True)
    led["exit_t"] = pd.to_datetime(led["exit_t"], utc=True)
    vegas = external_live().copy()
    vegas["time"] = pd.to_datetime(vegas["time"], utc=True)
    vegas["exit_t"] = pd.to_datetime(vegas["exit_t"], utc=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # 試驗次數：先登記 7 組
    n0 = trials(count_configs=True)
    if n0 < 1926 + len(REGISTERED):
        log_experiment(family="engulf_1d_opt", hypothesis="日線 EMA8/20 剛交叉＋吞沒，上線前優化 7 組（事先登記）",
                       params={"configs": REGISTERED}, data_source="binance um 1h->1D, PIT top100 by prior-month volume",
                       period="2020-2026", verdict="info", report=REPORT,
                       notes=f"prereg reports/signals/prereg_engulf_1d_{TAG}.md; configs={len(REGISTERED)}")
    n_trials = trials(count_configs=True)

    # 1) 驗證門檻
    rnd = _cfg(led, "R_cross5")
    nbrs = [_cfg(led, p) for p in PLATEAU]
    gate_rows, gate_js = [], {}
    for c in ["BASE", *REGISTERED]:
        g = _cfg(led, c)
        r = rnd[rnd["side"] < 0] if c == "L3" else rnd
        nb = [x[x["side"] < 0] for x in nbrs] if c == "L3" else nbrs
        res = evaluate(g, baseline=vegas, random_baseline=r, neighbors=nb, cfg=GateConfig(n_trials=n_trials, risk=RISK))
        ch = res["checks"]
        gate_js[c] = res
        gate_rows.append({"config": c, "trades": res["info"]["trades"], "mean_r": res["info"]["mean_r"],
                          "win": float((g["r_net"] > 0).mean()), "t_strict": ch["cluster_t"]["value"],
                          "t_required": ch["cluster_t"]["threshold"], "dsr": ch["deflated_sharpe"]["value"],
                          "corr_vegas": ch["baseline_corr"]["value"], "beats_random": ch["beats_random"]["status"],
                          "plateau": ch["plateau"]["status"], "overlap_vegas": overlap(g, vegas),
                          "passed": res["passed"], "verdict": res["verdict"]})
        if c != "BASE":
            log_experiment(family="engulf_1d_opt", hypothesis=f"validation_gate {c}", params={"config": c, "n_trials": n_trials},
                           data_source="binance um 1h->1D, PIT top100", period="2020-2026", trades=len(g),
                           exp_r=round(res["info"]["mean_r"], 4), t_stat=round(ch["cluster_t"]["value"], 3),
                           verdict="pass" if res["passed"] else "fail", report=REPORT,
                           notes=f"{res['verdict']}; configs=0")
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / f"engulf_opt_gate_{TAG}.csv", index=False)

    # 2) 每年
    yr = yearly(led)
    yr.to_csv(OUT / f"engulf_opt_yearly_{TAG}.csv", index=False)

    # 3) time_cv：7 組事先登記的當一個家族；另外各自單獨跑（只有一組 → 等於看它自己在每段的表現）
    fam = led[led["config"].isin(REGISTERED)]
    res = evaluate_family(fam, CVConfig(), FlipRule())
    cv_rows = [{"scope": "family_L0-L6", **summary_row("engulf_1d_opt", res)}]
    for c in ["BASE", *REGISTERED]:
        rr = evaluate_family(_cfg(led, c), CVConfig(), FlipRule())
        cv_rows.append({"scope": c, **summary_row(c, rr)})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / f"engulf_opt_cv_{TAG}.csv", index=False)
    wf_years = res["walk_forward"]["years"]

    # 4) 組合
    cb = combo(led, vegas)
    cb.to_csv(OUT / f"engulf_opt_combo_{TAG}.csv", index=False)

    js = {"n_trials": n_trials, "gate": gate.to_dict("records"), "cv": cv.to_dict("records"),
          "wf_years": wf_years.to_dict("records"), "combo": cb.to_dict("records")}
    Path("output/engulf_opt_20261122.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(gate.drop(columns=["verdict"]).round(3).to_string())
        print(yr[yr["year"].isin(["2020-23", "2024+", "all"])].round(3).to_string())
        print(cv[["scope", "wf_mean_r", "wf_t_strict", "wf_years_pos", "cpcv_frac_pos", "cpcv_paths_pos", "pbo", "flip",
                  "failed_checks"]].round(3).to_string())
        print(wf_years.to_string())
        print(cb.round(4).to_string())


def evaluate_universe() -> None:
    """全部永續的 2 組（補登記 2026-11-23）。python -c 'from research.engulf_1d_eval import evaluate_universe as f; f()'"""
    from research.engulf_1d_opt import LED_ALL
    led = pd.read_pickle(LED_ALL)
    for c in ("time", "exit_t"):
        led[c] = pd.to_datetime(led[c], utc=True)
    vegas = external_live().copy()
    for c in ("time", "exit_t"):
        vegas[c] = pd.to_datetime(vegas[c], utc=True)
    new = ["A_all", "A_all30"]
    if trials(count_configs=True) < 1935:
        log_experiment(family="engulf_1d_opt", hypothesis="全部幣安 USDT 永續（不限前 100）、有無 30M 成交額門檻（補登記）",
                       params={"configs": new}, data_source="binance um 1h->1D, all perps traded >=20d in month",
                       period="2020-2026", verdict="info", report=REPORT, notes="configs=2")
    n_trials = trials(count_configs=True)
    rnd = _cfg(led, "R_all")
    rows = []
    for c in new:
        g = _cfg(led, c)
        res = evaluate(g, baseline=vegas, random_baseline=rnd, cfg=GateConfig(n_trials=n_trials, risk=RISK))
        ch = res["checks"]
        log_experiment(family="engulf_1d_opt", hypothesis=f"validation_gate {c}", params={"config": c, "n_trials": n_trials},
                       data_source="binance um all perps", period="2020-2026", trades=len(g), exp_r=round(res["info"]["mean_r"], 4),
                       t_stat=round(ch["cluster_t"]["value"], 3), verdict="pass" if res["passed"] else "fail", report=REPORT,
                       notes=f"{res['verdict']}; configs=0")
        y = g["time"].dt.year
        for lab, m in (("all", np.ones(len(g), bool)), ("2020-23", y < 2024), ("2024+", y >= 2024),
                       ("top100", g["top100"].astype(bool)), ("outside100", ~g["top100"].astype(bool)),
                       ("outside100_2024+", ~g["top100"].astype(bool) & (y >= 2024)),
                       ("long", g["side"] > 0), ("short", g["side"] < 0)):
            gg = g[np.asarray(m)]
            rows.append({"config": c, "slice": lab, "trades": len(gg), "mean_r": gg["r_net"].mean(),
                         "win": (gg["r_net"] > 0).mean(), "t_strict": ledger_stats(gg)["t_strict"],
                         **({"t_required": ch["cluster_t"]["threshold"], "dsr": ch["deflated_sharpe"]["value"],
                             "corr_vegas": ch["baseline_corr"]["value"], "beats_random": ch["beats_random"]["status"],
                             "passed": res["passed"]} if lab == "all" else {})})
        cvr = evaluate_family(g, CVConfig(), FlipRule())
        rows[-8].update({"cpcv_frac_pos": cvr["cpcv"]["summary"]["frac_splits_pos"],
                         "wf_mean_r": cvr["walk_forward"]["summary"]["mean_r"]})
    rr = rnd
    rows.append({"config": "R_all", "slice": "all", "trades": len(rr), "mean_r": rr["r_net"].mean(), "win": (rr["r_net"] > 0).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"engulf_opt_universe_{TAG}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string())
