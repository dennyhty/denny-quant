"""公開策略（research/public_strategies.py）的跑法：算訊號與帳本、挑主設定（只看樣本內）、驗證關卡、和實盤維加斯合併。

python -m research.public_strategies_run [--log] [--reuse] [--draws 10]
--reuse：沿用 output/ps_ledger.pkl（已套成本的帳本），不重算訊號。
輸出：/mnt/project-files/signals/public_strats_{grid,gate,yearly,combo,filter}_20261122.csv、output/public_strats_summary.json。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import research.public_strategies as ps
import research.vegas_params as vp
from research.candidates_common import IS_END, compare_with, daily_series, pick_headline, split, stats, vegas_book
from research.experiments import log_experiment, read_ledger
from research.validation_gate import GateConfig, evaluate, report_md

TAG = "20261122"
OUT_CSV = "/mnt/project-files/signals/"
REPORT = f"signals/report_public_strategies_{TAG}.md"
DATA_SOURCE = "binance um 2h/4h klines, PIT top100 by prior-month volume"
LEDGER = Path("output/ps_ledger.pkl")
SUMMARY = Path("output/public_strats_summary.json")
GATE_DIR = Path("output/public_strats")


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _r(x, k=4):
    return round(float(x), k) if x is not None and np.isfinite(x) else ""


def _f(x):
    return float(x) if x is not None and np.isfinite(x) else None


def base_trials() -> int:
    total = 0
    for r in read_ledger():
        if r["family"] == ps.FAMILY:
            continue
        m = re.search(r"configs=(\d+)", r.get("notes") or "")
        total += int(m.group(1)) if m else 1
    return total


def cname(v, iv, ex):
    return f"{v}|{iv}|{ex}"


def neighbors(iv, ex):
    return [(iv, "fix15" if ex == "live" else "live"), ("4H" if iv == "2H" else "2H", ex)]


def build_ledger(workers=4) -> pd.DataFrame:
    mem, pit = ps.membership_table()
    raw = ps.run_signals(workers)
    raw = raw[[(s, m) in pit for s, m in zip(raw["symbol"], raw["time"].dt.strftime("%Y-%m"))]]
    funding = vp._funding(sorted(raw["symbol"].unique()))
    led = ps.to_ledger(raw, pit, funding)
    led.to_pickle(LEDGER)
    return led


def rand_ledger(heads: dict, led: pd.DataFrame, mem_pit: set, draws: int, workers=4) -> dict:
    """heads: 變體 -> (週期, 出場)。回傳 變體 -> 隨機進場帳本。"""
    months = {}
    for s, m in mem_pit:
        months.setdefault(s, set()).add(m)
    jobs = {}
    for v, (iv, ex) in heads.items():
        g = led[(led["variant"] == v) & (led["interval"] == iv) & (led["exit"] == ex)]
        for sym, gg in g.groupby("symbol"):
            jobs.setdefault((sym, iv), {})[v] = (ex, gg["time"].dt.year.to_numpy(), gg["side"].to_numpy())
    args = [(s, iv, j, months.get(s, set()), draws, 7) for (s, iv), j in jobs.items()]
    parts = []
    with ProcessPoolExecutor(workers) as ex_:
        for got in ex_.map(ps._rand_work, args, chunksize=4):
            if got is not None:
                parts.append(got)
    raw = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        raw[c] = pd.to_datetime(raw[c], utc=True)
    syms = sorted(raw["symbol"].unique())
    funding = vp._funding(syms)
    led_r = ps.to_ledger(raw, mem_pit, funding)
    return {v: led_r[led_r["variant"] == v] for v in heads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--draws", type=int, default=10)
    a = ap.parse_args()
    mem, pit = ps.membership_table()
    n_new = len(ps.VARIANTS) * 2 * 2   # 28
    n_trials = base_trials() + n_new
    _log(f"試驗數 = {n_trials}")
    led = pd.read_pickle(LEDGER) if a.reuse else build_ledger()
    base = vegas_book()
    base_day = daily_series(base)

    # ---------------- 網格
    rows = []
    for (v, iv, ex), tr in led.groupby(["variant", "interval", "exit"]):
        ins, oos = split(tr)
        rows.append({"variant": v, "interval": iv, "exit": ex, "config": cname(v, iv, ex),
                     "is_ref": v in ps.REFS, **{f"is_{k}": x for k, x in stats(ins).items()},
                     **{f"oos_{k}": x for k, x in stats(oos).items()}, **{f"all_{k}": x for k, x in stats(tr).items()}})
    grid = pd.DataFrame(rows)

    # ---------------- 主設定（只看樣本內）
    heads = {}
    for v in ps.VARIANTS:
        sub = grid[grid["variant"] == v].copy()
        h = pick_headline(sub, 100, "is_t")
        heads[v] = tuple(h.split("|")[1:])
        _log(f"[{v}] 主設定 {h}")
    rnd = rand_ledger(heads, led, pit, a.draws)

    if a.log:
        for _, r in grid[~grid["is_ref"]].iterrows():
            log_experiment(family=ps.FAMILY, hypothesis=f"public strategy {r['variant']} {r['interval']} {r['exit']} IS screen",
                           params={"variant": r["variant"], "interval": r["interval"], "exit": r["exit"]},
                           data_source=DATA_SOURCE, period="2020-2023 IS", trades=int(r["is_trades"]),
                           exp_r=_r(r.get("is_exp_r")), se_cluster=_r(r.get("is_se_cluster")), t_stat=_r(r.get("is_t"), 3),
                           sharpe=_r(r.get("is_sharpe"), 3), verdict="info", report=REPORT,
                           notes="IS screen of pre-registered public strategies (28); configs=1")

    gate_rows, year_rows, combo_rows, summary = [], [], [], {"n_trials": n_trials, "configs": {}, "headlines": {}}
    for _, r in grid[~grid["is_ref"]].iterrows():
        summary["configs"][r["config"]] = {k: _f(r.get(k)) for k in ("is_trades", "is_exp_r", "is_t", "is_sharpe",
                                                                      "oos_trades", "oos_exp_r", "oos_t", "oos_sharpe")}
    ledger_of = lambda v, iv, ex: led[(led["variant"] == v) & (led["interval"] == iv) & (led["exit"] == ex)].reset_index(drop=True)  # noqa: E731

    # ---------------- 和實盤合併（24 組全部）
    for _, r in grid[~grid["is_ref"]].iterrows():
        tr = ledger_of(r["variant"], r["interval"], r["exit"])
        cd = daily_series(tr)
        c = compare_with(base_day, cd)
        co = compare_with(base_day[base_day.index >= IS_END], cd[cd.index >= IS_END])
        combo_rows.append({"config": r["config"], "variant": r["variant"], "interval": r["interval"], "exit": r["exit"],
                           **c, **{f"oos_{k}": x for k, x in co.items()}})

    # ---------------- 關卡
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    for v, (iv, ex) in heads.items():
        name = cname(v, iv, ex)
        tr = ledger_of(v, iv, ex)
        nb = [ledger_of(v, i2, e2) for i2, e2 in neighbors(iv, ex)]
        g = evaluate(tr, baseline=base, random_baseline=rnd[v], cfg=GateConfig(n_trials=n_trials), rcol="r_net", neighbors=nb)
        (GATE_DIR / f"gate_{v}.md").write_text(report_md(g, name), encoding="utf-8")
        for k, c in g["checks"].items():
            gate_rows.append({"variant": v, "headline": name, "check": k, "label": c["label"], "status": c["status"],
                              "value": json.dumps(c["value"], default=float, ensure_ascii=False),
                              "threshold": c["threshold"], "detail": c["detail"]})
        yr = pd.DataFrame([{"variant": v, "headline": name, "year": int(y), **stats(gg)} for y, gg in tr.groupby(tr["time"].dt.year)])
        year_rows.append(yr)
        cr = next(x for x in combo_rows if x["config"] == name)
        summary["headlines"][v] = {"headline": name, "neighbors": [cname(v, i, e) for i, e in neighbors(iv, ex)],
                                   "corr_daily": cr["corr_daily"], "corr_daily_oos": cr["oos_corr_daily"],
                                   "sharpe_vegas": cr["sharpe_base"], "sharpe_cand": cr["sharpe_cand"],
                                   "sharpe_combined": cr["sharpe_combined"], "gate": {
                                       "verdict": g["verdict"], "passed": g["passed"], "incomplete": g["incomplete"],
                                       "failed": g["failed"], "skipped": g["skipped"], "info": g["info"],
                                       "checks": {k: {"label": c["label"], "value": c["value"], "threshold": c["threshold"],
                                                      "status": c["status"], "detail": c["detail"]}
                                                  for k, c in g["checks"].items()}}}
        if a.log:
            gr = grid.set_index("config").loc[name]
            log_experiment(family=ps.FAMILY, hypothesis="headline OOS (selected on IS only)",
                           params={"variant": v, "interval": iv, "exit": ex}, data_source=DATA_SOURCE,
                           period="2024-2026 OOS", trades=int(gr["oos_trades"]), exp_r=_r(gr.get("oos_exp_r")),
                           se_cluster=_r(gr.get("oos_se_cluster")), t_stat=_r(gr.get("oos_t"), 3),
                           sharpe=_r(gr.get("oos_sharpe"), 3), verdict="info", report=REPORT,
                           notes="same config as IS row, not a new trial; configs=0")
            log_experiment(family=ps.FAMILY, hypothesis="validation_gate on headline (baseline=vegas PIT book)",
                           params={"variant": v, "interval": iv, "exit": ex, "n_trials": n_trials}, data_source=DATA_SOURCE,
                           period="2020-2026", trades=g["info"]["trades"], exp_r=_r(g["info"]["mean_r"]),
                           t_stat=_r(g["checks"]["cluster_t"]["value"], 3), sharpe=_r(g["info"].get("sharpe_annual"), 3),
                           verdict="fail" if not g["passed"] else ("info" if g["incomplete"] else "pass"), report=REPORT,
                           notes=f"{g['verdict']}; corr_daily={cr['corr_daily']:.3f}; configs=0")
        _log(f"[{v}] {g['verdict']}")

    # ---------------- 濾網有沒有幫助：vegas 帶濾網 vs 原版 vs 被濾掉的單（同一批 K 棒、同一套模擬）
    frows = []
    for tag, v_f, rem in (("ma200", "vegas_ma200", "removed_ma200"), ("ema200", "vegas_ema200", "removed_ema200"),
                          ("trend", "vegas_trend_only", "removed_trend")):
        for iv in ps.INTERVALS:
            for ex in ps.EXITS:
                for lab, v in (("vegas_base", "vegas_base"), ("filtered", v_f), ("removed", rem)):
                    tr = ledger_of(v, iv, ex)
                    if not len(tr):
                        continue
                    ins, oos = split(tr)
                    frows.append({"filter": tag, "interval": iv, "exit": ex, "set": lab,
                                  **{f"is_{k}": x for k, x in stats(ins).items() if k in ("trades", "exp_r", "t", "sharpe")},
                                  **{f"oos_{k}": x for k, x in stats(oos).items() if k in ("trades", "exp_r", "t", "sharpe")}})
    filt = pd.DataFrame(frows)
    summary["vegas_filter"] = filt.to_dict("records")

    grid.assign(headline=lambda t: [heads.get(v) == (iv, ex) for v, iv, ex in zip(t["variant"], t["interval"], t["exit"])]
                ).to_csv(OUT_CSV + f"public_strats_grid_{TAG}.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT_CSV + f"public_strats_gate_{TAG}.csv", index=False)
    pd.concat(year_rows, ignore_index=True).to_csv(OUT_CSV + f"public_strats_yearly_{TAG}.csv", index=False)
    pd.DataFrame(combo_rows).to_csv(OUT_CSV + f"public_strats_combo_{TAG}.csv", index=False)
    filt.to_csv(OUT_CSV + f"public_strats_filter_{TAG}.csv", index=False)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
