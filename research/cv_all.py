"""把所有策略家族用滾動前進＋CPCV＋PBO 重新驗證（research/time_cv.py）。2026-11-22。

python -m research.cv_all [--only fam1,fam2]

每個家族先由各自的匯出程式（research/cv_export_*.py）產生「長帳本」：
    output/cv_ledgers/{family}.pkl    欄位 config、symbol、side、time、exit_t、r_net（＋可有 interval、r）
    output/cv_ledgers/{family}.json   {"family", "label"(中文名稱), "kind": "trades"|"portfolio",
                                       "source": 原報告, "configs_registered": 原本登記的組數, "notes"}
帳本一律：時點幣池（上個月成交額前 100，或該家族原本的實盤幣池）、同幣同時一張、手續費＋滑價＋資金費率。
kind=portfolio 的家族（橫斷面多空組合）一列是一期組合報酬（單位和原研究相同），symbol 填 "PORT"。

輸出：
    /mnt/project-files/signals/cv_summary_20261122.csv     每個家族一列
    /mnt/project-files/signals/cv_walkforward_20261122.csv 每家族每年選到誰、樣本外平均
    /mnt/project-files/signals/cv_cpcv_splits_20261122.csv 每家族 45 種切法
    /mnt/project-files/signals/cv_stable_configs_20261122.csv 每個設定在 10 段裡的表現（只列正的段數多的）
    output/cv_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.time_cv import CVConfig, FlipRule, evaluate_family, summary_row

LEDGERS = Path("output/cv_ledgers")
TAG = "20261122"
OUT = Path("/mnt/project-files/signals")


def _log(m):
    print(m, file=sys.stderr, flush=True)


# 「維加斯變體」家族：除了自己要穩，還要比實盤好才算翻案（time_cv.vs_baseline）。
# 帳本裡有實盤原樣那一組就用它；沒有的用外部實盤帳本（時點幣池、不設成交額門檻，和多數匯出程式同一個池）。
BASELINE = {
    "regime_filter": "live（不濾網）", "be_timing": "A live", "tp_ladder": "1/3/5R(live)", "vegas_divergence": "base",
    "vegas_params": "external", "vegas_improve": "external", "vegas_timeframes": "external",
    "universe_buckets": "external", "vegas_entry_filters": "external", "vegas_stoploss": "external",
    "public_strats": "external",
}
_EXT = None


def external_live() -> pd.DataFrame:
    """output/pit_trades.csv.gz → 時點幣池的實盤帳本（2H+4H，同週期同幣一張）。"""
    global _EXT
    if _EXT is None:
        from research.pit_backtest import book
        tr = pd.read_csv("output/pit_trades.csv.gz", parse_dates=["time", "exit_t"])
        _EXT = book(tr, "in_pit")[["symbol", "side", "time", "exit_t", "r_net"]]
    return _EXT


def load(family: str) -> tuple[pd.DataFrame, dict]:
    led = pd.read_pickle(LEDGERS / f"{family}.pkl")
    meta_p = LEDGERS / f"{family}.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"family": family}
    return led, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fams = sorted(p.stem for p in LEDGERS.glob("*.pkl"))
    if a.only:
        fams = [f for f in fams if f in set(a.only.split(","))]
    rows, wf_rows, sp_rows, pc_rows, js = [], [], [], [], {}
    for fam in fams:
        led, meta = load(fam)
        cfg = CVConfig()
        if meta.get("kind") == "portfolio":
            cfg.min_train_trades = 30   # 週組合一年約 52 期
            cfg.min_test_trades = 8
        _log(f"[{fam}] {len(led)} 列、{led['config'].nunique()} 組設定")
        base = BASELINE.get(fam)
        if base == "external":
            base = external_live()
        elif base is not None and base not in set(led["config"]):
            raise ValueError(f"{fam}: 實盤對照 {base} 不在帳本裡")
        res = evaluate_family(led, cfg, FlipRule(), baseline=base)
        row = {"label": meta.get("label", fam), "kind": meta.get("kind", "trades"),
               "configs_registered": meta.get("configs_registered"), "source": meta.get("source"),
               **summary_row(fam, res)}
        rows.append(row)
        for mode in ("walk_forward", "walk_forward_rolling", "walk_forward_gated"):
            y = res[mode]["years"].assign(family=fam, mode=mode)
            wf_rows.append(y)
        sp_rows.append(res["cpcv"]["splits"].assign(family=fam))
        pc = res["cpcv"]["per_config"].assign(family=fam)
        pc_rows.append(pc.sort_values(["blocks_pos", "mean_r"], ascending=False).head(15))
        js[fam] = {"meta": meta, "checks": res["checks"], "vs_baseline": res["vs_baseline"], "flip": res["flip"], "stable_configs": res["stable_configs"],
                   "wf": res["walk_forward"]["summary"], "wf_rolling": res["walk_forward_rolling"]["summary"],
                   "wf_gated": res["walk_forward_gated"]["summary"], "cpcv": res["cpcv"]["summary"],
                   "paths": res["cpcv"]["paths"].to_dict("records"), "pbo": res["pbo"]["summary"]}
        _log(f"[{fam}] WF {row['wf_mean_r']:+.3f}R t*{row['wf_t_strict']:.2f} 年 {row['wf_years_pos']}；"
             f"CPCV 正 {row['cpcv_frac_pos']:.0%} 路徑 {row['cpcv_paths_pos']}；PBO {row['pbo']}；翻案 {row['flip']}")
    if not rows:
        _log("沒有帳本")
        return
    pd.DataFrame(rows).to_csv(out / f"cv_summary_{TAG}.csv", index=False)
    pd.concat(wf_rows, ignore_index=True).to_csv(out / f"cv_walkforward_{TAG}.csv", index=False)
    pd.concat(sp_rows, ignore_index=True).to_csv(out / f"cv_cpcv_splits_{TAG}.csv", index=False)
    pd.concat(pc_rows, ignore_index=True).to_csv(out / f"cv_stable_configs_{TAG}.csv", index=False)
    Path("output/cv_results.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(rows)[["family", "n_configs", "wf_trades", "wf_mean_r", "wf_t_strict", "wf_years_pos",
                                  "cpcv_frac_pos", "cpcv_paths_pos", "pbo", "n_stable_configs", "flip"]].round(3).to_string())


if __name__ == "__main__":
    main()
