"""日線 EMA8/20 剛交叉＋吞沒：上線前優化（事先登記 7 組，見 reports/signals/prereg_engulf_1d_20261122.md）。

python -m research.engulf_1d_opt export   # → output/cv_ledgers/engulf_1d_opt.pkl（長帳本，給 time_cv / validation_gate）
python -m research.engulf_1d_opt evaluate # → 驗證門檻、time_cv、組合、重疊 → output/engulf_1d_opt_*.csv/json

資料、幣池、成本、模擬全部沿用 research.cv_export_old_patterns 的 threads 家族（時點幣池、幣安永續 1H 合成日線、
手續費 0.05%/邊 + 滑價 + 資金費率），所以 BASE 應該重現 PR #81 的 engulf_1D_cross5_half_atr|tp1|all。

config 命名：
  BASE           原設定（不加成交額門檻），對帳用
  L0..L6         事先登記的 7 組（L4 出場 h12 = 1R 平一半、保本、剩下 2R；其他都是 tp1 = 1R 全出）
  P_cross3 / P_cross10 / P_half1 / P_half5   參數高原用（L0 條件 + 原研究的敏感度設定）
  R_cross5       隨機基準（L0 條件 + 剛交叉 5 根內每 5 根抽一根、市價、1.5 ATR、tp1）
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import research.cv_export_old_patterns as cx
import research.threads_strategies as ts

OUT = Path("output")
LED = OUT / "cv_ledgers" / "engulf_1d_opt.pkl"
MIN_ADV = 3e7
DAY_CAP = 3
REGISTERED = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
EXITS = {"tp1": ((1.0,), (1.0,), 0.0), "h12": ((1.0, 2.0), (0.5, 0.5), 0.0)}


def _orders_df(df, sigs, entry_mode, wait=3):
    orders = ts.build_orders(df, sigs, entry_mode, "atr", wait)
    r = ts.run(df, orders, "1D")
    return r


def _features(df):
    v = df["volume"].to_numpy(float)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    body = np.abs(df["close"].to_numpy() - df["open"].to_numpy())
    return {"vol_ok": v > vavg, "body_ok": body >= 0.5 * df["atr"].to_numpy()}


def export(max_syms: int = 0) -> pd.DataFrame:
    ts.EXITS = EXITS
    mem, pit = cx.pit_index()
    symbols = cx.universe_symbols(mem)
    if max_syms:
        symbols = symbols[:max_syms]
    parts = []
    for n, sym in enumerate(symbols):
        df = cx._frame(sym, "1D")
        if df is None:
            continue
        adv = cx.FRAMES["adv"].copy()
        feats = _features(df)
        rng = np.random.default_rng(zlib.crc32(f"{sym}1D".encode()))
        jobs = {"S_cross5": (ts.engulf_signals(df, True), "half", 3),
                "P_cross3": (ts.engulf_signals(df, True, 3), "half", 3),
                "P_cross10": (ts.engulf_signals(df, True, 10), "half", 3),
                "P_half1": (ts.engulf_signals(df, True), "half", 1),
                "P_half5": (ts.engulf_signals(df, True), "half", 5),
                "R_cross5": (ts.regime_random(df, rng, True), "mkt", 3)}
        for name, (sigs, em, wait) in jobs.items():
            r = _orders_df(df, sigs, em, wait)
            if r.empty:
                continue
            r = r.copy()
            r["entry_bar"], r["busy_i"] = r["start"], r["start"] - 1
            sig = r["sig"].to_numpy(np.int64)
            exits = ["tp1", "h12"] if name == "S_cross5" else ["tp1"]
            led = cx._ledger_from_orders(r, sym, "1D", exits, lambda tt, ex, s=name: s + "|" + ex, True)
            k = len(exits)
            led["sig_adv"] = np.tile(adv[sig], k)
            led["vol_ok"] = np.tile(feats["vol_ok"][sig], k)
            led["body_ok"] = np.tile(feats["body_ok"][sig], k)
            led["sig_day"] = np.tile(pd.to_datetime(df["time"].to_numpy()[sig], utc=True).floor("D"), k)
            parts.append(led)
        if n % 25 == 0:
            cx.log(f"engulf_1d_opt {n}/{len(symbols)}")
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[(raw["time"] >= cx.START) & cx.in_pit(pit, raw["symbol"], raw["time"])]
    liq = raw["sig_adv"] >= MIN_ADV
    s1, s2 = raw["config"] == "S_cross5|tp1", raw["config"] == "S_cross5|h12"
    sel = {
        "BASE": s1,
        "L0": s1 & liq,
        "L1": s1 & liq & raw["vol_ok"],
        "L2": s1 & liq & raw["body_ok"],
        "L3": s1 & liq & (raw["side"] < 0),
        "L4": s2 & liq,
        "L5": s1 & liq,          # 每日上限在「同一張」之後做
        "L6": s1 & liq & raw["vol_ok"],
    }
    for p in ("P_cross3", "P_cross10", "P_half1", "P_half5", "R_cross5"):
        sel[p] = (raw["config"] == p + "|tp1") & liq
    tr = pd.concat([raw[m].assign(config=c) for c, m in sel.items()], ignore_index=True)
    tr = cx.one_at_a_time(tr, ["config", "symbol"])
    capped = tr["config"].isin(["L5", "L6"])
    c = tr[capped].sort_values(["config", "sig_day", "side", "sig_adv"], ascending=[True, True, True, False])
    c = c[c.groupby(["config", "sig_day", "side"]).cumcount() < DAY_CAP]
    tr = pd.concat([tr[~capped], c], ignore_index=True)
    led = cx.finish(tr)
    LED.parent.mkdir(parents=True, exist_ok=True)
    led.to_pickle(LED)
    cx.log("每組筆數 / 平均 r_net:\n" + led.groupby("config")["r_net"].agg(["size", "mean"]).to_string())
    return led


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    if cmd == "export":
        export(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        from research.engulf_1d_eval import evaluate_all
        evaluate_all()


if __name__ == "__main__":
    main()
