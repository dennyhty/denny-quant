"""新策略搜尋（2026-11-22）：跑預先登記的 4 個家族共 17 組設定，只用樣本內挑主設定，主設定過驗證關卡。

python -m research.new_strategies_run [--log] [--draws 20]

- N1 new_xsec_factors（8 組）：research/new_xsec.py，週組合、BTC beta 中性。
- N2 new_funding_extreme（4 組）：日線資金費率極端值的逐幣事件交易，R 單位。
- N3 new_cme_gap（3 組）：BTC、ETH 週末（CME 收盤到重開）缺口回補，1h。
- N4 new_intraday_hour（2 組）：池內等權指數（或 BTC/ETH）的日內時段多空，1h。

資料只用到訊號 K 棒收盤；進場在該收盤；出場從下一根開始檢查。樣本內 2020-2023，樣本外 2024 起。
所有結果扣手續費每邊 0.05%、滑價分級、資金費率（N4 持倉不到 8 小時，資金費率略過）。
輸出：/mnt/project-files/signals/new_strategies_*_20261122.csv、output/new_strategies_summary.json。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research.candidates_daily_trend as cdt
import research.candidates_xsec as cx
import research.new_xsec as nx
from grid_trading.costs import apply_costs, slippage_bps
from grid_trading.pit_universe import fetch_symbol
from research.candidates_common import (FEE, IS_END, RISK, START, ExitPlan, HALF_1R, adv30, compare_with, daily_series,
                                        finalize, load_daily, one_at_a_time, pick_headline, pool_table, random_entries,
                                        simulate, split, stats, trade_frame, vegas_book)
from research.experiments import log_experiment, read_ledger
from research.validation_gate import GateConfig, evaluate, report_md

TAG = "20261122"
OUT_CSV = "/mnt/project-files/signals/"
REPORT = f"signals/report_new_strategies_{TAG}.md"
DATA_SOURCE = "binance um 4h->1d / 1h, PIT top100 by prior-month volume"
SUMMARY = Path("output/new_strategies_summary.json")
GATE_MD = Path("output/new_strategies")
F_N1, F_N2, F_N3, F_N4 = "new_xsec_factors", "new_funding_extreme", "new_cme_gap", "new_intraday_hour"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _r(x, k=4):
    return round(float(x), k) if x is not None and np.isfinite(x) else ""


def _f(x):
    return float(x) if x is not None and np.isfinite(x) else None


# ---------------------------------------------------------------- N2：資金費率極端值

N2_CFG = {f"{th}|{ex}": {"th": th, "exit": ex} for th, ex in itertools.product("AB", ("X1", "X2"))}
N2_TH = {"A": (0.0005, 0.0003), "B": (0.001, 0.0005)}   # (做空門檻 T_hi, 做多門檻 T_lo)，每 8 小時


def n2_neighbors(name: str) -> list[str]:
    th, ex = name.split("|")
    return [f"{'B' if th == 'A' else 'A'}|{ex}", f"{th}|{'X2' if ex == 'X1' else 'X1'}"]


def n2_plan(ex: str) -> ExitPlan:
    return ExitPlan(stop_atr=2.0, **HALF_1R, max_bars=10) if ex == "X1" else ExitPlan(stop_atr=2.0, max_bars=5)


def f3_series(panel: cx.Panel, data: dict) -> dict[str, np.ndarray]:
    """每個幣、每個日 K（對齊該幣自己的日線）的近 3 天平均每 8 小時資金費率 = 3 天加總 / 9。
    第 a 天的值只含第 a-2..a 天的結算（都在 a 日收盤前發生）。沒有費率資料的幣回傳 NaN。"""
    out = {}
    for s, d in data.items():
        if s not in panel.syms:
            continue
        pos = panel.days.get_indexer(pd.DatetimeIndex(pd.to_datetime(d["time"], utc=True)))
        fd = panel.fund[:, panel.syms.index(s)]
        f = np.where(pos >= 0, fd[np.maximum(pos, 0)], 0.0)
        f3 = pd.Series(f).rolling(3).sum().to_numpy() / 9.0
        if not np.any(f):
            f3 = np.full(len(f), np.nan)
        out[s] = f3
    return out


def n2_raw(data, f3s, cfg) -> pd.DataFrame:
    t_hi, t_lo = N2_TH[cfg["th"]]
    plan = n2_plan(cfg["exit"])
    parts = []
    for s, d in data.items():
        if s not in f3s:
            continue
        arr = cdt.arrays(d)
        f3 = f3s[s]
        sig = np.where(f3 > t_hi, -1, np.where(f3 < -t_lo, 1, 0))
        ok = (np.arange(len(d)) >= 60) & (np.arange(len(d)) + 1 < len(d)) & np.isfinite(arr["atr"]) & np.isfinite(f3)
        idx = np.flatnonzero((sig != 0) & ok)
        if not len(idx):
            continue
        ds = sig[idx]
        sim = simulate(arr, idx, ds, plan)
        parts.append(trade_frame(s, d, idx, ds, sim, adv30(d["qv"].to_numpy())))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def n2_plan_of(data, cfg):
    cache = {}

    def f(sym):
        if sym not in cache:
            cache[sym] = (cdt.arrays(data[sym]), n2_plan(cfg["exit"]))
        return cache[sym]
    return f


# ---------------------------------------------------------------- N3：CME 缺口

N3_CFG = {f"G{g:g}": {"g": g} for g in (0.01, 0.02, 0.03)}
HOUR = pd.Timedelta(hours=1)
MAX_HOLD = 120


def n3_neighbors(name: str) -> list[str]:
    gs = list(N3_CFG)
    i = gs.index(name)
    return [gs[j] for j in (i - 1, i + 1) if 0 <= j < len(gs)]


def cme_events(sym: str) -> dict:
    """該幣所有週日 21:00 開盤的 1h K 線（= 到 22:00 收盤）與前一個週五 20:00 開盤那根（到 21:00 收盤）的位置與缺口。"""
    d = fetch_symbol(sym, "1h")
    t = pd.DatetimeIndex(d["time"])
    ix = pd.Series(np.arange(len(t)), index=t)
    sun = np.flatnonzero((t.dayofweek == 6) & (t.hour == 21))
    fri_t = t[sun] - pd.Timedelta(hours=49)
    fri = ix.reindex(fri_t).to_numpy()
    ok = np.isfinite(fri)
    sun, fri = sun[ok], fri[ok].astype(int)
    close = d["close"].to_numpy(float)
    return {"d": d, "t": t, "sun": sun, "fri": fri, "gap": close[sun] / close[fri] - 1}


def n3_trades(sym: str, ev: dict, g: float, sides: np.ndarray | None = None) -> pd.DataFrame:
    """sides=None：往缺口回補方向進場（缺口向上做空）；給定 sides（隨機基準）：同樣的進場點、自訂方向，
    目標與止損距離和真實交易一樣（|進場價 − 週五收盤|）。同一根同時碰到止損與目標算止損。"""
    d, t = ev["d"], ev["t"]
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    qv = pd.Series(d["volume"].to_numpy(float) * c).rolling(720, min_periods=100).mean().to_numpy() * 24
    rows = []
    for k, (i, f, gap) in enumerate(zip(ev["sun"], ev["fri"], ev["gap"])):
        if abs(gap) <= g or i + MAX_HOLD >= len(t) or t[i + MAX_HOLD] - t[i] != MAX_HOLD * HOUR:
            continue
        entry, tgt0 = c[i], c[f]
        dist = abs(entry - tgt0)
        s = int(-np.sign(gap)) if sides is None else int(sides[k])
        stop, tgt = entry - s * dist, entry + s * dist
        exit_px, j = c[i + MAX_HOLD], i + MAX_HOLD
        for jj in range(i + 1, i + MAX_HOLD + 1):
            hit_stop = l[jj] <= stop if s > 0 else h[jj] >= stop
            hit_tgt = h[jj] >= tgt if s > 0 else l[jj] <= tgt
            if hit_stop:
                exit_px, j = stop, jj
                break
            if hit_tgt:
                exit_px, j = tgt, jj
                break
        risk_pct = dist / entry
        rows.append({"symbol": sym, "coin": sym[:-4], "side": s, "time": t[i] + HOUR, "exit_t": t[j] + HOUR,
                     "r": s * (exit_px - entry) / dist - 2 * FEE / risk_pct, "risk_pct": risk_pct,
                     "adv_usdt": qv[i], "gap": gap})
    return pd.DataFrame(rows)


def n3_book(evs: dict, g: float, rng: np.random.Generator | None = None) -> pd.DataFrame:
    parts = []
    for sym, ev in evs.items():
        sides = None if rng is None else rng.choice([-1, 1], len(ev["sun"]))
        parts.append(n3_trades(sym, ev, g, sides))
    tr = pd.concat([p for p in parts if len(p)], ignore_index=True)
    tr = tr[tr["time"] >= START].sort_values("time").reset_index(drop=True)
    return apply_costs(tr)


# ---------------------------------------------------------------- N4：日內時段

N4_CFG = {"pool_ew": {"universe": "pool", "slip_bps": 4.0}, "btceth_ew": {"universe": "btceth", "slip_bps": 1.0}}


def n4_neighbors(name: str) -> list[str]:
    return [k for k in N4_CFG if k != name]


def n4_load(mem: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每根 1h K 線的簡單報酬（收/開 − 1）矩陣（時間 x 幣，以開盤時間為索引）與池內遮罩（該 K 線所在月在時點幣池）。"""
    rets = {}
    for s in symbols:
        d = fetch_symbol(s, "1h")
        if len(d):
            rets[s] = pd.Series((d["close"] / d["open"] - 1).to_numpy(float), index=pd.DatetimeIndex(d["time"]))
    ret = pd.DataFrame(rets).sort_index()
    ret = ret[ret.index >= START]
    month = ret.index.strftime("%Y-%m")
    pit = set(zip(mem["symbol"], mem["month"]))
    mask = pd.DataFrame({s: np.fromiter(((s, m) in pit for m in month), bool, len(month)) for s in ret.columns},
                        index=ret.index)
    return ret, mask


def n4_index(ret: pd.DataFrame, mask: pd.DataFrame, universe: str) -> pd.Series:
    if universe == "btceth":
        x = ret[["BTCUSDT", "ETHUSDT"]]
        return x.mean(axis=1).where(x.notna().all(axis=1))
    x = ret.where(mask)
    return x.mean(axis=1).where(x.notna().sum(axis=1) >= 10)


def n4_trades(idx: pd.Series, long_h: list[int], short_h: list[int], hstd: float, slip: float) -> pd.DataFrame:
    """每個日內時段一筆交易：進場在該小時開盤、出場在該小時收盤。1R = hstd（樣本內小時報酬標準差）。"""
    h = idx.index.hour
    side = np.where(np.isin(h, long_h), 1, np.where(np.isin(h, short_h), -1, 0))
    m = (side != 0) & idx.notna().to_numpy()
    t = idx.index[m]
    s, x = side[m], idx.to_numpy()[m]
    df = pd.DataFrame({"symbol": "INDEX", "coin": "INDEX", "side": s, "time": t, "exit_t": t + HOUR,
                       "r": (s * x - 2 * FEE) / hstd, "funding_r": 0.0, "slip_r": -2 * slip / 1e4 / hstd,
                       "risk_pct": hstd})
    df["r_net"] = df["r"] + df["slip_r"]
    df["gross_r"] = s * x / hstd
    return df


def n4_build(ret, mask, cfg, rng_draws: int = 0):
    idx = n4_index(ret, mask, cfg["universe"])
    ins = idx[idx.index < IS_END]
    mean_by_h = ins.groupby(ins.index.hour).mean()
    order = mean_by_h.sort_values()
    long_h, short_h = list(order.index[-3:]), list(order.index[:3])
    hstd = float(ins.std())
    tr = n4_trades(idx, long_h, short_h, hstd, cfg["slip_bps"])
    rnd = []
    if rng_draws:
        rng = np.random.default_rng(11)
        for k in range(rng_draws):
            pick = rng.permutation(24)
            rnd.append(n4_trades(idx, list(pick[:3]), list(pick[3:6]), hstd, cfg["slip_bps"]).assign(symbol_draw=k))
    return tr, (pd.concat(rnd, ignore_index=True) if rnd else None), {
        "long_hours": [int(x) for x in long_h], "short_hours": [int(x) for x in short_h], "hstd_is": hstd,
        "mean_by_hour_is_bp": {int(k): float(v * 1e4) for k, v in mean_by_h.items()}}


# ---------------------------------------------------------------- 共用

def stats_row(family: str, name: str, tr: pd.DataFrame) -> dict:
    ins, oos = split(tr)
    return {"family": family, "config": name, **{f"is_{k}": v for k, v in stats(ins).items()},
            **{f"oos_{k}": v for k, v in stats(oos).items()}, **{f"all_{k}": v for k, v in stats(tr).items()}}


def yearly(tr: pd.DataFrame, family: str) -> pd.DataFrame:
    return pd.DataFrame([{"family": family, "year": int(y), **stats(g)} for y, g in tr.groupby(tr["time"].dt.year)])


def base_trials() -> int:
    """帳本裡「這次以外」的設定數（排除 new_ 開頭的家族，重跑時才不會重複算）。"""
    import re
    total = 0
    for r in read_ledger():
        if r["family"].startswith("new_"):
            continue
        m = re.search(r"configs=(\d+)", r.get("notes") or "")
        total += int(m.group(1)) if m else 1
    return total


def gate_one(family, tr, base, rnd, nbrs, n_trials) -> dict:
    res = evaluate(tr, baseline=base, random_baseline=rnd, cfg=GateConfig(n_trials=n_trials), rcol="r_net",
                   neighbors=nbrs)
    GATE_MD.mkdir(parents=True, exist_ok=True)
    (GATE_MD / f"gate_{family}.md").write_text(report_md(res, family), encoding="utf-8")
    return res


def weekly_t(day: pd.Series) -> float:
    w = day.resample("W").sum()
    return float(w.mean() / w.std() * math.sqrt(len(w))) if w.std() > 0 else float("nan")


def n1_row(name: str, rows: pd.DataFrame) -> dict:
    rec = {"family": F_N1, "config": name}
    for lab, m in (("is", rows["time"] < IS_END), ("oos", rows["time"] >= IS_END), ("all", slice(None))):
        sub = rows[m]
        day, pw = cx.daily(sub), cx.position_weeks(sub)
        rec.update({f"{lab}_trades": len(pw), f"{lab}_exp_r": float(pw["r_net"].mean()), f"{lab}_t": weekly_t(day),
                    f"{lab}_sharpe": cx.sharpe(day), f"{lab}_sharpe_gross": cx.sharpe(cx.daily(sub, "r")),
                    f"{lab}_ann_ret_pct": day.mean() * 365 * 100, f"{lab}_ann_vol_pct": day.std() * math.sqrt(365) * 100,
                    f"{lab}_win_rate": float((pw["r_net"] > 0).mean()),
                    f"{lab}_long_r": float(pw.loc[pw["side"] > 0, "r_net"].mean()),
                    f"{lab}_short_r": float(pw.loc[pw["side"] < 0, "r_net"].mean()),
                    f"{lab}_funding_r_per_pw": float(sub["funding_r"].sum() / max(len(pw), 1)),
                    f"{lab}_cost_r_per_pw": float(sub["cost_r"].sum() / max(len(pw), 1))})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="寫進 research/experiments.csv（只跑一次）")
    ap.add_argument("--draws", type=int, default=20, help="隨機基準重複幾次")
    ap.add_argument("--skip", default="", help="略過的家族（逗號分隔 N1..N4），除錯用")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    mem = pool_table()
    pool = set(zip(mem["symbol"], mem["month"]))
    n_new = len(nx.configs()) + len(N2_CFG) + len(N3_CFG) + len(N4_CFG)
    n_trials = base_trials() + n_new
    _log(f"試驗數 = {n_trials}（帳本既有 {n_trials - n_new} + 這次 {n_new}）")
    base = vegas_book()
    base_day = daily_series(base)
    data = load_daily()
    panel = cx.build_panel(data)
    _log(f"面板：{len(panel.syms)} 個幣、{len(panel.days)} 天")

    fam = {}   # key -> dict(table, headline, trades(dict), rnd, nbr names, cand_day, ...)

    # ---------------- N1
    if "N1" not in skip:
        rows_by, table = {}, []
        for name, p in nx.configs().items():
            t0 = time.time()
            rows_by[name] = nx.run(panel, pool, p)
            table.append(n1_row(name, rows_by[name]))
            _log(f"[N1] {name}: IS 夏普 {table[-1]['is_sharpe']:.2f} OOS {table[-1]['oos_sharpe']:.2f} {time.time() - t0:.0f}s")
        table = pd.DataFrame(table)
        head = str(table.sort_values("is_sharpe", ascending=False)["config"].iloc[0])
        rnd = pd.concat([nx.run(panel, pool, nx.configs()[head], rng=np.random.default_rng(100 + k))
                         for k in range(a.draws)], ignore_index=True)
        port = lambda x: x.assign(side=1)  # noqa: E731   組合是一個整體，分群不要把多空當獨立
        fam["N1"] = {"family": F_N1, "table": table, "headline": head, "gate_tr": port(rows_by[head]),
                     "gate_rnd": port(rnd), "gate_nbrs": [port(rows_by[n]) for n in nx.neighbors(head)],
                     "cand_day": cx.daily(rows_by[head]), "nbrs": nx.neighbors(head), "configs": nx.configs(),
                     "yearly_tr": rows_by[head], "extra": {}}

    # ---------------- N2
    if "N2" not in skip:
        f3s = f3_series(panel, data)
        trades, table = {}, []
        for name, cfg in N2_CFG.items():
            t0 = time.time()
            trades[name] = finalize(n2_raw(data, f3s, cfg), mem)
            table.append(stats_row(F_N2, name, trades[name]))
            _log(f"[N2] {name}: {len(trades[name])} 筆 IS t {table[-1].get('is_t', float('nan')):.2f} {time.time() - t0:.0f}s")
        table = pd.DataFrame(table)
        head = pick_headline(table, 100, "is_t")
        rnd = random_entries(trades[head], data, mem, n2_plan_of(data, N2_CFG[head]), draws=min(a.draws, 10))
        fam["N2"] = {"family": F_N2, "table": table, "headline": head, "gate_tr": trades[head], "gate_rnd": rnd,
                     "gate_nbrs": [trades[n] for n in n2_neighbors(head)], "cand_day": daily_series(trades[head]),
                     "nbrs": n2_neighbors(head), "configs": N2_CFG, "yearly_tr": trades[head],
                     "extra": {"th": {k: list(v) for k, v in N2_TH.items()}}}

    # ---------------- N3
    if "N3" not in skip:
        evs = {s: cme_events(s) for s in ("BTCUSDT", "ETHUSDT")}
        trades, table = {}, []
        for name, cfg in N3_CFG.items():
            trades[name] = n3_book(evs, cfg["g"])
            table.append(stats_row(F_N3, name, trades[name]))
            _log(f"[N3] {name}: {len(trades[name])} 筆")
        table = pd.DataFrame(table)
        head = pick_headline(table, 30, "is_t")   # 週度事件筆數少，最少 30 筆（預先登記的 100 筆在 G=3% 不可能達到）
        rnd = pd.concat([n3_book(evs, N3_CFG[head]["g"], np.random.default_rng(300 + k)).assign(symbol_draw=k)
                         for k in range(a.draws)], ignore_index=True)
        fam["N3"] = {"family": F_N3, "table": table, "headline": head, "gate_tr": trades[head], "gate_rnd": rnd,
                     "gate_nbrs": [trades[n] for n in n3_neighbors(head)], "cand_day": daily_series(trades[head]),
                     "nbrs": n3_neighbors(head), "configs": N3_CFG, "yearly_tr": trades[head], "extra": {}}

    # ---------------- N4
    if "N4" not in skip:
        ret, mask = n4_load(mem, sorted(set(mem["symbol"])))
        _log(f"[N4] 1h 報酬矩陣 {ret.shape}")
        trades, rnds, table, extra = {}, {}, [], {}
        for name, cfg in N4_CFG.items():
            trades[name], rnds[name], extra[name] = n4_build(ret, mask, cfg, a.draws)
            row = stats_row(F_N4, name, trades[name])
            ins, oos = split(trades[name])
            row.update({"is_exp_gross_r": float(ins["gross_r"].mean()), "oos_exp_gross_r": float(oos["gross_r"].mean()),
                        "is_exp_gross_bp": float((ins["gross_r"] * extra[name]["hstd_is"]).mean() * 1e4),
                        "oos_exp_gross_bp": float((oos["gross_r"] * extra[name]["hstd_is"]).mean() * 1e4),
                        "long_hours": str(extra[name]["long_hours"]), "short_hours": str(extra[name]["short_hours"])})
            table.append(row)
            _log(f"[N4] {name}: 多 {extra[name]['long_hours']} 空 {extra[name]['short_hours']} OOS 毛 "
                 f"{row['oos_exp_gross_bp']:.2f}bp")
        table = pd.DataFrame(table)
        head = pick_headline(table, 100, "is_t")
        fam["N4"] = {"family": F_N4, "table": table, "headline": head, "gate_tr": trades[head], "gate_rnd": rnds[head],
                     "gate_nbrs": [trades[n] for n in n4_neighbors(head)], "cand_day": daily_series(trades[head]),
                     "nbrs": n4_neighbors(head), "configs": N4_CFG, "yearly_tr": trades[head], "extra": extra}

    # ---------------- 帳本：每組設定的樣本內
    if a.log:
        for k, f in fam.items():
            for _, row in f["table"].iterrows():
                log_experiment(family=f["family"], hypothesis=f"{f['family']} IS screen (pre-registered)",
                               params=f["configs"][row["config"]], data_source=DATA_SOURCE, period="2020-2023 IS",
                               trades=int(row["is_trades"]), exp_r=_r(row.get("is_exp_r")),
                               se_cluster=_r(row.get("is_se_cluster")), t_stat=_r(row.get("is_t"), 3),
                               sharpe=_r(row.get("is_sharpe"), 3), verdict="info", report=REPORT,
                               notes=f"IS screen of pre-registered grid ({len(f['configs'])}); configs=1")

    # ---------------- 主設定：關卡、相關性、逐年
    gate_rows, corr_rows, year_rows, summary = [], [], [], {"n_trials": n_trials, "configs": {}, "headlines": {}}
    for k, f in fam.items():
        head = f["headline"]
        _log(f"[{k}] 主設定（樣本內挑）：{head}")
        try:
            g = gate_one(f["family"], f["gate_tr"], base, f["gate_rnd"], f["gate_nbrs"], n_trials)
        except Exception as e:   # noqa: BLE001 - 關卡對極小樣本可能出錯，記錄下來
            _log(f"[{k}] 關卡錯誤：{e!r}")
            g = None
        cmp_ = compare_with(base_day, f["cand_day"])
        oos_ = compare_with(base_day[base_day.index >= IS_END], f["cand_day"][f["cand_day"].index >= IS_END])
        corr_rows.append({"family": f["family"], "headline": head, **cmp_, **{f"oos_{x}": v for x, v in oos_.items()}})
        ytr = f["yearly_tr"]
        if k == "N1":
            pw = cx.position_weeks(ytr)
            cd = f["cand_day"]
            yd = cd.groupby(cd.index.year)
            y = pw.groupby(pw["time"].dt.year).agg(trades=("r_net", "size"), exp_r=("r_net", "mean"),
                                                   win_rate=("r_net", lambda x: (x > 0).mean())).reset_index()
            y = y.rename(columns={"time": "year"}).merge(pd.DataFrame({"year": yd.mean().index,
                                                                        "sharpe": (yd.mean() / yd.std() * math.sqrt(365)).values,
                                                                        "ret_pct": (yd.sum() * 100).values}), on="year")
            year_rows.append(y.assign(family=f["family"]))
        else:
            year_rows.append(yearly(ytr, f["family"]))
        if g is not None:
            for c, v in g["checks"].items():
                gate_rows.append({"family": f["family"], "headline": head, "check": c, "label": v["label"],
                                  "status": v["status"], "value": json.dumps(v["value"], default=float, ensure_ascii=False),
                                  "threshold": v["threshold"], "detail": v["detail"]})
        summary["headlines"][f["family"]] = {
            "headline": head, "neighbors": f["nbrs"], "extra": f["extra"], "corr_vegas": cmp_, "corr_vegas_oos": oos_,
            "corr_daily": cmp_["corr_daily"], "sharpe_combined": cmp_["sharpe_combined"],
            "gate": None if g is None else {"verdict": g["verdict"], "passed": g["passed"], "incomplete": g["incomplete"],
                                            "failed": g["failed"], "skipped": g["skipped"], "info": g["info"],
                                            "checks": {c: {"label": v["label"], "value": v["value"],
                                                           "threshold": v["threshold"], "status": v["status"],
                                                           "detail": v["detail"]} for c, v in g["checks"].items()}}}
        for _, row in f["table"].iterrows():
            summary["configs"][f"{f['family']}/{row['config']}"] = {
                kk: _f(row.get(f"{lab}_{m}")) for lab in ("is", "oos") for m, kk in
                (("trades", f"{lab}_trades"), ("exp_r", f"{lab}_exp_r"), ("t", f"{lab}_t"), ("sharpe", f"{lab}_sharpe"))}
        if a.log:
            row = f["table"].set_index("config").loc[head]
            log_experiment(family=f["family"], hypothesis="headline OOS (selected on IS only)", params=f["configs"][head],
                           data_source=DATA_SOURCE, period="2024-2026 OOS", trades=int(row["oos_trades"]),
                           exp_r=_r(row.get("oos_exp_r")), se_cluster=_r(row.get("oos_se_cluster")),
                           t_stat=_r(row.get("oos_t"), 3), sharpe=_r(row.get("oos_sharpe"), 3), verdict="info",
                           report=REPORT, notes="same config as IS row, not a new trial; configs=0")
            if g is not None:
                log_experiment(family=f["family"], hypothesis="validation_gate on headline (baseline=vegas PIT book)",
                               params={**f["configs"][head], "n_trials": n_trials}, data_source=DATA_SOURCE,
                               period="2020-2026", trades=g["info"]["trades"], exp_r=_r(g["info"]["mean_r"]),
                               t_stat=_r(g["checks"]["cluster_t"]["value"], 3), sharpe=_r(g["info"].get("sharpe_annual"), 3),
                               verdict="fail" if not g["passed"] else ("info" if g["incomplete"] else "pass"),
                               report=REPORT, notes=f"{g['verdict']}; corr_daily={cmp_['corr_daily']:.3f}; configs=0")
        _log(f"[{k}] {None if g is None else g['verdict']}；日相關 {cmp_['corr_daily']:+.3f}")

    grid = pd.concat([f["table"].assign(headline=lambda t, f=f: t["config"] == f["headline"]) for f in fam.values()],
                     ignore_index=True)
    grid.to_csv(OUT_CSV + f"new_strategies_grid_{TAG}.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT_CSV + f"new_strategies_gate_{TAG}.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT_CSV + f"new_strategies_corr_{TAG}.csv", index=False)
    pd.concat(year_rows, ignore_index=True).to_csv(OUT_CSV + f"new_strategies_yearly_{TAG}.csv", index=False)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
