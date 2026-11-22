"""舊研究（時點幣池／成本框架之前做的）搬到「時點幣池 + r_net 成本」的長帳本，給 research/cv_all.py 用。2026-11-22。

python -m research.cv_export_old_indicators [--only fam] [--fresh]   → output/cv_ledgers/<fam>.pkl / .json

這些研究原本用固定的 84 幣（派網上市的市值前 100 名快照）或 TOP100 快照、只扣手續費；這裡全部改成
幣安 USDT 永續、時點幣池（進場月份屬於上個月成交額前 100 名，grid_trading.pit_universe.membership）、
2020-01-01 起、手續費每邊 0.05%（模擬內）+ grid_trading.costs.apply_costs（滑價分級 + 資金費率）= r_net。
同一個 config、同週期、同一個幣同時只一張（新訊號的 K 棒 i >= 前一張出場那根 j）。訊號只用已收盤 K 棒，
收盤後下一根開盤進場（照原研究）。單行程（workers=1）。每個 (群組, 幣) 的結果快取在 data/cv_cache/，可續跑（--fresh 重算）。

家族（名稱對應 research/experiments.csv）：
  basic_indicators_trend     順勢派/維加斯/背離/順勢回檔規則 x 4H/2H x 出場（live、tp1、自己的出場）。
                             重用 classic_indicator_study 的 STRATEGIES / features / run_exit（只取原本 45 個「預設」規則，
                             不含參數鄰近版本）。原登記 204 組，這裡 196 組（見 json notes）。
  basic_indicators_reversal  9 個反轉規則（RSI30/70、RSI2、布林回歸、KD、威廉、CCI、MFI、SMI、乖離）x 4H/2H x 6 種出場
                             （live、tp0.5/1/2/3、自己的出場）= 108 組，和登記一致。
  outside_day                Larry Williams 外包日反轉：日線/4H x 多空 x 止損（1/2/3 ATR、外包日極值）x 要不要低開 = 32 組（幣安部分）。
                             原程式 research/outside_day.py 不在 repo，依報告規則重寫（見 _outside_sim）。
  alt_meanrev                布林 + RSI 均值回歸 + 盤整過濾，2H/4H 各 80 組 = 160 組（alt_strategies.meanrev_signals）。
  alt_vwap                   週錨定 VWAP 偏離回歸 24 組 + VWAP 順勢回踩 4 組 = 28 組，4H（alt_strategies.vwap_signals）。
  alt_xsection_momentum      橫斷面強弱輪動 96 組（alt_momentum.run；每個再平衡日的多/空腿各算一筆，kind=trades）。
  regime_filter              實盤維加斯 + BTC 日線 EMA200 / 200 週均線牛熊濾網 3 種規則 = 6 組，另加不濾網的 live 基準 = 7 組。
  be_timing                  實盤維加斯 TP1/TP2 後保本時機 A（實盤）B C D E = 5 組。
  tp_ladder                  實盤維加斯止盈階梯 1/3/5R（實盤）、1/2/3R、2/4/6R、1/2/4R = 4 組。
  （後三者 4H、2H 兩個帳本各自一張一幣，合在同一個 config；原研究是 4H+2H 同幣同向只留一張，這裡不做。）

和原研究的共同差異：幣池 84 幣 / TOP100 快照 -> 時點幣池；資料 1H 月檔重取樣 -> 幣安直接的 2h/4h/1d 月檔；
只有手續費 -> 加滑價（依 30 日均成交額分級）和資金費率；2020 起（原本就是 2020 起）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import research.classic_indicator_study as cis
from grid_trading import indicators as ind
from grid_trading.altsim import Bars, run_signals
from grid_trading.costs import apply_costs
from grid_trading.pit_universe import fetch_symbol, membership
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from research import alt_momentum as am
from research import alt_strategies as ast_
from research.cv_export_vegas_live import save
from research.stop_research import LIVE_ATR, MAX_BARS, MIN_STOP_PCT, sim_many

START = pd.Timestamp("2020-01-01", tz="UTC")
CACHE = Path("data/cv_cache")
FEE = 0.0005
KL = {"4H": "4h", "2H": "2h", "1D": "1d"}
BPD = {"4H": 6, "2H": 12, "1D": 1}
KEEP = ["config", "symbol", "side", "time", "exit_t", "r", "r_net", "interval"]

REVERSAL_RULES = ["rsi_30_70", "rsi2", "bb_revert", "kd_cross", "williams_r", "cci", "mfi", "smi", "bias"]
REVERSAL_EXITS = ["live", "tp0.5", "tp1", "tp2", "tp3", "natural"]
TREND_EXITS = ["live", "tp1", "natural"]
TREND_RULES = [s["name"] for s in cis.STRATEGIES if s["base"] is None and s["name"] not in REVERSAL_RULES]
RULE_BY = cis.BY_NAME

FAMILIES = {
    "basic_indicators_trend": ("classic", "順勢派基礎指標（含維加斯、背離、順勢回檔）", "signals/report_basic_indicators_opus_max_20261013.md", 204),
    "basic_indicators_reversal": ("classic", "反轉派基礎指標（超買超賣）", "signals/report_basic_indicators_opus_max_20261013.md", 108),
    "outside_day": ("outside", "外包日反轉（Larry Williams）", "signals/report_outside_day_20261015.md", 48),
    "alt_meanrev": ("alt", "均值回歸（布林+RSI+盤整過濾）", "signals/report_alt_strategies_20261012.md", 160),
    "alt_vwap": ("alt", "VWAP 偏離回歸與順勢回踩", "signals/report_alt_strategies_20261012.md", 28),
    "alt_xsection_momentum": ("xsec", "橫斷面強弱輪動", "signals/report_alt_strategies_20261012.md", 96),
    "regime_filter": ("vegasvar", "維加斯 + BTC 牛熊濾網", "signals/report_regime_filter_20261018.md", 6),
    "be_timing": ("vegasvar", "維加斯保本時機", "signals/report_be_timing_20261018.md", 5),
    "tp_ladder": ("vegasvar", "維加斯止盈階梯", "signals/report_tp_ladder_20261018.md", 4),
}
NOTES = {
    "basic_indicators_trend": "幣池 84 幣快照 -> 時點幣池；只扣手續費 -> r_net；2H 由 1H 合成 -> 幣安直接 2h。config=規則|出場|週期。"
        "原登記 204 組，這裡用 36 個預設非反轉規則（順勢 22、維加斯 4、背離 4、順勢回檔 6）x 4H/2H x (live、tp1、有自己出場的才加 natural) = 196 組；"
        "不含 11 個參數鄰近版本（rsi_25_75 等）。登記數 204 的精確組成在報告裡沒有逐項列出，差 8 組無法對上。",
    "basic_indicators_reversal": "同 trend：時點幣池、r_net。config=規則|出場|週期，9 規則 x 2 週期 x 6 出場 = 108，和登記一致。",
    "outside_day": "原程式 outside_day.py 不在 repo，依報告規則重寫：外包日（高>前高、低<前低、收<前低；空單鏡像），下一根開盤進場，"
        "止損 1/2/3 ATR(14) 或外包日極值（=1R），第一個開盤價優於進場價的那根開盤出場，最多 20 日（4H 60 根）收盤出；gap=True 要求低開（空單高開）。"
        "登記 48 組 = 幣安 1d 16 + 幣安 4h 16 + 派網 1d 16；派網部分用不到時點幣池，這裡只有幣安 32 組。成本原本多算了 0.05% 滑價，現在改用 apply_costs。",
    "alt_meanrev": "幣池 83 檔快照 -> 時點幣池；2H/4H 各 80 組。config=原 cfg 名稱|週期。signals 用 alt_strategies.meanrev_signals，出場 altsim.sim（止損 + 回到布林中線，最多 30/48 根）。",
    "alt_vwap": "只有 4H：24 組 vwapRev + 4 組 vwapPull（不含成交量分布 vpReenter 12 組，原登記 28 組不含它）。其餘同 alt_meanrev。",
    "alt_xsection_momentum": "kind=trades：每個再平衡日的多/空各 k 檔各算一筆（和原研究一樣，不是組合報酬）；幣池改成「再平衡後進場月份屬於時點前 100 名」且 30 日均成交額 >= 10M。"
        "週一 00:00 UTC 週日日線收盤再平衡，到下次再平衡收盤平倉，中途碰止損先出。",
    "regime_filter": "牛熊濾網用 BTC 日線 EMA200（前一根已收的日線）與 200 週均線（BTC 現貨 + CMC 早期資料，同原研究）。3 規則 x 2 判斷 = 6 組 + 不濾網基準 live 共 7 組（登記 6 不含基準）。"
        "原研究 4H+2H 同幣同向只留一張，這裡改成各週期各一張。",
    "be_timing": "A 實盤（TP1 保本、TP2 移到 TP1）、B TP1 不動 TP2 保本、C TP1 -0.5R TP2 保本、D TP1 不動 TP2 移 TP1、E TP1 -0.5R TP2 移 TP1。時點幣池、r_net，各週期各一張。",
    "tp_ladder": "1/3/5R（實盤）、1/2/3R、2/4/6R、1/2/4R，每個 TP 1/4、保本與追蹤照實盤。不含原本多比的 1:1 / 1:1.5 全平兩個（登記 4 組不含）。時點幣池、r_net，各週期各一張。",
}


# ---------------------------------------------------------------- 共用

def _log(*a):
    print(*a, file=sys.stderr, flush=True)


@njit(cache=True)
def _oaat(g, i, j):
    keep = np.zeros(len(i), np.bool_)
    busy, prev = -1, -1
    for k in range(len(i)):
        if g[k] != prev:
            busy, prev = -1, g[k]
        if i[k] < busy:
            continue
        keep[k] = True
        busy = j[k]
    return keep


def one_at_a_time(tr: pd.DataFrame) -> pd.DataFrame:
    """每個 (config, interval) 內同一個幣同時只一張：i >= 前一張的出場那根 j；資料結束還沒出場（kind -1）的丟掉。
    tr 必須只含單一個幣（呼叫端逐幣處理）。"""
    if tr.empty:
        return tr
    tr = tr[tr["kind"] != -1]
    g = pd.factorize(tr["config"].astype(str) + "|" + tr["interval"].astype(str))[0]
    tr = tr.assign(_g=g).sort_values(["_g", "i"], kind="stable")
    keep = _oaat(tr["_g"].to_numpy(np.int64), tr["i"].to_numpy(np.int64), tr["j"].to_numpy(np.int64))
    return tr[keep].drop(columns="_g")


_MEM: dict | None = None


def pool() -> dict[str, set]:
    """symbol -> {進場月份 'YYYY-MM'}（上個月成交額前 100 名）。"""
    global _MEM
    if _MEM is None:
        mem = membership(n=100)
        mem["month"] = mem["month"].astype(str)
        _MEM = {s: set(g["month"]) for s, g in mem.groupby("symbol")}
    return _MEM


def load(sym: str, interval: str) -> pd.DataFrame | None:
    df = fetch_symbol(sym, KL[interval])
    if df is None or len(df) == 0:
        return None
    df = df.reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.as_unit("ns")
    return df


def ent_ok(df: pd.DataFrame, sym: str) -> np.ndarray:
    """ent_ok[i]：在第 i 根收盤訊號、第 i+1 根開盤進場時，進場時間在 2020 起且進場月份屬於時點幣池。"""
    t = pd.DatetimeIndex(df["time"])
    ok_bar = np.isin(t.strftime("%Y-%m"), list(pool().get(sym, ()))) & (t >= START)
    out = np.zeros(len(df), bool)
    out[:-1] = ok_bar[1:]
    return out


def adv_series(df: pd.DataFrame, interval: str) -> np.ndarray:
    b = BPD[interval]
    return ((df["volume"] * df["close"]).rolling(30 * b, min_periods=b).sum() / 30).to_numpy()


def frame(sym, interval, config, t, idx, ds, r, j, kind, risk_pct, adv) -> pd.DataFrame:
    """t：每根 K 棒的開盤時間（DatetimeIndex）。進場 = t[idx+1]，出場 = t[j]（出場那根的開盤時間，和 pit_backtest 同）。"""
    n = len(t)
    idx = np.asarray(idx, np.int64)
    j = np.asarray(j, np.int64)
    return pd.DataFrame({"config": config, "symbol": sym, "interval": interval, "i": idx, "j": j, "kind": np.asarray(kind, np.int64),
                         "side": np.asarray(ds).astype(np.int64), "time": t[np.minimum(idx + 1, n - 1)],
                         "exit_t": t[np.minimum(j, n - 1)], "r": np.asarray(r, float), "risk_pct": np.asarray(risk_pct, float),
                         "adv_usdt": adv[idx]})


def finish(raw: pd.DataFrame | None, sym: str) -> pd.DataFrame:
    """一個幣的原始交易 -> 一張一幣 -> 滑價與資金費率（r_net）。"""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=KEEP)
    raw = one_at_a_time(raw)
    if raw.empty:
        return pd.DataFrame(columns=KEEP)
    out = apply_costs(raw.assign(symbol=sym))
    return out[KEEP].reset_index(drop=True)


def cached(group: str, sym: str, fresh: bool, fn):
    path = CACHE / group / f"{sym}.pkl"
    if path.exists() and not fresh:
        return pd.read_pickle(path)
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, path)
    return res


def _cat(L):
    return pd.concat(L, ignore_index=True) if L else None


# ---------------------------------------------------------------- 群組 classic：基礎指標

def classic_unit(sym: str, tfs=("4H", "2H")) -> dict[str, pd.DataFrame]:
    trend, rev = [], []
    for tf in tfs:
        df = load(sym, tf)
        if df is None or len(df) < cis.WARMUP + 100:
            continue
        x = cis.features(df, tf)
        n = len(df)
        t = pd.DatetimeIndex(df["time"])
        atr, c = x["atr"], x["c"]
        valid = (np.arange(n) >= cis.WARMUP) & (np.arange(n) + 1 < n) & np.isfinite(atr) & (atr > 0) & ent_ok(df, sym)
        if not valid.any():
            continue
        arr = {k: x[k] for k in ("o", "h", "l", "c", "atr")}
        adv = adv_series(df, tf)
        for name in dict.fromkeys(TREND_RULES + REVERSAL_RULES):
            st = RULE_BY[name]
            if name == "vegas_live":
                lg, sh = cis.vegas_live_signals(df, tf)
                xl = xs = None
            else:
                lg, sh, xl, xs = st["fn"](x)
            lg, sh = np.asarray(lg, bool) & valid, np.asarray(sh, bool) & valid
            idx = np.r_[np.flatnonzero(lg), np.flatnonzero(sh)]
            if not len(idx):
                continue
            ds = np.r_[np.ones(lg.sum(), np.int64), -np.ones(sh.sum(), np.int64)]
            o = np.argsort(idx, kind="stable")
            idx, ds = idx[o], ds[o]
            is_rev = name in REVERSAL_RULES
            for ex in (REVERSAL_EXITS if is_rev else TREND_EXITS):
                if ex == "natural":
                    if not st["natural"] or xl is None:
                        continue
                    r, hit, j, kind = cis.run_exit(arr, idx, ds, tf, "natural", 0.0, np.asarray(xl, bool) & np.isfinite(c),
                                                   np.asarray(xs, bool) & np.isfinite(c), st["natural"])
                    kstop = cis.NATURAL_ATR
                else:
                    r, hit, j, kind = cis.run_exit(arr, idx, ds, tf, ex, 0.0)
                    kstop = cis.LIVE_ATR
                price = c[idx]
                rp = np.maximum(kstop * atr[idx], cis.MIN_STOP_PCT * price) / price
                f = frame(sym, tf, f"{name}|{ex}|{tf}", t, idx, ds, r, j, kind, rp, adv)
                (rev if is_rev else trend).append(f)
    return {"basic_indicators_trend": finish(_cat(trend), sym), "basic_indicators_reversal": finish(_cat(rev), sym)}


# ---------------------------------------------------------------- 群組 alt：均值回歸、VWAP

def make_bars(sym: str, tf: str, df: pd.DataFrame) -> Bars:
    a = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}
    return Bars(sym, KL[tf], df["time"].astype("int64").to_numpy(), a["open"], a["high"], a["low"],
                a["close"], a["volume"], a["high"].tolist(), a["low"].tolist(), a["close"].tolist(), df)


def _alt_rows(sym, tf, cfg, rows, t, adv):
    if not rows:
        return None
    cols = list(zip(*rows))  # cfg,sym,side,i,time,exit_time,r,risk_pct,tpk,bars
    idx = np.array(cols[3], np.int64)
    j = idx + np.array(cols[9], np.int64)
    return frame(sym, tf, f"{cfg}|{KL[tf]}", t, idx, np.array(cols[2]), np.array(cols[6]), j,
                 np.full(len(idx), 5), np.array(cols[7]), adv)


def alt_unit(sym: str) -> dict[str, pd.DataFrame]:
    mr, vw = [], []
    for tf in ("4H", "2H"):
        df = load(sym, tf)
        if df is None or len(df) < 400:
            continue
        b = make_bars(sym, tf, df)
        ok = ent_ok(df, sym)
        t = pd.DatetimeIndex(df["time"])
        adv = adv_series(df, tf)
        x = ast_.base_ind(b)
        mb = {"4H": 30, "2H": 48}[tf]

        def keep(sigs):
            return [s for s in sigs if ok[s[0]]]

        for cfg, p in ast_.meanrev_grid():
            f = _alt_rows(sym, tf, cfg, run_signals(b, cfg, keep(ast_.meanrev_signals(b, x, p)), FEE, mb), t, adv)
            if f is not None:
                mr.append(f)
        if tf == "4H":
            cache: dict = {}
            for cfg, p in ast_.vwap_grid():
                rows = run_signals(b, cfg, keep(ast_.vwap_signals(b, x, p, cache)), FEE, mb if p["kind"] == "rev" else 240)
                f = _alt_rows(sym, tf, cfg, rows, t, adv)
                if f is not None:
                    vw.append(f)
    return {"alt_meanrev": finish(_cat(mr), sym), "alt_vwap": finish(_cat(vw), sym)}


# ---------------------------------------------------------------- 群組 xsec：橫斷面強弱輪動

def xsec_ledger(syms: list[str], fresh: bool) -> pd.DataFrame:
    path = CACHE / "xsec" / "ledger.pkl"
    if path.exists() and not fresh:
        return pd.read_pickle(path)
    mem = pool()
    data = {}
    for s in syms:
        df = load(s, "1D")
        if df is not None and len(df) > 120:
            data[s] = make_bars(s, "1D", df)
    DAY = am.DAY
    days = pd.Index(sorted({int(t // DAY) for b in data.values() for t in b.t}))
    close = pd.DataFrame({c: pd.Series(b.c, index=b.t // DAY) for c, b in data.items()}).reindex(days)
    dvol_raw = pd.DataFrame({c: pd.Series(b.c * b.v, index=b.t // DAY) for c, b in data.items()}).reindex(days)
    dvol_raw = dvol_raw.rolling(30, min_periods=20).mean()
    atr = pd.DataFrame({c: pd.Series(ind.atr(b.df).to_numpy(), index=b.t // DAY) for c, b in data.items()}).reindex(days)
    hist = close.notna().cumsum()
    # 時點幣池：再平衡日（日線收盤）後隔天進場，進場日所在月份要屬於幣池；2020 起
    ent_day = pd.to_datetime((np.asarray(days) + 1) * 86400, unit="s", utc=True)
    months = ent_day.strftime("%Y-%m").to_numpy()
    started = np.asarray(ent_day >= START)
    mask = pd.DataFrame({c: np.isin(months, list(mem.get(c, ()))) & started for c in data}, index=days)
    dvol = dvol_raw.where(mask)
    parts = []
    for cfg, p in am.configs():
        rows = am.run(p, data, days, close, dvol, atr, hist)
        if not rows:
            continue
        cols = list(zip(*rows))  # c, side, i, t, xt, r, rp, tpk, bars
        sy = np.array(cols[0])
        idx = np.array(cols[2], np.int64)
        tns = np.array(cols[3], np.int64)
        dd = tns // DAY - 1
        adv = np.array([dvol_raw.at[d, s] if d in dvol_raw.index else np.nan for d, s in zip(dd, sy)])
        parts.append(pd.DataFrame({"config": cfg, "symbol": sy, "interval": "1D", "i": idx, "j": idx + np.array(cols[8], np.int64),
                                   "kind": 5, "side": np.array(cols[1], np.int64),
                                   "time": pd.to_datetime(tns, unit="ns", utc=True),
                                   "exit_t": pd.to_datetime(np.array(cols[4], np.int64), unit="ns", utc=True),
                                   "r": np.array(cols[5]), "risk_pct": np.array(cols[6]), "adv_usdt": adv}))
        _log("  xsec", cfg, len(rows))
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw["time"] >= START]
    res = pd.concat([finish(g, s) for s, g in raw.groupby("symbol", sort=False)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_pickle(path)
    return res


# ---------------------------------------------------------------- 群組 outside：外包日

STOPS = ("1atr", "2atr", "3atr", "bar")


@njit(cache=True)
def _outside_sim(op, hi, lo, cl, i, d, stop_mult, atr, olo, ohi, max_bars, fee, bar_stop):
    """第 i 根是外包日，第 i+1 根開盤進場。回傳 (r, 出場 K 棒 j, kind, risk_pct)；kind 1 止損、4 第一個優於進場價的開盤、5 超時、
    -1 資料結束、-2 風險距離無效（丟掉）。每根先看開盤是否優於進場價（j >= i+2），再看止損（止損價成交），最後看超時（收盤出）。"""
    n = len(op)
    e = op[i + 1]
    if bar_stop:
        risk = d * (e - (olo if d > 0 else ohi))
    else:
        risk = stop_mult * atr
    risk = max(risk, 0.001 * e)
    if not (risk > 0) or not np.isfinite(risk):
        return 0.0, i + 1, -2, 0.0
    stop = e - d * risk
    for j in range(i + 1, n):
        if j >= i + 2 and d * (op[j] - e) > 0:
            return (d * (op[j] - e) - fee * (e + op[j])) / risk, j, 4, risk / e
        if (d > 0 and lo[j] <= stop) or (d < 0 and hi[j] >= stop):
            return (d * (stop - e) - fee * (e + stop)) / risk, j, 1, risk / e
        if j - i >= max_bars:
            return (d * (cl[j] - e) - fee * (e + cl[j])) / risk, j, 5, risk / e
    return 0.0, n - 1, -1, risk / e


def outside_unit(sym: str) -> dict[str, pd.DataFrame]:
    parts = []
    for tf, mb in (("1D", 20), ("4H", 60)):
        df = load(sym, tf)
        if df is None or len(df) < 100:
            continue
        n = len(df)
        t = pd.DatetimeIndex(df["time"])
        op, hi, lo, cl = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = ind.atr(df, 14).to_numpy(float)
        ph, pl = np.r_[np.nan, hi[:-1]], np.r_[np.nan, lo[:-1]]
        ok = ent_ok(df, sym) & (np.arange(n) >= 30) & np.isfinite(atr) & (atr > 0)
        adv = adv_series(df, tf)
        outside = (hi > ph) & (lo < pl)
        nxt_open = np.r_[op[1:], np.nan]
        for side, base in ((1, outside & (cl < pl)), (-1, outside & (cl > ph))):
            for gap in (True, False):
                sig = base & ok
                if gap:
                    sig = sig & ((nxt_open < cl) if side > 0 else (nxt_open > cl))
                idx = np.flatnonzero(sig)
                if not len(idx):
                    continue
                for stop in STOPS:
                    mult = 0.0 if stop == "bar" else float(stop[0])
                    res = [_outside_sim(op, hi, lo, cl, int(i), side, mult, atr[i], lo[i], hi[i], mb, FEE, stop == "bar") for i in idx]
                    r, j, kind, rp = (np.array(v) for v in zip(*res))
                    m = kind != -2
                    parts.append(frame(sym, tf, f"{tf.lower()}|{'long' if side > 0 else 'short'}|{stop}|gap{int(gap)}", t,
                                       idx[m], np.full(int(m.sum()), side), r[m], j[m], kind[m], rp[m], adv))
    return {"outside_day": finish(_cat(parts), sym)}


# ---------------------------------------------------------------- 群組 vegasvar：維加斯變體

Q = (0.25, 0.25, 0.25)
TRAIL_K = 3.0
SIM_MODES = {  # 名稱 -> (TP 的 R, 各 TP 平倉比例, sim_many 的 protect)
    "live": ((1.0, 3.0, 5.0), Q, 0),
    "tp123": ((1.0, 2.0, 3.0), Q, 0),
    "tp246": ((2.0, 4.0, 6.0), Q, 0),
    "tp124": ((1.0, 2.0, 4.0), Q, 0),
    "beB": ((1.0, 3.0, 5.0), Q, 4),
    "beC": ((1.0, 3.0, 5.0), Q, 3),
    "beD": ((1.0, 3.0, 5.0), Q, 2),
    "beE": ((1.0, 3.0, 5.0), Q, 1),
}
TP_CFG = {"live": "1/3/5R(live)", "tp123": "1/2/3R", "tp246": "2/4/6R", "tp124": "1/2/4R"}
BE_CFG = {"live": "A live", "beB": "B TP1不動TP2保本", "beC": "C TP1-0.5R TP2保本", "beD": "D TP1不動TP2移TP1",
          "beE": "E TP1-0.5R TP2移TP1"}
REGIME_RULES = {
    "trend_only": lambda bull, s: np.where(bull, s > 0, s < 0),    # 牛市只做多、熊市只做空
    "no_long_bear": lambda bull, s: bull | (s < 0),                 # 熊市不做多
    "no_short_bull": lambda bull, s: ~bull | (s > 0),               # 牛市不做空
}
_REG = None


def regimes():
    """BTC 牛熊（前一根已收盤）：日線 EMA200、200 週均線。load_btc_long 失敗時改用 BTCUSDT 永續日線（早期缺的視為牛市）。"""
    global _REG
    if _REG is not None:
        return _REG
    from research.regime_filter import regimes as _regimes
    try:
        daily, weekly = _regimes()
        src = "spot+CMC"
    except Exception as e:  # noqa: BLE001
        _log("load_btc_long 失敗，改用 BTCUSDT 永續日線：", repr(e)[:200])
        d = fetch_symbol("BTCUSDT", "1d")
        d = d.set_index(pd.to_datetime(d["time"], utc=True))
        daily = pd.DataFrame({"avail": d.index + pd.Timedelta(days=1),
                              "ema200d": (d["close"] > ind.ema(d["close"], 200)).astype(float).to_numpy()})
        w = d["close"].resample("W-SUN", label="right", closed="right").last()
        ma = w.rolling(200).mean()
        weekly = pd.DataFrame({"avail": w.index + pd.Timedelta(days=1), "sma200w": np.where(ma.isna(), np.nan, (w > ma).astype(float))})
        src = "perp-fallback"
    daily["avail"] = pd.to_datetime(daily["avail"], utc=True).dt.as_unit("ns")
    weekly["avail"] = pd.to_datetime(weekly["avail"], utc=True).dt.as_unit("ns")
    _log("BTC 牛熊資料來源：", src)
    _REG = (daily, weekly, src)
    return _REG


def tag_regime(tr: pd.DataFrame) -> pd.DataFrame:
    daily, weekly, _ = regimes()
    tr = tr.sort_values("time").reset_index(drop=True)
    tr["time"] = pd.to_datetime(tr["time"], utc=True).dt.as_unit("ns")
    for reg in (daily, weekly):
        m = pd.merge_asof(tr[["time"]], reg.sort_values("avail"), left_on="time", right_on="avail", direction="backward")
        tr[reg.columns[1]] = m[reg.columns[1]].to_numpy()
    return tr


def vegas_signals(df: pd.DataFrame, interval: str):
    strat = VegasTrend()
    if len(df) < strat.min_bars + 50:
        return None
    p = strat.prepare(df).reset_index(drop=True)
    idx, ds = [], []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < strat.min_bars or i + 1 >= len(p):
            continue
        hit = strat.check(p, int(i))
        if hit is None or not strat.accept(p, int(i), interval):
            continue
        idx.append(int(i))
        ds.append(1 if hit[0] == LONG else -1)
    return p, np.array(idx, np.int64), np.array(ds, np.int64)


def vegasvar_unit(sym: str) -> dict[str, pd.DataFrame]:
    modes: dict[str, list] = {m: [] for m in SIM_MODES}
    for iv in ("4H", "2H"):
        df = load(sym, iv)
        if df is None:
            continue
        got = vegas_signals(df, iv)
        if got is None:
            continue
        p, idx, ds = got
        if not len(idx):
            continue
        keep = ent_ok(p, sym)[idx]
        idx, ds = idx[keep], ds[keep]
        if not len(idx):
            continue
        op, hi, lo, cl, atr = (p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
        price, a = cl[idx], atr[idx]
        risk = np.maximum(LIVE_ATR * a, MIN_STOP_PCT * price)
        stop0 = price - ds * risk
        zero = np.zeros(len(idx))
        isc = np.ones(len(hi), np.bool_)
        t = pd.DatetimeIndex(pd.to_datetime(p["time"], utc=True)).as_unit("ns")
        adv = adv_series(p, iv)
        for name, (tp_r, portions, protect) in SIM_MODES.items():
            tps = np.stack([price + ds * k * risk for k in tp_r], axis=1)
            r, hit, j, kind, _ = sim_many(op, hi, lo, cl, isc, idx + 1, ds, price, risk, stop0, tps, np.array(portions),
                                          protect, TRAIL_K * a, FEE, 0.0, MAX_BARS, 0, 0, zero, 0.0, 0.0, zero, 0, cl, cl, False)
            modes[name].append(frame(sym, iv, name, t, idx, ds, r, j, kind, risk / price, adv))
    modes = {m: pd.concat(L, ignore_index=True) for m, L in modes.items() if L}
    if not modes:
        return {k: finish(None, sym) for k in ("tp_ladder", "be_timing", "regime_filter")}

    def sel(table: dict) -> pd.DataFrame:
        return pd.concat([modes[m].assign(config=c) for m, c in table.items() if m in modes], ignore_index=True)

    tp, be = finish(sel(TP_CFG), sym), finish(sel(BE_CFG), sym)
    rg = []
    if "live" in modes:
        live = tag_regime(modes["live"])
        rg.append(live.assign(config="live（不濾網）"))
        for reg in ("ema200d", "sma200w"):
            bull = live[reg].fillna(1).astype(bool).to_numpy()
            for rn, f in REGIME_RULES.items():
                rg.append(live[f(bull, live["side"].to_numpy())].assign(config=f"{reg}|{rn}"))
    return {"tp_ladder": tp, "be_timing": be, "regime_filter": finish(_cat(rg), sym)}


# ---------------------------------------------------------------- 主程式

UNITS = {"classic": classic_unit, "alt": alt_unit, "outside": outside_unit, "vegasvar": vegasvar_unit}


def summarize(fam: str, led: pd.DataFrame) -> str:
    return (f"{fam}: rows={len(led)} configs={led['config'].nunique()} (登記 {FAMILIES[fam][3]}) "
            f"mean r_net={led['r_net'].mean():+.4f} r={led['r'].mean():+.4f} "
            f"{led['time'].min().date()}~{led['time'].max().date()} symbols={led['symbol'].nunique()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="只匯出這個家族（可重複）")
    ap.add_argument("--fresh", action="store_true", help="忽略 data/cv_cache 的快取")
    ap.add_argument("--max-symbols", type=int, default=0, help="除錯用：只跑前 N 個幣，不存檔、不快取")
    a = ap.parse_args()
    fams = a.only or list(FAMILIES)
    groups = sorted({FAMILIES[f][0] for f in fams})
    syms = sorted(pool())
    if a.max_symbols:
        syms = syms[:a.max_symbols]
    _log(f"{len(syms)} 個幣（時點幣池曾入選）；群組 {groups}")
    for g in groups:
        want = [f for f in fams if FAMILIES[f][0] == g]
        t0 = time.time()
        if g == "xsec":
            res = {"alt_xsection_momentum": xsec_ledger(syms, a.fresh)}
        else:
            acc: dict[str, list] = {f: [] for f in want}
            for k, s in enumerate(syms):
                out = UNITS[g](s) if a.max_symbols else cached(g, s, a.fresh, lambda s=s: UNITS[g](s))
                for f in want:
                    if f in out and len(out[f]):
                        acc[f].append(out[f])
                if k % 20 == 0:
                    _log(f"[{g}] {k}/{len(syms)} {s} {time.time() - t0:.0f}s")
            res = {f: pd.concat(L, ignore_index=True) for f, L in acc.items() if L}
        for f in want:
            led = res.get(f)
            if led is None or led.empty:
                _log(f"{f}: 沒有交易")
                continue
            for c in ("time", "exit_t"):
                led[c] = pd.to_datetime(led[c], utc=True)
            led = led.sort_values(["config", "time"]).reset_index(drop=True)
            _log(summarize(f, led))
            if a.max_symbols:
                continue
            save(f, led[KEEP], {"label": FAMILIES[f][1], "kind": "trades", "source": FAMILIES[f][2],
                                "configs_registered": FAMILIES[f][3],
                                "notes": "PIT top100 by prior-month volume, Binance USDT perps 2020+, net costs (apply_costs). " + NOTES[f]})
        _log(f"[{g}] 完成 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
