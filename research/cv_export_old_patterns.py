"""舊研究（時點幣池／成本框架之前做的）搬到「幣安 USDT 永續 + 時點幣池 + r_net」重跑，匯出長帳本給 research/cv_all.py 用。

python -m research.cv_export_old_patterns [--only fam] [--max-syms N]
  → output/cv_ledgers/<fam>.pkl / .json

家族（括號是原報告，設定網格照原研究事先登記的那一組，不新增、不挑）：
  vegas_entry_filters  維加斯進場濾網（entry_research）：41 個特徵 x 5 種去頭尾 x 4H/2H = 410 組 + 兩個週期的「未過濾實盤」= 412 組
  vegas_stoploss       維加斯止損變體（stop_research.variants() 96 種，2H、4H 的單合併成一組）
  smc_ict              SMC/ICT（smc_backtest）：3 週期 x 擺動點 k=3,5 x 42 種規則 = 252 組
  harmonic             諧波型態（harmonic_strategies）：9 型態 x 3 週期(1D/4H/1H) x 3 靈敏度 x 4 出場 = 324 組
  threads_strategies   Threads 三個想法（threads_strategies）：日線/4H 吞沒+EMA8/20、4H/1H 鯊魚諧波（每個策略 x 3 出場 x 有無 EMA200 濾網），
                       另加「AI 挑最近 14 天最強幣」機器人 4 組（1/3/7 天換、抱到止損）
  orderflow            訂單流（orderflow_backtest）：112 組 x 4H/2H = 224 組（需要主動買量與 OI／多空比 metrics，見 notes）

和原研究的共同差異（每個家族的 notes 也會寫）：
  1. 幣池：改成時點幣池（上個月成交額前 100 的幣安 USDT 永續，grid_trading.pit_universe.membership）；
     原研究用 84/100 個今天的市值前 100 快照。進場月份不在池內的訊號直接不做（先篩池，再做「同一個幣同時一張」）。
  2. 資料：直接用幣安原始代號的永續 K 線（fetch_symbol），2020-01-01 起進場；不接現貨、不做改名接續。
  3. 成本：模擬內每邊手續費 0.05%（原研究就有），再用 grid_trading.costs.apply_costs 加滑價（依 30 日均成交額分級）和資金費率 -> r_net。
  4. 同一個 config 內，同週期同一個幣同時只持一張。
  5. workers=1（單一行程）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.costs import apply_costs, fetch_funding
from grid_trading.pit_universe import coin_of, fetch_symbol, membership
from research.pit_backtest import KLINE, START

OUT = Path("output/cv_ledgers")
FEE = 0.0005
KEEP = ["config", "symbol", "side", "time", "exit_t", "r", "r_net", "interval"]
LOG_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - LOG_T0:7.0f}s] {msg}", file=sys.stderr, flush=True)


def save(family: str, led: pd.DataFrame, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    led.to_pickle(OUT / f"{family}.pkl")
    (OUT / f"{family}.json").write_text(json.dumps({"family": family, **meta}, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 共用：幣池、同一張、成本

def pit_index() -> tuple[pd.DataFrame, pd.MultiIndex]:
    mem = membership(n=100)
    mem["month"] = mem["month"].astype(str)
    return mem, pd.MultiIndex.from_arrays([mem["symbol"], mem["month"]])


def universe_symbols(mem: pd.DataFrame) -> list[str]:
    return sorted(set(mem.loc[mem["month"] >= "2020-01", "symbol"]))


def in_pit(pit: pd.MultiIndex, symbols, times) -> np.ndarray:
    months = pd.to_datetime(pd.Series(times), utc=True).dt.strftime("%Y-%m").to_numpy()
    return pd.MultiIndex.from_arrays([np.asarray(symbols), months]).isin(pit)


def adv30(df: pd.DataFrame, bars_per_day: int) -> np.ndarray:
    """近 30 天平均每日成交額（USDT），和 pit_backtest 同一個算法。"""
    return ((df["volume"] * df["close"]).rolling(30 * bars_per_day, min_periods=bars_per_day).sum() / 30).to_numpy()


def one_at_a_time(tr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """每個 by 群組內同時只持一張。tr 需要 i（進場訊號位置，bar 序號）、j（出場 bar 序號）、kind（-1 = 資料尾端未平倉）。
    和 stop_research.busy_filter 一樣：前一張出場那根之前的新訊號跳過。"""
    from research.stop_research import busy_filter
    tr = tr[tr["kind"] != -1].sort_values(by + ["i"], kind="stable").reset_index(drop=True)
    if tr.empty:
        return tr
    gid = tr.groupby(by, sort=False).ngroup().to_numpy(np.int64)
    take = busy_filter(gid, tr["i"].to_numpy(np.int64), tr["j"].to_numpy(np.int64), tr["kind"].to_numpy(np.int64))
    return tr[take]


def finish(tr: pd.DataFrame) -> pd.DataFrame:
    """交易表 -> 帳本：補滑價、資金費率。tr 需要 symbol、side、time、exit_t、r、risk_pct、adv_usdt、config、interval。"""
    tr = tr.copy()
    tr["time"] = pd.to_datetime(tr["time"], utc=True)
    tr["exit_t"] = pd.to_datetime(tr["exit_t"], utc=True)
    tr = tr[tr["time"] >= START].copy()
    tr["exit_t"] = tr["exit_t"].where(tr["exit_t"] >= tr["time"], tr["time"])
    funding = {}
    for s in tr["symbol"].unique():
        try:
            funding[s] = fetch_funding(s)
        except Exception as e:  # noqa: BLE001
            log(f"funding {s} 抓不到：{e}")
    out = apply_costs(tr, funding=funding, fetch=False)
    out["side"] = out["side"].astype(int)
    return out[KEEP + ["risk_pct"]].sort_values("exit_t").reset_index(drop=True)


def report(fam: str, led: pd.DataFrame, reg: int) -> None:
    log(f"{fam}: {len(led)} 列, {led['config'].nunique()} configs (登記 {reg}), 平均 r_net {led['r_net'].mean():+.4f}, "
        f"平均 r {led['r'].mean():+.4f}")


# ---------------------------------------------------------------- 維加斯：進場濾網 / 止損變體

LAST = {}


def _patched_load_frame(sym, interval, source):
    df = fetch_symbol(sym, KLINE[interval])
    bpd = {"4H": 6, "2H": 12}[interval]
    LAST["adv"] = adv30(df, bpd) if len(df) else np.array([])
    return df.reset_index(drop=True)


def _funding_daily(key: str):
    """entry_research.load_funding 的替代：用 costs.fetch_funding（幣安代號），每日 % = 每次費率 * 每日結算次數 * 100。"""
    sym = "BTCUSDT" if key == "BTC" else key
    d = fetch_funding(sym)
    if d is None or d.empty:
        return None
    d = d.sort_values("time").reset_index(drop=True)
    hrs = d["time"].diff().dt.total_seconds().div(3600).round().clip(1, 8).fillna(8)
    d["daily"] = d["rate"] * 24 / hrs * 100
    return d[["time", "daily"]]


def fam_vegas_entry(mem, pit, symbols):
    import research.entry_research as er
    er.load_frame = _patched_load_frame
    er.load_funding = _funding_daily
    from grid_trading.universe import TOP100
    mcap = {c: k for k, c in enumerate(TOP100)}
    rows = []
    for iv in ("4H", "2H"):
        feats, panels = [], {}
        for n, sym in enumerate(symbols):
            got = er._work((sym, iv, FEE, 0.0, "binance"))
            if got is None:
                continue
            _, f, panel = got
            key = "BTC" if sym == "BTCUSDT" else sym
            f["coin"] = key
            f["symbol"] = sym
            f["adv_usdt"] = panel["qvol30"].to_numpy()[f["i"].to_numpy()]
            feats.append(f)
            panels[key] = panel
            if n % 50 == 0:
                log(f"entry {iv} {n}/{len(symbols)}")
        f = pd.concat(feats, ignore_index=True)
        for c in ("time", "signal_close", "exit_t"):
            f[c] = pd.to_datetime(f[c], utc=True)
        f = f[(f["time"] >= START) & in_pit(pit, f["symbol"], f["time"])].reset_index(drop=True)
        f = er.add_cross_features(f, panels, iv)
        f["mcap_rank"] = f["symbol"].map(lambda s: mcap.get(coin_of(s)))  # 今天的市值排名（原研究的特徵，沒有時點版本）
        f = f.sort_values(["coin", "i"]).reset_index(drop=True)
        per = er.periods(f["time"])
        live = (f["live"] & (f["kind"] != -1)).to_numpy()
        log(f"entry {iv}: 候選 {len(f)}, 實盤訊號 {int(live.sum())}")

        def add(name, mask):
            sub = f[mask & (f["kind"] != -1).to_numpy()]
            sub = one_at_a_time(sub.assign(config=name, interval=iv), ["symbol"])
            rows.append(sub[["config", "symbol", "side", "time", "exit_t", "r", "risk_pct", "adv_usdt", "interval"]])

        add(f"{iv}|live", live)
        for feat in er.FEATURES:
            if feat not in f or f.loc[live, feat].notna().sum() < 100:
                continue
            q, _ = er.qcut_is(f[feat], live & per["2020-23"])
            for fname, keep in er.FILTERS.items():
                km = np.isin(q, keep) | (q == 0)
                add(f"{iv}|{feat}|{fname}", live & km)
    led = finish(pd.concat(rows, ignore_index=True))
    return led, ("維加斯進場濾網（41 特徵 x 5 切法 x 4H/2H + 未過濾實盤）", 410,
                 "時點幣池（不含 adv>=30M 門檻）；候選=趨勢+MACD 柱翻轉，live=RSI 54-70/30-46 且離通道 3 ATR 內；"
                 "出場用 entry_research 原本的實盤出場（4H：1R 平一半、保本、3ATR 追蹤；2H：1/3/5R 各 1/4）、最長 240 根；"
                 "五分位門檻照原研究用 2020-23 的實盤訊號定（有一點樣本內門檻，CV 時要留意）；"
                 "多出 2 組「未過濾實盤」（每週期一組），故 412 組 vs 登記 410；mcap_rank 特徵是今天（2026-10）的市值排名，"
                 "有存活者偏差，無時點版本；跨幣特徵（大盤廣度、BTC 狀態、強弱排名、同時訊號數）用全部時點池曾入選的合約計算；"
                 "資金費率特徵改用 costs.fetch_funding（同樣來自 data.binance.vision）。")


def fam_vegas_stop(mem, pit, symbols):
    import research.stop_research as sr
    sr.load_frame = _patched_load_frame
    vs = sr.variants()
    parts = []
    for iv in ("4H", "2H"):
        for n, sym in enumerate(symbols):
            data = sr.prepare(sym, iv, "binance", False)
            if data is None or not data["sigs"]:
                continue
            adv = LAST["adv"]
            got = sr.run_set(data, data["sigs"], vs, iv, FEE, 0.0, False)
            if got is None:
                continue
            res, meta, _ = got
            idx = meta["i"].to_numpy()
            mt = pd.to_datetime(meta["time"], utc=True)
            ok = (mt >= START).to_numpy() & in_pit(pit, [sym] * len(meta), mt)
            if not ok.any():
                continue
            for vi, v in enumerate(vs):
                parts.append(pd.DataFrame({
                    "config": v["name"], "symbol": sym, "side": meta["side"].to_numpy()[ok], "time": meta["time"].to_numpy()[ok],
                    "exit_t": res["exit_t"][vi][ok], "r": res["r"][vi][ok], "risk_pct": res["risk"][vi][ok],
                    "adv_usdt": adv[idx[ok]], "interval": iv, "i": idx[ok], "j": res["j"][vi][ok].astype(np.int64),
                    "kind": res["kind"][vi][ok].astype(np.int64)}))
            if n % 50 == 0:
                log(f"stop {iv} {n}/{len(symbols)}")
    tr = pd.concat(parts, ignore_index=True)
    tr["time"] = pd.to_datetime(tr["time"], utc=True)
    tr["exit_t"] = pd.to_datetime(tr["exit_t"], utc=True)
    tr = one_at_a_time(tr, ["config", "interval", "symbol"])
    led = finish(tr)
    return led, ("維加斯止損變體（96 種）", 96,
                 "時點幣池；進場同實盤維加斯訊號（RSI 54-70/30-46），2H、4H 的單合併成同一個 config（登記的 96 = variants() 的種類數）；"
                 "bar 層級模擬（沒有用 --sub 的 1H/15m 細 K 線判斷同一根先碰止損還 TP，一根 K 棒同時碰到兩邊時的先後用原研究的保守規則）；"
                 "出場 TP 結構用 stop_research.PLANS 原本設定（4H：1R 平一半+3ATR 追蹤；2H：1/3/5R 各 1/4），最長 240 根；"
                 "不含隨機進場基準；同週期同幣同時一張；單合併 2H+4H 時兩個週期的單各自獨立不互相去重。")


# ---------------------------------------------------------------- 共用：從 1H 合成 K 線（smc / harmonic / threads）

def fetch_1h(sym: str) -> pd.DataFrame:
    return fetch_symbol(sym, "1h").reset_index(drop=True)


# ---------------------------------------------------------------- SMC / ICT

def smc_simulate(df, setups, symbol, tf_h, fee, pool_ok, adv, expiry=20, cancel_r=2.0, max_days=40):
    """複製 smc_backtest.simulate，只多了「成交（進場）當下要在時點幣池內」的檢查和 adv、risk_pct 輸出。"""
    from grid_trading.signals.strategy import LONG, Signal
    from grid_trading.signals.tracker import track
    from research.smc_backtest import TP_R
    times = df["time"]
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    bar = pd.Timedelta(hours=tf_h)
    max_bars = max_days * 24 // tf_h
    busy_until = -1
    out = []
    for n, s in enumerate(setups):
        if s.bar <= busy_until:
            continue
        d = s.direction
        risk = d * (s.entry - s.sl)
        fill = None
        if s.market:
            fill = s.bar
        else:
            nxt = setups[n + 1].bar if n + 1 < len(setups) else len(df) - 1
            end = min(s.bar + expiry, nxt, len(df) - 1)
            cancel = s.entry + d * cancel_r * risk
            for t in range(s.bar + 1, end + 1):
                if (d > 0 and low[t] <= s.entry) or (d < 0 and high[t] >= s.entry):
                    fill = t
                    break
                if (d > 0 and high[t] >= cancel) or (d < 0 and low[t] <= cancel):
                    break
        if fill is None:
            continue
        t_in = times.iat[fill] + bar
        if t_in < START or not pool_ok(t_in):
            continue
        rec = {"symbol": symbol, "side": 1 if d > 0 else -1, "time": t_in, "risk_pct": risk / s.entry,
               "adv_usdt": adv[min(fill, len(adv) - 1)]}
        if not s.market and ((d > 0 and low[fill] <= s.sl) or (d < 0 and high[fill] >= s.sl)):
            cost = fee * (s.entry + s.sl)
            rec.update(r=-(risk + cost) / risk, exit_t=times.iat[fill])
            busy_until = fill
        else:
            sig = Signal(symbol, f"{tf_h}H", LONG if d > 0 else "SHORT", t_in, s.entry, s.sl,
                         [s.entry + d * k * risk for k in TP_R], "smc")
            res = track(sig, df, fee, max_bars)
            if res.status != "CLOSED":
                continue
            ex = res.events[-1].time
            rec.update(r=res.realized_r, exit_t=ex)
            busy_until = int(times.searchsorted(ex))
        out.append(rec)
    return out


def fam_smc(mem, pit, symbols, tfs=("1H", "2H", "4H")):
    from grid_trading import smc
    from research.smc_backtest import HTF_HOURS, TF_HOURS, config_name, configs, htf_trend_for, tf_bars
    cfgs = [(c, config_name(c)) for c in configs()]
    pitset = set(zip(mem["symbol"], mem["month"]))
    rows = []
    for n, sym in enumerate(symbols):
        h1 = fetch_1h(sym)
        if h1.empty:
            continue
        for tf in tfs:
            tf_h, htf_h = TF_HOURS[tf], HTF_HOURS[tf]
            df = tf_bars(h1, tf_h)
            if len(df) < 300:
                continue
            df["atr"] = ind.atr(df)
            adv = adv30(df, 24 // tf_h)
            htf = tf_bars(h1, htf_h)
            pool_ok = lambda t, s=sym: (s, t.strftime("%Y-%m")) in pitset  # noqa: E731
            for k in (3, 5):
                htf_tr = htf_trend_for(df, htf, tf_h, htf_h, k)
                _, breaks = smc.structure(df, k)
                for c, name in cfgs:
                    setups = smc.find_setups(df, breaks, htf_tr, **c)
                    for t in smc_simulate(df, setups, sym, tf_h, FEE, pool_ok, adv):
                        t.update(config=f"{tf}|k{k}|{name}", interval=tf)
                        rows.append(t)
        if n % 10 == 0:
            log(f"smc {n}/{len(symbols)} 累計 {len(rows)} 筆")
    tr = pd.DataFrame(rows)
    led = finish(tr)
    return led, ("SMC/ICT（252 組）", 252,
                 "時點幣池（以成交那根收盤的月份判斷）；規則、結構、限價單、出場（TP 1/2/3R 各 1/3、保本、TP2 後移到 TP1、40 天）照 smc_backtest；"
                 "同一個幣同時只有一張（和原研究一樣，在模擬內判斷，先篩池再排他）；config = 週期|k|setup|entry|htf|discount，3 週期 x k=3,5 x 42 規則；"
                 "只算手續費 0.05% 那一版（原研究另有 fee+slip 版，這裡改用 apply_costs 的分級滑價 + 資金費率）；"
                 "不含隨機進場基準。")


# ---------------------------------------------------------------- 諧波 / Threads 共用的 frame（從 1H 合成）

FRAMES = {}


def _frame(sym: str, tf: str):
    """複製 threads_strategies.frame：改用幣安原始代號；同時記下時間與 adv，匯出時用。"""
    from grid_trading.history import resample
    from research.threads_strategies import TFS
    df = fetch_1h(sym)
    if len(df) < 24 * 250:
        return None
    rule, hours = TFS[tf]
    base = df
    if rule:
        df = resample(df, rule, origin="epoch")
    df = df.reset_index(drop=True)
    c = df["close"]
    df["atr"] = ind.atr(df)
    df["e8"], df["e20"] = ind.ema(c, 8), ind.ema(c, 20)
    d = resample(base, "1D", origin="epoch")
    d["ema200"] = ind.ema(d["close"], 200).where(np.arange(len(d)) >= 200)
    d["avail"] = d["time"] + pd.Timedelta(days=1)
    bar_close = df["time"] + pd.Timedelta(hours=hours)
    m = pd.merge_asof(pd.DataFrame({"t": bar_close}), d[["avail", "close", "ema200"]], left_on="t", right_on="avail",
                      direction="backward")
    df["d_close"], df["d_ema200"] = m["close"].to_numpy(), m["ema200"].to_numpy()
    FRAMES["time"] = df["time"].to_numpy()
    FRAMES["adv"] = adv30(df, 24 // hours)
    return df


def _ledger_from_orders(t: pd.DataFrame, sym: str, tf: str, exits, name_of, start_minus_one: bool):
    """t：harmonic/threads 的訂單表（每列一筆進場，每個出場有 r_/kind_/j_ 欄），轉成每個 (策略, 出場) 的交易列。"""
    times, adv = FRAMES["time"], FRAMES["adv"]
    n = len(times)
    out = []
    for ex in exits:
        j = t[f"j_{ex}"].to_numpy(np.int64)
        e = t["entry_bar"].to_numpy(np.int64)
        out.append(pd.DataFrame({
            "config": name_of(t, ex), "symbol": sym, "side": t["side"].to_numpy(np.int64),
            "time": pd.to_datetime(times[np.minimum(e, n - 1)], utc=True), "exit_t": pd.to_datetime(times[np.clip(j, 0, n - 1)], utc=True), "r": t[f"r_{ex}"].to_numpy(),
            "risk_pct": t["risk_pct"].to_numpy(), "adv_usdt": adv[np.minimum(e, n - 1)], "interval": tf,
            "i": t["busy_i"].to_numpy(np.int64), "j": j, "kind": t[f"kind_{ex}"].to_numpy(np.int64),
            "sig_key": t["sig_key"].to_numpy() if "sig_key" in t else ""}))
    return pd.concat(out, ignore_index=True)


def fam_harmonic(mem, pit, symbols):
    import research.harmonic_strategies as hs
    hs.frame = lambda coin, tf: _frame(coin, tf)
    parts = []
    for n, sym in enumerate(symbols):
        for tf in ("1D", "4H", "1H"):
            got = hs._coin((sym, tf))
            if not got:
                continue
            t = got[0].copy()
            t["entry_bar"], t["busy_i"] = t["fill"], t["fill"]
            t["sig_key"] = t["pattern"] + "|k" + t["k"].map(lambda x: f"{x:g}")
            frame_t, adv = FRAMES["time"], FRAMES["adv"]  # 最後一次 _frame 的結果（這個 sym、tf）
            led = _ledger_from_orders(t, sym, tf, list(hs.EXITS), lambda tt, ex: tf + "|" + tt["sig_key"] + "|" + ex, True)
            led["config"] = led["config"].to_numpy()
            parts.append(led)
        if n % 10 == 0:
            log(f"harmonic {n}/{len(symbols)}")
    tr = pd.concat(parts, ignore_index=True)
    tr = tr[(tr["time"] >= START) & in_pit(pit, tr["symbol"], tr["time"])]
    tr = one_at_a_time(tr, ["config", "symbol"])
    led = finish(tr)
    return led, ("諧波型態（324 組，不含 EMA200 濾網的 +136 組）", 324,
                 "時點幣池；9 型態 x 1D/4H/1H x zigzag 2/3/5 ATR x 4 出場（harm/q135/tp1/tp2）；規則、限價單、停損、出場全照 harmonic_strategies；"
                 "1H 合成 4H/1D 的方式同原研究（epoch 起算）；config = 週期|型態|k|出場；原研究另測的「日線 EMA200 同向」濾網版（+136 組）沒有匯出，"
                 "因為不是 324 組的一部分；進場時間 = 成交那根 K 棒的開盤時間（限價單在那根內成交，模擬從下一根開始）；出場時間 = 出場那根的開盤時間；"
                 "同週期同幣同型態同 k 同出場同時一張。")


# ---------------------------------------------------------------- Threads

def fam_threads(mem, pit, symbols):
    import research.threads_strategies as ts
    ts.frame = lambda coin, tf: _frame(coin, tf)
    parts = []
    exits = list(ts.EXITS)
    for n, sym in enumerate(symbols):
        for tf in ("1D", "4H", "1H"):
            got = ts._coin((sym, tf))
            for r in got:
                if r["strategy"].iloc[0].startswith("random_"):   # 隨機基準，不是策略
                    continue
                r = r.copy()
                r["entry_bar"], r["busy_i"] = r["start"], r["start"] - 1
                strat = r["strategy"].iloc[0]
                # 要在 FRAMES 還是這個 sym、tf 時用（_coin 內 frame 只呼叫一次）
                led = _ledger_from_orders(r, sym, tf, exits, lambda tt, ex, s=strat: f"{s}|{ex}|all", True)
                led["ema200"] = np.tile(((r["side"] * (r["d_close"] - r["d_ema200"])) > 0).to_numpy(), len(exits))
                led["strategy"] = strat
                parts.append(led)
        if n % 10 == 0:
            log(f"threads {n}/{len(symbols)}")
    tr = pd.concat(parts, ignore_index=True)
    tr = tr[(tr["time"] >= START) & in_pit(pit, tr["symbol"], tr["time"])]
    # 無濾網版與 EMA200 濾網版各自做「同一張」
    a = tr.assign(config=tr["config"])
    b = tr[tr["ema200"]].assign(config=lambda d: d["config"].str.replace(r"\|all$", "|ema200", regex=True))
    tr = pd.concat([a, b], ignore_index=True)
    tr = one_at_a_time(tr, ["config", "symbol"])
    sig_led = finish(tr)
    bot_led = threads_bot(mem, pit, symbols)
    led = pd.concat([sig_led, bot_led], ignore_index=True).sort_values("exit_t").reset_index(drop=True)
    return led, ("Threads 策略（日線/4H 吞沒+EMA8/20、4H/1H 鯊魚諧波、14 天最強幣機器人）", None,
                 "時點幣池；信號策略：吞沒 16 個（1D/4H x 剛交叉5根/僅 EMA 空頭態 x 市價/掛單吞沒實體一半 x 結構停損/1.5ATR）+ 敏感度 4 個（1D）"
                 "+ 鯊魚 12 個（4H/1H x k=2,3,5 x loose/strict）= 32 個策略 x 3 出場（q135/tp2/tp1）x 有無日線 EMA200 同向濾網 = 192 組"
                 "（隨機基準不算）；機器人 4 組（1/3/7 天換、抱到止損）；config 命名：策略|出場|all 或 ema200、bot|hold…；"
                 "experiments.csv 沒有登記組數，上面是本次實際組數；進場時間：市價單 = 訊號下一根開盤，掛單 = 成交那根的下一根開盤（照原模擬）；"
                 "機器人：每天只在『上月成交額前 100 且近 30 天成交額前 30』的合約挑最強者、全倉做多、-20% 止損，r_net = 單筆報酬/0.20（即以 20% 止損為 1R），"
                 "滑價與資金費率用 risk_pct=0.20 換成 R；機器人是單一部位序列（同時只一個幣）。")


def threads_bot(mem, pit, symbols) -> pd.DataFrame:
    """14 天最強幣機器人（threads_strategies.bot），候選限制在時點幣池內。"""
    import research.threads_strategies as ts
    syms, panel = [], {}
    for sym in symbols:
        d = fetch_symbol(sym, "1d")
        if len(d) < 60:
            continue
        d = d.set_index("time")
        d["qv"] = d["volume"] * d["close"]
        panel["BTC" if sym == "BTCUSDT" else sym] = d
        syms.append(sym)
    keys = list(panel)
    A = ts.bot_arrays(panel)
    days = A["days"]
    # 候選限制：第 t+1 天（進場日）要在時點幣池
    mask = np.zeros_like(A["vol30"], bool)
    pit_set = set(zip(mem["symbol"], mem["month"]))
    months = np.asarray(days.strftime("%Y-%m"))
    nxt_months = np.concatenate([months[1:], ["9999-99"]])
    for c, s in enumerate(syms):
        mask[:, c] = [(s, m) in pit_set for m in nxt_months]
    A["vol30_raw"] = A["vol30"].copy()
    A["vol30"] = np.where(mask, A["vol30"], np.nan)
    # BTC 24h 跌幅仍用 BTC 全期（大盤保護，不是選幣）
    rows = []
    for hold, name in ((1, "bot|hold1d"), (3, "bot|hold3d"), (7, "bot|hold7d"), (None, "bot|hold_to_stop")):
        tr = ts.bot(A, hold, "best")
        for c_, t0, t1, ret, why in tr.itertuples(index=False):
            c = syms[c_]
            k = days.get_loc(t0)
            rows.append({"config": name, "symbol": c, "side": 1, "time": t0, "exit_t": t1, "r": ret / 0.20, "risk_pct": 0.20,
                         "adv_usdt": A["vol30_raw"][k - 1, c_] if k >= 1 else np.nan, "interval": "1D"})
    tr = pd.DataFrame(rows)
    tr["time"] = pd.to_datetime(tr["time"], utc=True)
    tr["exit_t"] = pd.to_datetime(tr["exit_t"], utc=True)
    return finish(tr)


# ---------------------------------------------------------------- 訂單流

SYM_DIR = Path("data/binance_sym")
METRIC_END = "2026-11-20"


def flow_1h(sym: str) -> pd.DataFrame:
    import os
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from grid_trading.binance_history import _months, _read_zip, clean
    path = SYM_DIR / f"{sym}_1h_flow.csv.gz"
    if not path.exists():
        session = requests.Session()
        keys = _months(sym, "1h", session)
        if not keys:
            return pd.DataFrame()
        with ThreadPoolExecutor(8) as ex:
            parts = list(ex.map(lambda k: _read_zip(k, session, True), keys))
        df = clean(pd.concat(parts))
        SYM_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def metrics_1h(sym: str, fetch: bool = True) -> pd.DataFrame:
    """每小時 metrics（OI、多空比…），照 binance_metrics.fetch_metrics 的做法，但用幣安原始代號。"""
    import os
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from grid_trading import binance_metrics as bm
    path = SYM_DIR / f"{sym}_metrics_1h.csv.gz"
    if not path.exists():
        if not fetch:
            return pd.DataFrame()
        session = requests.Session()
        ad = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        session.mount("https://", ad)
        first = bm._first_day(sym, session)
        parts = []
        if first is not None:
            days = pd.date_range(first, pd.Timestamp(METRIC_END), freq="D")
            with ThreadPoolExecutor(16) as ex:
                parts = [d for d in ex.map(lambda d: bm._read_day(sym, d, session), days) if d is not None]
        SYM_DIR.mkdir(parents=True, exist_ok=True)
        if parts:
            df = pd.concat(parts).drop_duplicates("time", keep="last").sort_values("time")
            for col in bm.FIELDS.values():
                df[col] = pd.to_numeric(df[col], errors="coerce")
            hourly = df.set_index("time").resample("1h", label="left", closed="left").last().dropna(how="all").reset_index()
            hourly["time"] = (hourly["time"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        else:
            hourly = pd.DataFrame({"time": pd.Series(dtype="int64")})
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
        hourly.to_csv(tmp, index=False)
        os.replace(tmp, path)
    m = pd.read_csv(path)
    if m.empty:
        return pd.DataFrame()
    m["time"] = pd.to_datetime(m["time"], unit="ms", utc=True)
    return m


def flow_bars_sym(sym: str, rule: str, with_oi: bool) -> pd.DataFrame:
    """orderflow.flow_bars 的幣安代號版。"""
    from grid_trading.orderflow import FLOW_AGG, HOURS, METRIC_COLS
    k = flow_1h(sym)
    if k.empty:
        return k
    m = metrics_1h(sym) if with_oi else pd.DataFrame()
    if not m.empty:
        k = k.merge(m[["time", *METRIC_COLS]], on="time", how="left")
    else:
        for c in METRIC_COLS:
            k[c] = np.nan
    agg = dict(FLOW_AGG, **{c: "last" for c in METRIC_COLS})
    out = k.set_index("time").resample(rule, label="left", closed="left", origin="start_day").agg(agg)
    out = out.dropna(subset=["close"]).reset_index()
    last_close = k["time"].iloc[-1] + pd.Timedelta(hours=1)
    out = out[out["time"] + pd.Timedelta(hours=HOURS[rule]) <= last_close].reset_index(drop=True)
    out["delta"] = 2 * out["taker_buy_quote"] - out["quote_volume"]
    out["cvd"] = out["delta"].cumsum()
    return out


def fam_orderflow(mem, pit, symbols, with_oi=True, tfs=("4h", "2h")):
    from grid_trading import signals
    from grid_trading.orderflow import add_features
    from grid_trading.signals import orderflow as _of  # noqa: F401
    from grid_trading.signals.tracker import track
    from research.orderflow_backtest import MAX_BARS, grid
    cfgs = grid()
    if not with_oi:
        cfgs = {k: v for k, v in cfgs.items() if not (v[0] == "of_oi" or v[0] == "of_crowd")}
    pitset = set(zip(mem["symbol"], mem["month"]))
    rows = []
    for n, sym in enumerate(symbols):
        for tf in tfs:
            df = flow_bars_sym(sym, tf, with_oi)
            if len(df) <= 250:
                continue
            df = add_features(df)
            bar = pd.Timedelta(tf)
            bpd = 24 // int(tf[:-1])
            adv = (df["quote_volume"].rolling(30 * bpd, min_periods=bpd).sum() / 30).to_numpy()
            atr = df["atr"].to_numpy()
            tpos = pd.DatetimeIndex(df["time"])
            for label, (name, params) in cfgs.items():
                strat = signals.get(name, **params)
                busy = None
                for sig in strat.scan(df, sym, tf):
                    if sig.time < START or (sym, sig.time.strftime("%Y-%m")) not in pitset:
                        continue
                    if busy is not None and sig.time < busy:
                        continue
                    res = track(sig, df, FEE, MAX_BARS[tf])
                    if res.status != "CLOSED":
                        continue
                    ex = res.events[-1].time
                    busy = ex
                    k = int(tpos.searchsorted(sig.time - bar))
                    rows.append({"config": f"{tf.upper()}|{label}", "symbol": sym, "side": 1 if sig.side == "LONG" else -1,
                                 "time": sig.time, "exit_t": ex, "r": res.realized_r,
                                 "risk_pct": 1.5 * atr[k] / sig.price, "adv_usdt": adv[k], "interval": tf.upper()})
        if n % 10 == 0:
            log(f"orderflow {n}/{len(symbols)} 累計 {len(rows)} 筆")
    led = finish(pd.DataFrame(rows))
    return led, ("訂單流（CVD 背離、主動買賣失衡、OI x 價格、多空比）", 224,
                 "時點幣池；主動買量取自幣安 K 線月檔（taker_buy_quote），OI/多空比取自 data.binance.vision 每日 metrics 檔"
                 + ("（BTC 2020-09 起、多數幣 2021-12 起，更早為 NaN，相關規則不觸發）" if with_oi else "；本次沒有 OI metrics，只重建不需要 OI 的 CVD 背離與主動買賣失衡組")
                 + "；112 組 x 4H/2H；出場同 vegas 舊式（1.5ATR 停損、TP 1/2/3R 各 1/3、40 天）照原研究；同一個幣同時一張在模擬內判斷（先篩池）；"
                 "不含隨機基準與『拿訂單流過濾維加斯』的子分析。")


# ---------------------------------------------------------------- main

FAMILIES = {
    "vegas_entry_filters": fam_vegas_entry,
    "vegas_stoploss": fam_vegas_stop,
    "smc_ict": fam_smc,
    "harmonic": fam_harmonic,
    "threads_strategies": fam_threads,
    "orderflow": fam_orderflow,
}
SOURCES = {
    "vegas_entry_filters": "signals/report_vegas_trend_entry_20261013.md",
    "vegas_stoploss": "signals/report_vegas_trend_stoploss_20261013.md",
    "smc_ict": "signals/report_smc_ict_20261012.md",
    "harmonic": "signals/report_harmonic_20261018.md",
    "threads_strategies": "signals/report_threads_strategies_20261018.md",
    "orderflow": "signals/report_orderflow_20261012.md",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某個家族（逗號分隔）")
    ap.add_argument("--max-syms", type=int, default=0, help="除錯用：只跑前 N 個合約")
    ap.add_argument("--no-oi", action="store_true", help="orderflow 不用 OI metrics")
    ap.add_argument("--of-tfs", default="4h,2h", help="orderflow 週期")
    ap.add_argument("--tag", default="", help="除錯用：輸出檔名加尾巴")
    a = ap.parse_args()
    mem, pit = pit_index()
    symbols = universe_symbols(mem)
    if a.max_syms:
        symbols = sorted(set(symbols[: a.max_syms]) | {"BTCUSDT"})
    log(f"{len(symbols)} 個合約（時點幣池曾入選）")
    todo = a.only.split(",") if a.only else list(FAMILIES)
    for fam in todo:
        t0 = time.time()
        kw = {"with_oi": not a.no_oi, "tfs": tuple(a.of_tfs.split(","))} if fam == "orderflow" else {}
        led, (label, reg, notes) = FAMILIES[fam](mem, pit, symbols, **kw)
        led = led[KEEP + ["risk_pct"]]
        report(fam, led, reg or 0)
        save(fam + a.tag, led, {"label": label, "kind": "trades", "source": SOURCES[fam], "configs_registered": reg,
                                "notes": notes, "configs_exported": int(led["config"].nunique())})
        log(f"{fam} 完成，{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
