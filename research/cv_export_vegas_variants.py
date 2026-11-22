"""維加斯變體家族的長帳本（給 research/cv_all.py 的滾動前進＋CPCV＋PBO 用）。2026-11-22。

    python -m research.cv_export_vegas_variants [--only fam] [--fresh] [--workers 1]
    → output/cv_ledgers/<fam>.pkl / .json

這些都是「現代」研究（已經用時點幣池＋滑價＋資金費率），所以直接重用原程式的訊號與模擬函式，只是把每一組
預先登記的設定各自存成一本帳本（欄 config），一律：同週期同幣同時只一張、r_net = 扣手續費（模擬內 0.05%/邊）
＋滑價＋資金費率。原始逐筆結果快取在 output/_cv_cache_variants/（--fresh 重算）。

家族與設定網格（和原研究事先登記的一樣，不新增、不挑）：
- vegas_params     research/vegas_params.py + vegas_params_extra.py：6 個變體（ch288、ch576、no_e12、e12_trig、stack2、
                   stack3）× {2H,4H} = 12 組，加 4 組「多出來的單」（add_ch288、add_ch576、add_ch288_576、add_e12_trig）
                   × {2H,4H} = 8 組，共 20 組。變體用時點幣池；多出來的單用原研究的「流動幣池」（時點幣池且 30 日日均
                   成交額 >= 5,000 萬，只看多出來的單、獨立帳本）。基準 base 不是登記的新設定，不放進網格。
- vegas_timeframes research/vegas_timeframes.py：15m、1H、12H、1D、2D、3D、1W（登記 7 組）加 2H、4H 當對照，共 9 組，
                   每個週期是一個 config（interval 欄 = 週期）。時點幣池、無持倉上限。原研究另做隨機進場基準，這裡不需要。
- universe_buckets research/universe_buckets.py：11 個幣池分組（每個分組一本實盤維加斯 2H+4H 帳本）。幣池就是研究對象：
                   CoinMarketCap 當月（上月底）市值名次（扣穩定幣）與訊號當下 30 日日均成交額，所以不再套時點成交額前
                   100 的過濾（那會破壞「市值 101-300」「成交額 500 萬」等分組）。原研究的第 12 組「現行實盤固定名單」
                   用的是 TOP100 快照，不是登記的 11 組之一，略過。
- vegas_improve    research/vegas_improve.py：登記 26 組裡的 21 組逐筆策略（多空拆開 2、BTC 對沖 6（對沖腿一律扣成本）、掛限價 4、止損
                   距離 3、流動性門檻 3、合併 3）。每筆 r_net = 原 R 的調整（對沖損益、省下的成本）× 風險倍數（0 的單
                   不進帳本；0.5 倍的單 r_net 減半），和原研究 line() 的算法相同。缺：搭配策略 3 組（山寨籃子×2、
                   資金費率排序多空）是週再平衡的組合報酬，單位和每筆 R 不同，混在同一本帳本會讓選設定的 t 值失真，所以不放。
                   幣池 in_pit（時點幣池），來源 output/pit_trades.csv.gz（需先跑 research.pit_backtest）。
- vegas_divergence research/vegas_divergence.py：9 個條件（rsi_div、macd_div、any_div、rsi_hdiv、macd_hdiv、bb_out、
                   bb_squeeze、bb_touch、no_opp_div；W=12）× {filter、size2x} = 18 組，加 base（實盤維加斯）共 19 組。
                   filter = 只做有條件的單（重新跑帳本）；size2x = 基準帳本裡有條件的單 r_net 乘 2。W=6/24 只是參數高原檢查，不放。
                   和原研究不同：幣池由 CMC 市值前 100 改成時點成交額前 100，仍保留「30 日日均成交額 >= 3,000 萬」。
- public_strats    research/public_strategies.py + public_strategies_run.build_ledger：7 個變體 × {2H,4H} × {live,fix15}
                   = 28 組（config = 變體|週期|出場），不含參照帳本 REFS。

差異總表見各家族 json 的 notes。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("output/cv_ledgers")
CACHE = Path("output/_cv_cache_variants")
START = pd.Timestamp("2020-01-01", tz="UTC")
KEEP = ["config", "symbol", "side", "time", "exit_t", "r", "r_net", "interval"]


def _log(m):
    print(m, file=sys.stderr, flush=True)


def save(family: str, led: pd.DataFrame, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    led = led[[c for c in KEEP if c in led.columns]].reset_index(drop=True)
    for c in ("time", "exit_t"):
        led[c] = pd.to_datetime(led[c], utc=True)
    led = led[led["time"] >= START].reset_index(drop=True)
    led.to_pickle(OUT / f"{family}.pkl")
    meta = {"family": family, **meta, "configs_exported": int(led["config"].nunique())}
    (OUT / f"{family}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    per = led.groupby("config")["r_net"].agg(["size", "mean"])
    _log(f"[{family}] {len(led)} 列，{led['config'].nunique()} 組（登記 {meta['configs_registered']}），"
         f"全期 r_net 平均 {led['r_net'].mean():.4f}；各組平均 {per['mean'].round(3).to_dict()}")


def cached(name: str, fn, fresh: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.pkl"
    if p.exists() and not fresh:
        return pd.read_pickle(p)
    x = fn()
    x.to_pickle(p)
    return x


def pit_set(mem: pd.DataFrame) -> set:
    return set(zip(mem["symbol"], mem["month"].astype(str)))


def flag_pit(tr: pd.DataFrame, pit: set) -> pd.DataFrame:
    tr = tr.copy()
    tr["in_pit"] = [(s, m) in pit for s, m in zip(tr["symbol"], tr["time"].dt.strftime("%Y-%m"))]
    return tr


def tag(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df.assign(config=name)


# ---------------------------------------------------------------- vegas_params

def fam_vegas_params(workers: int, fresh: bool) -> None:
    import research.vegas_params as vp
    from research.pit_backtest import book
    from research.vegas_params_extra import ADDS, KEY

    tr = cached("vp_raw", lambda: vp.run(workers=workers), fresh)
    base_raw = tr[tr["variant"] == "base"]
    base_keys = set(map(tuple, base_raw[KEY].to_numpy()))
    parts = []
    for v in vp.VARIANTS:
        if v == "base":
            continue
        b = book(tr[tr["variant"] == v], "in_pit")
        for iv, g in b.groupby("interval"):
            parts.append(tag(g, f"{v}|{iv}"))
    for name, vs in {"add_ch288": ["ch288"], "add_ch576": ["ch576"], "add_ch288_576": ["ch288", "ch576"],
                     "add_e12_trig": ["e12_trig"]}.items():
        ex = tr[tr["variant"].isin(vs)].drop_duplicates(KEY)
        ex = ex[[k not in base_keys for k in map(tuple, ex[KEY].to_numpy())]]
        eb = book(ex, "liquid")
        for iv, g in eb.groupby("interval"):
            parts.append(tag(g, f"{name}|{iv}"))
    save("vegas_params", pd.concat(parts, ignore_index=True), {
        "label": "維加斯換參數／加訊號（通道、EMA12、長通道排列）", "kind": "trades",
        "source": "signals/report_vegas_params_20261120.md", "configs_registered": 20,
        "notes": "12 變體組（ch288/ch576/no_e12/e12_trig/stack2/stack3 × 2H/4H，時點幣池）＋8 組多出來的單"
                 "（add_ch288/ch576/ch288_576/e12_trig × 2H/4H，原研究的流動幣池＝時點幣池∩30日均成交額>=5,000萬，獨立帳本）；"
                 "base 不是登記的新設定，不放。直接重用 research.vegas_params.run（出場同實盤、無持倉上限）。"})


# ---------------------------------------------------------------- vegas_timeframes

TFS = ["2H", "4H", "12H", "1D", "2D", "3D", "1W", "1H", "15m"]  # 慢的（1H、15m）放最後


def _tf_work(args):
    import research.vegas_timeframes as vt
    from grid_trading.signals.strategy import LONG
    from grid_trading.signals.vegas_trend import VegasTrend

    sym, tf, fee = args
    for k in range(6):
        try:
            df, first = vt.frame(sym, tf)
            break
        except Exception as e:  # noqa: BLE001  下載偶爾斷線
            if k == 5:
                _log(f"放棄 {sym} {tf}: {e}")
                return None
            time.sleep(5 * (k + 1))
    strat = VegasTrend()
    if len(df) < strat.min_bars + 5:
        return None
    p = strat.prepare(df).reset_index(drop=True)
    n = len(p)
    lo = max(strat.min_bars, first)
    idx, ds = [], []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < lo or i + 1 >= n:
            continue
        hit = strat.check(p, int(i))
        if hit is None or not strat.accept(p, int(i), tf):
            continue
        idx.append(int(i))
        ds.append(1 if hit[0] == LONG else -1)
    if not idx:
        return None
    bpd = vt.TF[tf][2]
    w = max(2, int(round(30 * bpd)))
    adv = ((p["volume"] * p["close"]).rolling(w, min_periods=2).mean() * bpd).to_numpy()
    idx, ds = np.array(idx), np.array(ds)
    return vt._frame_out(sym, tf, p, idx, ds, vt._sim(p, idx, ds, tf, fee), adv)


def _tf_run(workers: int, tfs: list[str]) -> None:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    import research.vegas_timeframes as vt
    from grid_trading.costs import apply_costs
    from grid_trading.pit_universe import membership
    from research.exit_unify import per_interval
    from research.vegas_params import _funding

    syms = vt.pool_symbols()
    mem = membership(n=100)
    pit = pit_set(mem)
    for tf in tfs:
        t0 = time.time()
        jobs = [(s, tf, 0.0005) for s in syms]
        with ThreadPoolExecutor(6) as tp:  # 下載是 I/O 等待，用執行緒預先抓好快取（不增加計算行程）
            errs = [e for e in tp.map(vt._fetch, [(s, tf) for s in syms]) if e]
        if errs:
            _log(f"  {tf} 預抓失敗 {len(errs)} 個：{errs[:3]}")
        _log(f"  {tf} 預抓完成 {time.time() - t0:.0f}s")
        parts = []
        with ProcessPoolExecutor(workers) as ex:
            for k, got in enumerate(ex.map(_tf_work, jobs, chunksize=2)):
                if got is not None:
                    parts.append(got)
                if k % 100 == 0:
                    _log(f"  {tf} {k}/{len(jobs)} {time.time() - t0:.0f}s")
        if not parts:
            continue
        tr = pd.concat(parts, ignore_index=True)
        for c in ("time", "exit_t"):
            tr[c] = pd.to_datetime(tr[c], utc=True)
        tr = flag_pit(tr[tr["time"] >= START], pit)
        tr = tr[tr["in_pit"]]
        tr = per_interval(tr.rename(columns={"coin": "_c", "symbol": "coin"})).rename(columns={"coin": "symbol", "_c": "coin"})
        tr = apply_costs(tr, funding=_funding(sorted(tr["symbol"].unique())))
        _log(f"  {tf}: {len(tr)} 筆，{time.time() - t0:.0f}s")
        tr.to_pickle(CACHE / f"tf_{tf}.pkl")


def fam_vegas_timeframes(workers: int, fresh: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [t for t in TFS if fresh or not (CACHE / f"tf_{t}.pkl").exists()]
    if todo:
        _tf_run(workers, todo)
    out = [pd.read_pickle(CACHE / f"tf_{t}.pkl") for t in TFS if (CACHE / f"tf_{t}.pkl").exists()]
    led = pd.concat(out, ignore_index=True)
    led["config"] = led["interval"]
    missing = [t for t in TFS if t not in set(led["config"])]
    save("vegas_timeframes", led, {
        "label": "維加斯換週期（15m 到 1W，含 2H/4H 對照）", "kind": "trades",
        "source": "signals/report_vegas_timeframes_20261120.md", "configs_registered": 7,
        "notes": "9 組（7 個登記週期 15m/1H/12H/1D/2D/3D/1W ＋ 2H/4H 對照），每個週期一個 config；時點幣池、無持倉上限；"
                 "12H 以上用現貨補暖機（同原研究）；未含原研究的隨機進場基準。" + (f" 缺週期：{missing}" if missing else "")})


# ---------------------------------------------------------------- universe_buckets

def fam_universe_buckets(workers: int, fresh: bool) -> None:
    import research.universe_buckets as ub
    from research.pit_backtest import book

    ub.fetch_cmc()
    tr = cached("ub_raw", lambda: ub.trades(0.0005, workers), fresh)
    parts = []
    for name, mask in ub.buckets(tr).items():
        if name.startswith("現行實盤"):
            continue
        b = book(tr.assign(_m=mask), "_m")
        parts.append(tag(b, name))
    save("universe_buckets", pd.concat(parts, ignore_index=True), {
        "label": "維加斯擴大幣種範圍（市值名次／成交額分組）", "kind": "trades",
        "source": "signals/report_universe_buckets_20261120.md", "configs_registered": 11,
        "notes": "11 個幣池分組，各自一本 2H+4H 實盤維加斯帳本（無持倉上限、扣成本）。幣池就是研究對象（CMC 當月市值名次、"
                 "訊號當下 30 日均成交額），所以不再套時點成交額前 100；略過原研究第 12 組『現行實盤固定名單』（TOP100 快照，未登記）。"})


# ---------------------------------------------------------------- vegas_divergence

def fam_vegas_divergence(workers: int, fresh: bool) -> None:
    from concurrent.futures import ProcessPoolExecutor

    import research.vegas_divergence as vd
    from grid_trading.costs import apply_costs
    from grid_trading.pit_universe import membership
    from research.pit_backtest import book

    def run():
        mem = membership(n=100)
        mem["month"] = mem["month"].astype(str)
        pit = pit_set(mem)
        syms = sorted(set(mem.loc[mem["month"] >= "2020-01", "symbol"]))
        jobs = [(s, iv, 0.0005) for iv in ("4H", "2H") for s in syms]
        parts = []
        with ProcessPoolExecutor(workers) as ex:
            for k, got in enumerate(ex.map(vd._work, jobs, chunksize=2)):
                if got is not None:
                    parts.append(got)
                if k % 100 == 0:
                    _log(f"  {k}/{len(jobs)}")
        tr = pd.concat(parts, ignore_index=True)
        for c in ("time", "exit_t"):
            tr[c] = pd.to_datetime(tr[c], utc=True)
        tr = flag_pit(tr[tr["time"] >= START], pit)
        tr = tr[tr["in_pit"] & (tr["adv_usdt"] >= vd.ADV_FLOOR)].reset_index(drop=True)
        return apply_costs(tr, funding=vd._funding(sorted(tr["symbol"].unique())))

    tr = cached("vd_raw", run, fresh)
    tr["all"] = True
    base = book(tr, "all")
    parts = [tag(base, "base")]
    for cond in vd.CONDS:
        col = f"{cond}_w{vd.MAIN_W}"
        fl = tr[col].astype(bool)
        parts.append(tag(book(tr.assign(_m=fl), "_m"), f"{cond}|filter"))
        bf = base[col].astype(bool).to_numpy()
        s = tag(base, f"{cond}|size2x")
        s["r_net"] = base["r_net"].to_numpy() * np.where(bf, 2.0, 1.0)
        s["r"] = base["r"].to_numpy() * np.where(bf, 2.0, 1.0)
        parts.append(s)
    save("vegas_divergence", pd.concat(parts, ignore_index=True), {
        "label": "維加斯 + 背離／布林共振", "kind": "trades",
        "source": "signals/report_vegas_divergence_20261122.md", "configs_registered": 18,
        "notes": "9 條件(W=12)×{filter,size2x}=18 組＋base 共 19 組（base 是對照，登記 18）。filter＝只做有條件的單（重跑同週期同幣一張）；"
                 "size2x＝基準帳本有條件的單 r、r_net 乘 2。W=6/24 高原檢查不放。幣池：原研究為 CMC 市值前 100∩日均成交額>=3,000萬，"
                 "這裡改時點成交額前 100（membership）∩日均成交額>=3,000萬。"})


# ---------------------------------------------------------------- vegas_improve

def fam_vegas_improve(workers: int, fresh: bool) -> None:
    import research.vegas_improve as vi

    if not Path("output/improve_trades.csv.gz").exists() or fresh:
        vi.enrich()
    tr = vi.book(vi.load(), "in_pit")
    r = tr["r_net"].to_numpy()
    side, rp = tr["side"].to_numpy(), tr["risk_pct"].to_numpy()
    adv, hit = tr["adv_usdt"].to_numpy(), tr["hit"].to_numpy()
    tk = tr["touched"].to_numpy().astype(bool)
    T, M = vi.TAKER, vi.MAKER
    hg = -side * tr["btc_seg_ret"].to_numpy() / rp
    hf = tr["hedge_funding_r"].to_numpy()
    hc = -2 * (T + 0.0001) / rp
    c_taker = hc
    sh = side < 0
    slip1 = -tr["slip_r"].to_numpy() / 2
    save_entry = (T - M) / rp + slip1
    save_tp = 0.25 * hit * ((T - M) / rp + slip1)
    liq = adv >= 5e7
    save_both = np.where(tk, save_entry, 0.0) + save_tp
    ones = np.ones(len(tr))
    cfgs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # A. 多空拆開
    cfgs["A|只做空單"] = (r, (sh).astype(float))
    cfgs["A|多單風險減半"] = (r, np.where(side > 0, 0.5, 1.0))
    # B. BTC 對沖
    # 對沖腿一律扣來回成本（帳本規則：所有設定都要扣成本）；原研究「不計對沖腿成本」那兩組是理想情況，不放
    for h in (0.5, 1.0):
        cfgs[f"B|對沖{h}倍 對沖腿來回成本"] = (r + h * (hg + hf + hc), ones)
    cfgs["B|多單基準"] = (r, (side > 0).astype(float))
    cfgs["B|多單+對沖1倍"] = (r + hg + hf + hc, (side > 0).astype(float))
    cfgs["B|空單基準"] = (r, sh.astype(float))
    cfgs["B|空單+對沖1倍"] = (r + hg + hf + hc, sh.astype(float))
    # C. 掛單
    cfgs["C|限價進場 沒成交不做"] = (r + save_entry, tk.astype(float))
    cfgs["C|限價進場 沒成交改市價追"] = (r + np.where(tk, save_entry, 0.0), ones)
    cfgs["C|TP 改限價"] = (r + save_tp, ones)
    cfgs["C|TP 限價 + 進場限價(追市價)"] = (r + save_tp + np.where(tk, save_entry, 0.0), ones)
    # D. 止損距離
    for th in (0.0075, 0.01, 0.015):
        cfgs[f"D|止損距離<{th:.2%}不做"] = (r, (rp >= th).astype(float))
    # F2. 流動性門檻
    for lab, v in (("1,000萬", 1e7), ("5,000萬", 5e7), ("2億", 2e8)):
        cfgs[f"F2|日均成交額<{lab}不做"] = (r, (adv >= v).astype(float))
    # F3. 合併（事後組合）
    cfgs["F3|掛單+流動性門檻"] = (r + save_both, liq.astype(float))
    cfgs["F3|掛單+流動性門檻+空單對沖0.5倍"] = (r + save_both + np.where(sh, 0.5 * (hg + hf + c_taker), 0.0), liq.astype(float))
    cfgs["F3|掛單+流動性門檻+空單對沖1倍"] = (r + save_both + np.where(sh, hg + hf + c_taker, 0.0), liq.astype(float))
    parts = []
    for name, (radj, w) in cfgs.items():
        m = w > 0
        g = tr[m].copy()
        g["r_net"] = (radj * w)[m]
        g["r"] = g["r"] * w[m]
        parts.append(tag(g, name))
    save("vegas_improve", pd.concat(parts, ignore_index=True), {
        "label": "維加斯優化方向（多空、BTC 對沖、掛單、止損距離、流動性門檻）", "kind": "trades",
        "source": "signals/report_vegas_improve_20261115.md", "configs_registered": 26,
        "notes": f"{len(cfgs)} 組逐筆策略（A 2、B 8、C 4、D 3、F2 3、F3 3）。缺 3 組搭配策略（放空山寨籃子×做多 BTC 1倍/beta、"
                 "資金費率排序多空）：週再平衡的組合報酬，單位和每筆 R 不同，混入會讓選設定失真。每筆 r_net＝原 r_net 加上該設定的調整"
                 "（對沖損益、掛單省下成本）再乘風險倍數，倍數 0 的單不進帳本。對沖腿的資金費率計入；『對沖腿來回成本』和 F3 的空單對沖"
                 "含對沖腿 taker 成本。F3 為事後組合（原研究也當假設）。B 的多單/空單基準與 A 的只做空單帳本重複，但原研究登記為不同設定。"
                 "幣池 in_pit（時點幣池），沿用原研究 enrich() 補欄位；原研究的 TP 掛單成本試算沿用其近似。"})


# ---------------------------------------------------------------- public_strats

def fam_public_strats(workers: int, fresh: bool) -> None:
    import research.public_strategies as ps
    import research.public_strategies_run as psr

    led = cached("ps_raw", lambda: psr.build_ledger(workers), fresh)
    led = led[~led["variant"].isin(ps.REFS)].copy()
    led["config"] = led["variant"] + "|" + led["interval"] + "|" + led["exit"]
    save("public_strats", led, {
        "label": "公開策略（MACD+EMA200、ATR 追蹤、通道震盪、維加斯加濾網等）", "kind": "trades",
        "source": "signals/report_public_strategies_20261122.md", "configs_registered": 28,
        "notes": "7 變體×{2H,4H}×{live 出場, fix15 出場}=28 組，config=變體|週期|出場；不含參照帳本（vegas_base 與被濾掉的單）。"
                 "直接重用 public_strategies_run.build_ledger（時點幣池、同週期同幣一張、滑價+資金費率）。"})


FAMS = {"vegas_params": fam_vegas_params, "vegas_timeframes": fam_vegas_timeframes,
        "universe_buckets": fam_universe_buckets, "vegas_improve": fam_vegas_improve,
        "vegas_divergence": fam_vegas_divergence, "public_strats": fam_public_strats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    for name, fn in FAMS.items():
        if a.only and name not in a.only.split(","):
            continue
        _log(f"=== {name}")
        fn(a.workers, a.fresh)


if __name__ == "__main__":
    main()
