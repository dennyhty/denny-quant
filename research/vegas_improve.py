"""維加斯還能怎麼改：之前沒測過的幾個方向（只做分析，不改實盤規則）。2026-11-15。

先跑 python -m research.pit_backtest（output/pit_trades.csv.gz），再跑
python -m research.vegas_improve enrich   → output/improve_trades.csv.gz（每筆加上下一根有沒有回到進場價、BTC 對沖損益）
python -m research.vegas_improve report   → 印出各項比較，output/improve_*.csv
python -m research.vegas_improve followup → 追加檢查
python -m research.vegas_improve baskets  → 兩個搭配策略（山寨籃子對 BTC、資金費率排序多空），output/improve_baskets.csv

基準是「時點幣池 + 滑價 + 資金費率」的實盤帳本（report_honest_backtest_20261020.md）。事先定好的測試（共 12 組設定）：
 A. 多空拆開：只做空、多單風險減半（2 組）
 B. BTC 對沖：每張單掛反向 BTC，名目 0.5 倍 / 1 倍，倉位隨 TP 減碼（2 組）
 C. 進場改掛限價單（訊號收盤價，掛一根 K 棒，沒回來就不做）（1 組）＋ TP 改限價單（成本試算，不是策略測試）
 D. 止損距離太近（成本佔 R 比例高）的單不做：門檻只用 2020-23 定（3 組：0.75% / 1% / 1.5%）
 E. 搭配策略：放空等權山寨籃子、做多 BTC（1 倍 / beta 調整，2 組）；資金費率排序多空（五分位，1 組）。
 F. 看完第一輪後追加（followup）：只對沖空單含對沖腿成本、日均成交額門檻、合併。事後挑的，當假設看。
報告：reports/signals/report_vegas_improve_20261115.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.costs import fetch_funding, funding_r_many
from grid_trading.pit_universe import fetch_symbol, membership
from research.pit_backtest import KLINE, RISK, book

OUT = Path("output")
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TP_R = (1.0, 3.0, 5.0)
TAKER, MAKER = 0.0005, 0.0002


# ---------------------------------------------------------------- 統計

def cl_stats(r: np.ndarray, key: np.ndarray) -> tuple[float, float]:
    """平均和群集穩健標準誤（同一群的單算一注）。"""
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return np.nan, np.nan
    mean = r.mean()
    s = pd.Series(r - mean).groupby(np.asarray(key)).sum().to_numpy()
    g = len(s)
    se = np.sqrt((s ** 2).sum()) / n * (np.sqrt(g / (g - 1)) if g > 1 else 1.0)
    return mean, se


def keys(tr: pd.DataFrame) -> dict[str, np.ndarray]:
    t = tr["time"]
    return {"day_side": (t.dt.floor("D").astype(str) + tr["side"].astype(str)).to_numpy(),
            "week": t.dt.strftime("%G-%V").to_numpy(), "month": t.dt.strftime("%Y-%m").to_numpy()}


def daily_sharpe(tr: pd.DataFrame, r: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """按出場日加總的日損益 → 年化 Sharpe、最大回撤（%）。w 是每筆的風險倍數。"""
    w = np.ones(len(tr)) if w is None else w
    d = pd.Series(r * w * RISK, index=tr["exit_t"].dt.floor("D").to_numpy()).groupby(level=0).sum()
    if len(d) < 2:
        return np.nan, np.nan
    days = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(days, fill_value=0.0)
    eq = (1 + d).cumprod()
    return d.mean() / d.std() * np.sqrt(365), (eq / eq.cummax() - 1).min() * 100


def line(name: str, tr: pd.DataFrame, r: np.ndarray, w: np.ndarray | None = None) -> dict:
    """一列摘要：全期、2020-23、2024+、2026，三種分群的 t。"""
    r = np.asarray(r, float)
    w = np.ones(len(tr)) if w is None else np.asarray(w, float)
    k = keys(tr)
    m, se = cl_stats(r * w, k["day_side"])
    _, se_w = cl_stats(r * w, k["week"])
    _, se_m = cl_stats(r * w, k["month"])
    sh, dd = daily_sharpe(tr, r, w)
    is_ = (tr["time"] < SPLIT).to_numpy()
    y26 = (tr["exit_t"].dt.year == 2026).to_numpy()
    years = (tr["exit_t"].max() - tr["time"].min()).days / 365.25
    out = {"rule": name, "trades": int((w > 0).sum()), "win": float((r[w > 0] > 0).mean()),
           "exp_r": float((r * w).sum() / max((w > 0).sum(), 1)), "t_day": m / se, "t_week": m / se_w, "t_month": m / se_m,
           "R_per_year": float((r * w).sum() / years), "sharpe": sh, "max_dd": dd}
    for nm, mask in (("is", is_), ("oos", ~is_), ("y26", y26)):
        sub = tr[mask]
        if len(sub):
            s2, _ = daily_sharpe(sub, r[mask], w[mask])
            out[f"R_yr_{nm}"] = float((r[mask] * w[mask]).sum() / max((sub["exit_t"].max() - sub["time"].min()).days / 365.25, 1e-9))
            out[f"sharpe_{nm}"] = s2
    return out


def show(rows: list[dict], title: str, name: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    print(f"\n=== {title}")
    with pd.option_context("display.width", 280, "display.max_columns", 40, "display.max_rows", 300):
        print(df.round(3).to_string(index=False))
    if name:
        df.to_csv(OUT / f"improve_{name}.csv", index=False)
    return df


# ---------------------------------------------------------------- 每筆補資料

def enrich() -> None:
    tr = pd.read_csv(OUT / "pit_trades.csv.gz")
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    tr = tr[tr["in_pit"] | tr["in_snapshot"]].reset_index(drop=True)
    btc = fetch_symbol("BTCUSDT", "2h")
    bt = btc["time"].to_numpy("datetime64[ns]")
    bo = btc["open"].to_numpy(float)

    def btc_at(ts: np.ndarray) -> np.ndarray:  # 那個時間點（K 棒開盤時間）的 BTC 價格
        pos = np.clip(np.searchsorted(bt, ts), 0, len(bt) - 1)
        return bo[pos]

    n = len(tr)
    touched = np.zeros(n, bool)      # 下一根有沒有穿過進場價（限價單會成交）
    deep = np.zeros(n)               # 下一根往不利方向最多走幾 R
    hedge = np.zeros(n)              # Σ 倉位比例 × BTC 報酬（進場到出場，隨 TP 減碼）
    held = np.zeros(n)               # 平均持倉比例（時間加權）
    step = {"4H": np.timedelta64(4, "h"), "2H": np.timedelta64(2, "h")}
    for (sym, iv), g in tr.groupby(["symbol", "interval"]):
        df = fetch_symbol(sym, KLINE[iv])
        hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
        t = df["time"].to_numpy("datetime64[ns]")
        for row, i, j, side, rp in zip(g.index, g["i"].to_numpy(int), g["j"].to_numpy(int), g["side"].to_numpy(int),
                                       g["risk_pct"].to_numpy(float)):
            px = cl[i]
            risk = rp * px
            j = min(j, len(cl) - 1)
            if side > 0:
                touched[row] = lo[i + 1] < px
                deep[row] = (px - lo[i + 1]) / risk
            else:
                touched[row] = hi[i + 1] > px
                deep[row] = (hi[i + 1] - px) / risk
            # TP 成交的 K 棒
            frac, seg_start, tot, wsum = 1.0, t[i + 1], 0.0, 0.0
            for k in TP_R:
                lvl = px + side * k * risk
                w = np.flatnonzero((hi[i + 1:j + 1] >= lvl) if side > 0 else (lo[i + 1:j + 1] <= lvl))
                if not len(w):
                    break
                seg_end = t[i + 1 + w[0]] + step[iv]
                a, b = btc_at(np.array([seg_start, seg_end]))
                tot += frac * (b / a - 1)
                wsum += frac * float((seg_end - seg_start) / np.timedelta64(1, "h"))
                frac -= 0.25
                seg_start = seg_end
            seg_end = max(t[j] + step[iv], seg_start)
            a, b = btc_at(np.array([seg_start, seg_end]))
            tot += frac * (b / a - 1)
            wsum += frac * float((seg_end - seg_start) / np.timedelta64(1, "h"))
            dur = float((seg_end - t[i + 1]) / np.timedelta64(1, "h"))
            hedge[row] = tot
            held[row] = wsum / dur if dur > 0 else 1.0
    tr["touched"], tr["deep_r"], tr["btc_seg_ret"], tr["held_frac"] = touched, deep, hedge, held
    # 對沖腿的資金費率（反向 BTC）：以 1 倍名目算，之後乘 h
    fb = fetch_funding("BTCUSDT")
    tr["hedge_funding_r"] = funding_r_many(fb, tr["time"], tr["exit_t"], -tr["side"].to_numpy(), tr["risk_pct"].to_numpy(),
                                           tr["held_frac"].to_numpy())
    tr.to_csv(OUT / "improve_trades.csv.gz", index=False)
    print(len(tr), "筆", file=sys.stderr)


def load() -> pd.DataFrame:
    tr = pd.read_csv(OUT / "improve_trades.csv.gz")
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    return tr


# ---------------------------------------------------------------- 報告

def report(col: str = "in_pit") -> None:
    tr = book(load(), col)
    tag = "pit" if col == "in_pit" else "snap"
    r = tr["r_net"].to_numpy()
    side = tr["side"].to_numpy()
    rp = tr["risk_pct"].to_numpy()
    print(f"帳本 {col}: {len(tr)} 筆，{tr['time'].min():%Y-%m-%d} ~ {tr['exit_t'].max():%Y-%m-%d}")

    # A. 多空
    rows = [line("基準（實盤規則，扣成本）", tr, r),
            line("只看多單", tr[side > 0], r[side > 0]), line("只看空單", tr[side < 0], r[side < 0]),
            line("多單風險減半", tr, r, np.where(side > 0, 0.5, 1.0))]
    show(rows, "A. 多空拆開", f"{tag}_A_sides")
    yr = tr.assign(r=r, y=tr["exit_t"].dt.year, s=np.where(side > 0, "long", "short"))
    p = yr.pivot_table(index="y", columns=["interval", "s"], values="r", aggfunc=["mean", "count"])
    print(p.round(3).to_string())
    p.to_csv(OUT / f"improve_{tag}_A_yearly.csv")

    # B. BTC 對沖
    rows = [line("基準", tr, r)]
    hg = -side * tr["btc_seg_ret"].to_numpy() / rp          # 1 倍反向 BTC 的損益（R）
    hf = tr["hedge_funding_r"].to_numpy()
    hc = -2 * (TAKER + 0.0001) * 1.0 / rp                    # 對沖腿完全不軋差時的交易成本（R，1 倍）
    for h in (0.5, 1.0):
        rows.append(line(f"BTC 對沖 {h} 倍（不計對沖腿交易成本）", tr, r + h * (hg + hf)))
        rows.append(line(f"BTC 對沖 {h} 倍（對沖腿每筆都付來回成本）", tr, r + h * (hg + hf + hc)))
    for nm, m in (("多單", side > 0), ("空單", side < 0)):
        rows.append(line(f"{nm} 基準", tr[m], r[m]))
        rows.append(line(f"{nm} + 對沖 1 倍（不計成本）", tr[m], (r + hg + hf)[m]))
    show(rows, "B. 每張單掛反向 BTC", f"{tag}_B_hedge")
    ep = (tr["exit_t"] >= "2026-03-01") & (tr["exit_t"] < "2026-05-06")
    print("2026-03-01~05-05 合計 R：基準", round(r[ep].sum(), 1), "對沖 0.5", round((r + 0.5 * (hg + hf))[ep].sum(), 1),
          "對沖 1.0", round((r + hg + hf)[ep].sum(), 1))
    yy = pd.DataFrame({"y": tr["exit_t"].dt.year, "base": r, "h05": r + 0.5 * (hg + hf), "h10": r + hg + hf,
                       "h10_cost": r + hg + hf + hc})
    print(yy.groupby("y").sum().round(1).to_string())

    # C. 限價進場
    tk = tr["touched"].to_numpy()
    slip1 = -tr["slip_r"].to_numpy() / 2                     # 一邊的滑價（R）
    save_entry = (TAKER - MAKER) / rp + slip1                # 進場改成掛單省下的（R）
    rows = [line("基準（市價進場）", tr, r),
            line("限價掛訊號收盤價，一根內沒成交就不做", tr, r + save_entry, tk.astype(float)),
            line("　沒成交的那些單（基準損益）", tr[~tk], r[~tk]),
            line("限價掛一根，沒成交改市價追（上限估計：省成本但不漏單）", tr, r + np.where(tk, save_entry, 0.0))]
    hit = tr["hit"].to_numpy()
    save_tp = 0.25 * hit * ((TAKER - MAKER) / rp + slip1)    # TP 改限價單省下的（R）：每個 TP 是 1/4 倉位
    rows.append(line("TP 改掛限價單（省手續費和滑價）", tr, r + save_tp))
    rows.append(line("TP 限價 + 進場限價（沒成交就追市價）", tr, r + save_tp + np.where(tk, save_entry, 0.0)))
    show(rows, "C. 掛單省成本", f"{tag}_C_maker")
    print(f"下一根有回到進場價的比例 {tk.mean():.1%}；TP 平均命中 {hit.mean():.2f} 個；"
          f"TP 改限價每筆省 {save_tp.mean():.4f}R，進場改限價每筆省 {save_entry.mean():.4f}R；"
          f"成本合計（滑價+資金費率）每筆 {(tr['slip_r'] + tr['funding_r']).mean():.4f}R")

    # D. 止損距離（成本佔比）
    rows = [line("基準", tr, r)]
    is_ = (tr["time"] < SPLIT).to_numpy()
    edges = np.quantile(rp[is_], [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(edges, rp)
    for q in range(5):
        m = b == q
        lo_, hi_ = rp[m].min(), rp[m].max()
        rows.append(line(f"止損距離第 {q + 1} 組（{lo_:.2%}~{hi_:.2%}）", tr[m], r[m]))
    for th in (0.0075, 0.01, 0.015):
        rows.append(line(f"止損距離 < {th:.2%} 不做", tr, r, (rp >= th).astype(float)))
    show(rows, "D. 止損距離（成本佔 1R 的比例）", f"{tag}_D_stopdist")
    g = pd.DataFrame({"b": b, "gross": tr["r"], "net": r, "is": is_})
    print(g.groupby(["b", "is"]).agg(n=("net", "size"), gross=("gross", "mean"), net=("net", "mean")).round(3).to_string())

    # 流動性分級（描述）
    rows = []
    for bps, g in tr.groupby("slip_bps"):
        rows.append(line(f"滑價級距 {bps:.0f}bp", g, g["r_net"].to_numpy()))
    show(rows, "流動性分級（描述，不是規則）", f"{tag}_liq")

    # 日損益（給搭配策略算相關用）
    d = pd.Series(r * RISK, index=tr["exit_t"].dt.floor("D").to_numpy()).groupby(level=0).sum()
    d.to_csv(OUT / f"improve_{tag}_daily.csv")


def followup(col: str = "in_pit") -> None:
    """看完第一輪後追加的檢查（事後挑的，要當成假設而不是結論）：空單對沖加成本、流動性門檻、合併。"""
    tr = book(load(), col)
    tag = "pit" if col == "in_pit" else "snap"
    r, side, rp = tr["r_net"].to_numpy(), tr["side"].to_numpy(), tr["risk_pct"].to_numpy()
    sh = side < 0
    hg = -side * tr["btc_seg_ret"].to_numpy() / rp + tr["hedge_funding_r"].to_numpy()
    c_taker = -2 * (TAKER + 0.0001) / rp
    c_maker = -2 * MAKER / rp
    rows = [line("基準（多空都做）", tr, r), line("空單 基準", tr[sh], r[sh])]
    for h in (0.5, 1.0):
        rows.append(line(f"空單 + BTC 多單對沖 {h} 倍，對沖腿市價", tr[sh], (r + h * (hg + c_taker))[sh]))
        rows.append(line(f"空單 + BTC 多單對沖 {h} 倍，對沖腿掛單", tr[sh], (r + h * (hg + c_maker))[sh]))
    for h in (0.5, 1.0):
        rows.append(line(f"多單照舊 + 空單對沖 {h} 倍（對沖腿市價）", tr, r + np.where(sh, h * (hg + c_taker), 0.0)))
    rows.append(line("不做多單 + 空單對沖 1 倍（對沖腿市價）", tr, r + hg + c_taker, sh.astype(float)))
    show(rows, "F1. 只對沖空單（含對沖腿成本）", f"{tag}_F1_short_hedge")
    yy = pd.DataFrame({"y": tr["exit_t"].dt.year, "base": r, "long": np.where(sh, 0, r), "short": np.where(sh, r, 0),
                       "short_h05": np.where(sh, r + 0.5 * (hg + c_taker), 0), "short_h10": np.where(sh, r + hg + c_taker, 0)})
    print(yy.groupby("y").sum().round(1).to_string())
    yy.groupby("y").sum().to_csv(OUT / f"improve_{tag}_F1_yearly.csv")

    adv = tr["adv_usdt"].to_numpy()
    liq = adv >= 5e7
    g = pd.DataFrame({"tier": tr["slip_bps"], "period": np.where(tr["time"] < SPLIT, "2020-23", "2024+"), "gross": tr["r"], "net": r})
    print(g.groupby(["tier", "period"]).agg(n=("net", "size"), gross=("gross", "mean"), net=("net", "mean")).round(3).to_string())
    rows = [line("基準", tr, r), line("日均成交額 < 5,000 萬 U 不做", tr, r, liq.astype(float)),
            line("日均成交額 < 1,000 萬 U 不做", tr, r, (adv >= 1e7).astype(float)),
            line("日均成交額 < 2 億 U 不做", tr, r, (adv >= 2e8).astype(float)),
            line("　被 5,000 萬門檻擋掉的單（未扣滑價資金費）", tr[~liq], tr["r"].to_numpy()[~liq])]
    show(rows, "F2. 流動性門檻", f"{tag}_F2_liquidity")
    lq = pd.DataFrame({"y": tr["exit_t"].dt.year, "liquid": np.where(liq, r, 0), "illiquid": np.where(liq, 0, r),
                       "n_illiq": (~liq).astype(int)})
    print(lq.groupby("y").sum().round(1).to_string())

    tk = tr["touched"].to_numpy()
    slip1 = -tr["slip_r"].to_numpy() / 2
    save = np.where(tk, (TAKER - MAKER) / rp + slip1, 0.0) + 0.25 * tr["hit"].to_numpy() * ((TAKER - MAKER) / rp + slip1)
    rows = [line("基準", tr, r),
            line("掛單（進場+TP）", tr, r + save),
            line("掛單 + 流動性門檻 5,000 萬", tr, r + save, liq.astype(float)),
            line("掛單 + 流動性門檻 + 空單對沖 0.5 倍", tr, r + save + np.where(sh, 0.5 * (hg + c_taker), 0.0), liq.astype(float)),
            line("掛單 + 流動性門檻 + 空單對沖 1 倍", tr, r + save + np.where(sh, hg + c_taker, 0.0), liq.astype(float))]
    show(rows, "F3. 合併（事後組合，僅供參考）", f"{tag}_F3_combined")


# ---------------------------------------------------------------- 搭配策略：籃子

def daily_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """時點幣池的日收盤、每日資金費率合計、每月成員。"""
    cache = OUT / "improve_daily_close.csv.gz"
    mem = membership(n=100)
    mem["month"] = mem["month"].astype(str)
    mem = mem[mem["month"] >= "2020-01"]
    syms = sorted(set(mem["symbol"]) | {"BTCUSDT"})
    if cache.exists():
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        fd = pd.read_csv(OUT / "improve_daily_funding.csv.gz", index_col=0, parse_dates=True)
        return px, fd, mem
    cols, fcols = {}, {}
    for s in syms:
        df = fetch_symbol(s, "4h")
        if len(df) < 12:
            continue
        cols[s] = df.set_index("time")["close"].resample("1D").last()
        try:
            f = fetch_funding(s)
        except Exception:  # noqa: BLE001
            f = None
        if f is not None and len(f):
            ft = pd.to_datetime(f["time"], unit="ms", utc=True)
            fcols[s] = pd.Series(f["rate"].to_numpy(float), index=ft).resample("1D").sum()
    px, fd = pd.DataFrame(cols), pd.DataFrame(fcols)
    px.to_csv(cache)
    fd.to_csv(OUT / "improve_daily_funding.csv.gz")
    return px, fd, mem


def baskets() -> None:
    px, fd, mem = daily_panel()
    px.index = pd.to_datetime(px.index, utc=True)
    fd.index = pd.to_datetime(fd.index, utc=True)
    fd = fd.reindex(px.index).fillna(0.0)
    weeks = pd.date_range(pd.Timestamp("2020-03-02", tz="UTC"), px.index.max(), freq="W-MON")
    pool = mem.groupby("month")["symbol"].apply(set).to_dict()
    cost = 0.0010  # 每週換倉成本，佔山寨腿名目（保守：每週整個籃子來回 5bp+滑價）
    rows = []
    for a, b in zip(weeks[:-1], weeks[1:]):
        if a not in px.index or b not in px.index:
            continue
        members = [s for s in pool.get(a.strftime("%Y-%m"), ()) if s in px.columns and s != "BTCUSDT"]
        p0, p1 = px.loc[a, members], px.loc[b, members]
        ok = p0.notna() & (p0 > 0)
        members = list(p0.index[ok])
        if len(members) < 10:
            continue
        ret = (p1[members] / p0[members] - 1).fillna(-1.0).clip(lower=-1.0)   # 一週內下市的當作歸零（對放空有利，極少）
        fsum = fd.loc[(fd.index >= a) & (fd.index < b)].sum()
        btc = px.loc[b, "BTCUSDT"] / px.loc[a, "BTCUSDT"] - 1
        fbtc = fsum.get("BTCUSDT", 0.0)
        # 過去 60 天對 BTC 的 beta（等權籃子）
        hist = px.loc[(px.index < a) & (px.index >= a - pd.Timedelta(days=60))]
        hr = hist.pct_change().iloc[1:]
        bk = hr[members].mean(axis=1)
        beta = float(np.cov(bk, hr["BTCUSDT"])[0, 1] / hr["BTCUSDT"].var()) if len(hr) > 20 else 1.0
        beta = float(np.clip(beta, 0.5, 2.0))
        alt, falt = ret.mean(), fsum.reindex(members).fillna(0.0).mean()
        # 資金費率排序：過去 7 天平均
        f7 = fd.loc[(fd.index < a) & (fd.index >= a - pd.Timedelta(days=7)), members].mean()
        f7 = f7[f7.index.isin(fd.columns)].dropna()
        q = max(int(len(f7) * 0.2), 3)
        hi_f, lo_f = f7.nlargest(q).index, f7.nsmallest(q).index
        carry = (ret[lo_f].mean() - ret[hi_f].mean()) + (fsum.reindex(hi_f).fillna(0).mean() - fsum.reindex(lo_f).fillna(0).mean())
        rows.append({"week": a, "n": len(members), "btc": btc, "alt": alt, "beta": beta,
                     "short_alt_long_btc": (btc - fbtc) - (alt - falt) - cost,
                     "short_alt_long_btc_beta": beta * (btc - fbtc) - (alt - falt) - cost,
                     "funding_sort_ls": carry - 2 * cost})
    w = pd.DataFrame(rows).set_index("week")
    w.to_csv(OUT / "improve_baskets_weekly.csv")
    veg = pd.read_csv(OUT / "improve_pit_daily.csv", index_col=0, parse_dates=True).iloc[:, 0]
    veg.index = pd.to_datetime(veg.index, utc=True)
    vw = veg.resample("W-MON", label="left", closed="left").sum().reindex(w.index).fillna(0.0)
    out = []
    for c in ("short_alt_long_btc", "short_alt_long_btc_beta", "funding_sort_ls"):
        x = w[c]
        row = {"strategy": c, "weeks": len(x), "mean_wk": x.mean(), "ann_ret": x.mean() * 52, "ann_vol": x.std() * np.sqrt(52),
               "sharpe": x.mean() / x.std() * np.sqrt(52), "t": x.mean() / x.std() * np.sqrt(len(x)),
               "worst_wk": x.min(), "max_dd": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min(),
               "corr_vegas": float(np.corrcoef(x, vw)[0, 1])}
        for y, g in x.groupby(x.index.year):
            row[f"y{y}"] = g.sum()
        for nm, m in (("is", x.index < SPLIT), ("oos", x.index >= SPLIT)):
            row[f"sharpe_{nm}"] = x[m].mean() / x[m].std() * np.sqrt(52)
        out.append(row)
    show(out, "E. 搭配策略（每週換倉，1 倍名目，已扣資金費率和換倉成本）", "baskets")
    ep = (w.index >= "2026-03-01") & (w.index < "2026-05-06")
    print("2026-03~05 期間合計：", w.loc[ep, ["short_alt_long_btc", "short_alt_long_btc_beta", "funding_sort_ls"]].sum().round(3).to_dict(),
          "維加斯", round(float(vw[ep].sum()), 3))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    OUT.mkdir(exist_ok=True)
    if cmd == "enrich":
        enrich()
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else "in_pit")
    elif cmd == "baskets":
        baskets()
    elif cmd == "followup":
        followup(sys.argv[2] if len(sys.argv) > 2 else "in_pit")
