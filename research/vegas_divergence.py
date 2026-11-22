"""維加斯 + 背離／布林帶「共振」：維加斯訊號出現時剛好也有同向背離或布林條件，才做（或加碼）。2026-11-22。

之前（2026-10-13 兩份基礎指標研究）測過 RSI 背離、MACD 背離、布林帶單獨用、加趨勢過濾、和維加斯並行下單，
都沒有加成。這裡測還沒測過的「同時成立」：只看實盤維加斯訊號本身，把它分成「近期有同向條件」和「沒有」兩群。

事先登記（跑之前就定好，不看結果再加）。條件都在訊號 K 棒收盤時就已知（不偷看未來），W = 12 根（4H 約 2 天、2H 約 1 天）：
  rsi_div    近 W 根內出現同向的 RSI(14) 一般背離（做多：價格創新低但 RSI 低點墊高；做空相反）
  macd_div   近 W 根內出現同向的 MACD 柱一般背離
  any_div    rsi_div 或 macd_div
  rsi_hdiv   近 W 根內出現同向的 RSI 隱藏背離（做多：價格低點墊高但 RSI 低點更低＝順勢回檔結束）
  macd_hdiv  近 W 根內出現同向的 MACD 柱隱藏背離
  bb_out     訊號 K 棒收在布林(20,2) 外側（做多：收在上軌之上）
  bb_squeeze 近 W 根內布林帶寬在近 120 根最窄 20%（收縮後啟動）
  bb_touch   近 W 根內碰過反向軌（做多：最低價碰到下軌＝回檔到下軌後再走趨勢）
  no_opp_div 反過來：近 W 根內有「反向」一般背離（RSI 或 MACD）的單不做
背離的定義和 2026-10-13 研究一樣（research.basic_indicators_research.divergence，look 30、recent 5）。
每個條件兩種用法：只做有條件的單（filter），或有條件的單風險加倍（size2x），共 9 × 2 = 18 組；
另外 W = 6、24 只當參數高原檢查（9 × 2 = 18 組），全部記進多重檢定帳本。

幣池：實盤現況＝每月 CoinMarketCap 市值前 100（扣穩定幣）∩ 近 30 天日均成交額 >= 3,000 萬 U，幣安 USDT 永續（含下架）。
出場同實盤：止損 1.5 ATR、TP 1/3/5R 各 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR，沒有持倉時間上限。
成本：手續費每邊 0.05%、依成交額分級的滑價、資金費率（grid_trading/costs.py）。2020-01 起，樣本外 2024-01 起。
帳本：同週期同幣同時只一張（被過濾掉的單不佔位子），2H、4H 合併。

判斷（事先定好）：一個條件算「有共振」要同時做到
  1. 有條件 vs 沒條件的每筆 R 差（同日同向算一群的群集 SE）t >= 3.0（18 組的 Bonferroni，雙尾 5%）；
  2. 樣本內（2020-23）和樣本外（2024+）差值都同號為正；
  3. 用法（filter 或 size2x）的日報酬 Sharpe 高於基準。

python -m research.universe_buckets fetch       # CMC 每月市值排名 → data/cmc_hist/
python -m research.vegas_divergence run         # → output/vd_trades.csv.gz
python -m research.vegas_divergence report      # → output/vd_*.csv，印出比較
報告：reports/signals/report_vegas_divergence_20261122.md
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import research.entry_research as er
from grid_trading import indicators as ind
from grid_trading.costs import apply_costs
from grid_trading.pit_universe import coin_of, fetch_symbol, list_um_symbols
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from research.basic_indicators_research import divergence
from research.pit_backtest import KLINE, PLAN
from research.universe_buckets import attach_rank, base_coin, cmc_ranks
from research.vegas_params import _funding

OUT = Path("output")
START = pd.Timestamp("2020-01-01", tz="UTC")
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
ADV_FLOOR = 30e6
WINDOWS = (6, 12, 24)
MAIN_W = 12
CONDS = ("rsi_div", "macd_div", "any_div", "rsi_hdiv", "macd_hdiv", "bb_out", "bb_squeeze", "bb_touch", "no_opp_div")


def hidden_divergence(price_lo, price_hi, osc, look: int = 30, recent: int = 5):
    """隱藏背離：最近 recent 根的低點比前一段高（價格墊高），但指標低點比前一段低；看空相反。回傳（看多, 看空）。"""
    plo, phi, o = pd.Series(price_lo), pd.Series(price_hi), pd.Series(osc)
    p_rl, p_pl = plo.rolling(recent).min(), plo.shift(recent).rolling(look - recent).min()
    o_rl, o_pl = o.rolling(recent).min(), o.shift(recent).rolling(look - recent).min()
    p_rh, p_ph = phi.rolling(recent).max(), phi.shift(recent).rolling(look - recent).max()
    o_rh, o_ph = o.rolling(recent).max(), o.shift(recent).rolling(look - recent).max()
    bull = (plo == p_rl) & (p_rl > p_pl) & (o_rl < o_pl)
    bear = (phi == p_rh) & (p_rh < p_ph) & (o_rh > o_ph)
    return bull.to_numpy(), bear.to_numpy()


def _recent(x: np.ndarray, w: int) -> np.ndarray:
    """近 w 根（含當根）內有沒有成立過。"""
    return pd.Series(x.astype(float)).rolling(w, min_periods=1).max().to_numpy() > 0


def features(p: pd.DataFrame, idx: np.ndarray, ds: np.ndarray) -> dict[str, np.ndarray]:
    """每個訊號在每個 W 下的條件旗標，欄名 {cond}_w{W}。"""
    c, h, lo = p["close"], p["high"], p["low"]
    rsi = ind.rsi(c)
    hist = ind.macd(c)["hist"]
    raw = {"rsi": divergence(lo, h, rsi), "macd": divergence(lo, h, hist),
           "rsi_h": hidden_divergence(lo, h, rsi), "macd_h": hidden_divergence(lo, h, hist)}
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    up_b, lo_b = (mid + 2 * sd).to_numpy(), (mid - 2 * sd).to_numpy()
    bw = (up_b - lo_b) / mid.to_numpy()
    sq = (pd.Series(bw) <= pd.Series(bw).rolling(120).quantile(0.2)).to_numpy()
    cl, hi, lw = c.to_numpy(), h.to_numpy(), lo.to_numpy()
    lg = ds > 0
    out = {}
    out["bb_out"] = np.where(lg, cl[idx] > up_b[idx], cl[idx] < lo_b[idx])
    for w in WINDOWS:
        rec = {k: (_recent(b, w)[idx], _recent(s, w)[idx]) for k, (b, s) in raw.items()}
        same = {k: np.where(lg, b, s) for k, (b, s) in rec.items()}
        opp = {k: np.where(lg, s, b) for k, (b, s) in rec.items()}
        out[f"rsi_div_w{w}"] = same["rsi"]
        out[f"macd_div_w{w}"] = same["macd"]
        out[f"any_div_w{w}"] = same["rsi"] | same["macd"]
        out[f"rsi_hdiv_w{w}"] = same["rsi_h"]
        out[f"macd_hdiv_w{w}"] = same["macd_h"]
        out[f"bb_squeeze_w{w}"] = _recent(sq, w)[idx]
        out[f"bb_touch_w{w}"] = np.where(lg, _recent(lw <= lo_b, w)[idx], _recent(hi >= up_b, w)[idx])
        out[f"no_opp_div_w{w}"] = ~(opp["rsi"] | opp["macd"])
        out[f"bb_out_w{w}"] = out["bb_out"]
    del out["bb_out"]
    return out


def _work(args):
    sym, interval, fee = args
    er.MAX_BARS = 10 ** 9  # 實盤沒有持倉時間上限
    for k in range(6):
        try:
            df = fetch_symbol(sym, KLINE[interval])
            break
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    else:
        print(f"放棄 {sym} {interval}", file=sys.stderr)
        return None
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
    if not idx:
        return None
    idx, ds = np.array(idx), np.array(ds)
    arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
    t = pd.to_datetime(p["time"], utc=True)
    er.PLANS = {interval: PLAN}
    s = er.simulate(arr, idx, ds, interval, fee, 0.0)
    bpd = 6 if interval == "4H" else 12
    adv = (p["volume"] * p["close"]).rolling(30 * bpd, min_periods=bpd).sum() / 30
    return pd.DataFrame({"symbol": sym, "coin": coin_of(sym), "interval": interval, "i": idx, "side": ds,
                         "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                         "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "risk_pct": s["risk_pct"],
                         "adv_usdt": adv.iloc[idx].to_numpy(), **features(p, idx, ds)})


def symbols() -> list[str]:
    """幣安 USDT 永續（含下架）裡，2020 年起曾經進過 CMC 市值前 100 的。"""
    rk = cmc_ranks()
    top = set(rk.loc[(rk["rank"] <= 100) & (rk["month"] >= "2020-01"), "symbol"])
    return [s for s in list_um_symbols() if any(c in top for c in base_coin(s))]


def run(fee: float = 0.0005, workers: int = 4) -> pd.DataFrame:
    syms = symbols()
    jobs = [(s, iv, fee) for iv in ("4H", "2H") for s in syms]
    print(f"{len(syms)} 個合約，{len(jobs)} 個工作", file=sys.stderr)
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(_work, jobs, chunksize=2)):
            if got is not None:
                parts.append(got)
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr, flush=True)
    tr = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    tr = tr[tr["time"] >= START]
    tr = attach_rank(tr)
    tr["live"] = (tr["mcap_rank"] <= 100) & (tr["adv_usdt"] >= ADV_FLOOR)
    tr = tr[tr["live"]].reset_index(drop=True)
    return apply_costs(tr, funding=_funding(sorted(tr["symbol"].unique())))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUT.mkdir(exist_ok=True)
    if cmd == "run":
        tr = run()
        tr.to_csv(OUT / "vd_trades.csv.gz", index=False)
        print(tr.groupby("interval").size())
    elif cmd == "report":
        from research.vegas_divergence_report import log, report
        s, verdicts = report()
        if "--log" in sys.argv:
            log(s, verdicts)


if __name__ == "__main__":
    main()
