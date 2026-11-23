"""「候選與新的非維加斯策略」家族的長帳本，給 research/cv_all.py（滾動前進＋CPCV＋PBO）用。2026-11-22。

python -m research.cv_export_candidates [--only fam1,fam2]  → output/cv_ledgers/{fam}.pkl/json

這些研究本來就用時點幣池（上個月成交額前 100）、幣安 USDT 永續、手續費每邊 0.05%＋滑價分級＋資金費率，
所以這裡直接重用原模組，每一組「預先登記」的設定各自產生一本帳（config 欄），不新增、不挑選：

- cand_daily_donchian（12 組，research/candidates_daily_trend.py，逐筆）：日線唐奇安突破 N∈{20,55,100} x 止損 x 出場。
- cand_daily_squeeze（8 組，research/candidates_squeeze.py，逐筆）：日線波動壓縮後突破。
- cand_xsec_portfolio（10 組，research/candidates_xsec.py，組合）：週再平衡橫斷面動能/低波動多空、BTC beta 中性。
- xsec_ext（8 組，research/candidates_xsec_ext.py，組合）：同上但三種對沖（btc/index/none）。
- new_xsec_factors（8 組，research/new_xsec.py，組合）：fund7/rev7/resrev7/max30 x q。
- new_funding_extreme（4 組，逐筆）：日線資金費率極端值事件交易。
- new_cme_gap（3 組，逐筆）：BTC/ETH 週末 CME 缺口回補（1h）。
- new_intraday_hour（2 組，逐筆）：池內等權指數 / BTC+ETH 的日內時段多空（1h）。

逐筆家族：呼叫原研究的 raw_trades → candidates_common.finalize（時點幣池、2020 起、同幣同時一張、apply_costs），
r_net 即原研究的扣成本 R。組合家族：原引擎輸出逐 (幣, 日) 列，這裡依再平衡期（period）加總成「一期一列」：
symbol="PORT"、side=1、time=該期第一列時間、exit_t=該期最後一列的 exit_t、r_net=該期組合報酬 / 0.25%
（和原研究的 r_net 同單位：帳戶報酬除以 RISK=0.25%，含 BTC/指數對沖腿、換手成本與資金費率）。

和原研究的差異（各家族 notes 內也有）：
- 沒有「原研究選主設定」這一步：CV 自己用訓練集 t 挑。
- xsec_ext 原研究從 2017-09 用現貨補長；time_cv 的資料起點是 2020-01-01，所以這裡只保留 2020-01-01 起的列
  （合約期；2017-2019 的現貨紙上多空不納入）。
- new_intraday_hour 的多/空時段原本就用 2020-2023 樣本內挑好（固定 3 多 3 空），整段帳本含這個樣本內挑選的偷看，
  CV 時 2020-23 的訓練/測試切法會偏樂觀（原研究的設定本來就固定）。
- new_cme_gap 只有 BTC、ETH，沒套時點幣池（兩者永遠在前 100，和原研究相同）。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from research.cv_export_vegas_live import save

KEEP = ["config", "symbol", "side", "time", "exit_t", "r", "r_net"]


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    led = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        led[c] = pd.to_datetime(led[c], utc=True)
    return led.sort_values(["config", "time"], kind="stable").reset_index(drop=True)


def port_rows(rows: pd.DataFrame, cfg: str, start: str = "2020-01-01") -> pd.DataFrame:
    """逐 (幣, 日) 列 → 一期（再平衡週）一列。"""
    rows = rows[rows["time"] >= pd.Timestamp(start, tz="UTC")]
    g = rows.groupby("period")
    out = pd.DataFrame({"time": g["time"].min(), "exit_t": g["exit_t"].max(), "r": g["r"].sum(),
                        "r_net": g["r_net"].sum()}).reset_index(drop=True)
    return out.assign(config=cfg, symbol="PORT", side=1)[KEEP]


# ---------------------------------------------------------------- 家族

def trade_fam(mod, data, mem) -> pd.DataFrame:
    from research.candidates_common import finalize
    parts = []
    for name, p in mod.configs().items():
        t0 = time.time()
        tr = finalize(mod.raw_trades(data, p), mem)
        parts.append(tr.assign(config=name)[KEEP])
        _log(f"[{mod.FAMILY}] {name}: {len(tr)} 筆 {time.time() - t0:.0f}s")
    return _concat(parts)


def fam_donchian(ctx):
    import research.candidates_daily_trend as m
    return trade_fam(m, ctx["data"], ctx["mem"])


def fam_squeeze(ctx):
    import research.candidates_squeeze as m
    return trade_fam(m, ctx["data"], ctx["mem"])


def fam_xsec(ctx):
    import research.candidates_xsec as m
    return _concat([port_rows(m.run(ctx["panel"], ctx["pool"], p), n) for n, p in m.configs().items()])


def fam_new_xsec(ctx):
    import research.new_xsec as m
    return _concat([port_rows(m.run(ctx["panel"], ctx["pool"], p), n) for n, p in m.configs().items()])


def fam_xsec_ext(ctx):
    import research.candidates_xsec_ext as fx
    import research.candidates_xsec_ext_run as er
    env = er.build()
    parts = []
    for name, cfg in fx.configs().items():
        rows = fx.run_ext(env["panel"], env["pool"], cfg)
        parts.append(port_rows(rows, name))
        _log(f"[xsec_ext] {name}: {len(parts[-1])} 期")
    return _concat(parts)


def fam_funding(ctx):
    import research.new_strategies_run as ns
    from research.candidates_common import finalize
    f3s = ns.f3_series(ctx["panel"], ctx["data"])
    parts = []
    for name, cfg in ns.N2_CFG.items():
        tr = finalize(ns.n2_raw(ctx["data"], f3s, cfg), ctx["mem"])
        parts.append(tr.assign(config=name)[KEEP])
        _log(f"[new_funding_extreme] {name}: {len(tr)} 筆")
    return _concat(parts)


def fam_cme(ctx):
    import research.new_strategies_run as ns
    evs = {s: ns.cme_events(s) for s in ("BTCUSDT", "ETHUSDT")}
    parts = []
    for name, cfg in ns.N3_CFG.items():
        tr = ns.n3_book(evs, cfg["g"])
        parts.append(tr.assign(config=name)[KEEP])
        _log(f"[new_cme_gap] {name}: {len(tr)} 筆")
    return _concat(parts)


def fam_hour(ctx):
    import research.new_strategies_run as ns
    ret, mask = ns.n4_load(ctx["mem"], sorted(set(ctx["mem"]["symbol"])))
    parts = []
    for name, cfg in ns.N4_CFG.items():
        tr, _, ex = ns.n4_build(ret, mask, cfg, 0)
        parts.append(tr.assign(config=name)[KEEP])
        _log(f"[new_intraday_hour] {name}: {len(tr)} 筆 多{ex['long_hours']} 空{ex['short_hours']}")
    return _concat(parts)


FAMS = {
    "cand_daily_donchian": (fam_donchian, "候選 A：日線唐奇安突破", "trades", "signals/report_candidates_20261020.md", 12, ""),
    "cand_daily_squeeze": (fam_squeeze, "候選 C：日線波動壓縮突破", "trades", "signals/report_candidates_20261020.md", 8, ""),
    "cand_xsec_portfolio": (fam_xsec, "候選 B：週橫斷面動能/低波動多空組合", "portfolio", "signals/report_candidates_20261020.md", 10,
                            "組合：一期（週）一列，r_net=該期含 BTC 對沖、成本、資金費率的組合報酬/0.25%"),
    "xsec_ext": (fam_xsec_ext, "候選 B 延伸版（三種對沖）", "portfolio", "signals/report_xsec_ext_20261020.md", 8,
                 "原研究用現貨補到 2017-09；這裡只保留 2020-01-01 起（time_cv 起點），2017-2019 現貨紙上組合不納入"),
    "new_xsec_factors": (fam_new_xsec, "新橫斷面因子（資金費率/反轉/MAX）", "portfolio", "signals/report_new_strategies_20261122.md", 8,
                         "組合：一期（週）一列，同 cand_xsec_portfolio 引擎"),
    "new_funding_extreme": (fam_funding, "資金費率極端值事件交易", "trades", "signals/report_new_strategies_20261122.md", 4, ""),
    "new_cme_gap": (fam_cme, "BTC/ETH 週末 CME 缺口回補", "trades", "signals/report_new_strategies_20261122.md", 3,
                    "只有 BTC、ETH（沒套時點幣池，和原研究相同）；1h 進出場"),
    "new_intraday_hour": (fam_hour, "日內時段多空（池內等權指數 / BTC+ETH）", "trades", "signals/report_new_strategies_20261122.md", 2,
                          "symbol=INDEX、1R=樣本內小時報酬標準差；多/空時段原研究就用 2020-23 樣本內挑好，整段帳本含此樣本內挑選"),
}
NEEDS_PANEL = {"cand_xsec_portfolio", "new_xsec_factors", "new_funding_extreme"}
NEEDS_DATA = NEEDS_PANEL | {"cand_daily_donchian", "cand_daily_squeeze"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    sel = [f for f in FAMS if not a.only or f in set(a.only.split(","))]
    ctx: dict = {}
    if set(sel) & (NEEDS_DATA | {"new_intraday_hour"}):
        from research.candidates_common import pool_table
        ctx["mem"] = pool_table()
        ctx["pool"] = set(zip(ctx["mem"]["symbol"], ctx["mem"]["month"]))
    if set(sel) & NEEDS_DATA:
        from research.candidates_common import load_daily
        ctx["data"] = load_daily()
    if set(sel) & NEEDS_PANEL:
        import research.candidates_xsec as cx
        ctx["panel"] = cx.build_panel(ctx["data"])
    for fam in sel:
        fn, label, kind, src, nreg, note = FAMS[fam]
        t0 = time.time()
        led = fn(ctx)
        ncfg = led["config"].nunique()
        notes = (f"時點幣池 top100、幣安 USDT 永續、費+滑價+資金費率，同原研究；設定網格=原預先登記全部 {nreg} 組"
                 f"（實際 {ncfg} 組）。{note}").strip()
        save(fam, led, {"label": label, "kind": kind, "source": src, "configs_registered": nreg, "notes": notes})
        print(f"{fam}: rows={len(led)} configs={ncfg}/{nreg} mean_r_net={led['r_net'].mean():+.4f} "
              f"span={led['time'].min().date()}..{led['time'].max().date()} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
