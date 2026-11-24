"""網站「績效」頁的日線 EMA8/20 吞沒回測（web/perf/engulf_backtest.json），並把 1D 摘要加進 web/perf/summary.json。

python -m research.engulf_1d_opt export          # 先產生 output/cv_ledgers/engulf_1d_opt.pkl
python -m research.site_performance_engulf       → web/perf/engulf_backtest.json

用帳本裡的 BASE（= 實盤規則）：時點幣池（上個月成交額前 100 的幣安 USDT 永續）、不設成交額門檻、
日線 EMA8/20 5 根內剛交叉＋吞沒 K、掛單在實體一半 3 天內成交、止損 1.5 ATR、1R 全出、60 天到期、同幣同時一張。
R 用 r_net：已扣手續費每邊 0.05%、滑價（依成交額分級）和資金費率（和維加斯頁不同，維加斯頁沒扣滑價和資金費率）。
每筆 0.25% 風險（和維加斯頁一樣，方便比較）；損益記在出場那天。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import site_performance as sp

LED = Path("output/cv_ledgers/engulf_1d_opt.pkl")
OUT = Path("web/perf/engulf_backtest.json")


def main() -> None:
    led = pd.read_pickle(LED)
    tr = led[led["config"] == "BASE"].copy()
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    tr["r"] = tr["r_net"]
    tr["coin"] = tr["symbol"]
    tr = tr.sort_values(["exit_t", "time"]).reset_index(drop=True)
    days = pd.date_range(tr["time"].min().floor("D"), tr["exit_t"].max().floor("D"), freq="D")
    btc = sp.btc_daily(days)
    out = {"source": "binance_um_pit", "risk": sp.RISK, "net_costs": True, "generated": str(pd.Timestamp.now(tz="UTC").date()),
           "dates": [str(d.date()) for d in days], "books": {}}
    btc_norm = (btc / btc.dropna().iloc[0] - 1) * 100 if btc.notna().any() else btc
    out["btc_pct"] = [None if pd.isna(v) else round(float(v), 1) for v in btc_norm]
    d = sp.daily_returns(tr, days)
    comp = np.exp(np.log1p(d).cumsum())
    out["books"]["1D"] = {
        "metrics": sp.metrics(tr, days), "yearly": sp.yearly(tr, btc), "monthly": sp.monthly(tr),
        "recent": sp.recent(tr, days), "rolling_avg_r": sp.rolling(tr, n=300, every=5), "histogram": sp.histogram(tr["r"].to_numpy()),
        "curve": {"simple_pct": (d.cumsum() * 100).round(2).tolist(), "compound_pct": ((comp - 1) * 100).round(2).tolist(),
                  "dd_pct": ((comp / comp.cummax() - 1) * 100).round(2).tolist()},
    }
    OUT.write_text(json.dumps(sp.clean(out), ensure_ascii=False, separators=(",", ":")))
    keys = ("trades", "win_rate", "avg_r", "profit_factor", "max_dd_pct", "cagr_pct", "simple_per_year_pct", "start", "end")
    sp_path = OUT.with_name("summary.json")
    summary = json.loads(sp_path.read_text(encoding="utf-8"))
    summary["1D"] = sp.clean({k: out["books"]["1D"]["metrics"].get(k) for k in keys})
    sp_path.write_text(json.dumps(summary, ensure_ascii=False))
    m = out["books"]["1D"]["metrics"]
    print(f"{m['trades']} 筆、勝率 {m['win_rate']:.1%}、每筆 {m['avg_r']:+.3f}R、CAGR {m['cagr_pct']:.1f}%、最大回撤 {m['max_dd_pct']:.1f}%")


if __name__ == "__main__":
    main()
