"""實盤維加斯（2H+4H、只 2H、只 4H）的長帳本，給 research/cv_all.py 用。

python -m research.pit_backtest          # 先產生 output/pit_trades.csv.gz
python -m research.cv_export_vegas_live  → output/cv_ledgers/vegas_live*.pkl/json

幣池：時點幣池（上個月成交額前 100）＋近 30 日日均成交額 >= 3,000 萬（實盤 --min-adv）；
同週期同幣同時一張；扣手續費、滑價、資金費率（r_net）。只有一組設定，所以沒有 PBO。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.pit_backtest import book

OUT = Path("output/cv_ledgers")


def save(family: str, led: pd.DataFrame, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    led.to_pickle(OUT / f"{family}.pkl")
    (OUT / f"{family}.json").write_text(json.dumps({"family": family, **meta}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")


def main() -> None:
    tr = pd.read_csv("output/pit_trades.csv.gz", parse_dates=["time", "exit_t"])
    tr["live_pool"] = tr["in_pit"] & (tr["adv_usdt"] >= 3e7)
    b = book(tr, "live_pool")
    keep = ["symbol", "side", "time", "exit_t", "r", "r_net", "interval"]
    for fam, sub, label in (("vegas_live", b, "實盤維加斯 2H+4H"), ("vegas_live_2h", b[b["interval"] == "2H"], "實盤維加斯 只 2H"),
                            ("vegas_live_4h", b[b["interval"] == "4H"], "實盤維加斯 只 4H")):
        save(fam, sub[keep].assign(config="live").reset_index(drop=True),
             {"label": label, "kind": "trades", "source": "signals/report_honest_backtest_20261020.md",
              "configs_registered": 1, "notes": "PIT top100 by prior-month volume ∩ adv30 >= 30M, live exits, net costs"})
        print(fam, len(sub), round(sub["r_net"].mean(), 4))


if __name__ == "__main__":
    main()
