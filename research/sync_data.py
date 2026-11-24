"""把研究常用的幣安 K 線一次抓齊／補到最新，存在共用快取（grid_trading/data_cache.py）。

    python -m research.sync_data                      # TOP100 的 1h、4h（回測最常用）
    python -m research.sync_data --intervals 1h 4h 1d --coins BTC ETH
    python -m research.sync_data --status             # 只看快取現況，不連網

已經是最新的檔案不會連網，所以每個月跑一次就能把全部補到上個月底；回測腳本直接用
grid_trading.binance_history.fetch_binance / load，不用先跑這支。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from grid_trading.binance_history import fetch_binance
from grid_trading.data_cache import cache_root, latest_complete_month
from grid_trading.universe import TOP100


def status(coins: list[str], intervals: list[str]) -> None:
    from grid_trading.binance_history import CACHE_DIR

    latest = latest_complete_month()
    print(f"快取：{cache_root()}（網路上最新月檔：{latest}）")
    for iv in intervals:
        have = stale = 0
        for c in coins:
            p = CACHE_DIR / f"{c}_{iv}.csv.gz"
            if not p.exists():
                continue
            have += 1
            last = pd.Timestamp(int(pd.read_csv(p, usecols=["time"])["time"].iloc[-1]), unit="ms").to_period("M")
            stale += last < latest
        print(f"{iv}: {have}/{len(coins)} 個幣有快取，其中最後一根 K 線早於 {latest} 的 {stale} 個（下架或需要更新）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", nargs="+", default=["1h", "4h"])
    ap.add_argument("--coins", nargs="+", default=TOP100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status(a.coins, a.intervals)

    def job(args):
        coin, iv = args
        try:
            return coin, iv, len(fetch_binance(coin, iv)), ""
        except Exception as e:  # 單一個幣失敗不影響其他幣
            return coin, iv, 0, repr(e)

    jobs = [(c, iv) for iv in a.intervals for c in a.coins]
    with ThreadPoolExecutor(a.workers) as ex:
        for coin, iv, n, err in ex.map(job, jobs):
            print(coin, iv, n, err, flush=True)
    status(a.coins, a.intervals)


if __name__ == "__main__":
    main()
