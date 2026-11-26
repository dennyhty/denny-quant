"""影子紀錄（只記錄不下單）累計結果：python -m research.shadow_report

資料是網站的 data/shadow/ledger.json（shadow_snapshot.py 每次部署更新）。R 只扣手續費，沒扣滑價和資金費；
回測裡這些成本平均約 -0.06R（成交額 1,000–2,000 萬）到 -0.09R，所以和回測對照時要先扣掉。
"""
from __future__ import annotations

import json
import urllib.request

BASE = "https://dennyhty.github.io/denny-quant/data/shadow/"


def main() -> None:
    with urllib.request.urlopen(BASE + "ledger.json", timeout=30) as r:
        book = json.load(r)
    groups: dict[str, list[dict]] = {}
    for e in book:
        groups.setdefault(e.get("group") or "?", []).append(e)
    print(f"{'組別':<16}{'已結束':>6}{'持倉中':>6}{'勝率':>8}{'平均R(未扣滑價)':>16}{'合計R':>8}")
    for g, es in sorted(groups.items()):
        closed = [e for e in es if e["status"] == "CLOSED"]
        n = len(closed)
        sr = sum(e["r"] for e in closed)
        print(f"{g:<16}{n:>6}{sum(e['status'] in ('OPEN', 'PENDING') for e in es):>6}"
              f"{(sum(e['r'] > 0 for e in closed) / n if n else float('nan')):>8.0%}"
              f"{(sr / n if n else float('nan')):>16.3f}{sr:>8.2f}")


if __name__ == "__main__":
    main()
