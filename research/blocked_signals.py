"""沒開成單的維加斯訊號，現在的結果（只做分析，唯讀）。

python -m research.blocked_signals [--since 2026-11-19] [--book live]

資料全是網站公開的：data/notifications.json（訊號與止盈止損事件）和 data/trader.json（實際開了哪些單）。
R 用現行出場規則換算：1/3/5R 各平 1/4，TP1 後保本，TP2 後止損移到 TP1，TP3 後最後 1/4 追蹤止損。
沒扣手續費、滑價、資金費（小幣價差大，實際會更差）。日線吞沒的出場規則不同，這裡不算。
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone

BASE = "https://dennyhty.github.io/denny-quant/data/"


def fetch(name: str):
    with urllib.request.urlopen(BASE + name, timeout=30) as r:
        return json.load(r)


def price(text: str) -> float:
    return float(re.findall(r"[\d.]+", text.split("@")[-1] if "@" in text else text.split()[-1])[-1])


def signal_r(entry: float, side: int, events: dict) -> tuple[float | None, str]:
    """回傳 (R, 狀態)；還沒收完是 (None, 'open')。"""
    if "TP1" not in events:
        return (-1.0, "止損") if "SL" in events else (None, "未結束")
    one_r = abs(price(events["TP1"]) - entry)
    if one_r == 0:
        return None, "未結束"
    r = 0.25
    if "BE_EXIT" in events:
        return r, "TP1 後保本"
    if "TP2" not in events:
        return None, "未結束"
    if "PROTECT_EXIT" in events:
        return 0.25 + 0.75 + 0.5, "TP2 後回到 TP1 出場"
    if "TP3" not in events:
        return None, "未結束"
    r = 0.25 * (1 + 3 + 5)
    if "TRAIL_EXIT" in events:
        r += 0.25 * side * (price(events["TRAIL_EXIT"]) - entry) / one_r * 1.0
        return r, "TP3 後追蹤出場"
    return None, "未結束"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-19")
    ap.add_argument("--book", default="live", choices=["live", "testnet"])
    a = ap.parse_args()
    since = int(datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    notes, books = fetch("notifications.json"), fetch("trader.json")["books"][a.book]
    opened = {(r["symbol"], r["interval"], r["side"], r["opened"] // 3600000) for r in books["ledger"] + books["positions"]}
    by: dict[str, dict] = {}
    for n in notes:
        by.setdefault(n["signal_id"], {})[n["kind"]] = n["text"]
        if n["kind"] == "NEW":
            by[n["signal_id"]]["_t"] = n["time"]
    rows, tot = [], []
    for sid, ev in sorted(by.items(), key=lambda kv: kv[1].get("_t", 0)):
        if "NEW" not in ev or ev["_t"] < since or "vegas_trend" not in sid:
            continue
        sym, iv = sid.split("|")[0].replace("_USDT_PERP", "USDT"), sid.split("|")[1]
        side = "SHORT" if "做空" in ev["NEW"] else "LONG"
        h = ev["_t"] // 3600000
        if any(s == sym and i == iv and sd == side and abs(hh - h) <= 1 for s, i, sd, hh in opened):
            continue
        r, st = signal_r(price(ev["NEW"]), -1 if side == "SHORT" else 1, ev)
        if r is not None:
            tot.append(r)
        t = datetime.fromtimestamp(ev["_t"] / 1000, timezone.utc).strftime("%m-%d %H:%M")
        rows.append(f"{t} {sym:<14}{iv:<3}{side:<6}{'' if r is None else f'{r:+.2f}R':<8}{st}")
    print(f"{a.book} 帳本沒開單的維加斯訊號（{a.since} 起）：{len(rows)} 個，已結束 {len(tot)} 個")
    print("\n".join(rows))
    if tot:
        print(f"已結束合計 {sum(tot):+.2f}R，平均 {sum(tot) / len(tot):+.2f}R／張（未扣成本）")


if __name__ == "__main__":
    main()
