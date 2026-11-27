"""只記錄不下單的影子紀錄：python shadow_snapshot.py --out site

實盤（交易程式）只做「市值前 100 名、近 30 天日均成交額 ≥ 3,000 萬 U」的維加斯訊號。這支程式另外掃描
CoinMarketCap 市值前 200 名，把「實盤不會做」的維加斯 4H / 2H 訊號照實盤規則追蹤到結束，只存紀錄：
不推播、不下單、不進網站訊號頁（2026-11-26 Denny 要求，用來累積真實樣本，再決定門檻要不要調）。

記錄的組別（group）：
  top100_lt5M / _5M / _10M / _20M          市值前 100，成交額 <500 萬 / 500–1,000 萬 / 1,000–2,000 萬 / 2,000–3,000 萬
  r101_200_lt10M / _10M / _20M / _30M      市值 101–200，成交額 <1,000 萬 / 1,000–2,000 萬 / 2,000–3,000 萬 / ≥3,000 萬
  （前 100 名且 ≥3,000 萬的是實盤做的，不記；量縮時「隨大盤縮放」會多放進來的單可由 adv_usdt 和當時大盤量比事後算）
每筆都存市值名次（rank）和訊號當下近 30 天日均成交額（adv_usdt），事後想換別的切法也行。

輸出：
  <out>/data/shadow/ledger.json    每個訊號的結果（接續上一次）
  <out>/data/shadow/meta.json      {訊號 id: {rank, adv_usdt, group}}
  <out>/data/shadow/summary.json   各組已結束筆數、勝率、平均 R（只扣手續費；沒扣滑價和資金費）
  <out>/data/shadow/state.json     內部狀態（接續掃描用）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grid_trading import signals
from grid_trading.binance_history import binance_symbols  # noqa: F401  (保持和 signal_snapshot 同一套代號規則)
from grid_trading.binance_public import BinancePublic, from_binance
from grid_trading.signals import ledger as ledger_mod
from grid_trading.signals.engine import Engine
from grid_trading.universe import binance_perp_for, fetch_cmc_top, monthly_top
from signal_server import _sanitize

LIVE_ADV = 3e7  # 交易程式的 --min-adv
LIVE_RANK = 100
UNIVERSE_N = 200


def group_of(rank: int, adv: float | None) -> str | None:
    """None = 實盤會做的（不記）或沒有成交額資料。"""
    if adv is None:
        return None
    if rank <= LIVE_RANK:
        if adv >= LIVE_ADV:
            return None
        return "top100_" + ("20M" if adv >= 2e7 else "10M" if adv >= 1e7 else "5M" if adv >= 5e6 else "lt5M")
    return "r101_200_" + ("30M" if adv >= 3e7 else "20M" if adv >= 2e7 else "10M" if adv >= 1e7 else "lt10M")


def rank_map(coins: list[str], listed: set[str]) -> dict[str, int]:
    """網站代號 → 市值名次（照 coins 的順序，從 1 開始）。"""
    out: dict[str, int] = {}
    for i, c in enumerate(coins, 1):
        b = binance_perp_for(c, listed)
        if b:
            out.setdefault(from_binance(b), i)
    return out


def summarize(book: list[dict], meta: dict) -> dict:
    groups: dict[str, dict] = {}
    for e in book:
        g = (meta.get(e["id"]) or {}).get("group")
        if not g:
            continue
        b = groups.setdefault(g, {"trades": 0, "wins": 0, "sum_r": 0.0, "open": 0})
        if e["status"] == "CLOSED":
            b["trades"] += 1
            b["wins"] += e["r"] > 0
            b["sum_r"] += e["r"]
        elif e["status"] not in ("EXPIRED",):
            b["open"] += 1
    for b in groups.values():
        n = b["trades"]
        b["win_rate"] = b["wins"] / n if n else None
        b["expectancy_r"] = b["sum_r"] / n if n else None
    return groups


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def load(path: str | None, default):
    try:
        return json.loads(Path(path).read_text()) if path and Path(path).exists() else default
    except Exception:
        return default


def main() -> int:
    p = argparse.ArgumentParser(description="影子紀錄（只記錄不下單）")
    p.add_argument("--out", default="site")
    p.add_argument("--previous-state", help="上一次的 shadow/state.json")
    p.add_argument("--ledger", help="上一次的 shadow/ledger.json")
    p.add_argument("--meta", help="上一次的 shadow/meta.json")
    p.add_argument("--intervals", nargs="+", default=["4H", "2H"])
    p.add_argument("--keep-days", type=float, default=7)
    p.add_argument("--max-failed", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out) / "data" / "shadow"
    prev_state = load(args.previous_state, {})
    previous = load(args.ledger, [])
    meta = load(args.meta, {})

    top = monthly_top(prev_state.get("universe"), n=UNIVERSE_N, fetch=fetch_cmc_top)
    if top["status"] in ("stale", "fallback"):
        print(f"注意：市值前 {UNIVERSE_N} 名抓不到（{top.get('error')}），{top['status']}", file=sys.stderr)
    if top["status"] == "fallback":
        print("沒有名單，這次不記錄", file=sys.stderr)
        return 1
    api = BinancePublic()
    ranks = rank_map(top["coins"], api.usdt_perps())
    symbols = sorted(ranks)
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine([signals.get("vegas_trend")], symbols, args.intervals, args.keep_days, state_dir=out)
    engine.previous = {s["id"]: s for s in prev_state.get("signals", []) if "id" in s}
    engine.blocked = dict(prev_state.get("blocked") or {})
    engine.scanned = dict(prev_state.get("scanned") or {})
    if prev_state.get("updated"):  # 新進名單的幣：當成上次更新時已經掃過，舊 K 線上的訊號不算新的
        for s in symbols:
            for iv in args.intervals:
                engine.scanned.setdefault(f"{s}|{iv}", prev_state["updated"])
    else:  # 第一次執行：只從現在開始記，不補過去 7 天的舊訊號
        import time as _t
        for s in symbols:
            for iv in args.intervals:
                engine.scanned.setdefault(f"{s}|{iv}", int(_t.time() * 1000))
    engine.scan_once()
    if len(engine.failed) > args.max_failed * len(symbols) * len(args.intervals):
        print("K 線抓不到的太多，這次不更新", file=sys.stderr)
        return 1

    state = engine.state()
    keep = []
    for s in state["signals"]:
        rank = ranks.get(s["symbol"])
        adv = (s.get("info") or {}).get("adv_usdt")
        g = group_of(rank, adv) if rank else None
        if g is None and s["id"] not in meta:
            continue  # 實盤會做的單不記；以前記過的照樣追蹤到結束
        if s["id"] not in meta:
            meta[s["id"]] = {"rank": rank, "adv_usdt": adv, "group": g}
        keep.append(s)
    book = ledger_mod.update(previous, keep, lambda sym, iv: engine.klines(sym, iv, 1000), engine.fee)
    book = [e for e in book if e["id"] in meta]
    for e in book:
        e.update({k: meta[e["id"]][k] for k in ("rank", "adv_usdt", "group")})
    write_json(out / "ledger.json", book)
    write_json(out / "meta.json", meta)
    write_json(out / "summary.json", {"updated": state["updated"], "groups": summarize(book, meta)})
    state["signals"] = keep
    state["notifications"] = []
    state["universe"] = {k: top[k] for k in ("month", "coins", "fetched_at")}
    write_json(out / "state.json", state)
    (out / "notifications.json").unlink(missing_ok=True)  # 不需要通知紀錄
    print(f"影子紀錄：{len(symbols)} 個幣，本輪 {len(keep)} 個實盤不做的訊號，累計 {len(book)} 筆", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
