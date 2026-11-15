"""產生靜態網站用的資料快照：python signal_snapshot.py --out site

給 GitHub Pages 用。GitHub Actions 每 15 分鐘跑一次：用幣安 USDT 永續合約的 K 線掃描實盤維加斯 4H / 2H、
寫出 JSON，網頁（web/）直接讀這些 JSON，不需要一直開著的伺服器。2026-10-20 起只用幣安（和 VPS 上的自動交易同一份合約名單），
派網、觀察組、美股/商品都不再掃描。

輸出：
  <out>/data/state.json                    訊號、通知、回測數字（同 /api/state）
  <out>/data/notifications.json            通知紀錄，下一次執行接著用
  <out>/data/ledger.json                   每個訊號的結果（實盤紀錄），下一次執行接著用
  <out>/data/chart/<symbol>_<interval>.json  圖表資料（同 /api/chart）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grid_trading import signals
from grid_trading.binance_history import binance_symbols
from grid_trading.binance_public import from_binance, universe
from grid_trading.signals import ledger as ledger_mod
from grid_trading.signals.engine import Engine
from signal_server import _sanitize

CHART_INTERVALS = ["15M", "60M", "4H", "1D"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def main() -> int:
    p = argparse.ArgumentParser(description="產生訊號網頁的靜態資料")
    p.add_argument("--out", default="site")
    p.add_argument("--web", default="web", help="網頁原始檔，會複製到 --out")
    p.add_argument("--previous", help="上一次的 notifications.json，用來接續通知紀錄")
    p.add_argument("--ledger", help="上一次的 ledger.json（實盤紀錄），沒有就從頭開始")
    p.add_argument("--strategies", nargs="+", default=["vegas_trend"], choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["4H", "2H"])
    p.add_argument("--keep-days", type=float, default=7)
    p.add_argument("--previous-state", help="上一次的 state.json；這次 K 線沒抓到的（幣, 週期）沿用裡面的訊號")
    p.add_argument("--max-failed", type=float, default=0.5,
                   help="K 線抓不到的（幣, 週期）超過這個比例（大概是幣安整個連不上）就不輸出，網站保留上一版")
    args = p.parse_args()

    out = Path(args.out)
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(args.web, out)
    # 策略研究頁讀的報告原文（reports/*.md）
    if Path("reports").is_dir():
        shutil.copytree("reports", out / "reports")
    # 每次部署給 app.js、style.css 加上版本號，瀏覽器才不會拿舊的頁面配新的程式（或反過來）
    # 版本號用網頁原始檔的內容算，網頁沒改就不變（一直開著的分頁只在真的改版時才重新載入）
    digest = hashlib.sha256()
    for f in sorted(Path(args.web).rglob("*")):
        if f.is_file():
            digest.update(str(f.relative_to(args.web)).encode() + f.read_bytes())
    version = digest.hexdigest()[:12]
    index = out / "index.html"
    index.write_text(index.read_text().replace('src="app.js"', f'src="app.js?v={version}"')
                     .replace('src="perf.js"', f'src="perf.js?v={version}"')
                     .replace('href="style.css"', f'href="style.css?v={version}"'))
    data.mkdir(parents=True)
    # 一直開著的分頁會用 data/version.json 發現網站改版，自己重新載入，不會繼續跑舊程式
    (data / "version.json").write_text(json.dumps({"version": version}))
    (out / ".nojekyll").write_text("")
    try:
        prev_state = json.loads(Path(args.previous_state).read_text()) if args.previous_state else {}
    except Exception:
        prev_state = {}
    previous = json.loads(Path(args.ledger).read_text()) if args.ledger and Path(args.ledger).exists() else []
    if args.previous and Path(args.previous).exists():
        (data / "notifications.json").write_text(json.dumps(json.loads(Path(args.previous).read_text()), ensure_ascii=False))

    symbols = universe()
    engine = Engine([signals.get(n) for n in args.strategies], symbols, args.intervals, args.keep_days, state_dir=data)
    engine.previous = {s["id"]: s for s in prev_state.get("signals", []) if "id" in s}
    engine.blocked = dict(prev_state.get("blocked") or {})
    engine.scanned = dict(prev_state.get("scanned") or {})
    # 派網時代的代號換成幣安合約名稱的（PEPE → 1000PEPE、SHIB → 1000SHIB）：沿用掃到哪根 K 線的紀錄，
    # 舊 K 線上的訊號才不會被當成新訊號
    for key, t in list(engine.scanned.items()):
        sym, iv = key.split("|", 1)
        new = from_binance(binance_symbols(sym.split("_")[0])[-1])
        if new != sym and new in symbols:
            engine.scanned.setdefault(f"{new}|{iv}", t)
    if not engine.scanned and prev_state.get("updated"):
        # 上一次的 state.json 還沒有 scanned：當成上一次每個幣、週期都掃到它更新的時間
        engine.scanned = {f"{s}|{iv}": prev_state["updated"] for s in symbols for iv in args.intervals}
    engine.scan_once()
    if engine.failed:
        sample = sorted(set(engine.errors.values()))[:3]
        print(f"注意：{len(engine.failed)} 組 K 線重試後仍抓不到，沿用上一次的 {engine.carried} 個訊號。"
              f"例如 {engine.failed[:5]}；錯誤 {sample}", file=sys.stderr)
    if len(engine.failed) > args.max_failed * len(symbols) * len(args.intervals):
        print("抓不到的太多（大概是幣安連不上），這次不更新網站", file=sys.stderr)
        return 1
    state = engine.state()
    state["static"] = True
    state["source"] = "binance"

    book = ledger_mod.update(previous, state["signals"], lambda sym, iv: engine.klines(sym, iv, 1000),
                             engine.fee)
    write_json(data / "ledger.json", book)
    state["ledger"] = ledger_mod.summary(book)
    write_json(data / "state.json", state)

    pairs = {(s["symbol"], s["interval"], s["strategy"]) for s in state["signals"]}
    charts: dict[tuple[str, str], tuple[str, int]] = {}  # (幣, 週期) → (策略, 根數)；訊號本身的週期多抓一些
    for sym, sig_iv, strat in sorted(pairs):
        for iv in dict.fromkeys([sig_iv, *CHART_INTERVALS]):
            bars = 1000 if iv == sig_iv else 500
            if charts.get((sym, iv), ("", 0))[1] < bars:
                charts[(sym, iv)] = (strat, bars)

    def write_chart(job):
        (sym, iv), (strat, bars) = job
        write_json(data / "chart" / f"{sym}_{iv}.json", engine.chart(sym, iv, strat, bars))

    with ThreadPoolExecutor(engine.workers) as ex:
        list(ex.map(write_chart, charts.items()))
    print(f"{len(state['signals'])} 個訊號，{len(pairs)} 個幣的圖表，輸出到 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
