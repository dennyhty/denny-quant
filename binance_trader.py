"""幣安自動交易（實盤維加斯 4H + 2H）。要在美國以外、固定 IP 的主機上跑，安裝方法見 docs/binance_bot.md。

  python binance_trader.py --mode paper              # 紙上交易：不用金鑰，只記帳
  python binance_trader.py --mode testnet            # 測試網：要測試網的 BINANCE_API_KEY / BINANCE_API_SECRET
  TRADER_ALLOW_LIVE=1 python binance_trader.py --mode live   # 正式帳戶（測試網跑過再開）

選項：--risk 0.0025（每筆風險）、--max-risk 0.01（最小下單量造成的風險上限）、--leverage 5（逐倉）、
--max-open 15（同時最多幾張）、--dd-half-r 20（回撤 20R 後風險減半）、--state-dir（狀態檔位置）、
--no-stream（不開即時推送）、--poll-stream 300（推送正常時的輪詢間隔）。
暫停開新倉：在狀態資料夾建立 pause 檔（touch trader_state/pause）；刪掉就恢復。已開的倉位照常管理。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from grid_trading.trader.binance import MAINNET, TESTNET, BinanceFutures
from grid_trading.trader.bot import LABEL, Config, LiveBroker, Notifier, PaperBroker, Trader
from grid_trading.trader.risk import RiskRules
from grid_trading.trader.sizing import parse_filters
from grid_trading.trader.stream import UserStream, url_templates


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["paper", "testnet", "live"], default="paper")
    ap.add_argument("--risk", type=float, default=0.0025)
    ap.add_argument("--max-risk", type=float, default=0.01)
    ap.add_argument("--leverage", type=int, default=5)
    ap.add_argument("--max-open", type=int, default=None)
    ap.add_argument("--dd-half-r", type=float, default=None)
    ap.add_argument("--paper-equity", type=float, default=100.0)
    ap.add_argument("--state-dir", type=Path, default=Path("trader_state"))
    ap.add_argument("--poll", type=float, default=60.0, help="持倉管理幾秒看一次（預設 60）")
    ap.add_argument("--poll-stream", type=float, default=300.0,
                    help="即時推送正常時幾秒輪詢一次（預設 300；成交會馬上叫醒，這只是保險）")
    ap.add_argument("--no-stream", action="store_true", help="不開即時推送（帳戶 websocket），只照 --poll 輪詢")
    ap.add_argument("--scan-delay", type=float, default=None,
                    help="收盤後幾秒再掃描（預設：測試網、正式 15 秒，紙上 75 秒，和測試網錯開）")
    ap.add_argument("--once", action="store_true", help="只跑一輪（測試用）")
    a = ap.parse_args(argv)

    if a.mode == "live" and os.environ.get("TRADER_ALLOW_LIVE") != "1":
        print("正式帳戶要先設 TRADER_ALLOW_LIVE=1（請先在測試網跑過）", file=sys.stderr)
        return 2
    cfg = Config(mode=a.mode, risk=a.risk, max_risk=a.max_risk, leverage=a.leverage, paper_equity=a.paper_equity,
                 rules=RiskRules(max_open=a.max_open, dd_half_r=a.dd_half_r), state_dir=a.state_dir,
                 poll=a.poll, poll_stream=a.poll_stream, scan_delay=(75.0 if a.mode == "paper" else 15.0) if a.scan_delay is None else a.scan_delay)
    data = BinanceFutures(MAINNET)
    stream = None
    if a.mode == "paper":
        venue = data
        broker = PaperBroker(data, cfg)
    else:
        key, secret = os.environ.get("BINANCE_API_KEY", ""), os.environ.get("BINANCE_API_SECRET", "")
        if not (key and secret):
            print("沒有設定 BINANCE_API_KEY / BINANCE_API_SECRET", file=sys.stderr)
            return 2
        venue = BinanceFutures(TESTNET if a.mode == "testnet" else MAINNET, key, secret,
                               ban_file=a.state_dir / f"{a.mode}.ban",
                               # 測試網的 418 多半是共用轉送節點被別人用爆，不是我們：停一分鐘就再試（正式站照幣安給的時間停）
                               shared_ban_pause=60.0 if a.mode == "testnet" else None)
        wait = venue.banned_until - time.time()
        if wait > 0:  # 上次被幣安封鎖還沒解封：等到解封才開始，封鎖期間送請求只會被封得更久
            print(f"上次被幣安封鎖（HTTP 418）還沒解封，等 {wait:.0f} 秒"
                  f"（到 {time.strftime('%m-%d %H:%M', time.gmtime(venue.banned_until))} UTC）才啟動", flush=True)
            time.sleep(wait)
        venue.sync_time()
        broker = LiveBroker(venue, data, cfg)
        if not a.no_stream:  # 推播只是叫醒主迴圈，持倉照舊用 REST 輪詢查；BINANCE_WS_URL 可以換網址
            stream = UserStream(venue, url_templates(testnet=a.mode == "testnet"))
    filters = parse_filters(venue.exchange_info())
    # Telegram 只推測試網和實盤的訊息；紙上交易只寫紀錄（Denny 2026-11-06）
    tg = a.mode != "paper"
    notify = Notifier(LABEL[a.mode], os.environ.get("TELEGRAM_BOT_TOKEN", "") if tg else "",
                      os.environ.get("TELEGRAM_CHAT_ID", "") if tg else "")
    # systemd 停止服務送的是 SIGTERM：轉成正常結束，atexit 才會把佇列裡還沒送出的推播送完
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Trader(cfg, data, broker, notify, filters, stream=stream).run(once=a.once)
    notify.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
