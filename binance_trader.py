"""幣安自動交易（實盤維加斯 4H + 2H）。要在美國以外、固定 IP 的主機上跑，安裝方法見 docs/binance_bot.md。

  python binance_trader.py --mode paper              # 紙上交易：不用金鑰，只記帳
  python binance_trader.py --mode testnet            # 測試網：要測試網的 BINANCE_API_KEY / BINANCE_API_SECRET
  TRADER_ALLOW_LIVE=1 python binance_trader.py --mode live   # 正式帳戶（測試網跑過再開）

選項：--risk 0.0025（每筆風險）、--max-risk 0.01（最小下單量造成的風險上限）、--leverage 5（逐倉）、
--max-open 15（同時最多幾張）、--dd-half-r 20（回撤 20R 後風險減半）、--state-dir（狀態檔位置）、
--no-stream（不開即時推送）、--poll-stream 300（推送正常時的輪詢間隔）、
--min-adv 3e7（近 30 天日均成交額低於 3,000 萬 U 的幣不做，0 = 不設）、
--fixed-coins（用固定的舊名單，不每月更新市值前 100 名）、--entry-wait 600（限價進場最多等幾秒）、--taker（進場和止盈改回市價）。
預設（2026-11-17 起）：進場掛限價單在訊號收盤價、逾時沒成交的部分改市價；止盈掛限價單；止損仍是市價條件單。
--max-equity 300（資金上限：算倉位的權益最多當 300 U；也可設環境變數 TRADER_MAX_EQUITY，命令列優先）。
--check：上線前唯讀檢查（只限 testnet / live）：對時、權益、雙向持倉、持倉和掛單、API 金鑰權限；不下單、不改任何帳戶設定、不推 Telegram。
暫停開新倉：在狀態資料夾建立 pause 檔（touch trader_state/pause）暫停所有模式，或 pause-live / pause-testnet 只暫停那個模式；
刪掉就恢復。已開的倉位照常管理。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from grid_trading.trader.binance import MAINNET, TESTNET, BinanceError, BinanceFutures
from grid_trading.trader.bot import LABEL, Config, LiveBroker, Notifier, PaperBroker, Trader
from grid_trading.trader.risk import RiskRules
from grid_trading.trader.sizing import parse_filters
from grid_trading.trader.stream import UserStream, url_templates


SPOT = "https://api.binance.com"  # 查 API 金鑰權限（/sapi）走現貨主機，不是合約主機


def preflight(venue: BinanceFutures, mode: str, risk: float, max_equity: float | None, out=print) -> int:
    """上線前唯讀檢查：只有 GET，不下單、不撤單、不改帳戶設定、不推 Telegram。回傳結束碼：0 沒有硬問題，1 金鑰不能用。"""
    warn = 0
    try:
        venue.sync_time()
        out("[OK] 對時完成")
        bal = venue.balance_usdt()
        hedge = venue.hedge_mode()
        pos = venue.positions()
        n_orders = len(venue.open_orders())
        n_algo = len(venue.open_conditionals())
    except BinanceError as e:
        out(f"[錯誤] 金鑰或帳戶檢查失敗：{e}")
        if e.code == -2015:
            out("  提示：-2015 是金鑰無效、權限不足或 IP 不在白名單；請到幣安 API 管理確認這台 VPS 的 IP 有加進白名單"
                "（IPv4 和 IPv6 都要看，curl -4 / curl -6 ifconfig.me 各查一次）")
        return 1
    except Exception as e:  # 網路等其他錯誤：沒辦法確認金鑰，視為不通過
        out(f"[錯誤] 檢查失敗：{type(e).__name__}: {e}")
        return 1
    out(f"[OK] USDT 權益：{bal:,.2f} U")
    if hedge:
        out("[OK] 雙向持倉（Hedge Mode）：已開啟")
    else:
        out("[提醒] 雙向持倉（Hedge Mode）：目前是關閉；程式會在第一次下單時自動切換成開啟。"
            "警告：帳戶有持倉時幣安會拒絕切換，請先確認帳戶沒有倉位")
        warn += 1
    if pos:
        out(f"[警告] 帳戶目前有 {len(pos)} 個持倉：" + "、".join(f"{s} {side} {q:g}" for (s, side), q in sorted(pos.items())))
        out("  不是這個程式開的倉位只會提醒、不會管理（不掛止損止盈）；建議用沒有手動倉位的專用帳戶")
        warn += 1
    else:
        out("[OK] 目前沒有持倉")
    out(f"[OK] 掛單：一般訂單 {n_orders} 張、條件單（algo）{n_algo} 張")
    if mode == "testnet":
        out("[略過] API 金鑰權限檢查：測試網沒有這個端點")
    else:
        try:
            r = venue.api_restrictions(SPOT)
            out(f"[OK] API 金鑰權限：提幣 enableWithdrawals={r.get('enableWithdrawals')}、"
                f"合約 enableFutures={r.get('enableFutures')}、IP 限制 ipRestrict={r.get('ipRestrict')}")
            if r.get("enableWithdrawals"):
                out("[嚴重警告] 這把金鑰可以提幣！請立刻到幣安 API 管理關掉「提幣」，或換一把只有讀取 + 合約的金鑰")
                warn += 1
            if not r.get("ipRestrict"):
                out("[嚴重警告] 這把金鑰沒有限制 IP！請在幣安 API 管理設成只允許這台 VPS 的 IP")
                warn += 1
            if r.get("enableFutures") is False:
                out("[警告] 這把金鑰沒有開「合約」權限，無法下單")
                warn += 1
        except Exception as e:  # 權限查不到不擋：印出錯誤、照樣往下
            out(f"[提醒] 查不到 API 金鑰權限（{e}）；請自己到幣安 API 管理確認：不能提幣、只勾讀取 + 合約、有限制 IP")
    eff = min(bal, max_equity) if max_equity is not None else bal
    cap = f"{max_equity:g} U" if max_equity is not None else "未設"
    out(f"[OK] 算倉位用的權益：{eff:,.2f} U（資金上限 {cap}）")
    out(f"[OK] 每筆風險 {risk * 100:g}%：約 {eff * risk:,.2f} U")
    out("檢查完成：" + ("沒有硬性問題" if not warn else f"有 {warn} 項要留意（見上面）") + "；這次沒有下單、沒有改任何設定")
    return 0


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
    ap.add_argument("--min-adv", type=float, default=3e7,
                    help="近 30 天日均成交額（USDT）低於這個數字的幣不做（預設 3,000 萬，2026-11-21 起；0 = 不設門檻）")
    ap.add_argument("--fixed-coins", action="store_true",
                    help="用 grid_trading/universe.py 的固定名單，不每月從 CoinMarketCap 更新市值前 100 名")
    ap.add_argument("--entry-wait", type=float, default=600.0,
                    help="限價進場最多等幾秒，沒成交完的部分改市價（預設 600）")
    ap.add_argument("--taker", action="store_true", help="進場和止盈都用市價（舊做法），不掛限價單")
    ap.add_argument("--max-equity", type=float, default=None,
                    help="資金上限（USDT）：算倉位的權益最多當這個數字（預設不設；也可用環境變數 TRADER_MAX_EQUITY，這個選項優先）")
    ap.add_argument("--check", action="store_true", help="上線前唯讀檢查（testnet / live）：不下單、不改設定、不推 Telegram，然後結束")
    ap.add_argument("--once", action="store_true", help="只跑一輪（測試用）")
    a = ap.parse_args(argv)

    if a.mode == "live" and os.environ.get("TRADER_ALLOW_LIVE") != "1":
        print("正式帳戶要先設 TRADER_ALLOW_LIVE=1（請先在測試網跑過）", file=sys.stderr)
        return 2
    max_equity = a.max_equity
    if max_equity is None and os.environ.get("TRADER_MAX_EQUITY", "").strip():
        try:
            max_equity = float(os.environ["TRADER_MAX_EQUITY"])
        except ValueError:
            print("TRADER_MAX_EQUITY 不是數字", file=sys.stderr)
            return 2
    if max_equity is not None and max_equity <= 0:
        print("資金上限要大於 0", file=sys.stderr)
        return 2
    if a.check:
        if a.mode == "paper":
            print("--check 只用在 testnet / live：紙上交易不連帳戶，沒有東西可檢查")
            return 0
        key, secret = os.environ.get("BINANCE_API_KEY", ""), os.environ.get("BINANCE_API_SECRET", "")
        if not (key and secret):
            print("沒有設定 BINANCE_API_KEY / BINANCE_API_SECRET", file=sys.stderr)
            return 1
        venue = BinanceFutures(TESTNET if a.mode == "testnet" else MAINNET, key, secret)  # 不寫 ban 檔、不用 Telegram
        print(f"== 幣安{'測試網' if a.mode == 'testnet' else '實盤'}上線前檢查（唯讀）==")
        return preflight(venue, a.mode, a.risk, max_equity)
    cfg = Config(mode=a.mode, risk=a.risk, max_risk=a.max_risk, leverage=a.leverage, paper_equity=a.paper_equity,
                 rules=RiskRules(max_open=a.max_open, dd_half_r=a.dd_half_r), state_dir=a.state_dir,
                 poll=a.poll, poll_stream=a.poll_stream, min_adv=a.min_adv, entry_wait=a.entry_wait, auto_coins=not a.fixed_coins,
                 max_equity=max_equity, maker_entry=not a.taker, maker_tp=not a.taker, scan_delay=(75.0 if a.mode == "paper" else 15.0) if a.scan_delay is None else a.scan_delay)
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
    if a.mode == "live":  # 實盤啟動推一則：確認權益、資金上限和風險設定是不是照預期
        try:
            eq = f"{broker.equity(None):,.2f}"
        except Exception as e:  # 查不到不擋啟動
            eq = f"讀取失敗（{type(e).__name__}）"
        cap = f"{max_equity:g} U" if max_equity is not None else "未設"
        notify(f"幣安實盤啟動，權益 {eq} U（資金上限 {cap}）、每筆風險 {a.risk * 100:g}%")
    # systemd 停止服務送的是 SIGTERM：轉成正常結束，atexit 才會把佇列裡還沒送出的推播送完
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Trader(cfg, data, broker, notify, filters, stream=stream).run(once=a.once)
    notify.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
