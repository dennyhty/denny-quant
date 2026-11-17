"""自動交易主程式：每根 2H / 4H 收盤後找新訊號、開倉；平時管理已開的倉位。

進場（和網站的實盤維加斯一樣）：
- 只看剛收盤那一根的訊號；程式停機期間錯過的舊訊號不補做。
- 同一個幣、同一個週期一次只做一張；2H 和 4H 可以同時各有一張，同方向也可以（Denny 2026-10-20 決定，和網站一致）。
- 每筆風險 = 權益 × 0.25%（回撤減半時 × 0.5），最小下單量造成的實際風險超過 1% 就不做。

出場（和回測的 tracker 一樣）：止損 1.5 ATR（1R）；TP 1R / 3R / 5R 各平 1/4；TP1 後止損移到保本
（進場價 + 來回手續費）、TP2 後移到 TP1；TP3 後最後 1/4 用 3 ATR 追蹤止損（TP3 觸及當下就從 TP3 價起算，
之後每根 K 線收盤時更新，和 tracker 一樣只往有利方向移）；
240 根 K 線還沒結束就市價平掉。

- paper：PaperBroker 直接用 tracker 算結果，和回測完全一致，只記帳。
- testnet / live：LiveBroker 開倉後馬上在交易所掛止損（這張單的數量）和三個止盈（各 1/4），
  之後每分鐘（--poll）查各條件單的狀態判斷哪個 TP 成交了，再移動止損。程式停了，交易所上的止損還在。
  有帳戶推播（stream.py）時，成交事件會馬上叫醒這個輪詢，推播正常時輪詢間隔放寬到 --poll-stream。
  價位以實際成交價為準（止損距離照訊號的比例），所以 1R 的大小和訊號一致。
  平倉時盡量用交易所的成交明細算損益（帳本 pnl_source = fills），對不上就用觸發價估算（estimate）。
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..binance_history import binance_symbols
from ..signals import get
from ..signals.strategy import LONG, Signal
from ..signals.tracker import breakeven, track
from ..universe import TOP100
from .binance import BinanceError, BinanceFutures, OrderUnknown
from .risk import RiskRules, can_open, risk_multiplier
from .sizing import Filters, fmt, plan_size

BAR = {"2H": pd.Timedelta(hours=2), "4H": pd.Timedelta(hours=4)}
ORDER = ("2H", "4H")  # 同一個時間點兩個週期都有訊號時，2H 先做
MAX_BARS = 240
LABEL = {"paper": "幣安紙上交易", "testnet": "幣安測試網", "live": "幣安實盤"}
AUDIT_EVERY = 3600.0  # 多久比對一次交易所倉位和程式記錄（秒）；一次只用一個 positions 請求
OUTAGE_GRACE = 600.0  # 連續失敗超過幾秒才推警報（短暫斷線不吵人）
RETRY_PAUSE = 5.0     # 掃描時抓不到 K 線的幣，等幾秒再補抓一次
BACKUPS_KEEP = 7      # 狀態檔每天備份一份，留幾份
DEBOUNCE = 1.0        # 推播叫醒後等一下再跑：一批事件只跑一輪，幣安那邊也已經查得到
STREAM_GAP = 3.0      # 推播觸發的兩輪之間至少隔幾秒（事件很多時不要一直打 API）
HEALTH_CHECK = 5.0    # 等待時多久重看一次推播是否健康（斷了要馬上改回短的輪詢間隔）


@dataclass
class Config:
    mode: str = "paper"
    risk: float = 0.0025
    max_risk: float = 0.01
    leverage: int = 5
    fee: float = 0.0005
    paper_equity: float = 100.0
    rules: RiskRules = field(default_factory=RiskRules)
    state_dir: Path = Path("trader_state")
    coins: list[str] = field(default_factory=lambda: list(TOP100))
    bars: int = 1000
    poll: float = 60.0         # 持倉管理多久看一次（秒）；止損、止盈掛在交易所，這裡只負責追蹤和記帳
    poll_stream: float = 300.0  # 即時推送健康時多久輪詢一次（成交會叫醒主迴圈，這只是保險）
    scan_delay: float = 15.0   # 收盤後等幾秒再掃描（紙上和測試網錯開，免得同一秒打光 IP 的額度）
    paper_check: float = 300.0  # 紙上交易每張單多久重算一次（用的是 K 線，晚一點算結果一樣）


def bar_close(now: pd.Timestamp, interval: str) -> pd.Timestamp:
    """now 之前最近一次 interval 收盤的時間（從 00:00 UTC 起算，和幣安 K 線一致）。"""
    return now.floor(BAR[interval])


def signal_from(d: dict) -> Signal:
    return Signal(d["symbol"], d["interval"], d["side"], pd.Timestamp(d["time"], unit="ms", tz="UTC"),
                  d["price"], d["sl"], list(d["tps"]), d["strategy"], d.get("info") or {}, d.get("exit") or {})


class StateCorrupt(RuntimeError):
    """狀態檔壞了而且找不到能用的備份：不能用空狀態開始（已開的倉位會變成沒人管）。"""


def _utc_day() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def _read_state(path: Path) -> dict:
    """讀狀態檔；不是 JSON 或格式不對就丟 ValueError（UnicodeDecodeError、JSONDecodeError 都是它的子類）。"""
    data = json.loads(path.read_text())
    if not (isinstance(data, dict) and isinstance(data.get("positions", {}), dict)
            and isinstance(data.get("ledger", []), list)):
        raise ValueError("狀態檔格式不對")
    return data


def _write_atomic(path: Path, text: str) -> None:
    """先寫暫存檔、fsync 後才改名：當機或斷電時不會留下寫到一半的檔案。"""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class Store:
    """狀態檔：持倉、已平倉紀錄、每個週期處理到哪根 K 線。每次改完立刻寫檔（先寫暫存檔再改名）。

    壞掉的狀態檔會改名保留（.corrupt-時間），改用最新的備份（.bak-YYYYMMDD，每天第一次存檔時寫一份，留 7 份）；
    發生了什麼放在 notices，由 Trader 啟動時推播。都讀不了就丟 StateCorrupt（原檔不動）。"""

    def __init__(self, path: Path):
        self.path = path
        self.notices: list[str] = []
        self._last: str | None = None  # 上一次寫進去的內容：沒變就不重寫
        self._backup_day = ""
        try:
            data = _read_state(path)
        except FileNotFoundError:
            data = {}
        except (ValueError, OSError) as e:
            data = self._recover(e)
        self.positions: dict[str, dict] = data.get("positions", {})
        self.ledger: list[dict] = data.get("ledger", [])
        self.processed: dict[str, int] = data.get("processed", {})
        self.setup: list[str] = data.get("setup", [])
        self.leverage: dict[str, int] = data.get("leverage", {})

    def backups(self) -> list[Path]:
        """備份檔，新的在前（檔名帶日期，字串排序就是時間順序）。"""
        return sorted(self.path.parent.glob(self.path.name + ".bak-*"), reverse=True)

    def _recover(self, err: Exception) -> dict:
        for bak in self.backups():
            try:
                data = _read_state(bak)
            except (ValueError, OSError):
                continue
            kept = self.path.with_name(f"{self.path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
            self.path.replace(kept)  # 壞檔留著給人看，不覆蓋
            self.notices.append(
                f"狀態檔 {self.path.name} 壞了（{err}），已改名保留為 {kept.name}，改用備份 {bak.name} 開始。"
                f"備份是那一天第一次存檔時的內容，之後開的倉、平的倉可能不在裡面，請到幣安核對倉位")
            return data
        raise StateCorrupt(f"狀態檔 {self.path} 壞了（{err}），也沒有能用的備份"
                           f"（{self.path.name}.bak-*）。不能用空狀態開始，否則已開的倉位會沒人管；原檔沒動，請人工處理")

    def save(self) -> None:
        text = json.dumps({"positions": self.positions, "ledger": self.ledger,
                           "processed": self.processed, "setup": self.setup,
                           "leverage": self.leverage}, ensure_ascii=False, indent=1)
        if text == self._last:  # 每分鐘每張單都存一次，沒變就不必重寫
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.path, text)
        self._last = text
        self._daily_backup(text)

    def _daily_backup(self, text: str) -> None:
        day = _utc_day()
        if day == self._backup_day:
            return
        try:
            bak = self.path.with_name(f"{self.path.name}.bak-{day}")
            if not bak.exists():
                _write_atomic(bak, text)
            for old in self.backups()[BACKUPS_KEEP:]:
                old.unlink(missing_ok=True)
            self._backup_day = day
        except OSError as e:  # 備份失敗不能影響交易，明天再試
            print("狀態檔備份失敗:", e, flush=True)


class Notifier:
    """推播：呼叫時只放進佇列，由一條背景執行緒照順序送 Telegram（send 每則要睡 1.1 秒、Telegram 慢時一則可能卡 20 秒，
    不能拖住交易迴圈）。stdout 照舊馬上印；sink（測試用）照舊同步。結束前用 flush() 等佇列送完（有上限）。"""
    MAX_QUEUE = 200
    FLUSH_WAIT = 30.0

    def __init__(self, prefix: str, token: str = "", chat_id: str = "", sink=None):
        self.prefix, self.token, self.chat_id = prefix, token, chat_id
        self.sink = sink  # 測試用：收集訊息
        self._q: queue.Queue = queue.Queue(self.MAX_QUEUE)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._atexit = False

    def __call__(self, text: str) -> None:
        msg = f"［{self.prefix}］{text}"
        print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)
        if self.sink is not None:
            self.sink.append(msg)
        elif self.token and self.chat_id:
            self._enqueue(msg)

    def _enqueue(self, msg: str) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, name="notifier", daemon=True)
                self._thread.start()
            if not self._atexit:
                atexit.register(self.flush)
                self._atexit = True
        try:
            self._q.put_nowait(msg)
        except queue.Full:  # Telegram 長時間不通：丟掉新的，stdout 紀錄還在
            print("Telegram 佇列已滿，訊息只留在紀錄", flush=True)

    def _worker(self) -> None:
        while True:
            msg = self._q.get()
            try:
                from signal_telegram import send
                send([msg], self.token, self.chat_id)
            except Exception as e:  # 推播失敗不影響交易，也不能讓執行緒死掉
                print("Telegram 失敗:", e, flush=True)
            finally:
                self._q.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """等佇列送完，最多等 timeout 秒（預設 FLUSH_WAIT）。回傳是否送完。"""
        deadline = time.monotonic() + (self.FLUSH_WAIT if timeout is None else timeout)
        while self._q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
        return self._q.unfinished_tasks == 0


class Outage:
    """連續失敗超過 grace 秒才推一次警報，恢復時推一次「已恢復」（含中斷多久）；同一段中斷不重複推。
    Telegram 不在幣安上，所以幣安封鎖期間也推得出去。"""

    def __init__(self, label: str, notify, grace: float = OUTAGE_GRACE):
        self.label, self.notify, self.grace = label, notify, grace
        self.since: float | None = None
        self.reason = ""
        self.alerted = False

    def fail(self, reason: str, now: float) -> None:
        if self.since is None:
            self.since = now
        self.reason = reason
        if not self.alerted and now - self.since >= self.grace:
            self.alerted = True
            self.notify(f"警報：交易程式{self.label}已超過 {self.grace / 60:.0f} 分鐘"
                        f"（{time.strftime('%H:%M', time.gmtime(self.since))} UTC 起）：{reason}")

    def ok(self, now: float) -> None:
        if self.since is not None and self.alerted:
            mins = int((now - self.since) // 60)
            dur = f"{mins // 60} 小時 {mins % 60} 分鐘" if mins >= 60 else f"{mins} 分鐘"
            self.notify(f"已恢復：{self.label}，中斷了 {dur}")
        self.since, self.reason, self.alerted = None, "", False


def _pair(sym: str) -> str:
    return sym.replace("USDT", "/USDT")


def _side_text(side: str) -> str:
    return "做多" if side == LONG else "做空"


class Trader:
    def __init__(self, cfg: Config, data: BinanceFutures, broker, notify: Notifier, filters: dict[str, Filters],
                 stream=None):
        self.cfg = cfg
        self.stream = stream      # 帳戶推播（stream.UserStream）：只用來叫醒主迴圈；None = 只輪詢
        self.data = data          # 行情（正式站）
        self.broker = broker
        self.notify = notify
        self.filters = filters    # 下單的那個交易所（測試網 / 正式站）有的合約
        try:
            self.store = Store(cfg.state_dir / f"{cfg.mode}.json")
        except StateCorrupt as e:
            self._abort_corrupt(e)
        self._audit_at = 0.0  # 上次比對倉位的時間（秒）
        self._clock = time.time  # 測試可以換掉
        self._wait = lambda ev, t: ev.wait(t)  # 等推播事件（測試換成假的，不真的睡）
        self._sleep = time.sleep
        self._idle_ev = threading.Event()      # 沒有推播時拿來等的（永遠不會被設）
        self._poll = cfg.poll
        self._last_pass = 0.0
        self._stream_announced = False
        self._stream_out = Outage("即時推送", notify)
        self._manage_out = Outage("無法管理持倉", notify)
        self._loop_out = Outage("主迴圈無法正常執行", notify)
        self.last_loop: float | None = None
        self.last_manage_ok: float | None = None
        self.strategy = get("vegas_trend")
        self.symbols = {}
        for coin in cfg.coins:
            sym = binance_symbols(coin)[-1]
            if sym in filters:
                self.symbols[sym] = coin

    def _abort_corrupt(self, err: StateCorrupt) -> None:
        """狀態檔壞了又沒備份：推播後結束。systemd 每 30 秒重開一次，同一個壞檔只推一次（記在 .alerted）。"""
        path = self.cfg.state_dir / f"{self.cfg.mode}.json"
        mark = path.with_name(path.name + ".alerted")
        try:
            stamp = str(path.stat().st_mtime_ns)
            seen = mark.exists() and mark.read_text() == stamp
        except OSError:
            stamp, seen = "", False
        if seen:
            print(f"啟動中止：{err}", flush=True)
        else:
            self.notify(f"啟動中止：{err}")
            try:
                mark.write_text(stamp)
            except OSError:
                pass
        getattr(self.notify, "flush", lambda: None)()
        raise SystemExit(f"啟動中止：{err}") from err

    # ---------- 進場 ----------
    def paused(self) -> bool:
        return (self.cfg.state_dir / "pause").exists()

    def due(self, now: pd.Timestamp) -> list[str]:
        """這次要處理的週期：收盤時間比上次處理的新。第一次啟動只記下時間、不開單（避免開到舊訊號）。"""
        out = []
        for iv in ORDER:
            t = int(bar_close(now, iv).timestamp() * 1000)
            last = self.store.processed.get(iv)
            if last is None:
                self.store.processed[iv] = t
            elif t > last:
                out.append(iv)
        self.store.save()
        return out

    def find_signals(self, interval: str, close: pd.Timestamp) -> list[Signal]:
        def one(sym):
            try:
                df = self.data.klines(sym, interval, self.cfg.bars)
            except Exception as e:
                print(f"{sym} {interval} K 線失敗: {e}", flush=True)
                return None  # None = 抓不到（和「沒訊號」的 [] 分開）
            closed = df[df["time"] + BAR[interval] <= close].reset_index(drop=True)
            if len(closed) < 250 or closed["time"].iloc[-1] + BAR[interval] != close:
                return []
            return [s for s in self.strategy.scan(closed, sym, interval, start=len(closed) - 1) if s.time == close]
        def scan(syms):
            with ThreadPoolExecutor(4) as ex:
                return list(zip(syms, ex.map(one, syms)))
        res = scan(list(self.symbols))
        failed = [sym for sym, r in res if r is None]
        if failed:  # 抓不到的幣補抓一次，不然這根的訊號就永遠漏掉了
            time.sleep(RETRY_PAUSE)
            retry = dict(scan(failed))
            res = [(sym, retry.get(sym, r)) for sym, r in res]
            failed = [sym for sym, r in res if r is None]
        if failed:  # 還是抓不到：這根照樣記成已處理（不無限重掃），但要讓人知道漏了誰
            names = "、".join(_pair(x) for x in failed[:5]) + ("…" if len(failed) > 5 else "")
            self.notify(f"{interval} 掃描有 {len(failed)} 個幣抓不到 K 線，這根的訊號可能漏掉：{names}")
        found = [s for _, batch in res for s in batch or []]
        return sorted(found, key=lambda s: s.symbol)

    def blocked(self, sig: Signal) -> str:
        for p in self.store.positions.values():
            if p["symbol"] != sig.symbol:
                continue
            if p["interval"] == sig.interval:
                return f"{_pair(sig.symbol)} {sig.interval} 已有{_side_text(p['side'])}單"
        return ""

    def step_entries(self, now: pd.Timestamp) -> None:
        for iv in self.due(now):
            close = bar_close(now, iv)
            sigs = self.find_signals(iv, close)
            # 沒訊號也印一行，看紀錄就知道每根收盤都有掃（不推 Telegram）
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  f"{iv} 收盤 {close:%m-%d %H:%M} UTC 掃描完成：{len(sigs)} 個訊號，持倉 {len(self.store.positions)} 張",
                  flush=True)
            for sig in sigs:
                self.try_open(sig)
            self.store.processed[iv] = int(close.timestamp() * 1000)
            self.store.save()

    def try_open(self, sig: Signal) -> None:
        head = f"{_pair(sig.symbol)} {sig.interval} {_side_text(sig.side)}訊號 @ {sig.price:.6g}"
        reason = self.blocked(sig)
        if not reason and self.paused():
            reason = "已暫停開新倉（pause 檔存在）"
        if not reason and not can_open(len(self.store.positions), self.cfg.rules):
            reason = f"同時持倉已達上限 {self.cfg.rules.max_open} 張"
        if reason:
            self.notify(f"{head}，不做：{reason}")
            return
        mult = risk_multiplier(self.store.ledger, self.cfg.rules)
        try:
            equity = self.broker.equity(self.store)
        except Exception as e:  # 讀不到餘額也不能讓整輪掃描中斷（沒存「已處理」會每 30 秒重掃一次）
            self.notify(f"{head}，不做：讀不到帳戶餘額（{e}）")
            return
        size, why = plan_size(equity, self.cfg.risk * mult, sig.price, sig.sl, self.filters[sig.symbol],
                              tuple(sig.exit.get("portions") or (0.25, 0.25, 0.25)), self.cfg.max_risk)
        if size is None:
            self.notify(f"{head}，不做：{why}")
            return
        try:
            pos = self.broker.open(sig, size, self.filters[sig.symbol], self.store)
        except EntryUnknown as e:  # 不能說成「開倉失敗」：交易所上可能真的有倉位
            self.notify(f"{head}，下單結果不明（注意）：{e}")
            return
        except Exception as e:
            self.notify(f"{head}，開倉失敗：{e}")
            return
        if pos is None:
            return
        pos["base_usdt"] = equity * self.cfg.risk
        pos["risk_mult"] = mult
        self.store.positions[pos["id"]] = pos
        self.store.save()
        extra = "（風險減半）" if mult < 1 else ""
        extra += f"（最小下單量，實際風險 {size.risk_frac:.2%}）" if size.bumped else ""
        self.notify(f"{head}，已開倉 {pos['qty']} @ {pos['entry']:.6g}，止損 {pos['stop']:.6g}，"
                    f"TP {' / '.join(f'{t:.6g}' for t in pos['tps'])}{extra}")
        if pos.get("warn"):  # 例如槓桿調不下來
            self.notify(f"{_pair(pos['symbol'])} {pos['interval']}：{pos.pop('warn')}")
            self.store.save()

    # ---------- 持倉管理 ----------
    def _ban_text(self, t: float, e: Exception | None = None) -> str:
        """被幣安封鎖的說明（含解封時間）；沒被封回空字串。banned_until 在下單 / 查詢用的那個用戶端上。"""
        until = getattr(getattr(self.broker, "client", None), "banned_until", 0.0)
        if until > t:
            return f"被幣安封鎖（HTTP 418），預計 {time.strftime('%H:%M', time.gmtime(until))} UTC 解封"
        return f"被幣安封鎖（HTTP 418）：{e}" if e is not None else ""

    def _manage_ok(self, t: float) -> None:
        self.last_manage_ok = t
        self._manage_out.ok(t)

    def step_manage(self, now: pd.Timestamp) -> None:
        t = self._clock()
        if not self.store.positions:
            ban = self._ban_text(t)  # 沒持倉也要看：被封時開不了新倉，過了十分鐘要讓人知道
            if ban:
                self._manage_out.fail(ban, t)
            else:
                self._manage_ok(t)
            return
        self.broker.begin_step()
        failed = ""  # 這輪第一個管理失敗的原因
        for pid in list(self.store.positions):
            pos = self.store.positions[pid]
            # 同幣同方向的其他單：在交易所上是同一個倉位，這張單的單子多平時要從它們扣
            siblings = [p for q, p in self.store.positions.items()
                        if q != pid and p["symbol"] == pos["symbol"] and p["side"] == pos["side"]]
            try:
                result = self.broker.manage(pos, now, self.notify, siblings)
            except Exception as e:
                if isinstance(e, BinanceError) and e.status == 418:  # 被封：其餘持倉也一樣會失敗，這輪就到這裡
                    print(f"持倉管理暫停：{e}", flush=True)
                    self._manage_out.fail(self._ban_text(t, e), t)
                    return
                msg = f"{_pair(pos['symbol'])} {pos['interval']} 管理失敗：{e}"
                if isinstance(e, BinanceError):
                    msg += f"（程式記錄剩 {pos.get('remaining'):g}，止損 {pos.get('stop'):.6g}）"
                failed = failed or msg
                if pos.get("last_error") != msg:  # 同樣的錯誤只推一次
                    pos["last_error"] = msg
                    self.notify(msg)
                self.store.save()
                continue
            if pos.get("warn"):
                self.notify(f"{_pair(pos['symbol'])} {pos['interval']}：{pos.pop('warn')}")
            pos.pop("last_error", None)  # 這輪沒出錯，之後同樣的錯誤再出現要再通知一次
            if result is not None:  # 已平倉：result 是實現損益（USDT）
                r = result / pos["base_usdt"] if pos.get("base_usdt") else 0.0
                row = {"id": pid, "symbol": pos["symbol"], "interval": pos["interval"],
                       "side": pos["side"], "opened": pos["opened"],
                       "closed": int(now.timestamp() * 1000), "pnl": result, "r": r,
                       "pnl_source": pos.get("pnl_source", "estimate")}
                if "pnl_est" in pos:  # 用成交明細算的，也留著估算值方便對照
                    row["pnl_est"] = pos["pnl_est"]
                self.store.ledger.append(row)
                del self.store.positions[pid]
                cum = sum(x["r"] for x in self.store.ledger)
                self.notify(f"{_pair(pos['symbol'])} {pos['interval']} {_side_text(pos['side'])}已平倉："
                            f"{result:+.4f} USDT（{r:+.2f}R），累計 {cum:+.2f}R")
            self.store.save()
        if failed:
            self._manage_out.fail(failed, t)
        else:
            self._manage_ok(t)

    def step_audit(self, now: pd.Timestamp) -> None:
        """大約每小時比對一次交易所倉位：比程式記錄多（例如下單結果不明、其實成交了）就提醒，同一個差異只提醒一次。"""
        if now.timestamp() - self._audit_at < AUDIT_EVERY:
            return
        self._audit_at = now.timestamp()
        try:
            self.broker.audit(self.store, self.notify)
        except Exception as e:  # 查不到就下一小時再查，不影響交易
            print("倉位比對失敗:", e, flush=True)

    def heartbeat_path(self) -> Path:
        return self.cfg.state_dir / f"{self.cfg.mode}.heartbeat.json"

    def write_heartbeat(self) -> None:
        """每輪寫一次：最後一輪的時間、最後一次完整管理成功的時間、目前的問題。給 trader_feed 和網站看程式還活著沒。"""
        problems = [o.reason for o in (self._manage_out, self._loop_out) if o.since is not None]
        ms = lambda x: None if x is None else int(x * 1000)
        body = json.dumps({"last_loop": ms(self.last_loop), "last_manage_ok": ms(self.last_manage_ok),
                           "problem": "；".join(problems) or None, "stream": self.stream_state(),
                           "stream_last_event": ms(getattr(self.stream, "last_event", None))},
                          ensure_ascii=False)
        try:
            self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(self.heartbeat_path(), body)
        except OSError as e:  # 寫不出心跳不能影響交易
            print("心跳檔寫入失敗:", e, flush=True)

    def run_once(self, now: pd.Timestamp) -> None:
        """一輪：掃描、管理、比對各自獨立，前面出錯不會害後面的持倉管理被跳過。"""
        steps = []
        # 收盤後等一下再抓 K 線，幣安的最後一根才會是完整的
        if (now - bar_close(now, "2H")) >= pd.Timedelta(seconds=self.cfg.scan_delay):
            steps.append(("掃描", self.step_entries))
        steps += [("持倉管理", self.step_manage), ("倉位比對", self.step_audit)]
        errors = []
        for name, fn in steps:
            try:
                fn(now)
            except Exception as e:  # 網路斷線之類，下一輪再試
                print(f"本輪失敗（{name}）:", e, flush=True)
                errors.append(f"{name}失敗：{e}")
        t = self._clock()
        if errors:
            self._loop_out.fail("；".join(errors), t)
        else:
            self._loop_out.ok(t)
        self.last_loop = t
        self._watch_stream(t)
        self.write_heartbeat()

    # ---------- 即時推送 ----------
    def stream_state(self) -> str:
        """healthy（連著且證明過）/ connected（連著但還沒收到帳戶事件）/ down / off（沒開或沒裝套件）。"""
        return "off" if self.stream is None else self.stream.state

    def _watch_stream(self, t: float) -> None:
        """第一次確認可用推一次；確認過之後不健康超過 10 分鐘推一次警報、恢復推一次（重連迴圈不洗版）。"""
        s = self.stream
        if s is None or not s.enabled:
            return
        if s.healthy:
            if not self._stream_announced:
                self._stream_announced = True
                self.notify(f"即時推送已啟用：成交、止損止盈觸發會馬上處理，沒有事件時每 {self.cfg.poll_stream:.0f} 秒輪詢一次")
            self._stream_out.ok(t)
        elif self._stream_announced:
            self._stream_out.fail(f"中斷（{s.last_error or s.state}），持倉管理改回每 {self._poll:.0f} 秒輪詢，"
                                  f"止損止盈照常掛在交易所", t)

    def next_scan(self, t: float) -> float:
        """下一次收盤掃描的時間（Unix 秒）：2H 收盤 + scan_delay（4H 收盤一定也是 2H 收盤）。"""
        period = BAR["2H"].total_seconds()
        nxt = t - t % period + self.cfg.scan_delay
        while nxt <= t:
            nxt += period
        return nxt

    def idle(self, poll: float) -> str:
        """等到下一輪：推播叫醒（"event"）、輪詢間隔到或收盤掃描時間到（"poll"），哪個先到。
        推播健康時輪詢間隔用 poll_stream，否則照舊用 poll。"""
        started = self._clock()
        wake = self.stream.wake if self.stream is not None else self._idle_ev
        while True:
            now = self._clock()
            healthy = self.stream is not None and self.stream.healthy
            deadline = min(started + (self.cfg.poll_stream if healthy else poll), self.next_scan(now))
            if now >= deadline:
                return "poll"
            if self._wait(wake, min(deadline - now, HEALTH_CHECK)):
                break
        # 一批事件（例如止盈成交帶出好幾個事件）只跑一輪；離上一輪太近就等到滿 STREAM_GAP 秒
        self._sleep(max(DEBOUNCE, self._last_pass + STREAM_GAP - self._clock()))
        wake.clear()  # 清在跑之前：跑的時候又來的事件會再叫醒一次
        return "event"

    def run(self, poll: float | None = None, once: bool = False) -> None:
        poll = self.cfg.poll if poll is None else poll
        self.notify(f"啟動：{len(self.symbols)} 個幣，持倉 {len(self.store.positions)} 張")
        for text in self.store.notices:  # 狀態檔壞了、改用備份
            self.notify(text)
        try:
            self.broker.reconcile(self.store, self.notify)
        except Exception as e:  # 啟動檢查出錯不能讓程式當掉重開（每次重開都會推播一次「啟動」）
            print("啟動檢查失敗:", e, flush=True)
        self._audit_at = pd.Timestamp.now(tz="UTC").timestamp()  # 啟動檢查剛比對過
        self.store.save()
        self._poll = poll
        streaming = self.stream is not None and not once  # 單次模式不必開推播
        if streaming:
            self.stream.start()
        try:
            while True:
                self._last_pass = self._clock()
                self.run_once(pd.Timestamp.now(tz="UTC"))
                if once:
                    getattr(self.notify, "flush", lambda: None)()  # 單次模式：等推播送完再結束
                    return
                self.idle(poll)
        finally:
            if streaming:
                self.stream.stop()


# ---------- 紙上交易 ----------
class PaperBroker:
    """不下單：開倉價 = 訊號收盤價，出場用 tracker 逐根 K 線算（和回測一樣）。"""

    def __init__(self, data: BinanceFutures, cfg: Config):
        self.data, self.cfg = data, cfg

    def equity(self, store: Store) -> float:
        return self.cfg.paper_equity + sum(x["pnl"] for x in store.ledger)

    def reconcile(self, store: Store, notify) -> None:
        pass

    def audit(self, store: Store, notify) -> None:
        pass

    def begin_step(self) -> None:
        pass

    def open(self, sig: Signal, size, f: Filters, store: Store) -> dict:
        qty = size.steps * f.step
        return {"id": f"{sig.symbol}|{sig.interval}|{sig.side}|{int(sig.time.timestamp())}",
                "symbol": sig.symbol, "interval": sig.interval, "side": sig.side,
                "signal": sig.to_dict(), "qty": fmt(qty, f.step), "qty_f": qty,
                "entry": sig.price, "stop": sig.sl, "tps": list(sig.tps),
                "risk_usdt": qty * abs(sig.price - sig.sl), "events": 0,
                "opened": int(sig.time.timestamp() * 1000)}

    def manage(self, pos: dict, now: pd.Timestamp, notify, siblings=()) -> float | None:
        if now.timestamp() - pos.get("checked", 0) < self.cfg.paper_check:
            return None  # 用 K 線重算，隔幾分鐘算結果一樣，不必每輪都抓
        pos["checked"] = now.timestamp()
        sig = signal_from(pos["signal"])
        df = self.data.klines(sig.symbol, sig.interval, MAX_BARS + 20)
        res = track(sig, df[df["time"] <= now], self.cfg.fee, MAX_BARS)
        for e in res.events[pos["events"]:]:
            if e.kind in ("TP1", "TP2", "TP3", "STOP_MOVED", "TRAIL_MOVED"):
                notify(f"{_pair(sig.symbol)} {sig.interval} {_side_text(sig.side)}：{e.text}")
        pos["events"] = len(res.events)
        pos["stop"] = res.stop
        if res.status == "CLOSED":
            return res.realized_r * pos["risk_usdt"]
        return None


# ---------- 測試網 / 實盤 ----------
def client_tag(sig: Signal) -> str:
    """掛單用的 client id 前綴。幣安要求同一帳戶內不能重複，所以要帶幣名（同一根 K 線常有好幾個幣同時出訊號）；
    只能用 [.A-Za-z0-9:/_-]，最長 36 字元，後面還要接 e / s12 / t1 等尾巴。
    幣名不是純英數字（例如 币安人生USDT）或太長就改用幣名的雜湊。"""
    name = sig.symbol
    if not re.fullmatch(r"[A-Za-z0-9]+", name) or len(name) > 14:
        name = hashlib.md5(sig.symbol.encode()).hexdigest()[:10]
    return f"v{sig.interval}{int(sig.time.timestamp())}{sig.side[0]}{name}"


class EntryUnknown(RuntimeError):
    """進場市價單送出後不知道有沒有成交：交易所上可能有沒有止損的倉位，要明確推播，不能當成開倉失敗。"""


def _max_leverage(price: float, sl: float, cap: int) -> int:
    """止損越遠槓桿越低，讓強平價離進場至少是止損距離的 2 倍（槓桿 ≤ 1 / (2 × 止損%)），最多 cap。"""
    stop_pct = abs(price - sl) / price
    return max(1, min(cap, int(1 / (2 * stop_pct)) if stop_pct > 0 else cap))


class LiveBroker:
    """同一個幣同方向可能同時有 2H 和 4H 兩張單，在幣安上會合成同一個倉位（雙向持倉只有一個 LONG / SHORT），
    所以每張單的止損、止盈都用自己的數量掛，成交與否看各自條件單的狀態，不看合併後的持倉數量。
    損益優先用這張單自己的訂單（orderId）的成交明細算，對不上時用觸發價估算（成交價 × 數量，扣來回手續費）。"""

    FILLED = ("TRIGGERED", "FINISHED")

    def __init__(self, client: BinanceFutures, data: BinanceFutures, cfg: Config):
        self.client, self.data, self.cfg = client, data, cfg
        self._open: dict[str, set] | None = None
        self._extra: dict[tuple, float] = {}  # 已經提醒過的「交易所比記錄多」：同一個差異只推一次

    def equity(self, store: Store) -> float:
        return self.client.balance_usdt()

    def begin_step(self) -> None:
        self._open = None  # 每輪重抓一次掛單（所有幣一次抓完，不是每個幣打一個請求）

    def open_ids(self, symbol: str) -> set:
        if self._open is None:
            self._open = {}
            for o in self.client.open_conditionals():
                self._open.setdefault(o.get("symbol"), set()).add(o.get("algoId"))
        return self._open.setdefault(symbol, set())

    def status(self, pos: dict, oid) -> tuple[str, float]:
        """條件單的狀態：("open" / "filled" / "pending" / "gone", 成交價)。
        pending 是剛觸發、幣安還在送出市價單（TRIGGERING）：這時掛單清單裡已經沒有它，但還不算成交也不是不見，下一輪再看。"""
        if oid is None:
            return "gone", 0.0
        if oid in self.open_ids(pos["symbol"]):
            return "open", 0.0
        o = self.client.algo_order(oid)
        st = str(o.get("algoStatus") or "").upper()
        price = float(o.get("actualPrice") or 0) or float(o.get("triggerPrice") or 0)
        pos["last_algo_status"] = st  # 補掛時寫進推播，之後才查得出止損單是怎麼不見的
        if st in self.FILLED:
            # 觸發後實際送出的市價單 id：平倉時用它查成交明細算損益（不用另外打請求）
            aid = o.get("actualOrderId")
            if aid:
                ids = pos.setdefault("close_orders", [])
                if aid not in ids:
                    ids.append(aid)
            return "filled", price
        return ("pending" if st == "TRIGGERING" else "gone"), price

    def ensure_setup(self, symbol: str, store: Store, leverage: int) -> str:
        """雙向持倉、逐倉；設好槓桿。回傳要推播的警告（沒有就是空字串）。

        槓桿是整個幣共用的（LONG、SHORT 和 2H、4H 都一樣），所以目標 = 這張單止損距離允許的倍數，
        和這個幣其他還開著的單允許的倍數，取最小：絕不調到比任何一張開著的單允許的還高（會讓它強平價比止損近）；
        能調低就調低。和目前設定不同才送請求（沒有其他倉位時也會調回較高、但仍在允許範圍內的倍數）。
        逐倉有倉位時幣安不准調低槓桿（-4161）：沿用目前的槓桿照樣開倉，回傳警告，不讓整筆進場失敗。"""
        if "*hedge" not in store.setup:
            if not self.client.hedge_mode():
                self.client.set_hedge_mode()
            store.setup.append("*hedge")
        if symbol not in store.setup:
            self.client.set_isolated(symbol)
            store.setup.append(symbol)
        for p in store.positions.values():
            if p["symbol"] == symbol:
                s = p["signal"]
                leverage = min(leverage, _max_leverage(s["price"], s["sl"], self.cfg.leverage))
        cur, warn = store.leverage.get(symbol), ""
        if cur != leverage:
            try:
                self.client.set_leverage(symbol, leverage)
                store.leverage[symbol] = leverage
            except BinanceError as e:
                if e.code != -4161 or cur is None:
                    raise
                if cur > leverage:
                    warn = (f"槓桿應降到 {leverage} 倍，但這個幣已有逐倉倉位，幣安不准調低（-4161），"
                            f"沿用 {cur} 倍開倉：強平價會比止損距離要求的近，請留意")
        store.save()
        return warn

    def leverage_for(self, sig: Signal) -> int:
        """止損越遠槓桿越低，讓強平價離進場至少是止損距離的 2 倍（槓桿 ≤ 1 / (2 × 止損%)），最多 cfg.leverage。"""
        return _max_leverage(sig.price, sig.sl, self.cfg.leverage)

    def _check_extra(self, store: Store, live: dict, notify) -> None:
        """交易所有、狀態檔沒有的數量只提醒，不碰。同一個 (幣, 方向, 數量) 只提醒一次，數量變了或消失後再出現才再推。"""
        expected: dict[tuple, float] = {}
        tol: dict[tuple, float] = {}
        for p in store.positions.values():
            key = (p["symbol"], p["side"])
            expected[key] = expected.get(key, 0.0) + p["remaining"]
            tol[key] = min(tol.get(key, 1.0), p["step"] / 2)  # 剩餘數量是浮點數相減，差半個單位以內不算
        seen = {}
        for key, amt in live.items():
            if amt > expected.get(key, 0.0) + tol.get(key, 1e-9):
                seen[key] = amt
                if self._extra.get(key) != amt:
                    notify(f"注意：{_pair(key[0])} {key[1]} 有 {amt}，比這個程式記錄的多，多出來的不會管理")
        self._extra = seen

    def audit(self, store: Store, notify) -> None:
        """定期比對（Trader 大約每小時叫一次）：一個 positions 請求。"""
        self._check_extra(store, self.client.positions(), notify)

    def reconcile(self, store: Store, notify) -> None:
        """啟動時比對：交易所有、狀態檔沒有的倉位只提醒，不碰；舊版狀態檔的單補上新欄位。"""
        self.begin_step()
        migrated = [p for p in store.positions.values() if "remaining" not in p]
        for p in migrated:
            # 2026-10-20 以前開的單：沒有 remaining / fills，止損是「全部平倉」。補上欄位，止損改成用自己的數量
            p["remaining"] = max(0.0, p["qty_f"] - sum(p["slices"][:p["tps_hit"]]))
            p["fills"] = [[p["slices"][n], p["tps"][n]] for n in range(p["tps_hit"])]
        self._check_extra(store, self.client.positions(), notify)
        for pos in migrated:
            try:
                if self.status(pos, pos.get("stop_id"))[0] == "open" and pos["remaining"] > pos["step"] / 2:
                    self.move_stop(pos, pos["stop"])
            except Exception as e:  # 啟動時不能因為一張單出錯就整個程式當掉，之後每輪管理會再處理
                print(f"{pos['symbol']} 啟動時移止損失敗: {e}", flush=True)
        for pos in store.positions.values():
            try:
                if self.status(pos, pos.get("stop_id"))[0] == "gone":
                    pos["stop_version"] = pos.get("stop_version", 0) + 1
                    pos["stop_id"] = self._place_stop(pos, pos["stop"], pos["stop_version"])
                    notify(f"{_pair(pos['symbol'])} 止損單不見了，已補掛在 {pos['stop']:.6g}")
            except Exception as e:  # 補不上的（例如倉位已經不在）交給每輪的 manage() 處理，啟動不能因此當掉
                print(f"{pos['symbol']} 啟動時補掛止損失敗，交給管理迴圈: {e}", flush=True)

    def open(self, sig: Signal, size, f: Filters, store: Store) -> dict | None:
        warn = self.ensure_setup(sig.symbol, store, self.leverage_for(sig))
        d = sig.direction
        ps = sig.side  # 雙向持倉：LONG / SHORT
        buy, sell = ("BUY", "SELL") if d > 0 else ("SELL", "BUY")
        tag = client_tag(sig)
        cid = f"{tag}e"
        unknown = (f"交易所上可能有 {sig.symbol} {ps} 的倉位卻沒有止損，請馬上到幣安檢查"
                   f"（client id {cid}）；每小時的倉位比對發現多出來的數量也會再提醒")
        try:
            order = self.client.market_order(sig.symbol, buy, ps, fmt(size.steps * f.step, f.step), cid)
        except OrderUnknown as e:
            raise EntryUnknown(f"{e}。{unknown}") from e
        qty = float(order.get("executedQty") or 0)
        entry = float(order.get("avgPrice") or 0) or self.client.price(sig.symbol)
        if qty <= 0:
            if str(order.get("status") or "").upper() in ("NEW", "PARTIALLY_FILLED"):  # 查單時還沒成交完
                raise EntryUnknown(f"市價單狀態 {order.get('status')}。{unknown}")
            raise RuntimeError(f"市價單沒有成交：{order}")
        # 價位以實際成交價重算，止損距離照訊號的比例
        risk = abs(sig.price - sig.sl) / sig.price * entry
        r_mult = [abs(t - sig.price) / abs(sig.price - sig.sl) for t in sig.tps]
        stop = entry - d * risk
        tps = [entry + d * k * risk for k in r_mult]
        pos = {"id": f"{sig.symbol}|{sig.interval}|{sig.side}|{int(sig.time.timestamp())}",
               "symbol": sig.symbol, "interval": sig.interval, "side": sig.side, "signal": sig.to_dict(),
               "qty": fmt(qty, f.step), "qty_f": qty, "remaining": qty, "fills": [],
               "step": f.step, "tick": f.tick,
               "slices": [s * f.step for s in size.slices], "entry": entry, "stop": stop, "tps": tps,
               "be": breakeven(Signal(sig.symbol, sig.interval, sig.side, sig.time, entry, stop, tps, ""),
                               self.cfg.fee),
               "trail": float(sig.exit.get("trail") or 0) / sig.price * entry,
               "risk_usdt": qty * risk, "tps_hit": 0, "trail_from": None,
               "opened": int(time.time() * 1000), "close_side": sell, "tag": tag,
               "entry_order": order.get("orderId"), "close_orders": []}
        if warn:
            pos["warn"] = warn
        try:
            pos["stop_id"] = self._place_stop(pos, stop, 0)
        except Exception as e:
            # 掛不上止損就馬上平掉這張單，絕不留沒有止損的倉位
            try:
                self.client.market_order(sig.symbol, sell, ps, pos["qty"], f"{tag}x")
            except Exception as e2:  # 平倉也沒成功（或結果不明）：倉位可能還在而且沒有止損，要講清楚
                raise EntryUnknown(f"止損掛單失敗（{e}），市價平倉（{tag}x）也失敗（{e2}）。{unknown}") from e2
            raise RuntimeError(f"止損掛單失敗，已市價平倉：{e}")
        pos["tp_ids"] = []
        for n, (tp, q) in enumerate(zip(tps, pos["slices"][:3])):
            try:
                r = self.client.conditional_order(sig.symbol, sell, ps, "TAKE_PROFIT_MARKET", fmt(tp, f.tick),
                                                  quantity=fmt(q, f.step), client_id=f"{tag}t{n + 1}")
                pos["tp_ids"].append(r.get("algoId"))
            except Exception as e:
                pos["tp_ids"].append(None)
                print(f"{sig.symbol} TP{n + 1} 掛單失敗: {e}", flush=True)
        return pos

    def _place_stop(self, pos: dict, price: float, version: int):
        """止損用這張單剩下的數量掛（雙向持倉下 SELL+LONG / BUY+SHORT 只會減倉，不會反向開倉）。
        掛上去的數量記在 stop_qty：止損成交時用來判斷有沒有多平到同幣同向另一張單的數量。"""
        qty = pos["remaining"]
        try:
            r = self.client.conditional_order(pos["symbol"], pos["close_side"], pos["side"], "STOP_MARKET",
                                              fmt(price, pos["tick"]), quantity=fmt(qty, pos["step"]),
                                              client_id=f"{pos['tag']}s{version}")
        except BinanceError as e:
            if e.code != -2022:
                raise
            # 止損數量比交易所上的倉位大（記錄的剩餘數量和實際倉位對不上）：先用實際倉位掛，保住保護，並推播數字
            live = self.client.positions().get((pos["symbol"], pos["side"]), 0.0)
            if live <= pos["step"] / 2 or live >= qty:
                raise
            pos["warn"] = f"交易所上只有 {live:g}，程式記錄剩 {qty:g}，止損先用 {live:g} 掛"
            r = self.client.conditional_order(pos["symbol"], pos["close_side"], pos["side"], "STOP_MARKET",
                                              fmt(price, pos["tick"]), quantity=fmt(live, pos["step"]),
                                              client_id=f"{pos['tag']}s{version}c")
            qty = live
        pos["stop_qty"] = float(fmt(qty, pos["step"]))
        if self._open is not None:
            self._open.setdefault(pos["symbol"], set()).add(r.get("algoId"))  # 這一輪的掛單快取也要有新止損
        return r.get("algoId")

    def _close_rest(self, pos: dict, reason: str) -> None:
        """市價平掉這張單剩下的數量，並記下成交。"""
        if pos["remaining"] <= pos["step"] / 2:
            return
        cid = f"{pos['tag']}{reason}"
        # 上一次送同一個 id 沒有確定結果（例如中途出錯）：先查單，免得已經平過又再平一次（會平到同幣同向另一張單）
        kw = {"check_first": True} if pos.get("pending_close") == cid else {}
        pos["pending_close"] = cid
        try:
            r = self.client.market_order(pos["symbol"], pos["close_side"], pos["side"],
                                         fmt(pos["remaining"], pos["step"]), cid, **kw)
        except BinanceError as e:
            if e.code != -2022 or not self._flat(pos):  # -2022：倉位已經不在，沒東西可平
                raise
            r = {}  # 被別的單平掉了（止損或止盈已成交）：照目前價格記帳
        pos.pop("pending_close", None)
        price = float(r.get("avgPrice") or 0) or self.client.price(pos["symbol"])
        pos["fills"].append([pos["remaining"], price])
        pos["remaining"] = 0.0
        if r.get("orderId"):
            pos.setdefault("close_orders", []).append(r["orderId"])

    def _flat(self, pos: dict) -> bool:
        """交易所上這個幣這一邊已經沒有倉位。"""
        return self.client.positions().get((pos["symbol"], pos["side"]), 0.0) <= pos["step"] / 2

    def move_stop(self, pos: dict, price: float) -> None:
        """先掛新止損再取消舊的；掛不上時先取消再掛；還是不行（價格已越過）就市價平掉。"""
        version = pos.get("stop_version", 0) + 1
        old = pos.get("stop_id")
        try:
            new = self._place_stop(pos, price, version)
            if old:
                self.client.cancel_conditional(old)
        except BinanceError:
            if old:
                self.client.cancel_conditional(old)
            try:
                new = self._place_stop(pos, price, version)
            except BinanceError as e:
                pos["stop_id"], pos["stop"], pos["stop_version"] = None, price, version
                self._close_rest(pos, f"m{version}")
                raise RuntimeError(f"止損移不上去，已市價平倉：{e}")
        pos["stop_id"], pos["stop"], pos["stop_version"] = new, price, version

    def pnl(self, pos: dict) -> float:
        """估算：觸發價（或幣安回的成交價）× 數量，手續費照 cfg.fee 算。"""
        d = 1 if pos["side"] == LONG else -1
        gross = sum(d * (p - pos["entry"]) * q for q, p in pos["fills"])
        fees = self.cfg.fee * (pos["entry"] * pos["qty_f"] + sum(p * q for q, p in pos["fills"]))
        return gross - fees

    def fills_pnl(self, pos: dict) -> float | None:
        """用這張單自己的訂單（進場單 + 各個止損 / 止盈 / 市價平倉觸發的訂單）的成交明細算實現損益，扣實際手續費。
        不用 realizedPnl 欄位：同幣同向兩張單合成一個倉位時，幣安是用合併後的平均開倉價算的，不是這張單的。
        任何一點對不上（缺訂單 id、進出數量和這張單的數量不符、手續費不是 USDT、請求失敗）就回 None，改用估算。"""
        entry_id, closes = pos.get("entry_order"), list(dict.fromkeys(pos.get("close_orders") or []))
        if entry_id is None or not closes:
            return None
        try:
            ent = self.client.order_trades(pos["symbol"], entry_id)
            out = [t for oid in closes for t in self.client.order_trades(pos["symbol"], oid)]
        except Exception as e:
            print(f"{pos['symbol']} 查成交明細失敗，損益改用估算: {e}", flush=True)
            return None
        rows = ent + out
        if not ent or any(str(t.get("commissionAsset") or "USDT") != "USDT" for t in rows):
            return None
        if any(t.get("positionSide", pos["side"]) != pos["side"] for t in rows):
            return None
        eq = sum(float(t["qty"]) for t in ent)
        cq = sum(float(t["qty"]) for t in out)
        tol = pos["step"] / 2
        # 出場數量多於這張單（例如止損多平到另一張單）或少於（有一部分是別的方式平掉的）：歸屬不可靠
        if abs(eq - pos["qty_f"]) > tol or abs(cq - pos["qty_f"]) > tol:
            return None
        d = 1 if pos["side"] == LONG else -1
        entry_px = sum(float(t["price"]) * float(t["qty"]) for t in ent) / eq
        gross = sum(d * (float(t["price"]) - entry_px) * float(t["qty"]) for t in out)
        return gross - sum(float(t.get("commission") or 0) for t in rows)

    def _cancel_orders(self, pos: dict) -> None:
        """取消這張單還掛著的止損、止盈，並從這一輪的掛單快取拿掉（之後查狀態會直接問交易所，不會誤判成還掛著）。"""
        ids = self.open_ids(pos["symbol"])
        for oid in [pos.get("stop_id")] + list(pos.get("tp_ids") or []):
            if oid and oid in ids:
                self.client.cancel_conditional(oid)
                ids.discard(oid)

    def finish(self, pos: dict) -> float:
        self._cancel_orders(pos)
        est = self.pnl(pos)
        real = self.fills_pnl(pos)
        if real is None:
            pos["pnl_source"] = "estimate"
            return est
        pos["pnl_source"], pos["pnl_est"] = "fills", est
        return real

    def _absorb_excess(self, pos: dict, extra: float, price: float, siblings) -> None:
        """這張單的止損 / 止盈一共平掉的比它自己剩下的多 extra（止損是用 TP 成交前的數量掛的，或止損成交後
        剩下的止盈也成交了）。只有一個倉位時幣安會把超出的部分擋掉或只平到倉位為止；但同幣同向還有別張單時，
        合併倉位夠大，多出來的就會平到別張單。幣安實際怎麼做沒有驗證過，所以兩種都處理：
        用 positions 看合併倉位，比其他單記錄的少才從它們扣（照這張單的止損成交價記帳），否則不動。"""
        others = [s for s in siblings if s["remaining"] > s["step"] / 2]
        if not others:
            return
        # 其他單本身有條件單不在掛單清單（可能剛成交、還沒處理）：倉位變少不一定是被多平，不自己猜，只提醒
        unsure = any(oid is not None and oid not in self.open_ids(s["symbol"])
                     for s in others for oid in [s.get("stop_id")] + list(s.get("tp_ids") or [])[s["tps_hit"]:])
        try:
            live = None if unsure else self.client.positions().get((pos["symbol"], pos["side"]), 0.0)
        except Exception as e:
            print(f"{pos['symbol']} 查倉位失敗: {e}", flush=True)
            live = None
        if live is None:
            for s in others:
                s["warn"] = (f"同方向的 {pos['interval']} 單出場時可能多平了最多 {extra:g}，無法確認，"
                             f"請到幣安核對倉位（程式記錄剩 {s['remaining']:g}）")
            return
        short = min(extra, sum(s["remaining"] for s in others) - live)
        if short <= pos["step"] / 2:  # 交易所沒有多平（超出的部分被擋掉）
            return
        for s in others:
            take = min(short, s["remaining"])
            if take <= s["step"] / 2:
                continue
            s["fills"].append([take, price])
            s["remaining"] = max(0.0, s["remaining"] - take)
            short -= take
            s["warn"] = (f"同方向的 {pos['interval']} 單出場時多平了 {take:g}（交易所上只剩 {live:g}），"
                         f"已從這張單扣掉、照 {price:.6g} 記帳，止損數量跟著改小")

    def manage(self, pos: dict, now: pd.Timestamp, notify, siblings=()) -> float | None:
        name = f"{_pair(pos['symbol'])} {pos['interval']} {_side_text(pos['side'])}"
        if pos["remaining"] <= pos["step"] / 2:  # 上一輪已經平掉（例如移止損失敗時市價平倉）
            return self.finish(pos)
        st, stop_price = self.status(pos, pos.get("stop_id"))
        if st == "pending":  # 止損剛觸發，等下一輪看成交結果
            return None
        stop_filled = st == "filled"
        stop_price = stop_price or pos["stop"]
        if stop_filled:
            # 止損成交了：先取消剩下的止盈（越早越好，同幣同向另一張單的倉位還在，留著的止盈會平到它），
            # 再處理上一輪之後成交的止盈，最後才記止損：止損是用當時的數量掛的，實際只平到剩下的部分
            self._cancel_orders(pos)
        # 止損單不見了（不是成交）：先處理止盈成交再補掛。止盈成交後倉位變小，
        # 用舊的剩餘數量補掛會被幣安以 -2022 拒絕；止盈的處理會順便用新的剩餘數量重掛止損。
        stop_missing, stop_version = st == "gone", pos.get("stop_version", 0)
        placed = pos.get("stop_qty", pos["remaining"])  # 舊狀態檔沒有 stop_qty：止損是用目前剩餘數量掛的
        d = 1 if pos["side"] == LONG else -1
        while pos["tps_hit"] < 3:
            n = pos["tps_hit"]
            st, price = self.status(pos, pos["tp_ids"][n])
            if st == "pending" and stop_filled:  # 止盈也剛觸發：等它有結果再一起結算
                return None
            if st != "filled":
                break
            q = pos["slices"][n]
            pos["fills"].append([q, price or pos["tps"][n]])
            pos["remaining"] = max(0.0, pos["remaining"] - q)
            pos["tps_hit"] += 1
            notify(f"{name}：TP{n + 1} 觸及 {pos['tps'][n]:.6g}")
            if stop_filled:  # 止損已經成交，不必再移
                continue
            if n == 0:
                self.move_stop(pos, pos["be"])
                notify(f"{name}：保護止損設於 {pos['be']:.6g}")
            elif n == 1:
                self.move_stop(pos, pos["tps"][0])
                notify(f"{name}：保護止損設於 {pos['tps'][0]:.6g}")
            else:
                pos["trail_from"] = int(now.timestamp() * 1000)
                # 和 tracker 一樣 TP3 那根就開始追蹤：最有利價至少到過 TP3，止損 = TP3 往回 trail，只往有利方向移；
                # 這根收盤後再用整根 K 線的最高 / 最低價更新。同時把數量改成最後剩下的那一份
                ts = pos["tps"][2] - d * pos["trail"]
                if pos["trail"] > 0 and d * (ts - pos["stop"]) > pos["tick"]:
                    self.move_stop(pos, ts)
                    notify(f"{name}：追蹤止損移到 {ts:.6g}")
                else:
                    self.move_stop(pos, pos["stop"])
        if stop_filled:
            extra = placed - pos["remaining"]
            if pos["remaining"] > 0:
                pos["fills"].append([pos["remaining"], stop_price])
            pos["remaining"] = 0.0
            if extra > pos["step"] / 2 and siblings:
                self._absorb_excess(pos, extra, stop_price, siblings)
            return self.finish(pos)
        if pos["remaining"] <= pos["step"] / 2:
            return self.finish(pos)
        # 止盈成交後移止損那一步若中途失敗（例如斷線），tps_hit 已經加上去、止損卻還在舊價位：每輪補移到該在的位置
        want = pos["be"] if pos["tps_hit"] == 1 else pos["tps"][0] if pos["tps_hit"] == 2 else None
        if not stop_missing and want is not None and d * (want - pos["stop"]) > pos["tick"]:
            self.move_stop(pos, want)
            notify(f"{name}：保護止損設於 {want:.6g}")
        if (not stop_missing and pos.get("stop_qty") is not None
                and pos["stop_qty"] > pos["remaining"] + pos["step"] / 2):
            # 數量被同幣同向另一張單多平而扣掉了：止損改成用新的剩餘數量掛
            self.move_stop(pos, pos["stop"])
        if stop_missing and pos.get("stop_version", 0) == stop_version:  # 止盈沒有順便重掛
            why = f"（交易所狀態 {pos.get('last_algo_status') or '查無'}）"
            pos["stop_version"] = stop_version + 1
            try:
                pos["stop_id"] = self._place_stop(pos, pos["stop"], pos["stop_version"])
            except BinanceError as e:
                if e.code == -2021:  # 止損價已經被價格越過，掛不上去
                    pos["stop_id"] = None
                    self._close_rest(pos, f"m{pos['stop_version']}")
                    notify(f"{name}：止損單不見了{why}，而且價格已越過 {pos['stop']:.6g}，已市價平倉")
                    return self.finish(pos)
                if e.code == -2022 and self._flat(pos):  # 倉位已經不在交易所
                    pos["stop_id"] = None
                    self._close_rest(pos, f"m{pos['stop_version']}")
                    notify(f"{name}：止損單不見了{why}，交易所上已經沒有這個倉位，照目前價格記帳")
                    return self.finish(pos)
                raise
            notify(f"{name}：止損單不見了{why}，已補掛在 {pos['stop']:.6g}")
        bar = BAR[pos["interval"]]
        if pos["tps_hit"] >= 3 and pos["trail"] > 0:
            # 追蹤止損：TP3 那根 K 線起的最高價（做多）/ 最低價（做空）往回 3 ATR，每根收盤後更新
            last_close = int(bar_close(now, pos["interval"]).timestamp() * 1000)
            if pos.get("trail_checked") != last_close:
                df = self.client.klines(pos["symbol"], pos["interval"], MAX_BARS + 20)
                start = pd.Timestamp(pos["trail_from"], unit="ms", tz="UTC").floor(bar)
                seg = df[(df["time"] >= start) & (df["time"] + bar <= now)]
                if len(seg):
                    ext = float(seg["high"].max() if d > 0 else seg["low"].min())
                    ts = ext - d * pos["trail"]
                    if d * (ts - pos["stop"]) > pos["tick"]:
                        self.move_stop(pos, ts)
                        notify(f"{name}：追蹤止損移到 {ts:.6g}")
                pos["trail_checked"] = last_close
        opened = pd.Timestamp(pos["signal"]["time"], unit="ms", tz="UTC")
        if now - opened >= MAX_BARS * bar:
            self._close_rest(pos, "x")
            notify(f"{name}：超過 {MAX_BARS} 根未結束，市價平倉")
            return self.finish(pos)
        return None
