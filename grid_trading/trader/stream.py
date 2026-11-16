"""幣安 USDⓈ-M 合約的帳戶推播（user data stream）：只用來「叫醒」主迴圈，不拿推播內容記帳。

- 成交、條件單狀態改變、倉位變動時幣安會推事件；收到就設 wake，主迴圈馬上跑一輪平常的持倉管理（LiveBroker.manage）。
  推播可能漏、可能晚、格式也可能改，所以成交、數量、損益一律照舊用 REST 查，推播只決定「什麼時候查」。
- 只有「連著」而且「這個網址範本收過訂單 / 條件單 / 帳戶事件」才算健康：舊網址可能連得上卻什麼都不推，
  和「帳戶剛好沒事」分不出來，沒證明過就照舊每 --poll 秒輪詢。
- listenKey：POST 取得、每 30 分鐘 PUT 延長（60 分鐘沒延長會失效）；被幣安封鎖期間不送（BinanceFutures 會擋）。
- 連不上、斷線、listenKeyExpired：等一下再連（指數退避，最多 60 秒）；連滿 23 小時主動重連（幣安 24 小時會切斷）。
- 幣安 2026 年改了網址格式，細節沒辦法確認，所以網址是一串範本（BINANCE_WS_URL 可以換），連不上或連上馬上斷就換下一個。
- 整個只有一條背景執行緒；不碰狀態檔、持倉、broker，只設 wake 和自己的狀態欄位。listenKey 不寫進紀錄。
"""

from __future__ import annotations

import json
import os
import threading
import time

from .binance import BinanceError

PRIVATE_EVENTS = "events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE/ALGO_UPDATE"
MAINNET_WS = [f"wss://fstream.binance.com/private/ws?listenKey={{listenKey}}&{PRIVATE_EVENTS}",
              "wss://fstream.binance.com/ws/{listenKey}"]
TESTNET_WS = [f"wss://fstream.binancefuture.com/private/ws?listenKey={{listenKey}}&{PRIVATE_EVENTS}",
              "wss://fstream.binancefuture.com/ws/{listenKey}"]

PROVING = ("ORDER_TRADE_UPDATE", "ALGO_UPDATE", "ACCOUNT_UPDATE")  # 收到這些才證明這個網址真的有推帳戶事件
KEEPALIVE = 1800.0       # listenKey 多久延長一次（幣安：60 分鐘沒延長就失效）
KEEPALIVE_RETRY = 60.0   # 延長失敗（斷網、被封）隔多久再試；60 分鐘內成功都還來得及
MAX_AGE = 23 * 3600.0    # 連線滿這麼久就主動重連（幣安 24 小時會切斷）
QUICK = 60.0             # 連上不到這麼久就斷：當作這個網址不行，換下一個範本
BACKOFF_START = 1.0
BACKOFF_MAX = 60.0
RECV_TIMEOUT = 10.0      # 收資料最多等幾秒，回來檢查延長、停止、重連
PING_EVERY = 120.0       # 自己送 ping：不靠幣安的 ping 間隔也知道連線還活著
SILENT_LIMIT = 600.0     # 這麼久沒收到任何東西（含 pong）就當作斷線
PROVEN_FAILS = 3         # 證明過的範本連續失敗幾次才換（短暫斷網不要換到沒證明過的網址）
RETURN_AFTER = 600.0     # 連在沒證明過的範本、但有別的範本證明過：這麼久後換回去


def url_templates(testnet: bool, env: str | None = None, log=None) -> list[str]:
    """網址範本：BINANCE_WS_URL（逗號分隔，每個都要有 {listenKey}）優先，沒有就用預設。"""
    raw = os.environ.get("BINANCE_WS_URL", "") if env is None else env
    custom = [t.strip() for t in raw.split(",") if t.strip()]
    good = [t for t in custom if "{listenKey}" in t]
    if len(good) < len(custom):
        (log or _print)(f"BINANCE_WS_URL 有 {len(custom) - len(good)} 個範本沒有 {{listenKey}}，略過")
    return good or list(TESTNET_WS if testnet else MAINNET_WS)


def _print(*a) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


class _Conn:
    """websocket-client 的連線包一層：recv() 逾時回 None、ping / pong 回 ""（表示連線還活著），伺服器關閉就丟錯。"""

    def __init__(self, ws, mod):
        self.ws, self.mod = ws, mod

    def recv(self) -> str | None:
        try:
            opcode, data = self.ws.recv_data(control_frame=True)
        except self.mod.WebSocketTimeoutException:
            return None
        abnf = self.mod.ABNF
        if opcode == abnf.OPCODE_CLOSE:
            code = int.from_bytes(data[:2], "big") if len(data or b"") >= 2 else None
            raise ConnectionError(f"伺服器關閉連線（code={code}）")
        if opcode in (abnf.OPCODE_PING, abnf.OPCODE_PONG):
            return ""
        return data.decode("utf-8", "replace") if isinstance(data, bytes) else data

    def ping(self) -> None:
        self.ws.ping()

    def close(self) -> None:
        self.ws.close()


def _default_connect(url: str) -> _Conn:
    import websocket
    return _Conn(websocket.create_connection(url, timeout=RECV_TIMEOUT), websocket)


class UserStream:
    """client 要有 new_listen_key() / keepalive_listen_key()（BinanceFutures）。
    connect(url) 回傳有 recv() / ping() / close() 的連線；clock、sleep 可換掉（測試用假的）。"""

    def __init__(self, client, templates: list[str], connect=None, clock=time.time, sleep=None, log=None):
        self.client = client
        self.templates = list(templates) or list(MAINNET_WS)
        self._connect = connect
        self._clock = clock
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait  # 停止時不必等退避睡完
        self._log = log or _print
        self.wake = threading.Event()   # 主迴圈等的就是它
        self._lock = threading.Lock()
        self.enabled = True
        self.index = 0
        self.connected = False
        self.proven: set[int] = set()   # 收過帳戶事件的範本（換範本後再換回來不用重新證明）
        self.last_event: float | None = None
        self.last_error = ""
        self._key: str | None = None
        self._key_at = 0.0
        self._keys: list[str] = []      # 用過的 listenKey，只用來把紀錄裡的遮掉
        self._ws = None
        self._thread: threading.Thread | None = None
        self._backoff = BACKOFF_START
        self._fails = 0                 # 目前範本連續失敗幾次
        self._logged = ""               # 上一行斷線紀錄：同樣的原因不重複印

    # ---------- 給主執行緒看的狀態 ----------
    @property
    def healthy(self) -> bool:
        with self._lock:
            return self.enabled and self.connected and self.index in self.proven

    @property
    def state(self) -> str:
        if not self.enabled:
            return "off"
        with self._lock:
            if self.connected:
                return "healthy" if self.index in self.proven else "connected"
        return "down"

    # ---------- 啟動 / 停止 ----------
    def start(self) -> bool:
        if self._connect is None:
            try:
                import websocket  # noqa: F401
            except ImportError:
                # VPS 可能還沒裝套件就先更新了程式：不能讓交易程式因此當掉，照舊輪詢
                self.enabled = False
                self._log("沒有安裝 websocket-client，即時推送關閉，持倉管理照舊輪詢（請在 venv 執行 pip install -r requirements.txt）")
                return False
            self._connect = _default_connect
        self._thread = threading.Thread(target=self.run, name="user-stream", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()  # 讓卡在 recv 的執行緒馬上出來
            except Exception:
                pass

    # ---------- 背景執行緒 ----------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                outcome = self._cycle()
            except Exception as e:  # 保險：執行緒絕不能死，也不能把錯誤丟出去
                outcome = "failed_key"
                self._down(f"未預期的錯誤：{type(e).__name__}: {e}")
            if self._stop.is_set():
                break
            if outcome == "planned":
                continue
            if outcome == "failed_url":
                self._fails += 1
                self._maybe_rotate()
            if outcome == "dropped":  # 連了一陣子才斷：網址沒問題，退避從頭算
                self._fails = 0
                self._backoff = BACKOFF_START
                self._sleep(BACKOFF_START)
            else:
                self._sleep(self._backoff)
                self._backoff = min(self._backoff * 2, BACKOFF_MAX)

    def _maybe_rotate(self) -> None:
        """沒證明過的範本失敗就換；證明過的要連續失敗 PROVEN_FAILS 次才換（多半只是斷網）。"""
        if len(self.templates) < 2:
            return
        if self.index in self.proven and self._fails < PROVEN_FAILS:
            return
        with self._lock:
            self.index = (self.index + 1) % len(self.templates)
        self._fails = 0

    def _cycle(self) -> str:
        """連一次，回傳怎麼結束的：planned（馬上重連）、dropped（連了一陣子才斷）、failed_url（換範本）、failed_key。"""
        if self._key is None:
            try:
                self._new_key()
            except Exception as e:
                self._down(f"取得 listenKey 失敗：{e}")
                return "failed_key"
        i = self.index
        url = self.templates[i].replace("{listenKey}", self._key)
        try:
            ws = self._connect(url)
        except Exception as e:
            self._down(f"連線失敗（範本 #{i}）：{type(e).__name__}: {e}")
            return "failed_url"
        self._ws = ws
        opened = last_rx = last_ping = self._clock()
        with self._lock:
            self.connected = True
        self._logged = ""
        self._log(f"即時推送已連線（範本 #{i}：{self._show(i)}）"
                  + ("" if i in self.proven else "，收到第一個帳戶事件前照舊輪詢"))
        reason, outcome = "", "dropped"
        try:
            while not self._stop.is_set():
                now = self._clock()
                if now - opened >= MAX_AGE:
                    reason, outcome = "連線滿 23 小時，主動重連", "planned"
                    break
                if now - last_rx >= SILENT_LIMIT:
                    reason = f"{SILENT_LIMIT / 60:.0f} 分鐘沒有收到任何資料"
                    break
                if i not in self.proven and self.proven and now - opened >= RETURN_AFTER:
                    with self._lock:
                        self.index = min(self.proven)
                    reason, outcome = f"範本 #{i} 沒收到事件，換回證明過的範本 #{self.index}", "planned"
                    break
                if now - self._key_at >= KEEPALIVE and not self._keepalive(now):
                    reason, outcome = "listenKey 失效或換了，用新的重連", "planned"
                    break
                if now - last_ping >= PING_EVERY:
                    ws.ping()
                    last_ping = now
                msg = ws.recv()
                if msg is None:
                    continue
                last_rx = self._clock()
                if msg and self._handle(msg, i) == "expired":
                    self._key = None
                    reason, outcome = "listenKeyExpired，換新的 listenKey 重連", "planned"
                    break
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            outcome = "failed_url" if self._clock() - opened < QUICK else "dropped"
        finally:
            self._ws = None
            try:
                ws.close()
            except Exception:
                pass
        if self._stop.is_set():
            reason = "程式停止"
        self._down(f"範本 #{i}：{reason}")
        return outcome

    def _handle(self, msg: str, i: int) -> str | None:
        try:
            data = json.loads(msg)
        except ValueError:
            return None
        if isinstance(data, dict) and "e" not in data and isinstance(data.get("data"), dict):
            data = data["data"]  # 合併串流格式 {"stream": ..., "data": {...}}
        if not isinstance(data, dict):
            return None
        ev = data.get("e")
        if not ev:
            if "code" in data or "error" in data:  # 新網址的錯誤回應：記下來才知道網址哪裡不對
                self._log(f"即時推送回應（範本 #{i}）：{self._mask(msg[:200])}")
            return None
        if ev == "listenKeyExpired":
            return "expired"
        with self._lock:
            self.last_event = self._clock()
            newly = ev in PROVING and i not in self.proven
            if newly:
                self.proven.add(i)
        self.wake.set()
        if newly:
            self._log(f"即時推送確認可用：範本 #{i} 收到 {ev}")
        return None

    # ---------- listenKey ----------
    def _new_key(self) -> None:
        key = self.client.new_listen_key()
        if not key:
            raise RuntimeError("幣安沒有回 listenKey")
        self._set_key(key)

    def _set_key(self, key: str) -> None:
        self._key, self._key_at = key, self._clock()
        if key not in self._keys:
            self._keys = (self._keys + [key])[-5:]

    def _keepalive(self, now: float) -> bool:
        """延長 listenKey。回 False 表示要換新的重連（幣安說不存在、或回了別的 key）。"""
        try:
            got = self.client.keepalive_listen_key()
        except BinanceError as e:
            if e.code == -1125:  # This listenKey does not exist
                self._key = None
                return False
            self._keepalive_failed(now, e)
            return True
        except Exception as e:
            self._keepalive_failed(now, e)
            return True
        self._key_at = now
        if got and got != self._key:
            self._set_key(got)
            return False
        return True

    def _keepalive_failed(self, now: float, e: Exception) -> None:
        self._key_at = now - KEEPALIVE + KEEPALIVE_RETRY  # 等一下再試，不必斷線
        self._note(f"延長 listenKey 失敗，{KEEPALIVE_RETRY:.0f} 秒後再試：{e}")

    # ---------- 紀錄 ----------
    def _mask(self, text: str) -> str:
        for k in self._keys:
            text = text.replace(k, "***")
        return text

    def _show(self, i: int) -> str:
        return self.templates[i].replace("{listenKey}", "***")

    def _note(self, text: str) -> None:
        text = self._mask(text)
        if text != self._logged:  # 重連迴圈裡同樣的原因只印一次
            self._logged = text
            self._log(text)

    def _down(self, reason: str) -> None:
        reason = self._mask(reason)
        with self._lock:
            was = self.connected
            self.connected = False
            self.last_error = reason
        self._note(("即時推送斷線：" if was else "即時推送連不上：") + reason)
