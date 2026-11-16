"""幣安 USDT 本位合約 REST API 的最小用戶端：行情（不用金鑰）和下單（要金鑰）。

- 行情一律走正式站 fapi.binance.com（訊號要用真實價格算）；測試網模式的下單和持倉管理走測試網。
- 美國的 IP 會被擋（HTTP 451），所以只能在美國以外的主機上跑（GitHub Actions 不行）。
- 條件單（止損、止盈）從 2025-12 起改走 Algo Order：POST/DELETE /fapi/v1/algoOrder、GET /fapi/v1/openAlgoOrders。
- 下單（市價單、條件單）不盲目重送：逾時或 5xx 時幣安可能其實已經成交（官方文件：執行狀態不明），
  先用 client id 查這張單，查得到就當作成功，查無此單（-2013）才重送；查也查不到結果就丟 OrderUnknown。
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
import urllib.parse
from collections import Counter, deque

import pandas as pd
import requests

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"
INTERVALS = {"2H": "2h", "4H": "4h"}


def clean_id(client_id: str | None) -> str | None:
    """client id 只能用 [.A-Za-z0-9:/_-]、最長 36 字元，否則整張單會被 -1100 拒絕；不合的字元換成底線。"""
    return None if client_id is None else re.sub(r"[^.A-Za-z0-9:/_-]", "_", client_id)[:36]


MAX_WAIT = 30   # 被限流時單次最多等幾秒
BAN_MARGIN = 120  # 418 的 Retry-After 到了以後再多等幾秒才送請求
NOT_FOUND = -2013   # Order does not exist
DUPLICATE = -4116   # ClientOrderId is duplicated
DEAD = ("CANCELED", "EXPIRED", "REJECTED")


class BinanceError(RuntimeError):
    def __init__(self, status: int, code, msg: str):
        super().__init__(f"HTTP {status} code={code} {msg}")
        self.status, self.code, self.msg = status, code, msg


class OrderUnknown(RuntimeError):
    """下單請求送出去了，但不知道幣安有沒有執行（逾時、5xx），用 client id 也查不出結果。"""


class BinanceFutures:
    def __init__(self, base: str = MAINNET, key: str = "", secret: str = "", timeout: float = 15,
                 session: requests.Session | None = None, stream_session: requests.Session | None = None):
        self.base = base.rstrip("/")
        self.key, self.secret = key, secret
        self.timeout = timeout
        self.session = session or requests.Session()
        # requests.Session 沒有保證多執行緒安全：即時推送的執行緒（listenKey）用自己的 session，不和交易迴圈共用
        self.stream_session = stream_session or requests.Session()
        self.banned_until = 0.0   # 418（被封）時，到這個時間為止不再送需要簽名的請求
        self._offset = 0  # 本機時間和幣安伺服器時間的差（毫秒），簽名用
        self._recent: deque = deque(maxlen=600)  # 最近送出的請求（時間, 方法, 路徑）：被擋時印出來，查是不是真的送太多

    # ---------- 底層 ----------
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False, tries: int = 3,
                 resend: bool = True, keyed: bool = False, session: requests.Session | None = None):
        """resend=False（下單用）：逾時、5xx 不自己重送，丟 OrderUnknown 讓呼叫的人先查單再決定。
        429、-1021 和連線根本沒建立（ConnectTimeout）幣安沒有收到 / 沒有執行，照樣重送。
        keyed：只帶 API 金鑰、不簽名（listenKey 用）；和簽名請求一樣，被封期間不送。"""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}
        if keyed:
            if not self.key:
                raise RuntimeError("沒有設定 BINANCE_API_KEY")
            headers["X-MBX-APIKEY"] = self.key
        if (signed or keyed) and time.time() < self.banned_until:
            # 被封期間再送請求只會把封鎖拉長，直接失敗，讓上層記一行就好
            raise BinanceError(418, -1003, f"被幣安暫時封鎖，{self.banned_until - time.time():.0f} 秒後才會再送請求")
        for i in range(tries):
            q = dict(params)
            if signed:
                if not (self.key and self.secret):
                    raise RuntimeError("沒有設定 BINANCE_API_KEY / BINANCE_API_SECRET")
                q["timestamp"] = int(time.time() * 1000) + self._offset
                q["recvWindow"] = 10_000
                query = urllib.parse.urlencode(q)
                q["signature"] = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
                headers["X-MBX-APIKEY"] = self.key
            self._recent.append((time.time(), method, path))
            try:
                resp = (session or self.session).request(method, self.base + path, params=q, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if not resend and not isinstance(e, requests.ConnectTimeout):
                    raise OrderUnknown(f"{path} 送出後沒有回應：{e}") from e
                if i == tries - 1:
                    raise
                time.sleep(2 ** i)
                continue
            if resp.status_code == 418:
                raw = resp.headers.get("Retry-After")
                try:
                    ban = float(raw)
                except (TypeError, ValueError):
                    ban = 120.0
                # 多等 BAN_MARGIN 秒才再送：本機時鐘和幣安差一點、剛好在解封前一刻送出的請求會被當成
                # 「封鎖期間還在送」而再封一次，而且一次比一次久（2026-11-16 04:43 解封那一刻馬上又被封到 12:38）
                self.banned_until = time.time() + ban + BAN_MARGIN
                until = time.strftime("%H:%M", time.gmtime(self.banned_until))
                print(f"幣安回 418（{path}），Retry-After={raw}，暫停送簽名請求到 {until} UTC", flush=True)
                print(self._diagnose(method, path, q, resp), flush=True)
                raise BinanceError(418, -1003, f"被幣安暫時封鎖到 {until} UTC")
            if resp.status_code >= 500 and not resend:
                raise OrderUnknown(f"{path} 回 HTTP {resp.status_code}（執行狀態不明）")
            if resp.status_code == 429 or resp.status_code >= 500:
                if i < tries - 1:  # 被限流或幣安自己出錯，等一下再試
                    # Retry-After 可能是幾小時（418 封 IP）；整個交易迴圈只有一條執行緒，
                    # 睡下去就整個停擺且日誌沒有任何一行，所以最多等 MAX_WAIT 秒，之後照常往外丟錯誤。
                    try:
                        wait = float(resp.headers.get("Retry-After", 2 ** (i + 1)))
                    except ValueError:
                        wait = 2 ** (i + 1)
                    wait = min(wait, MAX_WAIT)
                    print(f"幣安回 {resp.status_code}（{path}），{wait:.0f} 秒後重試", flush=True)
                    if resp.status_code == 429:
                        print(self._diagnose(method, path, q, resp), flush=True)
                    time.sleep(wait)
                    continue
            try:
                data = resp.json()
            except ValueError:
                data = {"msg": resp.text[:200]}
            if resp.status_code >= 400:
                code = data.get("code") if isinstance(data, dict) else None
                if code == -1007 and not resend:  # 幣安：等後端回應逾時，執行狀態不明
                    raise OrderUnknown(f"{path} 回 -1007（執行狀態不明）")
                if code == -1021 and signed and i < tries - 1:  # 時間差太大：對時後重送
                    self.sync_time()
                    continue
                raise BinanceError(resp.status_code, code, (data.get("msg") if isinstance(data, dict) else "") or "")
            return data
        raise RuntimeError("unreachable")

    def _diagnose(self, method: str, path: str, params: dict, resp) -> str:
        """被限流 / 封鎖時的診斷：幣安回報的用量、回應內容、這個請求帶了哪些參數（只列名稱，不含簽名和金鑰），
        以及最近 1 分鐘、10 分鐘各送了幾個請求。2026-11-15 被封那次事前沒有任何 429，要靠這些才查得出原因。"""
        now = time.time()
        recent = list(self._recent)
        def count(window: float) -> str:
            c = Counter(f"{m} {p}" for t, m, p in recent if now - t <= window)
            return f"{sum(c.values())} 個（" + "、".join(f"{k}×{v}" for k, v in c.most_common(6)) + "）"
        usage = {k: v for k, v in resp.headers.items() if k.lower().startswith("x-mbx-")}
        names = sorted(k for k in params if k != "signature")
        return (f"診斷：{method} {path} 參數 {names}；幣安回報用量 {usage or '無'}；回應 {str(getattr(resp, 'text', ''))[:200]!r}；"
                f"最近 1 分鐘送了 {count(60)}，10 分鐘 {count(600)}")

    def sync_time(self) -> None:
        server = self._request("GET", "/fapi/v1/time")["serverTime"]
        self._offset = int(server) - int(time.time() * 1000)

    # ---------- 行情（不用金鑰） ----------
    def exchange_info(self) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """最近 limit 根 K 線（time 是開盤時間，UTC）。最後一根通常還沒收盤。"""
        rows = self._request("GET", "/fapi/v1/klines",
                             {"symbol": symbol, "interval": INTERVALS.get(interval, interval), "limit": limit})
        df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        return df

    def price(self, symbol: str) -> float:
        return float(self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})["price"])

    # ---------- 帳戶推播的 listenKey（只要 API 金鑰，不簽名；由即時推送的執行緒呼叫） ----------
    def new_listen_key(self) -> str:
        """取得 listenKey（已經有有效的 key 時幣安回同一個並延長 60 分鐘）。"""
        data = self._request("POST", "/fapi/v1/listenKey", keyed=True, session=self.stream_session)
        return (data or {}).get("listenKey", "") if isinstance(data, dict) else ""

    def keepalive_listen_key(self) -> str | None:
        """延長 listenKey 60 分鐘。回傳幣安回的 key（可能沒有）；key 不存在時幣安回 -1125。"""
        data = self._request("PUT", "/fapi/v1/listenKey", keyed=True, session=self.stream_session)
        return data.get("listenKey") if isinstance(data, dict) else None

    # ---------- 帳戶（要金鑰） ----------
    def balance_usdt(self) -> float:
        """USDT 的帳戶權益（錢包餘額 + 未實現損益）。"""
        for row in self._request("GET", "/fapi/v2/balance", signed=True):
            if row.get("asset") == "USDT":
                return float(row["balance"]) + float(row.get("crossUnPnl") or 0)
        return 0.0

    def positions(self) -> dict[tuple[str, str], float]:
        """(代號, LONG/SHORT) → 持倉數量（正數）。只列有倉位的。"""
        out = {}
        for p in self._request("GET", "/fapi/v2/positionRisk", signed=True):
            amt = abs(float(p.get("positionAmt") or 0))
            if amt > 0:
                side = p.get("positionSide") or ("LONG" if float(p["positionAmt"]) > 0 else "SHORT")
                out[(p["symbol"], side)] = amt
        return out

    def hedge_mode(self) -> bool:
        return bool(self._request("GET", "/fapi/v1/positionSide/dual", signed=True).get("dualSidePosition"))

    def set_hedge_mode(self) -> None:
        self._request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true"}, signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def set_isolated(self, symbol: str) -> None:
        try:
            self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        except BinanceError as e:
            if e.code != -4046:  # -4046：本來就是逐倉
                raise

    def _place(self, path: str, params: dict, lookup, dup_ok: bool = False) -> dict:
        """送出下單，結果不明時先用 client id 查單：查到就當作已下單，查無此單才重送，最多送 3 次。
        dup_ok：幣安回「client id 重複」時去查那張單，還掛著或已成交就當作已下單（前一次其實成功了）。"""
        err = None
        for i in range(3):
            try:
                return self._request("POST", path, params, signed=True, resend=False)
            except OrderUnknown as e:
                err = e
            except BinanceError as e:
                if not (dup_ok and e.code == DUPLICATE):
                    raise
                found = lookup()
                if str(found.get("algoStatus") or found.get("status") or "").upper() in DEAD:
                    raise
                return found
            time.sleep(2 ** i)  # 剛送出的單可能要一下子才查得到，馬上查容易誤判成「查無此單」
            try:
                found = lookup()
            except BinanceError as q:
                if q.code == NOT_FOUND:  # 前一次沒有下成，重送安全
                    print(f"{path} 結果不明，查無此單，重送（{err}）", flush=True)
                    continue
                raise OrderUnknown(f"{err}；查單也失敗：{q}") from q
            except (requests.RequestException, OrderUnknown) as q:
                raise OrderUnknown(f"{err}；查單也失敗：{q}") from q
            if not isinstance(found, dict) or not (found.get("orderId") or found.get("algoId")):
                raise OrderUnknown(f"{err}；查單回應看不懂：{str(found)[:200]}")
            print(f"{path} 結果不明，查單後確認已下單（{err}）", flush=True)
            return found
        raise err

    def order(self, symbol: str, client_id: str) -> dict:
        """用 client id 查一般訂單（含已成交的）。查無此單時幣安回 -2013。"""
        return self._request("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": clean_id(client_id)},
                             signed=True)

    def market_order(self, symbol: str, side: str, position_side: str, quantity: str, client_id: str,
                     check_first: bool = False) -> dict:
        """市價單。side 是 BUY/SELL，position_side 是 LONG/SHORT（雙向持倉）。回傳含 avgPrice、executedQty、orderId。
        幣安只要求 client id 在「掛著的單」之間不重複，已成交的市價單 id 可以再用，所以不能靠重複 id 擋掉重送：
        結果不明時一律先查單。check_first：上一次送這個 id 結果不明（例如程式在中間出錯），送之前先查。"""
        cid = clean_id(client_id)

        def lookup():
            return self.order(symbol, cid)

        if check_first:
            try:
                found = lookup()
                if found.get("orderId"):
                    return found
            except BinanceError as e:
                if e.code != NOT_FOUND:
                    raise
        return self._place("/fapi/v1/order", {
            "symbol": symbol, "side": side, "positionSide": position_side, "type": "MARKET",
            "quantity": quantity, "newClientOrderId": cid, "newOrderRespType": "RESULT"}, lookup)

    def conditional_order(self, symbol: str, side: str, position_side: str, kind: str, trigger: str,
                          quantity: str | None = None, client_id: str | None = None) -> dict:
        """條件單：kind 是 STOP_MARKET 或 TAKE_PROFIT_MARKET，用標記價格觸發。quantity 為 None 時全部平倉。"""
        params = {"algoType": "CONDITIONAL", "symbol": symbol, "side": side, "positionSide": position_side,
                  "type": kind, "triggerPrice": trigger, "workingType": "MARK_PRICE",
                  "clientAlgoId": clean_id(client_id)}
        if quantity is None:
            params["closePosition"] = "true"
        else:
            params["quantity"] = quantity
        if client_id is None:  # 沒有 client id 查不了單，只能照舊送一次
            return self._request("POST", "/fapi/v1/algoOrder", params, signed=True, resend=False)
        return self._place("/fapi/v1/algoOrder", params, lambda: self.algo_order_by_client(params["clientAlgoId"]),
                           dup_ok=True)

    def cancel_conditional(self, algo_id) -> None:
        try:
            self._request("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True)
        except BinanceError as e:
            if e.status != 400:  # 已經觸發或取消過的單會回 400，當作取消成功
                raise

    def algo_order(self, algo_id) -> dict:
        """查一張條件單（含已觸發、已取消的）；algoStatus 是 TRIGGERED / FINISHED 表示已成交。"""
        return self._request("GET", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True)

    def algo_order_by_client(self, client_id: str) -> dict:
        return self._request("GET", "/fapi/v1/algoOrder", {"clientAlgoId": clean_id(client_id)}, signed=True)

    def open_conditionals(self, symbol: str | None = None) -> list[dict]:
        """掛著的條件單。不給 symbol 就一次列出所有幣（權重 40，單一幣是 1）：幣多時一個請求比每個幣各打一次省得多。"""
        data = self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)
        return data if isinstance(data, list) else data.get("orders", [])

    def order_trades(self, symbol: str, order_id) -> list[dict]:
        """某一張訂單的成交明細（userTrades 用 orderId 篩，不受 7 天時間範圍限制）。"""
        return self._request("GET", "/fapi/v1/userTrades", {"symbol": symbol, "orderId": order_id}, signed=True)

    def realized_since(self, symbol: str, position_side: str, start_ms: int) -> float:
        """某個代號、某一邊（LONG/SHORT）從 start_ms 起的已實現損益減手續費（USDT）。"""
        total = 0.0
        for t in self._request("GET", "/fapi/v1/userTrades",
                               {"symbol": symbol, "startTime": start_ms, "limit": 1000}, signed=True):
            if t.get("positionSide", position_side) != position_side:
                continue
            total += float(t.get("realizedPnl") or 0) - float(t.get("commission") or 0)
        return total
