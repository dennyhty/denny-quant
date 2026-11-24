"""Telegram 推播：只剩 send()，由 VPS 上的自動交易（grid_trading/trader/bot.py）呼叫。
網站（GitHub Actions 的 pages.yml）2026-11-06 起不再推播。

Bot Token 和 Chat ID 由呼叫的人從環境變數 TELEGRAM_BOT_TOKEN、TELEGRAM_CHAT_ID 讀，不寫進程式或檔案。
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.parse
import urllib.request

SITE_URL = "https://dennyhty.github.io/denny-quant/"
MAX_MESSAGES = 30  # 一次最多送幾則，避免洗版；其餘請看網站


def send(messages: list[str], token: str, chat_id: str) -> tuple[int, int]:
    """回傳（送出則數, 失敗則數）。"""
    if len(messages) > MAX_MESSAGES:
        extra = len(messages) - MAX_MESSAGES + 1
        messages = messages[:MAX_MESSAGES - 1] + [f"還有 {extra} 則通知，請到網站查看"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    failed = 0
    for i, text in enumerate(messages):
        body = urllib.parse.urlencode({"chat_id": chat_id, "text": f"{text}\n{SITE_URL}",
                                       "disable_web_page_preview": "true"}).encode()
        for attempt in range(3):  # 這一輪沒送到，下一輪也不會再送，所以失敗時等一下重試
            try:
                urllib.request.urlopen(url, body, timeout=20).read()
                break
            except Exception as e:  # 錯誤訊息不含 token（只印類型和狀態碼）
                code = getattr(e, "code", "")
                if _timed_out(e):
                    # 請求可能已經送達、只是回應沒等到（2026-10-18 XMR TP1 就是這樣收到兩次）：重送會重複，寧可不重送
                    print(f"第 {i + 1} 則等不到回應（{type(e).__name__}），可能已送達，不重送", file=sys.stderr)
                    break
                print(f"第 {i + 1} 則送出失敗：{type(e).__name__} {code}", file=sys.stderr)
                if attempt == 2 or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                    failed += 1  # 400/401/403 之類重試也沒用
                    break
                time.sleep(_retry_after(e) if code == 429 else 3)
        time.sleep(1.1)  # Telegram 對同一個聊天大約每秒 1 則（群組每分鐘 20 則，超過會回 429）
    return len(messages) - failed, failed


def _timed_out(e: Exception) -> bool:
    """逾時（連線建立後等不到回應）。連線階段逾時也一併當成可能已送達，不重送。"""
    return isinstance(e, (socket.timeout, TimeoutError)) or isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError))


def _retry_after(e: Exception) -> float:
    """Telegram 回 429 時，回應裡的 parameters.retry_after 是要等幾秒。"""
    try:
        return min(60.0, float(json.loads(e.read())["parameters"]["retry_after"]) + 1)
    except Exception:
        return 5.0
