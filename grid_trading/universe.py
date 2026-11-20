"""要交易的幣：CoinMarketCap 市值前 100 名（扣掉穩定幣），每月自動更新（2026-11-21 起）。

- monthly_top()：每個月（UTC）第一次用到時從 CoinMarketCap 抓一次當下的前 100 名，整個月沿用；
  抓不到就沿用上個月的名單，下次再試；連上個月的都沒有才用下面的 TOP100 備用名單。
  回測（research/universe_buckets.py）用的是同一個規則：當月用上個月底的市值前 100 名，幣安沒有合約的不遞補。
- binance_perp_for()：CMC 代號對應到幣安 USDT 永續合約（1000PEPE、1000BONK 這類加前綴的也找得到）。
- TOP100 是 2026-10-12 的舊快照（當時只留派網也有合約的 88 個），現在只當備用名單和舊研究用。
"""

from __future__ import annotations

from datetime import datetime, timezone

import requests

CMC_LISTING = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"

TOP100 = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ZEC", "HYPE", "DOGE", "XMR",
    "LINK", "ADA", "NEAR", "XLM", "BCH", "LTC", "CC", "AVAX", "UNI", "SUI",
    "HBAR", "GRAM", "TAO", "SHIB", "ENA", "CRO", "QNT", "OKB", "PUMP", "AAVE",
    "ONDO", "DOT", "WLD", "ASTER", "ICP", "MNT", "SKY", "PAXG", "WLFI", "PEPE",
    "ETC", "ARB", "VVV", "JUP", "ALGO", "POL", "KAS", "RENDER", "FIL", "ATOM",
    "ZRO", "LIT", "PI", "AERO", "INJ", "NIGHT", "APT", "CAKE", "STABLE", "STRK",
    "ETHFI", "STX", "PYTH", "DASH", "VET", "RAY", "CRV", "币安人生", "TRUMP", "FET",
    "TIA", "SEI", "PENGU", "VIRTUAL", "XPL", "BSV", "PIEVERSE", "FF", "PENDLE", "IMX",
    "LDO", "XTZ", "SPX", "KITE", "CFX", "SUN", "OP", "KAIA",
]


# 派網的 RWA 永續合約（2026-10-12 整理）：貴金屬、原油等商品，以及美股/ETF 代幣（代號後面加 X）。
# 商品全部列入；股票/ETF 只列 24 小時成交額約 30 萬 USDT 以上的。
RWA_COMMODITIES = ["WTI", "XAU", "XAG", "XPT", "XPD", "COPPER", "USOX"]
RWA_STOCKS = [
    "AAOIX", "AAPLX", "ALABX", "AMATX", "AMDX", "AMZNX", "ARMX", "ASMLX", "ASTSX", "AVGOX",
    "AXTIX", "BEX", "BMNRX", "COHRX", "COINX", "CRCLX", "CRDOX", "CRWVX", "DELLX", "DRAMX",
    "EWYX", "GLWX", "GOOGLX", "HIMSX", "HOODX", "INTCX", "IRENX", "KORUX", "LITEX", "LLYX",
    "LRCXX", "METAX", "MRNAX", "MRVLX", "MSFTX", "MSTRX", "MUX", "NBISX", "NOKX", "NVDAX",
    "ONDSX", "PLTRX", "QCOMX", "QQQX", "RKLBX", "SKHX", "SNDKX", "SNXXX", "SOXLX", "SOXSX",
    "SPCX", "SPYX", "STXX", "TQQQX", "TSLAX", "TSMX", "URNMX", "WDCX",
]
RWA = RWA_COMMODITIES + RWA_STOCKS


def fetch_cmc_top(n: int = 100) -> list[str]:
    """CoinMarketCap 市值前 n 名（不含穩定幣）的代號，依市值排序。不需要 API Key。"""
    params = {"start": 1, "limit": n + 50, "sortBy": "market_cap", "sortType": "desc", "convert": "USD"}
    data = requests.get(CMC_LISTING, params=params, timeout=20).json()["data"]["cryptoCurrencyList"]
    coins = [c["symbol"] for c in data if not {"stablecoin", "fiat-stablecoin"} & set(c.get("tags") or [])]
    return coins[:n]


def perp_symbols(coins: list[str] = TOP100) -> list[str]:
    return [f"{c}_USDT_PERP" for c in coins]


def binance_perp_for(coin: str, listed: set[str]) -> str | None:
    """CMC 代號 → 幣安 USDT 永續合約代號（listed 裡有的）；沒有就回傳 None。"""
    from .binance_history import binance_symbols  # 避免循環匯入

    cands = list(reversed(binance_symbols(coin))) + [f"{p}{coin}USDT" for p in ("", "1000", "1000000", "1M")]
    return next((s for s in cands if s in listed), None)


def monthly_top(cached: dict | None, n: int = 100, now: datetime | None = None, fetch=None) -> dict:
    """這個月的市值前 n 名。cached 是上一次回傳的 dict（呼叫端自己存），同一個月直接沿用。

    回傳 {"month": "YYYY-MM", "coins": [...], "fetched_at": ISO 時間, "status": ...}；status：
    cached（沿用本月）、fresh（剛抓）、stale（抓不到，沿用舊月份，下次再試）、fallback（用 TOP100 備用名單）。
    stale / fallback 會多一個 error 欄位。"""
    now = now or datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    if cached and cached.get("month") == month and cached.get("coins"):
        return {**cached, "status": "cached"}
    try:
        coins = (fetch or fetch_cmc_top)(n)
        if len(coins) < n * 0.8:
            raise ValueError(f"只抓到 {len(coins)} 個")
        return {"month": month, "coins": list(coins), "fetched_at": now.isoformat(timespec="seconds"), "status": "fresh"}
    except Exception as e:  # noqa: BLE001
        if cached and cached.get("coins"):
            return {**cached, "status": "stale", "error": str(e)}
        return {"month": None, "coins": list(TOP100), "fetched_at": None, "status": "fallback", "error": str(e)}
