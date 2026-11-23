"""追蹤一個訊號之後的走勢：TP 觸及、保護止損移動、止損出場。回測和即時卡片共用。

出場規則：
- 預設在 TP1、TP2、TP3 各平 1/3；訊號的 exit["portions"] 可以改成別的比例和 TP 數量。
- TP1 觸及後，剩餘倉位的止損移到 BE（進場價加上來回手續費，出場不賠錢）。
- TP2 觸及後，止損再移到 TP1。
- 所有 TP 都觸及後還有剩下的倉位（exit["trail"] 有設）：止損改成追蹤止損，
  = 進場後最高價（做多）/最低價（做空）往回 trail 的距離，只會往有利方向移動。
- 同一根 K 線同時碰到止損和 TP 時，保守假設先碰止損。
- 限價進場的訊號（exit["limit"]，例如日線吞沒）：訊號價是掛單價，訊號後的 exit["fill_bars"] 根內價格碰到掛單價才算成交
  （狀態 PENDING = 還在等，成交後 OPEN），沒成交就 EXPIRED（未成交取消，不算交易）。成交那根只看止損（碰到就算止損），
  止盈、保護止損從下一根起算；exit["max_bars"] 是成交後（不含成交那根）最多幾根，超過就在那根收盤平倉。
- max_bars 預設 None：沒有時間上限，只有止損、止盈、追蹤止損出場（和實盤一致）。傳入數字才會在超過那根數後收盤平倉。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .strategy import Signal

PORTIONS = (1 / 3, 1 / 3, 1 / 3)


@dataclass
class Event:
    kind: str      # FILLED / EXPIRED / TP1 / TP2 / TP3 / STOP_MOVED / TRAIL_MOVED / SL / BE_EXIT / PROTECT_EXIT / TRAIL_EXIT / TIMEOUT
    time: pd.Timestamp
    price: float
    text: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "time": int(self.time.timestamp() * 1000), "price": self.price, "text": self.text}


@dataclass
class TrackResult:
    status: str               # OPEN / CLOSED；限價進場另有 PENDING（掛單中）/ EXPIRED（沒成交，取消）
    events: list[Event] = field(default_factory=list)
    stop: float = 0.0         # 目前的止損價
    remaining: float = 1.0    # 剩餘倉位比例
    realized_r: float = 0.0   # 已實現（扣手續費），以 R 計
    realized_pct: float = 0.0  # 已實現（扣手續費），以 1 倍槓桿下的倉位報酬 % 計
    tps_hit: int = 0
    bars: int = 0
    be: float = 0.0
    trailing: bool = False    # 剩下的倉位正在用追蹤止損
    filled_at: pd.Timestamp | None = None  # 限價進場成交的那根 K 線（開盤時間）

    def to_dict(self) -> dict:
        d = self._base()
        if self.filled_at is not None:
            d["filled_at"] = int(self.filled_at.timestamp() * 1000)
        return d

    def _base(self) -> dict:
        return {"status": self.status, "events": [e.to_dict() for e in self.events], "stop": self.stop,
                "remaining": self.remaining, "realized_r": self.realized_r, "realized_pct": self.realized_pct,
                "tps_hit": self.tps_hit, "bars": self.bars, "be": self.be,
                "trailing": self.trailing}


def breakeven(sig: Signal, fee: float) -> float:
    return sig.price * (1 + sig.direction * 2 * fee)


def track(sig: Signal, candles: pd.DataFrame, fee: float = 0.0005, max_bars: int | None = None) -> TrackResult:
    """candles：K 線（time 是開盤時間），由舊到新。只看開盤時間 >= 訊號時間的部分。"""
    d = sig.direction
    risk = abs(sig.price - sig.sl)
    be = breakeven(sig, fee)
    plan = getattr(sig, "exit", None) or {}
    limit = bool(plan.get("limit"))
    fill_bars = int(plan.get("fill_bars") or 0)
    if limit and max_bars is None:
        max_bars = plan.get("max_bars")  # 限價進場：成交後最多幾根（不含等成交的那幾根）
    res = TrackResult(status="PENDING" if limit else "OPEN", stop=sig.sl, be=be)
    portions = plan.get("portions") or PORTIONS
    trail = plan.get("trail") or 0.0
    extreme = sig.price
    announced = None  # 上次通知的追蹤止損價；移動超過 trail/6（約 0.5 ATR）才再通知，免得每根 K 線都推播
    # 只取訊號後最多 max_bars 根，轉成 list 逐根跑（iterrows 很慢，回測時會被呼叫上千次）
    start = int(candles["time"].searchsorted(sig.time, side="left"))
    end = len(candles) if (max_bars is None or limit) else min(len(candles), start + max_bars)
    times = candles["time"].iloc[start:end].tolist()
    highs = candles["high"].iloc[start:end].tolist()
    lows = candles["low"].iloc[start:end].tolist()
    closes = candles["close"].iloc[start:end].tolist()

    def close_part(portion: float, price: float):
        move = d * (price - sig.price)
        cost = fee * (sig.price + price)  # 進出場各一次手續費
        res.realized_r += portion * (move - cost) / risk
        res.realized_pct += portion * (move - cost) / sig.price * 100
        res.remaining -= portion

    n = len(times)
    for idx, (t, high, low, close) in enumerate(zip(times, highs, lows, closes)):
        if res.status == "PENDING":
            if idx >= fill_bars:  # 等了 fill_bars 根都沒成交（那幾根都收完了）：取消
                res.status = "EXPIRED"
                res.events.append(Event("EXPIRED", t, sig.price, f"{fill_bars} 根內沒成交，掛單 {_fmt(sig.price)} 取消"))
                break
            if (d < 0 and high >= sig.price) or (d > 0 and low <= sig.price):
                res.status, res.filled_at = "OPEN", t
                res.events.append(Event("FILLED", t, sig.price, f"掛單成交 {_fmt(sig.price)}"))
                if d * ((low if d > 0 else high) - res.stop) <= 0:  # 成交那根就碰到止損：算止損（和研究一樣）
                    close_part(res.remaining, res.stop)
                    res.events.append(Event("SL", t, res.stop, f"止損出場 {_fmt(res.stop)}"))
                    res.status = "CLOSED"
                    break
            continue
        res.bars += 1
        adverse = low if d > 0 else high
        if d * (adverse - res.stop) <= 0:
            price = res.stop
            if res.tps_hit == 0:
                kind, text = "SL", f"止損出場 {_fmt(price)}"
            elif res.trailing and price != be:
                kind, text = "TRAIL_EXIT", f"追蹤止損出場 {_fmt(price)}"
            elif res.tps_hit == 1:
                kind, text = "BE_EXIT", f"保本出場 {_fmt(price)}"
            else:
                kind, text = "PROTECT_EXIT", f"保護止損出場 {_fmt(price)}"
            close_part(res.remaining, price)
            res.events.append(Event(kind, t, price, text))
            res.status = "CLOSED"
            break
        favorable = high if d > 0 else low
        while res.tps_hit < len(sig.tps) and d * (favorable - sig.tps[res.tps_hit]) >= 0:
            n = res.tps_hit
            tp = sig.tps[n]
            close_part(portions[n], tp)
            res.tps_hit += 1
            res.events.append(Event(f"TP{n + 1}", t, tp, f"TP{n + 1} 觸及 {_fmt(tp)}"))
            new_stop = be if n == 0 else (sig.tps[n - 1] if n == 1 else None)
            if new_stop is not None and (n + 1 < len(sig.tps) or trail):  # 最後一個 TP 又沒有追蹤：已經全平，不必再移止損
                res.stop = new_stop
                res.events.append(Event("STOP_MOVED", t, new_stop, f"保護止損設於 {_fmt(new_stop)}"))
        if res.tps_hit == len(sig.tps) and (not trail or res.remaining <= 1e-9):
            res.remaining = 0.0
            res.status = "CLOSED"
            break
        if trail and res.tps_hit == len(sig.tps):
            # 剩下的倉位用追蹤止損：進場後最有利的價格往回 trail，不低於目前止損
            res.trailing = True
            if d * (favorable - extreme) > 0:
                extreme = favorable
            ts = extreme - d * trail
            if d * (ts - res.stop) > 0:
                res.stop = ts
                if d * (ts - (announced if announced is not None else be)) >= trail / 6:
                    announced = ts
                    res.events.append(Event("TRAIL_MOVED", t, ts, f"追蹤止損移到 {_fmt(ts)}"))
        # 限價進場的時間出場：那根要已經收完（後面還有 K 線）才算，免得用還在跑的那根的價格平倉
        if max_bars is not None and res.bars >= max_bars and (not limit or idx < n - 1):
            close_part(res.remaining, float(close))
            res.events.append(Event("TIMEOUT", t, float(close), f"超過 {max_bars} 根未結束，收盤平倉"))
            res.status = "CLOSED"
            break
    return res


def _fmt(p: float) -> str:
    return f"{p:.6g}"
