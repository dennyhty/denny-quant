import pandas as pd
import pytest

from grid_trading.signals.strategy import LONG, SHORT, Signal
from grid_trading.signals.tracker import track
from grid_trading.trader.bot import Config, LiveBroker, Notifier, PaperBroker, Trader, bar_close
from grid_trading.trader.risk import RiskRules, can_open, risk_multiplier
from grid_trading.trader.sizing import Filters, fmt, parse_filters, plan_size, split

T0 = pd.Timestamp("2026-10-19 08:00", tz="UTC")
F = Filters(step=0.001, min_qty=0.001, max_qty=1e6, tick=0.01, min_notional=5.0)


# ---------- 倉位大小 ----------
def test_size_follows_risk():
    size, _ = plan_size(1000, 0.0025, 100.0, 96.0, F)
    assert size.steps * F.step == pytest.approx(0.625)  # 2.5U / 4
    assert size.risk_frac == pytest.approx(0.0025)
    assert sum(size.slices) == size.steps and not size.bumped


def test_min_notional_bumps_then_caps_risk():
    btc = Filters(step=0.001, min_qty=0.001, max_qty=1e6, tick=0.1, min_notional=100.0)
    size, why = plan_size(100, 0.0025, 70000.0, 68600.0, btc)  # 2% 止損，最少 100U → 風險 2%
    assert size is None and "超過上限" in why
    size, _ = plan_size(100, 0.0025, 70000.0, 68600.0, btc, max_risk=0.06)  # 要 4 個單位才分得成四段
    assert size.bumped and size.steps * btc.step * 70000 >= 100


def test_split_keeps_a_remainder_for_trailing():
    assert split(4) == [1, 1, 1, 1]
    assert split(10) == [2, 2, 2, 4]
    assert all(p >= 1 for p in split(5))


def test_fmt_rounds_to_unit():
    assert fmt(0.12345, 0.001) == "0.123"
    assert fmt(123.456, 0.1) == "123.5"
    assert fmt(5.0, 1.0) == "5"


def test_parse_filters_keeps_usdt_perps():
    info = {"symbols": [
        {"symbol": "XUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "1000"},
                     {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "100"},
                     {"filterType": "PRICE_FILTER", "tickSize": "0.001"},
                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]},
        {"symbol": "YUSDT", "status": "SETTLING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "filters": []}]}
    f = parse_filters(info)
    assert list(f) == ["XUSDT"] and f["XUSDT"].max_qty == 100 and f["XUSDT"].min_notional == 5


# ---------- 風控 ----------
def test_half_risk_after_drawdown_until_new_high():
    rules = RiskRules(dd_half_r=20)
    led = [{"r": 10}] + [{"r": -1}] * 20
    assert risk_multiplier(led, rules) == 0.5
    assert risk_multiplier(led + [{"r": 5}], rules) == 0.5      # 回撤縮小但還沒新高，照樣減半
    assert risk_multiplier(led + [{"r": 20}], rules) == 1.0     # 回到新高
    assert risk_multiplier(led, RiskRules()) == 1.0
    assert can_open(14, RiskRules(max_open=15)) and not can_open(15, RiskRules(max_open=15))


# ---------- 測試用的假交易所 ----------
def candles(rows, start=T0, freq="2h"):
    df = pd.DataFrame(rows, columns=["high", "low", "close"])
    df["open"] = df["close"]
    df["volume"] = 1.0
    df["time"] = pd.date_range(start, periods=len(df), freq=freq, tz="UTC")
    return df


class FakeExchange:
    def __init__(self, price=100.0, df=None):
        self.px = price
        self.df = df
        self.pos = {}
        self.algo = {}
        self.next_id = 1
        self.pnl = 0.0
        self.hedge = False
        self.fail_stop = False
        self.done = {}
        self.client_ids = set()
        self.open_calls = 0
        self.fee = 0.0005          # 成交明細的手續費率
        self.trades = {}           # orderId → 成交明細（userTrades 的格式）
        self.next_order = 1000
        self.by_cid = {}           # client id → 市價單回應（check_first 查單用）

    def _trade(self, ps, qty, price):
        oid = self.next_order
        self.next_order += 1
        self.trades[oid] = [{"orderId": oid, "qty": str(qty), "price": str(price), "positionSide": ps,
                             "commission": str(qty * price * self.fee), "commissionAsset": "USDT"}]
        return oid

    # 行情
    def klines(self, symbol, interval, limit=500):
        return self.df

    def price(self, symbol):
        return self.px

    # 帳戶
    def balance_usdt(self):
        return 1000.0

    def positions(self):
        return {k: v for k, v in self.pos.items() if v > 1e-12}

    def hedge_mode(self):
        return self.hedge

    def set_hedge_mode(self):
        self.hedge = True

    def set_leverage(self, s, lev):
        pass

    def set_isolated(self, s):
        pass

    def market_order(self, symbol, side, ps, quantity, client_id, check_first=False):
        if check_first and client_id in self.by_cid:
            return self.by_cid[client_id]
        q = float(quantity)
        opening = (side == "BUY") == (ps == "LONG")
        if not opening and self.pos.get((symbol, ps), 0.0) < q - 1e-9:
            from grid_trading.trader.binance import BinanceError
            raise BinanceError(400, -2022, "ReduceOnly Order is rejected.")
        self.pos[(symbol, ps)] = self.pos.get((symbol, ps), 0.0) + (q if opening else -q)
        oid = self._trade(ps, q, self.px)
        self.by_cid[client_id] = {"executedQty": quantity, "avgPrice": str(self.px), "orderId": oid}
        return self.by_cid[client_id]

    def conditional_order(self, symbol, side, ps, kind, trigger, quantity=None, client_id=None):
        if kind == "STOP_MARKET" and self.fail_stop:
            from grid_trading.trader.binance import BinanceError
            raise BinanceError(400, -2021, "Order would immediately trigger.")
        if client_id in self.client_ids:
            from grid_trading.trader.binance import BinanceError
            raise BinanceError(400, -4116, "ClientOrderId is duplicated.")
        if quantity is not None and float(quantity) > self.pos.get((symbol, ps), 0.0) + 1e-9:
            from grid_trading.trader.binance import BinanceError
            raise BinanceError(400, -2022, "ReduceOnly Order is rejected.")
        self.client_ids.add(client_id)  # 通過檢查、真的下單後才算用掉這個 id
        oid = self.next_id
        self.next_id += 1
        self.algo[oid] = {"algoId": oid, "symbol": symbol, "ps": ps, "type": kind, "trigger": float(trigger),
                          "qty": None if quantity is None else float(quantity)}
        return {"algoId": oid}

    def cancel_conditional(self, oid):
        o = self.algo.pop(oid, None)
        if o:
            self.done[oid] = {**o, "algoStatus": "CANCELED"}

    def algo_order(self, oid):
        o = self.done[oid]
        return {"algoStatus": o["algoStatus"], "triggerPrice": str(o["trigger"]),
                "actualPrice": str(o.get("fill") or 0), "actualOrderId": o.get("actualOrderId")}

    def order_trades(self, symbol, order_id):
        return self.trades.get(order_id, [])

    def open_conditionals(self, symbol=None):
        self.open_calls += 1
        return [o for o in self.algo.values() if symbol is None or o["symbol"] == symbol]

    def realized_since(self, symbol, ps, start):
        return self.pnl

    def trigger(self, kind, n=0, oid=None, fill=None):
        """讓第 n 張某類條件單（或指定 id）成交（fill：實際成交價，預設是觸發價）。
        只會減倉：數量比交易所上的合併倉位大時只平到倉位為止。"""
        o = self.algo[oid] if oid else [o for o in self.algo.values() if o["type"] == kind][n]
        key = (o["symbol"], o["ps"])
        amt = self.pos[key] if o["qty"] is None else min(o["qty"], self.pos[key])
        self.pos[key] -= amt
        del self.algo[o["algoId"]]
        aid = self._trade(o["ps"], amt, fill or o["trigger"])
        self.done[o["algoId"]] = {**o, "algoStatus": "FINISHED", "fill": fill, "actualOrderId": aid}


def make_signal(side=LONG, price=100.0, risk=4.0, interval="2H", t=T0):
    d = 1 if side == LONG else -1
    return Signal("XUSDT", interval, side, t, price, price - d * risk, [price + d * k * risk for k in (1, 3, 5)],
                  "vegas_trend", {"atr": risk / 1.5}, {"portions": [0.25, 0.25, 0.25], "trail": 8.0, "trail_atr": 3})


def make_trader(tmp_path, broker, ex, rules=RiskRules()):
    sink = []
    cfg = Config(mode="testnet", state_dir=tmp_path, coins=["X"], rules=rules)
    tr = Trader(cfg, ex, broker, Notifier("test", sink=sink), {"XUSDT": F})
    return tr, sink


# ---------- 實盤 / 測試網 ----------
def test_live_open_places_stop_and_three_tps(tmp_path):
    ex = FakeExchange(price=101.0)
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pos = next(iter(tr.store.positions.values()))
    assert ex.hedge
    # 價位以成交價 101 重算：止損距離照訊號比例 4%
    assert pos["entry"] == 101.0 and pos["stop"] == pytest.approx(101 - 4.04)
    assert pos["tps"] == pytest.approx([101 + 4.04, 101 + 3 * 4.04, 101 + 5 * 4.04])
    kinds = sorted(o["type"] for o in ex.algo.values())
    assert kinds == ["STOP_MARKET"] + ["TAKE_PROFIT_MARKET"] * 3
    stop = next(o for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    assert stop["qty"] == pytest.approx(pos["qty_f"])  # 止損用這張單自己的數量
    assert "已開倉" in sink[-1]


def test_live_moves_stop_after_each_tp_then_trails(tmp_path):
    rows = [(101, 99, 100)] * 3
    ex = FakeExchange(price=100.0, df=candles(rows))
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    stop_of = lambda: next(o["trigger"] for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    now = T0 + pd.Timedelta(minutes=30)

    ex.trigger("TAKE_PROFIT_MARKET", 0)
    tr.step_manage(now)
    assert stop_of() == pytest.approx(100 * 1.001)  # 保本 = 進場價 + 來回手續費
    ex.trigger("TAKE_PROFIT_MARKET", 0)
    tr.step_manage(now)
    assert stop_of() == pytest.approx(104.0)        # TP2 後移到 TP1
    assert len([o for o in ex.algo.values() if o["type"] == "STOP_MARKET"]) == 1

    ex.trigger("TAKE_PROFIT_MARKET", 0)
    tr.step_manage(now)
    assert tr.store.positions[pid]["tps_hit"] == 3
    # 下一根收盤後：TP3 那根起最高 130 → 追蹤止損 130 − 8 = 122
    ex.df = candles([(101, 99, 100), (130, 118, 125)], start=T0)
    tr.step_manage(T0 + pd.Timedelta(hours=4, seconds=20))
    assert stop_of() == pytest.approx(122.0)
    assert any("追蹤止損移到 122" in m for m in sink)

    stop = next(o for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    assert stop["qty"] == pytest.approx(0.625 - 3 * 0.156)  # 最後剩下的那一份
    ex.trigger("STOP_MARKET")
    tr.step_manage(T0 + pd.Timedelta(hours=5))
    assert not tr.store.positions and not ex.algo and not ex.positions()
    q, rest = 0.156, 0.625 - 3 * 0.156
    gross = q * 4 + q * 12 + q * 20 + rest * 22
    fees = 0.0005 * (100 * 0.625 + q * (104 + 112 + 120) + rest * 122)
    assert tr.store.ledger[-1]["pnl"] == pytest.approx(gross - fees)
    assert tr.store.ledger[-1]["r"] == pytest.approx((gross - fees) / 2.5)


def test_live_stop_failure_closes_position(tmp_path):
    ex = FakeExchange()
    ex.fail_stop = True
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    assert not tr.store.positions and not ex.positions()
    assert "已市價平倉" in sink[-1]


def test_one_per_coin_and_interval(tmp_path):
    ex = FakeExchange()
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(side=SHORT, interval="2H"))     # 同幣同週期：一次一張
    assert len(tr.store.positions) == 1 and "已有做多單" in sink[-1]
    tr.try_open(make_signal(interval="4H"))                 # 另一個週期同方向：可以（2026-10-20 起）
    assert len(tr.store.positions) == 2


def test_two_same_side_positions_are_managed_separately(tmp_path):
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(interval="4H", t=T0 + pd.Timedelta(hours=4)))
    p2h, p4h = sorted(tr.store.positions.values(), key=lambda p: p["interval"])
    assert ex.positions()[("XUSDT", "LONG")] == pytest.approx(1.25)  # 幣安上合成一個倉位
    ex.trigger("STOP_MARKET", oid=p2h["stop_id"])           # 只有 2H 那張止損
    tr.step_manage(T0 + pd.Timedelta(hours=1))
    assert list(tr.store.positions) == [p4h["id"]]
    assert ex.positions()[("XUSDT", "LONG")] == pytest.approx(0.625)
    assert tr.store.ledger[-1]["interval"] == "2H" and tr.store.ledger[-1]["pnl"] < 0
    assert sum(o["type"] == "TAKE_PROFIT_MARKET" for o in ex.algo.values()) == 3  # 2H 的止盈已取消


def test_max_open_and_pause(tmp_path):
    ex = FakeExchange()
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex, RiskRules(max_open=0))
    tr.try_open(make_signal())
    assert not tr.store.positions and "上限" in sink[-1]
    tr.cfg.rules = RiskRules()
    (tmp_path / "pause").touch()
    tr.try_open(make_signal())
    assert not tr.store.positions and "暫停" in sink[-1]


def test_reconcile_replaces_missing_stop(tmp_path):
    ex = FakeExchange()
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    stop_id = next(o["algoId"] for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    ex.cancel_conditional(stop_id)
    broker.reconcile(tr.store, tr.notify)
    assert any(o["type"] == "STOP_MARKET" for o in ex.algo.values())
    assert "補掛" in sink[-1]


# ---------- 紙上交易 = 回測 ----------
def test_paper_matches_tracker(tmp_path):
    rows = [(101, 99, 100), (105, 99, 104), (113, 103, 112), (121, 111, 120), (121, 110, 112)]
    ex = FakeExchange(df=candles(rows))
    cfg = Config(mode="paper", state_dir=tmp_path, coins=["X"])
    broker = PaperBroker(ex, cfg)
    sink = []
    tr = Trader(cfg, ex, broker, Notifier("paper", sink=sink), {"XUSDT": F})
    sig = make_signal()
    tr.try_open(sig)
    tr.step_manage(T0 + pd.Timedelta(hours=10))
    expect = track(sig, candles(rows), 0.0005, 240)
    assert expect.status == "CLOSED"
    row = tr.store.ledger[-1]
    pos_risk = 0.062 * 4.0  # 權益 100U × 0.25% ÷ 4，捨去到 0.001
    assert row["pnl"] == pytest.approx(expect.realized_r * pos_risk)
    assert row["r"] == pytest.approx(row["pnl"] / 0.25)
    assert broker.equity(tr.store) == pytest.approx(100 + row["pnl"])


# ---------- 進場時間 ----------
def test_first_run_does_not_trade_old_bars(tmp_path):
    ex = FakeExchange()
    tr, _ = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    now = T0 + pd.Timedelta(minutes=1)
    assert tr.due(now) == []                               # 第一次只記時間
    assert tr.due(now + pd.Timedelta(hours=2)) == ["2H"]   # 下一根 2H 收盤
    assert tr.due(now + pd.Timedelta(hours=4)) == ["2H", "4H"]
    assert bar_close(T0 + pd.Timedelta(hours=3), "4H") == pd.Timestamp("2026-10-19 08:00", tz="UTC")


def test_leverage_lowered_for_wide_stops(tmp_path):
    ex = FakeExchange()
    calls = []
    ex.set_leverage = lambda s, lev: calls.append(lev)
    tr, _ = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(risk=4.0))                       # 止損 4% → 5 倍
    tr.try_open(make_signal(risk=16.0, interval="4H"))       # 止損 16% → 3 倍（強平在約 33% 外）
    assert calls == [5, 3]
    assert tr.store.leverage == {"XUSDT": 3}


def test_reconcile_migrates_old_positions(tmp_path):
    ex = FakeExchange()
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    pos = next(iter(tr.store.positions.values()))
    # 模擬舊版的單：沒有 remaining / fills，止損是全部平倉（沒有數量）
    del pos["remaining"], pos["fills"]
    ex.algo[pos["stop_id"]]["qty"] = None
    pos["tps_hit"] = 1
    broker.reconcile(tr.store, tr.notify)
    assert pos["remaining"] == pytest.approx(0.625 - 0.156)
    assert pos["fills"] == [[pytest.approx(0.156), pytest.approx(104.0)]]
    stops = [o for o in ex.algo.values() if o["type"] == "STOP_MARKET"]
    assert len(stops) == 1 and stops[0]["qty"] == pytest.approx(0.625 - 0.156)


def test_same_bar_different_symbols_get_unique_client_ids(tmp_path):
    import dataclasses
    ex = FakeExchange()
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.filters["YUSDT"] = F
    a = make_signal(side=SHORT)
    b = dataclasses.replace(a, symbol="YUSDT")
    tr.try_open(a)
    tr.try_open(b)
    assert len(tr.store.positions) == 2, sink
    assert not any("失敗" in m for m in sink)
    assert len(ex.client_ids) == 8  # 每張單 1 止損 + 3 止盈


def test_stop_triggering_waits_and_crossed_stop_closes_at_market(tmp_path):
    from grid_trading.trader.binance import BinanceError
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    pos = tr.store.positions[pid]
    now = T0 + pd.Timedelta(minutes=30)
    # 止損剛觸發：已不在掛單清單，狀態 TRIGGERING，不能當成「不見了」去補掛
    o = ex.algo.pop(pos["stop_id"])
    ex.done[pos["stop_id"]] = {**o, "algoStatus": "TRIGGERING"}
    tr.step_manage(now)
    assert pid in tr.store.positions and not any("失敗" in m for m in sink)
    assert len([o for o in ex.algo.values() if o["type"] == "STOP_MARKET"]) == 0
    # 狀態是 CANCELED（真的不見了），補掛時價格已越過止損（-2021）：市價平倉，不要一直失敗
    ex.done[pos["stop_id"]]["algoStatus"] = "CANCELED"
    real = ex.conditional_order

    def crossed(symbol, side, ps, kind, trigger, quantity=None, client_id=None):
        if kind == "STOP_MARKET":
            raise BinanceError(400, -2021, "Order would immediately trigger.")
        return real(symbol, side, ps, kind, trigger, quantity, client_id)

    ex.conditional_order = crossed
    tr.step_manage(now)
    assert pid not in tr.store.positions and not any("失敗" in m for m in sink)
    assert any("市價平倉" in m for m in sink) and len(tr.store.ledger) == 1


def test_missing_stop_after_tp_fill_is_replaced_with_reduced_quantity(tmp_path):
    """止盈成交後，交易所把原本全數量的止損單取消了：要先處理止盈，再用剩下的數量重掛，不能 -2022。"""
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    pos = tr.store.positions[pid]
    ex.trigger("TAKE_PROFIT_MARKET", 0)
    ex.cancel_conditional(pos["stop_id"])
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    assert not any("失敗" in m for m in sink), sink
    assert pos["tps_hit"] == 1 and pos["stop_id"] in ex.algo
    assert ex.algo[pos["stop_id"]]["qty"] == pytest.approx(pos["remaining"])


def test_missing_stop_on_flat_position_is_booked_not_raised(tmp_path):
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    pos = tr.store.positions[pid]
    ex.pos[(pos["symbol"], pos["side"])] = 0.0   # 倉位被別的單平掉了
    ex.cancel_conditional(pos["stop_id"])
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    assert pid not in tr.store.positions and len(tr.store.ledger) == 1
    assert not any("失敗" in m for m in sink), sink
    assert any("已經沒有這個倉位" in m and "CANCELED" in m for m in sink)


def test_client_tag_is_ascii_and_unique_for_non_ascii_symbols():
    import dataclasses
    import re
    from grid_trading.trader.bot import client_tag
    a = make_signal(side=LONG)
    tags = {client_tag(dataclasses.replace(a, symbol=sym))
            for sym in ("APTUSDT", "1000PEPEUSDT", "币安人生USDT", "我踏马来了USDT", "SOMEVERYLONGSYMBOLNAMEUSDT")}
    assert len(tags) == 5
    for t in tags:
        assert re.fullmatch(r"[.A-Za-z0-9:/_-]{1,30}", t), t
        assert re.fullmatch(r"[.A-Za-z0-9:/_-]{1,36}", t + "s12"), t


def test_live_open_with_non_ascii_symbol_uses_legal_client_ids(tmp_path):
    import dataclasses
    import re
    ex = FakeExchange()
    ids = []
    real_market, real_cond = ex.market_order, ex.conditional_order

    def market(symbol, side, ps, quantity, client_id):
        ids.append(client_id)
        return real_market(symbol, side, ps, quantity, client_id)

    def cond(symbol, side, ps, kind, trigger, quantity=None, client_id=None):
        ids.append(client_id)
        return real_cond(symbol, side, ps, kind, trigger, quantity, client_id)

    ex.market_order, ex.conditional_order = market, cond
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.filters["币安人生USDT"] = F
    tr.try_open(dataclasses.replace(make_signal(), symbol="币安人生USDT"))
    assert len(tr.store.positions) == 1, sink
    assert ids and all(re.fullmatch(r"[.A-Za-z0-9:/_-]{1,36}", i) for i in ids), ids


def test_binance_client_clean_id():
    from grid_trading.trader.binance import clean_id
    assert clean_id("v2H1L币安人生s1") == "v2H1L____s1"
    assert all(c.isascii() for c in clean_id("币安人生"))
    assert len(clean_id("a" * 50)) == 36 and clean_id(None) is None


def _client_with(status, retry_after, monkeypatch):
    from grid_trading.trader import binance as bmod

    class Resp:
        headers = {"Retry-After": retry_after}
        text = "x"

        def __init__(self):
            self.status_code = status

        def json(self):
            return {"code": -1003, "msg": "x"}

    class Sess:
        calls = 0

        def request(self, *a, **k):
            Sess.calls += 1
            return Resp()

    waits = []
    monkeypatch.setattr(bmod.time, "sleep", waits.append)
    return bmod, bmod.BinanceFutures(key="k", secret="s", session=Sess()), Sess, waits


def test_binance_client_418_fails_fast_and_stops_sending_while_banned(monkeypatch):
    """418 帶 Retry-After 幾小時：不睡（單執行緒迴圈會整個停擺、日誌沒有一行），也不在封鎖期間繼續送簽名請求。"""
    bmod, cli, sess, waits = _client_with(418, "10800", monkeypatch)
    with pytest.raises(bmod.BinanceError):
        cli.open_conditionals("BTCUSDT")
    assert sess.calls == 1 and not waits
    with pytest.raises(bmod.BinanceError) as e:
        cli.positions()
    assert e.value.status == 418 and sess.calls == 1   # 封鎖期間沒有再送出去


def test_binance_client_429_waits_at_most_max_wait(monkeypatch):
    bmod, cli, sess, waits = _client_with(429, "10800", monkeypatch)
    with pytest.raises(bmod.BinanceError):
        cli.exchange_info()
    assert waits and max(waits) <= bmod.MAX_WAIT, waits


def test_stop_quantity_above_live_position_is_capped_and_reported(tmp_path):
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    pos = tr.store.positions[pid]
    ex.pos[(pos["symbol"], pos["side"])] = pos["remaining"] * 0.5   # 交易所倉位比記錄少
    ex.cancel_conditional(pos["stop_id"])
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    assert pos["stop_id"] in ex.algo
    assert ex.algo[pos["stop_id"]]["qty"] == pytest.approx(pos["remaining"] * 0.5, abs=0.001)
    assert any("交易所上只有" in m for m in sink), sink


def test_startup_survives_position_that_is_already_flat_on_exchange(tmp_path):
    """APT 的情況：程式記錄還有倉位、止損 id 是空的，交易所上已經空倉。啟動不能當掉，第一輪管理要結清。"""
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    pos = tr.store.positions[pid]
    for oid in [pos["stop_id"]] + pos["tp_ids"]:
        ex.cancel_conditional(oid)
    ex.pos[(pos["symbol"], pos["side"])] = 0.0
    pos["stop_id"] = None
    pos["tps_hit"] = 3
    pos["remaining"] = pos["slices"][3]
    tr.broker.reconcile(tr.store, tr.notify)   # 不能丟例外
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    assert pid not in tr.store.positions and len(tr.store.ledger) == 1
    assert any("已經沒有這個倉位" in m for m in sink), sink


def test_manage_reads_all_open_orders_once_per_poll(tmp_path):
    import dataclasses
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    for sym in ("YUSDT", "ZUSDT"):
        tr.filters[sym] = F
    sig = make_signal()
    for sym in ("XUSDT", "YUSDT", "ZUSDT"):
        tr.try_open(dataclasses.replace(sig, symbol=sym))
    assert len(tr.store.positions) == 3, sink
    ex.open_calls = 0
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    assert ex.open_calls == 1   # 三個幣、一個請求


def test_paper_manage_skips_recheck_within_interval(tmp_path):
    sink = []
    df = candles([(101, 99, 100)] * 3)
    cfg = Config(mode="paper", state_dir=tmp_path, coins=["X"])
    calls = []

    class Data:
        def klines(self, symbol, interval, limit=500):
            calls.append(1)
            return df

        def price(self, symbol):
            return 100.0

    tr = Trader(cfg, Data(), PaperBroker(Data(), cfg), Notifier("test", sink=sink), {"XUSDT": F})
    tr.try_open(make_signal())
    now = T0 + pd.Timedelta(minutes=30)
    tr.step_manage(now)
    tr.step_manage(now + pd.Timedelta(seconds=60))
    assert len(calls) == 1
    tr.step_manage(now + pd.Timedelta(seconds=cfg.paper_check + 1))
    assert len(calls) == 2


# ---------- 下單不盲目重送 ----------
class _Resp:
    def __init__(self, status, body):
        self.status_code, self.body = status, body
        self.headers = {}
        self.text = str(body)

    def json(self):
        return self.body


class _Script:
    """照順序回應的假 session：每一步是 (HTTP 狀態, 內容) 或要丟出的例外。"""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), []

    def request(self, method, url, params=None, headers=None, timeout=None):
        self.calls.append((method, url.split("/fapi")[-1], dict(params or {})))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return _Resp(*step)


def _scripted(steps, monkeypatch):
    from grid_trading.trader import binance as bmod
    monkeypatch.setattr(bmod.time, "sleep", lambda s: None)
    sess = _Script(steps)
    return bmod, bmod.BinanceFutures(key="k", secret="s", session=sess), sess


def _posts(sess):
    return [c for c in sess.calls if c[0] == "POST"]


def test_market_order_timeout_is_looked_up_not_resent(monkeypatch):
    """第一次其實成交了但回應掉了：查單查到就用它，絕不再送一張（會變成兩倍倉位）。"""
    import requests
    filled = {"orderId": 5, "status": "FILLED", "executedQty": "0.625", "avgPrice": "100"}
    bmod, cli, sess = _scripted([requests.ReadTimeout("x"), (200, filled)], monkeypatch)
    assert cli.market_order("XUSDT", "BUY", "LONG", "0.625", "v2Habe") == filled
    assert len(_posts(sess)) == 1
    assert sess.calls[1][:2] == ("GET", "/v1/order") and sess.calls[1][2]["origClientOrderId"] == "v2Habe"


def test_market_order_5xx_then_not_found_is_resent_once(monkeypatch):
    ok = {"orderId": 6, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}
    bmod, cli, sess = _scripted([(503, {"code": -1001, "msg": "x"}),
                                 (400, {"code": -2013, "msg": "Order does not exist."}),
                                 (200, ok)], monkeypatch)
    assert cli.market_order("XUSDT", "BUY", "LONG", "1", "v2Habe") == ok
    assert len(_posts(sess)) == 2


def test_market_order_unknown_when_lookup_also_fails(monkeypatch):
    import requests
    bmod, cli, sess = _scripted([requests.ReadTimeout("x")] + [requests.ConnectionError("y")] * 3, monkeypatch)
    with pytest.raises(bmod.OrderUnknown):
        cli.market_order("XUSDT", "BUY", "LONG", "1", "v2Habe")
    assert len(_posts(sess)) == 1


def test_conditional_order_lookup_by_client_algo_id(monkeypatch):
    bmod, cli, sess = _scripted([(502, {}), (200, {"algoId": 7, "algoStatus": "NEW"})], monkeypatch)
    r = cli.conditional_order("XUSDT", "SELL", "LONG", "STOP_MARKET", "96", quantity="1", client_id="v2Habs1")
    assert r["algoId"] == 7 and len(_posts(sess)) == 1
    assert sess.calls[1][2]["clientAlgoId"] == "v2Habs1"


def test_conditional_order_duplicate_on_resend_means_already_placed(monkeypatch):
    import requests
    bmod, cli, sess = _scripted([requests.ReadTimeout("x"),
                                 (400, {"code": -2013, "msg": "Order does not exist."}),   # 還查不到
                                 (400, {"code": -4116, "msg": "ClientOrderId is duplicated."}),
                                 (200, {"algoId": 9, "algoStatus": "NEW"})], monkeypatch)
    r = cli.conditional_order("XUSDT", "SELL", "LONG", "STOP_MARKET", "96", quantity="1", client_id="v2Habs1")
    assert r["algoId"] == 9 and len(_posts(sess)) == 2


def test_order_post_429_is_still_retried(monkeypatch):
    """429 是被拒絕（沒有執行），可以直接重送。"""
    ok = {"orderId": 6, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}
    bmod, cli, sess = _scripted([(429, {"code": -1003, "msg": "x"}), (200, ok)], monkeypatch)
    assert cli.market_order("XUSDT", "BUY", "LONG", "1", "v2Habe") == ok
    assert len(_posts(sess)) == 2 and len(sess.calls) == 2


def test_entry_with_unknown_result_is_reported_and_hourly_audit_flags_once(tmp_path):
    from grid_trading.trader.binance import OrderUnknown
    from grid_trading.trader.bot import client_tag
    ex = FakeExchange()
    real = ex.market_order

    def lost(*a, **k):
        real(*a, **k)   # 交易所其實成交了，回應掉了
        raise OrderUnknown("/fapi/v1/order 送出後沒有回應")

    ex.market_order = lost
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    sig = make_signal()
    tr.try_open(sig)
    assert not tr.store.positions
    msg = sink[-1]
    assert "下單結果不明" in msg and "XUSDT" in msg and "LONG" in msg and f"{client_tag(sig)}e" in msg
    assert "開倉失敗" not in msg
    now = T0 + pd.Timedelta(minutes=5)
    tr.step_audit(now)
    assert "比這個程式記錄的多" in sink[-1]
    n = len(sink)
    tr.step_audit(now + pd.Timedelta(minutes=30))   # 還不到一小時：不查
    tr.step_audit(now + pd.Timedelta(hours=2))      # 同一個差異：不再推
    assert len(sink) == n
    ex.pos[("XUSDT", "LONG")] = 1.0                  # 數量變了：再推一次
    tr.step_audit(now + pd.Timedelta(hours=4))
    assert len(sink) == n + 1 and "比這個程式記錄的多" in sink[-1]


# ---------- 止損成交前先處理止盈 ----------
def test_tp_fill_is_booked_before_stop_fill(tmp_path):
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    ex.trigger("TAKE_PROFIT_MARKET", 0)    # TP1 成交後、程式還沒移止損，價格就回頭打到原始止損
    ex.trigger("STOP_MARKET")              # 止損是用 0.625 掛的，交易所只平到剩下的 0.469
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    assert not tr.store.positions and not ex.algo and not ex.positions()
    assert any("TP1 觸及" in m for m in sink)
    q, rest = 0.156, 0.625 - 0.156
    gross = q * 4 - rest * 4
    fees = 0.0005 * (100 * 0.625 + q * 104 + rest * 96)
    row = tr.store.ledger[-1]
    assert row["pnl"] == pytest.approx(gross - fees)
    assert row["pnl_source"] == "fills"


@pytest.mark.parametrize("tp_first", [True, False])
def test_oversized_exit_on_shared_position_is_taken_from_sibling(tmp_path, tp_first):
    """2H、4H 同幣同向合成一個倉位：2H 的止損用 TP 前的數量成交（或止損後 2H 剩下的止盈又成交），
    多平的部分其實是 4H 的數量，要從 4H 扣掉、4H 的止損改小，帳本總數和交易所一致。"""
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(interval="4H", t=T0 + pd.Timedelta(hours=4)))
    p2h, p4h = sorted(tr.store.positions.values(), key=lambda p: p["interval"])
    tp1 = p2h["tp_ids"][0]
    if tp_first:
        ex.trigger("", oid=tp1)
        ex.trigger("", oid=p2h["stop_id"])
    else:
        ex.trigger("", oid=p2h["stop_id"])
        ex.trigger("", oid=tp1)
    live = ex.positions()[("XUSDT", "LONG")]
    assert live == pytest.approx(0.625 - 0.156)
    tr.step_manage(T0 + pd.Timedelta(hours=5))
    assert list(tr.store.positions) == [p4h["id"]]
    assert p4h["remaining"] == pytest.approx(live)
    assert p4h["fills"] == [[pytest.approx(0.156), pytest.approx(96.0)]]
    assert ex.algo[p4h["stop_id"]]["qty"] == pytest.approx(live)      # 4H 止損數量跟著改小
    assert sum(o["type"] == "TAKE_PROFIT_MARKET" for o in ex.algo.values()) == 3  # 只剩 4H 的止盈
    assert any("多平了" in m for m in sink), sink
    row = tr.store.ledger[-1]
    assert row["interval"] == "2H" and row["pnl_source"] == "estimate"  # 2H 的訂單平了比自己多：不用成交明細


def test_close_with_unknown_result_is_looked_up_before_resending(tmp_path):
    """超時平倉的市價單結果不明：下一輪先查單，不能再送一次（會平到同幣同向的 4H）。"""
    from grid_trading.trader.binance import OrderUnknown
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(interval="4H", t=T0 + pd.Timedelta(hours=4)))
    real = ex.market_order
    calls = []

    def flaky(*a, **k):
        calls.append(k.get("check_first", False))
        r = real(*a, **k)
        if len(calls) == 1:
            raise OrderUnknown("timeout")
        return r

    ex.market_order = flaky
    now = T0 + pd.Timedelta(hours=480, minutes=1)   # 2H 超過 240 根，4H 還沒
    tr.step_manage(now)
    assert any("管理失敗" in m for m in sink)
    tr.step_manage(now + pd.Timedelta(minutes=1))
    assert calls == [False, True]
    assert len(tr.store.positions) == 1 and tr.store.ledger[-1]["interval"] == "2H"
    assert ex.positions()[("XUSDT", "LONG")] == pytest.approx(0.625)   # 4H 的數量沒被動到


# ---------- 帳本用實際成交 ----------
def test_ledger_uses_actual_fills_and_commission(tmp_path):
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    ex.fee = 0.0002
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pos = next(iter(tr.store.positions.values()))
    ex.trigger("", oid=pos["stop_id"], fill=95.5)   # 止損滑價
    tr.step_manage(T0 + pd.Timedelta(minutes=30))
    row = tr.store.ledger[-1]
    real = 0.625 * (95.5 - 100) - 0.0002 * (0.625 * 100 + 0.625 * 95.5)
    est = 0.625 * (95.5 - 100) - 0.0005 * (0.625 * 100 + 0.625 * 95.5)
    assert row["pnl_source"] == "fills" and row["pnl"] == pytest.approx(real)
    assert row["pnl_est"] == pytest.approx(est)
    assert row["r"] == pytest.approx(real / 2.5)


def test_ledger_falls_back_to_estimate(tmp_path):
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(interval="4H", t=T0 + pd.Timedelta(hours=4)))
    p2h, p4h = sorted(tr.store.positions.values(), key=lambda p: p["interval"])
    del p4h["entry_order"]                           # 舊狀態檔的單：沒有進場單 id

    def broken(symbol, order_id):
        raise RuntimeError("HTTP 500")

    ex.order_trades = broken
    ex.trigger("", oid=p2h["stop_id"])
    ex.trigger("", oid=p4h["stop_id"])
    tr.step_manage(T0 + pd.Timedelta(hours=5))
    assert not tr.store.positions
    assert [r["pnl_source"] for r in tr.store.ledger] == ["estimate", "estimate"]
    assert all("pnl_est" not in r for r in tr.store.ledger)
    assert tr.store.ledger[0]["pnl"] == pytest.approx(-0.625 * 4 - 0.0005 * 0.625 * 196)


# ---------- 槓桿 ----------
def test_leverage_refused_with_open_position_keeps_entry_and_warns(tmp_path):
    from grid_trading.trader.binance import BinanceError
    ex = FakeExchange()
    calls = []

    def set_lev(s, lev):
        calls.append(lev)
        if ex.positions() and lev < 5:
            raise BinanceError(400, -4161, "Leverage reduction is not supported in Isolated Margin Mode with open positions")

    ex.set_leverage = set_lev
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(risk=4.0))
    tr.try_open(make_signal(risk=16.0, interval="4H"))
    assert calls == [5, 3] and len(tr.store.positions) == 2, sink
    assert not any("開倉失敗" in m for m in sink)
    assert any("沿用 5 倍" in m for m in sink), sink
    assert tr.store.leverage == {"XUSDT": 5}


def test_leverage_never_raised_above_what_open_positions_allow(tmp_path):
    ex = FakeExchange()
    calls = []
    ex.set_leverage = lambda s, lev: calls.append(lev)
    tr, _ = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(risk=16.0, interval="4H"))   # 3 倍
    tr.try_open(make_signal(risk=4.0, interval="2H"))    # 自己可以 5 倍，但 4H 還開著：維持 3 倍
    assert calls == [3] and len(tr.store.positions) == 2
    tr.store.positions.clear()
    tr.try_open(make_signal(risk=4.0, interval="2H", t=T0 + pd.Timedelta(hours=2)))  # 沒有其他倉位：回到 5 倍
    assert calls == [3, 5]


# ---------- 追蹤止損和回測一致 ----------
def test_trailing_starts_on_tp3_like_tracker(tmp_path):
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pos = next(iter(tr.store.positions.values()))
    now = T0 + pd.Timedelta(minutes=30)
    for _ in range(3):
        ex.trigger("TAKE_PROFIT_MARKET", 0)
        tr.step_manage(now)
    stop = ex.algo[pos["stop_id"]]
    assert stop["trigger"] == pytest.approx(120 - 8)          # TP3 那根就從 TP3 往回 trail，不等收盤
    assert stop["qty"] == pytest.approx(0.625 - 3 * 0.156)
    assert any("追蹤止損移到 112" in m for m in sink)


def test_trailing_on_tp3_never_moves_stop_backwards(tmp_path):
    import dataclasses
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    sig = make_signal()
    tr.try_open(dataclasses.replace(sig, exit={**sig.exit, "trail": 20.0}))   # 120 − 20 = 100，比 TP1 104 差
    pos = next(iter(tr.store.positions.values()))
    now = T0 + pd.Timedelta(minutes=30)
    for _ in range(3):
        ex.trigger("TAKE_PROFIT_MARKET", 0)
        tr.step_manage(now)
    stop = ex.algo[pos["stop_id"]]
    assert stop["trigger"] == pytest.approx(104.0)
    assert stop["qty"] == pytest.approx(0.625 - 3 * 0.156)
    assert not any("追蹤止損移到" in m for m in sink)


def test_stop_move_that_failed_after_tp_is_retried_next_poll(tmp_path):
    ex = FakeExchange(price=100.0, df=candles([(101, 99, 100)] * 3))
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    stop_of = lambda: next(o["trigger"] for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    now = T0 + pd.Timedelta(minutes=30)
    old_stop = stop_of()

    ex.trigger("TAKE_PROFIT_MARKET", 0)
    real = ex.conditional_order
    ex.conditional_order = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("斷線"))
    tr.step_manage(now)   # TP1 記到了，但移止損時斷線
    assert tr.store.positions[pid]["tps_hit"] == 1 and stop_of() == old_stop
    ex.conditional_order = real
    tr.step_manage(now)   # 下一輪補移到保本，數量也改成剩下的
    assert stop_of() == pytest.approx(100 * 1.001)
    stops = [o for o in ex.algo.values() if o["type"] == "STOP_MARKET"]
    assert len(stops) == 1 and stops[0]["qty"] == pytest.approx(tr.store.positions[pid]["remaining"])


# ---------- 可靠性：警報、心跳、推播佇列、掃描補抓、狀態檔保護 ----------
import importlib.util
import json
import time
from pathlib import Path

from grid_trading.trader import bot as botmod
from grid_trading.trader.binance import BinanceError
from grid_trading.trader.bot import StateCorrupt, Store

NOW = T0 + pd.Timedelta(minutes=30)
CLK = 1_800_000_000.0   # 假時鐘的起點（Unix 秒）


def _clocked(tmp_path):
    """有一張持倉的測試網 trader，時鐘可以手動撥。"""
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    ex.banned_until = 0.0
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    sink.clear()
    clock = [CLK]
    tr._clock = lambda: clock[0]
    return tr, ex, sink, clock


def _break_exchange(ex, exc):
    def boom(*a, **k):
        raise exc
    ex.open_conditionals = boom


def test_ban_alert_once_after_grace_then_recovered(tmp_path):
    tr, ex, sink, clock = _clocked(tmp_path)
    ex.banned_until = CLK + 3600
    _break_exchange(ex, BinanceError(418, -1003, "被幣安暫時封鎖"))
    for dt in (0, 300, 599):
        clock[0] = CLK + dt
        tr.run_once(NOW)
    assert not any("警報" in m for m in sink)
    for dt in (601, 660, 900):  # 超過 10 分鐘只推一次
        clock[0] = CLK + dt
        tr.run_once(NOW)
    alerts = [m for m in sink if "警報" in m]
    assert len(alerts) == 1 and "418" in alerts[0] and "UTC 解封" in alerts[0]
    hb = json.loads((tmp_path / "testnet.heartbeat.json").read_text())
    assert "418" in hb["problem"] and hb["last_manage_ok"] is None and hb["last_loop"] == int((CLK + 900) * 1000)
    # 恢復：推一次「已恢復」和中斷多久，心跳的問題清掉
    del ex.open_conditionals
    ex.banned_until = 0.0
    clock[0] = CLK + 1260
    tr.run_once(NOW)
    clock[0] = CLK + 1320
    tr.run_once(NOW)
    rec = [m for m in sink if "已恢復" in m]
    assert len(rec) == 1 and "21 分鐘" in rec[0]
    hb = json.loads((tmp_path / "testnet.heartbeat.json").read_text())
    assert hb["problem"] is None and hb["last_manage_ok"] == int((CLK + 1320) * 1000)


def test_short_outage_is_silent(tmp_path):
    tr, ex, sink, clock = _clocked(tmp_path)
    _break_exchange(ex, RuntimeError("斷線"))
    tr.step_manage(NOW)
    del ex.open_conditionals
    clock[0] = CLK + 120
    tr.step_manage(NOW)
    assert not any("警報" in m or "已恢復" in m for m in sink)


def test_repeated_manage_failure_alerts_with_reason(tmp_path):
    tr, ex, sink, clock = _clocked(tmp_path)
    _break_exchange(ex, RuntimeError("斷線了"))
    for dt in (0, 400, 700, 800):
        clock[0] = CLK + dt
        tr.step_manage(NOW)
    alerts = [m for m in sink if "警報" in m]
    assert len(alerts) == 1 and "斷線了" in alerts[0]


def test_scan_failure_does_not_skip_manage_and_alerts(tmp_path):
    tr, ex, sink, clock = _clocked(tmp_path)

    def bad(now):
        raise RuntimeError("掃描炸了")
    tr.step_entries = bad
    for dt in (0, 700):
        clock[0] = CLK + dt
        tr.run_once(NOW)
    assert tr.last_manage_ok == CLK + 700   # 掃描失敗，持倉管理照跑
    alerts = [m for m in sink if "警報" in m]
    assert len(alerts) == 1 and "掃描炸了" in alerts[0]
    tr.step_entries = lambda now: None
    clock[0] = CLK + 760
    tr.run_once(NOW)
    assert sum("已恢復" in m for m in sink) == 1


def test_heartbeat_file_written_each_loop(tmp_path):
    tr, ex, sink, clock = _clocked(tmp_path)
    tr.run_once(NOW)
    hb = json.loads((tmp_path / "testnet.heartbeat.json").read_text())
    assert hb == {"last_loop": int(CLK * 1000), "last_manage_ok": int(CLK * 1000), "problem": None,
                  "stream": "off", "stream_last_event": None}
    assert not list(tmp_path.glob("*.tmp"))


def test_feed_includes_heartbeat_or_null(tmp_path):
    spec = importlib.util.spec_from_file_location("trader_feed", Path(__file__).parent.parent / "deploy" / "trader_feed.py")
    feed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feed)
    (tmp_path / "testnet.json").write_text(json.dumps({"positions": {}, "ledger": [], "processed": {}}))
    (tmp_path / "paper.json").write_text(json.dumps({"positions": {}, "ledger": [], "processed": {}}))
    (tmp_path / "testnet.heartbeat.json").write_text(json.dumps(
        {"last_loop": 5, "last_manage_ok": 4, "problem": "x", "extra": 1}))
    books = feed.build(tmp_path)["books"]
    assert books["testnet"]["heartbeat"] == {"last_loop": 5, "last_manage_ok": 4, "problem": "x"}
    assert books["paper"]["heartbeat"] is None
    (tmp_path / "paper.heartbeat.json").write_text("{壞")
    assert feed.build(tmp_path)["books"]["paper"]["heartbeat"] is None


# ---------- Notifier 佇列 ----------
def _fake_send(monkeypatch, delay=0.0, fail_first=False):
    import signal_telegram
    got, state = [], {"n": 0}

    def send(messages, token, chat_id):
        state["n"] += 1
        time.sleep(delay)
        if fail_first and state["n"] == 1:
            raise RuntimeError("Telegram 掛了")
        got.extend(messages)
        return len(messages), 0
    monkeypatch.setattr(signal_telegram, "send", send)
    return got


def test_notifier_queues_in_order_without_blocking(monkeypatch):
    got = _fake_send(monkeypatch, delay=0.2)
    n = Notifier("p", "tok", "chat")
    t0 = time.monotonic()
    for i in range(3):
        n(f"m{i}")
    assert time.monotonic() - t0 < 0.3   # 呼叫端不等 Telegram
    assert n.flush(5) and got == ["［p］m0", "［p］m1", "［p］m2"]


def test_notifier_survives_failing_send(monkeypatch):
    got = _fake_send(monkeypatch, fail_first=True)
    n = Notifier("p", "tok", "chat")
    n("a")
    n("b")
    assert n.flush(5) and got == ["［p］b"]
    assert n._thread.is_alive()


def test_notifier_flush_is_bounded(monkeypatch):
    _fake_send(monkeypatch, delay=1.0)
    n = Notifier("p", "tok", "chat")
    n("a")
    t0 = time.monotonic()
    assert n.flush(0.2) is False and time.monotonic() - t0 < 0.8
    n.flush(5)


def test_notifier_sink_stays_synchronous():
    sink = []
    Notifier("p", "tok", "chat", sink=sink)("x")
    assert sink == ["［p］x"]


def test_run_once_mode_flushes_notifier(tmp_path, monkeypatch):
    got = _fake_send(monkeypatch, delay=0.1)
    ex = FakeExchange()
    cfg = Config(mode="testnet", state_dir=tmp_path, coins=["X"])
    tr = Trader(cfg, ex, LiveBroker(ex, ex, cfg), Notifier("t", "tok", "chat"), {"XUSDT": F})
    tr.run(once=True)
    assert got and "啟動" in got[0]   # run 回來前已經送完


# ---------- 掃描補抓 ----------
def _scan_trader(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(botmod, "RETRY_PAUSE", 0)
    calls = {}

    class Data:
        def klines(self, symbol, interval, limit=500):
            calls[symbol] = calls.get(symbol, 0) + 1
            if bad(symbol, calls[symbol]):
                raise RuntimeError("timeout")
            return candles([(101, 99, 100)] * 3)   # 不到 250 根：沒訊號
    ex = FakeExchange()
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.data = Data()
    tr.symbols = {f"S{i}USDT": f"S{i}" for i in range(7)}
    return tr, sink, calls


def test_scan_retries_failed_symbols_once(tmp_path, monkeypatch):
    tr, sink, calls = _scan_trader(tmp_path, monkeypatch, lambda s, n: s == "S3USDT" and n == 1)
    tr.find_signals("2H", T0)
    assert calls["S3USDT"] == 2 and calls["S0USDT"] == 1 and not sink


def test_scan_notifies_once_with_count_and_five_names(tmp_path, monkeypatch):
    tr, sink, calls = _scan_trader(tmp_path, monkeypatch, lambda s, n: s != "S0USDT")
    tr.store.processed = {"2H": int((T0 - pd.Timedelta(hours=2)).timestamp() * 1000), "4H": int(T0.timestamp() * 1000)}
    tr.step_entries(T0 + pd.Timedelta(minutes=1))
    assert all(c == 2 for s, c in calls.items() if s != "S0USDT")   # 只補抓一次
    assert len(sink) == 1 and "6 個幣" in sink[0] and sink[0].count("/USDT") == 5
    assert tr.store.processed["2H"] == int(T0.timestamp() * 1000)   # 照樣記成已處理


# ---------- 狀態檔保護 ----------
def _saved_store(path, day, monkeypatch, mark="a"):
    monkeypatch.setattr(botmod, "_utc_day", lambda: day)
    st = Store(path)
    st.processed["2H"] = ord(mark)
    st.save()
    return st


def test_store_daily_backup_keeps_seven(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    st = _saved_store(path, "20261101", monkeypatch)
    st.processed["4H"] = 1
    st.save()   # 同一天第二次存檔：不再備份
    assert [p.name for p in st.backups()] == ["t.json.bak-20261101"]
    assert json.loads((tmp_path / "t.json.bak-20261101").read_text())["processed"] == {"2H": 97}
    for d in range(2, 10):
        monkeypatch.setattr(botmod, "_utc_day", lambda d=d: f"2026110{d}")
        st.processed["2H"] = d
        st.save()
    names = [p.name for p in st.backups()]
    assert len(names) == 7 and names[0] == "t.json.bak-20261109" and names[-1] == "t.json.bak-20261103"


def test_store_corrupt_uses_newest_backup_and_keeps_bad_file(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    _saved_store(path, "20261101", monkeypatch)
    (tmp_path / "t.json.bak-20261102").write_text(json.dumps({"positions": {"p": {"symbol": "X"}}}))
    (tmp_path / "t.json.bak-20261103").write_text("{壞")   # 最新的備份也壞：往前找
    path.write_text('{"positions": {"p"')
    st = Store(path)
    assert "p" in st.positions and len(st.notices) == 1 and "t.json.bak-20261102" in st.notices[0]
    kept = list(tmp_path.glob("t.json.corrupt-*"))
    assert len(kept) == 1 and kept[0].read_text() == '{"positions": {"p"'
    assert not path.exists()
    st.save()   # 還原後第一次存檔一定寫回正式檔
    assert json.loads(path.read_text())["positions"] == {"p": {"symbol": "X"}}


def test_store_corrupt_without_backup_refuses_empty_state(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("")
    with pytest.raises(StateCorrupt):
        Store(path)
    assert path.read_text() == ""   # 原檔不動，下次重開還是會擋住
    path.write_text("[1, 2]")       # 不是物件也算壞
    with pytest.raises(StateCorrupt):
        Store(path)


def test_trader_notifies_and_exits_on_unrecoverable_state_once(tmp_path):
    (tmp_path / "testnet.json").write_text("{壞")
    ex = FakeExchange()
    sink = []
    cfg = Config(mode="testnet", state_dir=tmp_path, coins=["X"])
    with pytest.raises(SystemExit):
        Trader(cfg, ex, LiveBroker(ex, ex, cfg), Notifier("t", sink=sink), {"XUSDT": F})
    assert len(sink) == 1 and "不能用空狀態" in sink[0]
    with pytest.raises(SystemExit):   # systemd 重開：同一個壞檔不再重複推播
        Trader(cfg, ex, LiveBroker(ex, ex, cfg), Notifier("t", sink=sink), {"XUSDT": F})
    assert len(sink) == 1


def test_trader_run_tells_user_about_backup_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(botmod, "_utc_day", lambda: "20261101")
    (tmp_path / "testnet.json.bak-20261101").write_text(json.dumps({"positions": {}, "processed": {"2H": 1}}))
    (tmp_path / "testnet.json").write_text("{壞")
    ex = FakeExchange()
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.run(once=True)
    assert any("改用備份" in m for m in sink)
    assert json.loads((tmp_path / "testnet.json").read_text())["processed"]["2H"] == 1


def test_store_fsyncs_tmp_before_rename(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"processed": {"2H": 1}}))
    st = Store(path)
    seen = []
    real = botmod.os.fsync
    monkeypatch.setattr(botmod.os, "fsync", lambda fd: (seen.append((path.read_text(), (tmp_path / "t.tmp").exists())), real(fd)))
    st.processed["2H"] = 2
    st.save()
    assert json.loads(seen[0][0])["processed"]["2H"] == 1 and seen[0][1]   # 改名前：舊檔還在、暫存檔已存在
    assert json.loads(path.read_text())["processed"]["2H"] == 2


def test_store_skips_rewrite_when_unchanged_but_writes_every_change(tmp_path, monkeypatch):
    writes = []
    real = botmod._write_atomic
    monkeypatch.setattr(botmod, "_write_atomic", lambda p, t: (writes.append(p.name), real(p, t)))
    st = Store(tmp_path / "t.json")
    st.save()
    st.save()
    assert writes.count("t.json") == 1
    st.positions["a"] = {"x": 1}
    st.save()
    st.positions["a"]["x"] = 2   # 巢狀內容改了也要寫
    st.save()
    assert writes.count("t.json") == 3
    assert json.loads((tmp_path / "t.json").read_text())["positions"]["a"]["x"] == 2


def test_step_manage_does_not_rewrite_unchanged_state(tmp_path, monkeypatch):
    tr, ex, sink, clock = _clocked(tmp_path)
    tr.step_manage(NOW)
    writes = []
    real = botmod._write_atomic
    monkeypatch.setattr(botmod, "_write_atomic", lambda p, t: (writes.append(p.name), real(p, t)))
    tr.step_manage(NOW)
    tr.step_manage(NOW)
    assert "testnet.json" not in writes


def test_binance_client_418_waits_past_retry_after_and_prints_diagnosis(monkeypatch, capsys):
    bmod, client, Sess, _ = _client_with(418, "100", monkeypatch)
    t0 = bmod.time.time()
    with pytest.raises(bmod.BinanceError):
        client.positions()
    # Retry-After 到了還要多等一段才送，免得剛好在解封前一刻送出又被封
    assert client.banned_until >= t0 + 100 + bmod.BAN_MARGIN - 1
    out = capsys.readouterr().out
    assert "診斷：GET /fapi/v2/positionRisk" in out and "最近 1 分鐘送了 1 個" in out
    assert "signature" not in out and "'s'" not in out  # 不印簽名和金鑰


def test_binance_ban_survives_restart_and_blocks_unsigned_requests(monkeypatch, tmp_path):
    bmod, client, Sess, _ = _client_with(418, "100", monkeypatch)
    client.ban_file = tmp_path / "testnet.ban"
    with pytest.raises(bmod.BinanceError):
        client.positions()
    calls = Sess.calls
    again = bmod.BinanceFutures(key="k", secret="s", session=Sess(), ban_file=tmp_path / "testnet.ban")
    assert again.banned_until == pytest.approx(client.banned_until)
    with pytest.raises(bmod.BinanceError):
        again.klines("BTCUSDT", "2H")   # 不簽名的請求在封鎖期間也不送
    assert Sess.calls == calls
