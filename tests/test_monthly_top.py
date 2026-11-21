from datetime import datetime, timezone

from grid_trading import binance_public as bp
from grid_trading.trader.bot import Config, Notifier, Trader
from grid_trading.universe import TOP100, binance_perp_for, monthly_top

NOV = datetime(2026, 11, 21, tzinfo=timezone.utc)
DEC = datetime(2026, 12, 1, 0, 5, tzinfo=timezone.utc)


def coins(n):
    return [f"C{i}" for i in range(n)]


def test_monthly_top_fetches_once_per_month():
    calls = []
    fetch = lambda n: calls.append(n) or coins(n)
    rec = monthly_top(None, now=NOV, fetch=fetch)
    assert rec["status"] == "fresh" and rec["month"] == "2026-11" and len(rec["coins"]) == 100
    again = monthly_top(rec, now=NOV, fetch=fetch)
    assert again["status"] == "cached" and len(calls) == 1
    nxt = monthly_top(rec, now=DEC, fetch=lambda n: ["NEW"] + coins(n - 1))
    assert nxt["status"] == "fresh" and nxt["month"] == "2026-12" and nxt["coins"][0] == "NEW"


def test_monthly_top_keeps_old_list_when_fetch_fails():
    def down(n):
        raise RuntimeError("timeout")
    old = {"month": "2026-11", "coins": coins(100), "fetched_at": "x"}
    rec = monthly_top(old, now=DEC, fetch=down)
    assert rec["status"] == "stale" and rec["month"] == "2026-11" and rec["coins"] == old["coins"]
    assert monthly_top(None, now=DEC, fetch=down)["coins"] == TOP100
    assert monthly_top(None, now=DEC, fetch=lambda n: ["BTC"])["status"] == "fallback"  # 抓到太少不算數


def test_binance_perp_for_prefixes():
    listed = {"BTCUSDT", "1000BONKUSDT", "1000PEPEUSDT", "POLUSDT", "MATICUSDT", "1MBABYDOGEUSDT"}
    assert binance_perp_for("BTC", listed) == "BTCUSDT"
    assert binance_perp_for("BONK", listed) == "1000BONKUSDT"
    assert binance_perp_for("PEPE", listed) == "1000PEPEUSDT"
    assert binance_perp_for("POL", listed) == "POLUSDT"
    assert binance_perp_for("BABYDOGE", listed) == "1MBABYDOGEUSDT"
    assert binance_perp_for("LEO", listed) is None


def test_site_universe_uses_given_coins():
    class Api:
        def usdt_perps(self):
            return {"BTCUSDT", "1000BONKUSDT", "LEOUSDT"}
    assert bp.universe(Api(), coins=["BTC", "BONK", "XYZ"]) == ["1000BONK_USDT_PERP", "BTC_USDT_PERP"]


def test_trader_refreshes_coins_monthly(tmp_path, monkeypatch):
    lists = {"2026-11": ["AAA", "BBB"], "2026-12": ["BBB", "CCC"]}
    clock = {"t": NOV}
    import grid_trading.trader.bot as bot
    real = bot.monthly_top
    monkeypatch.setattr(bot, "monthly_top", lambda cached, now=None: real(
        cached, n=2, now=now, fetch=lambda n: lists[now.strftime("%Y-%m")]))
    sink = []
    filters = {"AAAUSDT": None, "BBBUSDT": None, "CCCUSDT": None}
    tr = Trader(Config(mode="testnet", state_dir=tmp_path, auto_coins=True), None, None,
                Notifier("test", sink=sink), filters)
    tr._clock = lambda: clock["t"].timestamp()
    tr.refresh_coins(announce=False)
    assert set(tr.symbols) == {"AAAUSDT", "BBBUSDT"}
    assert (tmp_path / "universe.json").exists()
    clock["t"] = DEC
    tr.refresh_coins()
    assert set(tr.symbols) == {"BBBUSDT", "CCCUSDT"}
    msg = " ".join(str(m) for m in sink)
    assert "新增 CCC/USDT" in msg and "移除 AAA/USDT" in msg


def test_turnover_at_uses_bars_before_signal_close():
    import pandas as pd
    from grid_trading.signals.engine import turnover_at
    bar = pd.Timedelta(hours=4)
    t0 = pd.Timestamp("2026-10-01", tz="UTC")
    df = pd.DataFrame({"time": [t0 + i * bar for i in range(200)], "close": 2.0, "volume": 10.0})
    sig_close = df["time"].iloc[179] + bar
    assert turnover_at(df, sig_close, bar) == 2.0 * 10.0 * 6  # 每天 6 根
    assert turnover_at(df, t0 + bar, bar) is None  # 不夠 30 天
