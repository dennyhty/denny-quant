import pandas as pd

from grid_trading import binance_history as bh
from grid_trading import data_cache as dc


def _bars(start: str, n: int) -> pd.DataFrame:
    t = pd.date_range(start, periods=n, freq="D", tz="UTC")
    ms = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    return pd.DataFrame({"time": ms, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0})


def test_cache_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GRID_DATA_CACHE", str(tmp_path))
    assert dc.data_dir("binance") == tmp_path / "binance"


def test_incremental_update_fetches_only_new_months(monkeypatch, tmp_path):
    monkeypatch.setenv("GRID_DATA_CACHE", str(tmp_path))
    monkeypatch.setattr(bh, "CACHE_DIR", tmp_path / "binance")
    monkeypatch.setattr(dc, "latest_complete_month", lambda now=None: pd.Period("2026-02", "M"))
    monkeypatch.setattr(bh, "latest_complete_month", lambda now=None: pd.Period("2026-02", "M"))
    calls = []

    def fake_download(coin, interval, flow, since=None):
        calls.append(since)
        return [_bars("2026-01-01", 59)]  # 1~2 月

    monkeypatch.setattr(bh, "_download", fake_download)
    # 第一次：全抓
    bh.fetch_binance("BTC", "1d")
    assert calls == [None]
    # 快取已經到最新月：不連網
    bh.fetch_binance("BTC", "1d")
    assert calls == [None]
    # 最新月前進：只補 since=最後一根所在月
    monkeypatch.setattr(bh, "latest_complete_month", lambda now=None: pd.Period("2026-03", "M"))
    monkeypatch.setattr(bh, "_download", lambda *a, since=None, **k: calls.append(since) or [_bars("2026-02-01", 59)])
    out = bh.fetch_binance("BTC", "1d")
    assert calls[-1] == pd.Period("2026-02", "M")
    assert out["time"].is_unique and out["time"].is_monotonic_increasing
    assert out["time"].iloc[-1] == pd.Timestamp("2026-03-31", tz="UTC")
    # 檢查過這個月就不再連網（例如已下架的幣）
    n = len(calls)
    bh.fetch_binance("BTC", "1d")
    assert len(calls) == n
