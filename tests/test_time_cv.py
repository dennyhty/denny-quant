import numpy as np
import pandas as pd

from research.time_cv import CVConfig, _blocks, _masks, cpcv, evaluate_family, pbo, walk_forward


def _ledger(edge: dict[str, float], n_per_config=3000, seed=0, sd=1.5):
    rng = np.random.default_rng(seed)
    parts = []
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    span = (pd.Timestamp("2026-11-01", tz="UTC") - t0).total_seconds()
    for c, mu in edge.items():
        t = t0 + pd.to_timedelta(np.sort(rng.uniform(0, span, n_per_config)), unit="s")
        parts.append(pd.DataFrame({"config": c, "symbol": rng.choice(["A", "B", "C", "D"], n_per_config),
                                   "side": rng.choice([1, -1], n_per_config), "time": t,
                                   "exit_t": t + pd.Timedelta(days=3), "r_net": rng.normal(mu, sd, n_per_config)}))
    return pd.concat(parts, ignore_index=True)


def test_walk_forward_picks_real_edge():
    led = _ledger({"real": 0.15, **{f"noise{i}": 0.0 for i in range(9)}})
    wf = walk_forward(led)
    assert (wf["years"]["picked"] == "real").mean() >= 0.8
    assert wf["summary"]["mean_r"] > 0.08
    # 訓練集只收測試年前已出場的單
    assert wf["years"]["year"].min() == 2021


def test_pbo_low_for_real_edge_and_high_for_noise():
    real = pbo(_ledger({"real": 0.2, **{f"n{i}": 0.0 for i in range(9)}}))
    noise = pbo(_ledger({f"n{i}": 0.0 for i in range(10)}, seed=3))
    assert real["pbo"] < 0.1
    assert 0.25 < noise["pbo"] < 0.8


def test_cpcv_paths_and_single_config():
    res = cpcv(_ledger({"only": 0.1}))
    assert res["summary"]["splits"] == 45 and res["summary"]["paths"] == 9
    assert res["paths"]["trades"].min() > 2900  # 每條路徑都涵蓋全部 10 段
    assert pbo(_ledger({"only": 0.1}))["summary"]["n_configs"] == 1


def test_purge_and_embargo():
    cfg = CVConfig(n_groups=4, embargo_days=10)
    t = pd.to_datetime(["2020-01-05", "2021-09-01", "2021-09-20", "2023-01-01"], utc=True)
    df = pd.DataFrame({"time": t, "exit_t": t + pd.Timedelta(days=200)})
    df.loc[0, "exit_t"] = df.loc[0, "time"] + pd.Timedelta(days=1)
    cfg.end = "2024-01-01"
    b_e, b_x, emb, edges = _blocks(df, cfg)
    train, test = _masks((1,), b_e, b_x, emb, 4)
    # 第 2 筆進場在第 1 段之後、持有跨進下一段 → 不是測試段時要看有沒有碰到測試段
    assert test.tolist() == [b == 1 for b in b_e]
    # 持有期間碰到測試段（第 1 段）的單不能當訓練
    for i in range(4):
        if b_e[i] <= 1 <= b_x[i]:
            assert not train[i]


def test_evaluate_family_flip():
    res = evaluate_family(_ledger({"real": 0.25, "n1": 0.0, "n2": 0.0}))
    assert res["flip"]
    res2 = evaluate_family(_ledger({"n1": 0.0, "n2": -0.02, "n3": 0.0}, seed=5))
    assert not res2["flip"]
