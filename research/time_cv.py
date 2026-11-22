"""時間序列交叉驗證：滾動前進（walk-forward）、清除重疊的組合式 K-fold（CPCV）、過度擬合機率（PBO）。2026-11-22。

為什麼要這個：以前每個策略只切一刀（樣本內 2020-2023 挑參數、樣本外 2024 起看一次），等於只做了一折，
某一段行情剛好好或剛好壞就會誤判。這裡用三種方法重看「同一家族、預先登記的那些設定」：

1. 滾動前進：每年 1 月 1 日，用「之前」的交易（anchored 從 2020 起累積；rolling 只用前 2 年）挑出
   樣本內 t 值最高的設定，拿去做「這一年」。訓練集只收在測試年開始前就已出場的單（清除重疊）。
   把每一年選到的設定在那一年的交易接起來，就是一條完全沒偷看的樣本外績效。
2. CPCV（López de Prado）：整段時間切成 N=10 段，每次拿 k=2 段當測試、其餘當訓練，共 45 種切法。
   訓練集清除「持有期間碰到測試段」的單，並在每個測試段後面禁用 embargo 天（避免指標暖機期的資訊外洩）。
   每種切法都在訓練集挑最好的設定、看它在測試段的表現 → 樣本外的分布；45 種切法拼成 9 條完整的回測路徑。
3. PBO（Bailey 等人的 CSCV）：同樣 10 段，5 段訓練、5 段測試，252 種切法。訓練集挑出的最佳設定，
   在測試集的排名落在後半的比例 = 過度擬合機率。只有一組設定的家族沒有 PBO。

輸入：一個家族的「長帳本」DataFrame：config（設定名稱）、symbol、side(+1/-1)、time(進場)、exit_t、r_net
（已套時點幣池、同幣同時一張、手續費＋滑價＋資金費率）。每個 config 自己就是一本帳。

挑設定一律用訓練集的 t = 平均 R / (標準差/√筆數)，筆數不到 min_train_trades 的設定不參加。
這是事先定好的規則，不要為了讓某個策略過關而改。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from research.stop_research import cluster_key, cluster_stats

REQUIRED = ("config", "symbol", "side", "time", "exit_t")


@dataclass
class CVConfig:
    start: str = "2020-01-01"          # 資料起點（UTC）
    end: str | None = None             # 資料終點；None = 最後一筆出場
    wf_first_test_year: int = 2021     # 滾動前進的第一個測試年
    wf_rolling_years: int = 2          # rolling 模式只用前幾年訓練
    min_train_trades: int = 100        # 參加挑選的最少訓練筆數
    min_test_trades: int = 20          # 測試段排名時，筆數太少的設定排在最後
    n_groups: int = 10                 # CPCV / CSCV 的時間段數
    k_test: int = 2                    # CPCV 每次幾段當測試
    embargo_days: int = 30             # 每個測試段結束後禁用的天數
    gate_t: float = 2.0                # 「有門檻」的滾動前進：訓練 t 不到這個就那年空手


# ---------------------------------------------------------------- 小工具

def _prep(led: pd.DataFrame, rcol: str) -> pd.DataFrame:
    miss = [c for c in (*REQUIRED, rcol) if c not in led.columns]
    if miss:
        raise ValueError(f"帳本缺少欄位：{miss}")
    df = led[list(dict.fromkeys([*REQUIRED, rcol]))].copy()
    for c in ("time", "exit_t"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df[df[rcol].notna()]
    df["config"] = df["config"].astype(str)
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def _agg(codes: np.ndarray, r: np.ndarray, mask: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = codes[mask]
    x = r[mask]
    return (np.bincount(c, minlength=m).astype(float), np.bincount(c, x, minlength=m),
            np.bincount(c, x * x, minlength=m))


def _tstat(n, s, ss, min_n) -> np.ndarray:
    """iid t = mean / (sd/√n)；筆數不足或標準差 0 的回傳 -inf（不會被選到）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
        var = (ss - n * mean ** 2) / (n - 1)
        t = mean / np.sqrt(var / n)
    t = np.where((n >= min_n) & np.isfinite(t), t, -np.inf)
    return t


def ledger_stats(tr: pd.DataFrame, rcol: str = "r_net") -> dict:
    """筆數、平均 R、群集 t（進場日+方向）、以及四種分群中最小的 t（和驗證關卡同一套）。"""
    if not len(tr):
        return {"trades": 0, "mean_r": float("nan"), "t": float("nan"), "t_strict": float("nan")}
    from research.validation_gate import cluster_t_values  # 延遲載入，避免循環 import
    x = tr[rcol].to_numpy(float)
    mean, se, *_ = cluster_stats(x, cluster_key(tr))
    ts = [v for v in cluster_t_values(tr, rcol).values() if np.isfinite(v)]
    return {"trades": len(tr), "mean_r": float(mean), "t": float(mean / se) if se and se > 0 else float("nan"),
            "t_strict": float(min(ts)) if ts else float("nan")}


# ---------------------------------------------------------------- 1. 滾動前進

def walk_forward(led: pd.DataFrame, cfg: CVConfig | None = None, mode: str = "anchored", rcol: str = "r_net",
                 gated: bool = False) -> dict:
    """每年用之前的資料挑設定、做這一年。gated=True：訓練 t < cfg.gate_t（或平均 <= 0）那年就不做。

    回傳 years（每年選到誰、訓練 t、測試筆數與平均）、trades（接起來的樣本外交易）、summary。
    """
    cfg = cfg or CVConfig()
    df = _prep(led, rcol)
    configs = np.array(sorted(df["config"].unique()))
    codes = pd.Categorical(df["config"], categories=configs).codes.astype(np.int64)
    r = df[rcol].to_numpy(float)
    t_in = df["time"].to_numpy("datetime64[ns]")
    t_out = df["exit_t"].to_numpy("datetime64[ns]")
    start = np.datetime64(pd.Timestamp(cfg.start).tz_localize(None))
    last_year = int(df["time"].dt.year.max())
    rows, parts = [], []
    for y in range(cfg.wf_first_test_year, last_year + 1):
        lo = np.datetime64(f"{y}-01-01")
        hi = np.datetime64(f"{y + 1}-01-01")
        tr_lo = start if mode == "anchored" else max(start, np.datetime64(f"{y - cfg.wf_rolling_years}-01-01"))
        train = (t_in >= tr_lo) & (t_out < lo)          # 清除重疊：只收測試年開始前就出場的單
        n, s, ss = _agg(codes, r, train, len(configs))
        t = _tstat(n, s, ss, cfg.min_train_trades)
        row = {"year": y, "picked": None, "train_t": float("nan"), "train_mean": float("nan"), "train_n": 0,
               "test_n": 0, "test_mean": float("nan")}
        if np.isfinite(t).any():
            k = int(np.argmax(t))
            row.update(picked=str(configs[k]), train_t=float(t[k]), train_mean=float(s[k] / n[k]), train_n=int(n[k]))
            go = not gated or (t[k] >= cfg.gate_t and s[k] > 0)
            if go:
                test = (codes == k) & (t_in >= lo) & (t_in < hi)
                sub = df[test]
                row.update(test_n=int(test.sum()), test_mean=float(r[test].mean()) if test.any() else float("nan"))
                parts.append(sub)
            else:
                row["picked"] = f"(空手) {configs[k]}"
        rows.append(row)
    trades = pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]
    years = pd.DataFrame(rows)
    yr = years[years["test_n"] > 0]
    summary = {**ledger_stats(trades, rcol), "years": int(len(years)), "years_traded": int(len(yr)),
               "years_pos": int((yr["test_mean"] > 0).sum()), "n_configs": int(len(configs)),
               "distinct_picks": int(years["picked"].dropna().nunique())}
    return {"years": years, "trades": trades, "summary": summary, "mode": mode, "gated": gated}


# ---------------------------------------------------------------- 時間分段與清除重疊

def _blocks(df: pd.DataFrame, cfg: CVConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每筆的進場段 b_e、出場段 b_x、是否在進場段開頭的 embargo 天內，以及各段的起點。"""
    lo = pd.Timestamp(cfg.start, tz="UTC")
    hi = pd.Timestamp(cfg.end, tz="UTC") if cfg.end else df["exit_t"].max() + pd.Timedelta(seconds=1)
    edges = pd.date_range(lo, hi, periods=cfg.n_groups + 1)
    e = edges.to_numpy("datetime64[ns]")
    t_in = df["time"].to_numpy("datetime64[ns]")
    t_out = df["exit_t"].to_numpy("datetime64[ns]")
    b_e = np.clip(np.searchsorted(e, t_in, side="right") - 1, 0, cfg.n_groups - 1)
    b_x = np.clip(np.searchsorted(e, t_out, side="right") - 1, 0, cfg.n_groups - 1)
    emb = t_in < e[b_e] + np.timedelta64(cfg.embargo_days, "D")
    return b_e, b_x, emb, e


def _masks(test_blocks, b_e, b_x, emb, n_groups):
    """test：進場在測試段；train：持有期間沒碰到任何測試段、也不在測試段後面的 embargo 內。"""
    is_test = np.zeros(n_groups, bool)
    is_test[list(test_blocks)] = True
    test = is_test[b_e]
    # 持有期間跨過的段：b_e..b_x，任何一段是測試段就清除。用前綴和查區間內有沒有測試段
    pref = np.concatenate([[0], np.cumsum(is_test)])
    touches = (pref[b_x + 1] - pref[b_e]) > 0
    prev_test = np.concatenate([[False], is_test[:-1]])[b_e]
    train = ~touches & ~(emb & prev_test)
    return train, test


def _rank_frac(test_metric: np.ndarray, k: int, valid: np.ndarray) -> float:
    """被選中的設定 k 在測試集的相對排名 ω（0~1，越大越好），只跟有效的設定比。"""
    vals = np.where(valid, test_metric, -np.inf)
    m = int(valid.sum())
    if m < 2 or not valid[k]:
        return float("nan")
    better_or_eq = np.sum(vals[valid] <= vals[k])     # 1..m
    return float(better_or_eq / (m + 1))


# ---------------------------------------------------------------- 2. CPCV

def cpcv(led: pd.DataFrame, cfg: CVConfig | None = None, rcol: str = "r_net") -> dict:
    """組合式清除 K-fold。回傳 splits（每種切法：選到誰、訓練 t、測試筆數、平均、t、排名）、paths（9 條路徑的
    平均 R）、per_config（每個設定在 10 段各自的平均 R 與正的段數）、summary。"""
    cfg = cfg or CVConfig()
    df = _prep(led, rcol)
    configs = np.array(sorted(df["config"].unique()))
    m = len(configs)
    codes = pd.Categorical(df["config"], categories=configs).codes.astype(np.int64)
    r = df[rcol].to_numpy(float)
    b_e, b_x, emb, edges = _blocks(df, cfg)
    G = cfg.n_groups
    combos = list(itertools.combinations(range(G), cfg.k_test))
    rows = []
    # path 組裝：每一段當測試會出現 C(G-1, k-1) 次，第 j 次分給第 j 條路徑
    n_paths = math.comb(G - 1, cfg.k_test - 1)
    seen = np.zeros(G, int)
    path_sum = np.zeros((n_paths, G))
    path_n = np.zeros((n_paths, G))
    for ci, tb in enumerate(combos):
        train, test = _masks(tb, b_e, b_x, emb, G)
        n, s, ss = _agg(codes, r, train, m)
        t = _tstat(n, s, ss, cfg.min_train_trades)
        nt, st, sst = _agg(codes, r, test, m)
        row = {"split": ci, "test_blocks": "+".join(map(str, tb)), "picked": None, "train_t": float("nan"),
               "test_n": 0, "test_mean": float("nan"), "test_t": float("nan"), "omega": float("nan")}
        if np.isfinite(t).any():
            k = int(np.argmax(t))
            tt = _tstat(nt, st, sst, 2)
            row.update(picked=str(configs[k]), train_t=float(t[k]), test_n=int(nt[k]),
                       test_mean=float(st[k] / nt[k]) if nt[k] else float("nan"),
                       test_t=float(tt[k]) if np.isfinite(tt[k]) else float("nan"))
            with np.errstate(divide="ignore", invalid="ignore"):
                tm = st / nt
            row["omega"] = _rank_frac(tm, k, nt >= cfg.min_test_trades)
            # 被選中的設定在每個測試段的合計 → 路徑
            sel = (codes == k) & test
            for b in tb:
                mb = sel & (b_e == b)
                j = seen[b]
                path_sum[j, b] += r[mb].sum()
                path_n[j, b] += mb.sum()
        for b in tb:
            seen[b] += 1
        rows.append(row)
    splits = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        path_mean = path_sum.sum(1) / path_n.sum(1)
    paths = pd.DataFrame({"path": range(n_paths), "trades": path_n.sum(1).astype(int), "mean_r": path_mean})
    # 每個設定在 10 段各自的表現（不做挑選，看穩不穩）
    blk_n = np.zeros((m, G))
    blk_s = np.zeros((m, G))
    np.add.at(blk_n, (codes, b_e), 1)
    np.add.at(blk_s, (codes, b_e), r)
    with np.errstate(divide="ignore", invalid="ignore"):
        blk_mean = blk_s / blk_n
    ok = blk_n >= cfg.min_test_trades
    per_config = pd.DataFrame({"config": configs, "trades": blk_n.sum(1).astype(int),
                               "mean_r": blk_s.sum(1) / np.maximum(blk_n.sum(1), 1),
                               "blocks_with_trades": ok.sum(1),
                               "blocks_pos": (ok & (blk_mean > 0)).sum(1),
                               "worst_block_mean": np.where(ok, blk_mean, np.inf).min(1)})
    per_config["worst_block_mean"] = per_config["worst_block_mean"].replace(np.inf, np.nan)
    for b in range(G):
        per_config[f"b{b}"] = np.where(ok[:, b], blk_mean[:, b], np.nan)
    tm = splits["test_mean"].dropna()
    summary = {"splits": len(splits), "n_configs": m, "test_mean_median": float(tm.median()) if len(tm) else float("nan"),
               "test_mean_p10": float(tm.quantile(0.1)) if len(tm) else float("nan"),
               "frac_splits_pos": float((tm > 0).mean()) if len(tm) else float("nan"),
               "paths": n_paths, "paths_pos": int((paths["mean_r"] > 0).sum()),
               "path_mean_median": float(paths["mean_r"].median()), "path_mean_min": float(paths["mean_r"].min()),
               "distinct_picks": int(splits["picked"].dropna().nunique()),
               "block_edges": [str(pd.Timestamp(x).date()) for x in edges]}
    return {"splits": splits, "paths": paths, "per_config": per_config, "summary": summary}


# ---------------------------------------------------------------- 3. PBO（CSCV）

def pbo(led: pd.DataFrame, cfg: CVConfig | None = None, rcol: str = "r_net") -> dict:
    """過度擬合機率：10 段取 5 段訓練、5 段測試（252 種），同樣清除重疊與 embargo。

    λ = logit(ω)，ω = 被選中的設定在測試集的相對排名；PBO = λ <= 0（排名在中位數以下）的比例。
    也回傳「被選中的設定」在測試集的平均 R 分布、以及它掉到負的比例（prob_oos_loss）。
    """
    cfg = cfg or CVConfig()
    df = _prep(led, rcol)
    configs = np.array(sorted(df["config"].unique()))
    m = len(configs)
    if m < 2:
        return {"pbo": float("nan"), "summary": {"pbo": float("nan"), "n_configs": m, "note": "只有一組設定，沒有 PBO"},
                "splits": pd.DataFrame()}
    codes = pd.Categorical(df["config"], categories=configs).codes.astype(np.int64)
    r = df[rcol].to_numpy(float)
    b_e, b_x, emb, _ = _blocks(df, cfg)
    G = cfg.n_groups
    rows = []
    for tb in itertools.combinations(range(G), G // 2):
        train, test = _masks(tb, b_e, b_x, emb, G)
        n, s, ss = _agg(codes, r, train, m)
        t = _tstat(n, s, ss, cfg.min_train_trades)
        if not np.isfinite(t).any():
            continue
        k = int(np.argmax(t))
        nt, st, sst = _agg(codes, r, test, m)
        tt = _tstat(nt, st, sst, cfg.min_test_trades)
        omega = _rank_frac(tt, k, np.isfinite(tt))
        rows.append({"test_blocks": "+".join(map(str, tb)), "picked": str(configs[k]), "train_t": float(t[k]),
                     "test_t": float(tt[k]) if np.isfinite(tt[k]) else float("nan"),
                     "test_mean": float(st[k] / nt[k]) if nt[k] else float("nan"), "omega": omega})
    sp = pd.DataFrame(rows)
    om = sp["omega"].dropna() if len(sp) else pd.Series(dtype=float)
    lam = np.log(om / (1 - om)) if len(om) else om
    p = float((lam <= 0).mean()) if len(om) else float("nan")
    tm = sp["test_mean"].dropna() if len(sp) else pd.Series(dtype=float)
    # 表現衰退：訓練 t 越高，測試 t 有沒有跟著高（斜率 < 0 表示越挑越差）
    slope = float("nan")
    if len(sp) > 3 and sp["test_t"].notna().sum() > 3:
        g = sp.dropna(subset=["test_t"])
        if g["train_t"].std() > 0:
            slope = float(np.polyfit(g["train_t"], g["test_t"], 1)[0])
    summary = {"pbo": p, "n_configs": m, "splits": len(sp), "prob_oos_loss": float((tm <= 0).mean()) if len(tm) else float("nan"),
               "test_mean_median": float(tm.median()) if len(tm) else float("nan"),
               "test_t_median": float(sp["test_t"].median()) if len(sp) else float("nan"),
               "train_t_median": float(sp["train_t"].median()) if len(sp) else float("nan"),
               "degradation_slope": slope}
    return {"pbo": p, "summary": summary, "splits": sp}


# ---------------------------------------------------------------- 一次跑完＋判定

@dataclass
class FlipRule:
    """事先登記的「翻案」條件（2026-11-22，跑任何家族之前定好）。全部成立才算「值得再看」，
    之後仍要過完整驗證關卡（多重檢定門檻）和 Denny 同意，才可能進測試網／實盤。"""

    wf_t: float = 2.0            # 滾動前進（anchored）樣本外群集 t（四種分群取最小）
    wf_years_frac: float = 0.6   # 有交易的測試年裡，平均為正的比例
    cpcv_pos_frac: float = 0.8   # CPCV 45 種切法裡，被選中設定的測試平均為正的比例
    cpcv_paths_frac: float = 0.8  # 9 條路徑裡為正的比例
    max_pbo: float = 0.25        # 多組設定的家族：PBO 上限


def _iid_t(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def vs_baseline(led: pd.DataFrame, wf: dict, cp: dict, base, cfg: CVConfig, rcol: str = "r_net") -> dict:
    """變體家族（同一個維加斯換濾網／出場／幣池）：被選中的設定有沒有贏過實盤（同一年、同一組測試段比）。

    比的是 t 值（= 總 R / √總變異，和每筆風險相同時的帳戶夏普成正比），不是每筆平均 R：
    濾網或只做空單這種「少做一些單」的變體，每筆平均一定比較好看，但帳戶不一定比較賺。
    base：帳本裡實盤那一組的 config 名稱，或外部的實盤帳本 DataFrame（變體家族裡沒有實盤那組時用）。
    選到的就是實盤本身時不算贏。另外附上總 R 的差（資訊用）。"""
    df = _prep(led, rcol)
    if isinstance(base, pd.DataFrame):
        b = _prep(base.assign(config="__base__"), rcol)
        name = "外部實盤帳本"
    else:
        b = df[df["config"] == base]
        name = str(base)
    tr = wf["trades"]
    wins, n_y, dt, dsum = 0, 0, [], []
    for _, y in wf["years"][wf["years"]["test_n"] > 0].iterrows():
        pk = tr.loc[(tr["config"] == y["picked"]) & (tr["time"].dt.year == y["year"]), rcol].to_numpy(float)
        bb = b.loc[b["time"].dt.year == y["year"], rcol].to_numpy(float)
        tp, tb = _iid_t(pk), _iid_t(bb)
        n_y += 1
        if y["picked"] != name and np.isfinite(tp) and (not np.isfinite(tb) or tp > tb + 1e-9):
            wins += 1
        dt.append(tp - tb)
        dsum.append(pk.sum() - bb.sum())
    blocks_df = pd.concat([df, b], ignore_index=True) if isinstance(base, pd.DataFrame) else df
    edges_cfg = cfg
    b_e_all, _, _, _ = _blocks(blocks_df, edges_cfg)
    b_e = b_e_all[len(df):] if isinstance(base, pd.DataFrame) else b_e_all[(df["config"] == name).to_numpy()]
    rb = b[rcol].to_numpy(float)
    better, ddt = [], []
    for _, sp in cp["splits"].iterrows():
        if sp["picked"] is None or not np.isfinite(sp["test_t"]):
            continue
        tb_ = [int(x) for x in sp["test_blocks"].split("+")]
        tb = _iid_t(rb[np.isin(b_e, tb_)])
        better.append(sp["picked"] != name and (not np.isfinite(tb) or sp["test_t"] > tb + 1e-9))
        ddt.append(sp["test_t"] - tb)
    return {"baseline": name, "wf_years_better": f"{wins}/{n_y}", "wf_years_better_frac": wins / n_y if n_y else float("nan"),
            "wf_t_diff_median": float(np.nanmedian(dt)) if dt else float("nan"),
            "wf_total_r_diff": float(np.nansum(dsum)) if dsum else float("nan"),
            "cpcv_frac_better": float(np.mean(better)) if better else float("nan"),
            "cpcv_t_diff_median": float(np.nanmedian(ddt)) if ddt else float("nan")}


def evaluate_family(led: pd.DataFrame, cfg: CVConfig | None = None, rule: FlipRule | None = None,
                    rcol: str = "r_net", baseline=None) -> dict:
    """baseline：變體家族裡「實盤原樣」那一組的 config 名稱。有給的話，翻案還要多兩條：
    滾動前進選到的設定在 >= 60% 的測試年贏過實盤，且 CPCV >= 80% 的切法贏過實盤（2026-11-22 補登記：
    變體家族裡每一組都是維加斯，光看「選到的還是賺錢」等於在重測實盤本身，要比的是有沒有比實盤好）。"""
    cfg, rule = cfg or CVConfig(), rule or FlipRule()
    wf_a = walk_forward(led, cfg, "anchored", rcol)
    wf_r = walk_forward(led, cfg, "rolling", rcol)
    wf_g = walk_forward(led, cfg, "anchored", rcol, gated=True)
    cp = cpcv(led, cfg, rcol)
    pb = pbo(led, cfg, rcol)
    sa = wf_a["summary"]
    yrs = max(sa["years_traded"], 1)
    checks = {
        "wf_t": bool(np.isfinite(sa["t_strict"]) and sa["t_strict"] >= rule.wf_t and sa["mean_r"] > 0),
        "wf_years": sa["years_pos"] / yrs >= rule.wf_years_frac,
        "cpcv_splits": bool(cp["summary"]["frac_splits_pos"] >= rule.cpcv_pos_frac),
        "cpcv_paths": cp["summary"]["paths_pos"] / max(cp["summary"]["paths"], 1) >= rule.cpcv_paths_frac,
        "pbo": bool(not np.isfinite(pb["pbo"]) or pb["pbo"] <= rule.max_pbo),
    }
    vb = None
    if baseline is not None:
        vb = vs_baseline(led, wf_a, cp, baseline, cfg, rcol)
        checks["beats_live_wf"] = bool(vb["wf_years_better_frac"] >= rule.wf_years_frac)
        checks["beats_live_cpcv"] = bool(vb["cpcv_frac_better"] >= rule.cpcv_pos_frac)
    # 每個設定自己（不挑選）在 10 段裡幾段為正：給「有沒有漏掉的單一設定」看
    pc = cp["per_config"]
    stable = pc[(pc["blocks_with_trades"] >= cfg.n_groups - 1) & (pc["blocks_pos"] >= pc["blocks_with_trades"] - 1)
                & (pc["mean_r"] > 0)]
    return {"walk_forward": wf_a, "walk_forward_rolling": wf_r, "walk_forward_gated": wf_g, "cpcv": cp, "pbo": pb,
            "checks": checks, "flip": all(checks.values()), "stable_configs": stable["config"].tolist(), "vs_baseline": vb,
            "config": asdict(cfg), "rule": asdict(rule)}


def summary_row(name: str, res: dict) -> dict:
    """給總表用的一列。"""
    a, ro, g = (res[k]["summary"] for k in ("walk_forward", "walk_forward_rolling", "walk_forward_gated"))
    cp, pb = res["cpcv"]["summary"], res["pbo"]["summary"]
    return {"family": name, "n_configs": cp["n_configs"],
            "wf_trades": a["trades"], "wf_mean_r": a["mean_r"], "wf_t": a["t"], "wf_t_strict": a["t_strict"],
            "wf_years_pos": f"{a['years_pos']}/{a['years_traded']}", "wf_distinct_picks": a["distinct_picks"],
            "wf_roll_mean_r": ro["mean_r"], "wf_roll_t_strict": ro["t_strict"],
            "wf_gated_trades": g["trades"], "wf_gated_mean_r": g["mean_r"],
            "cpcv_test_mean_median": cp["test_mean_median"], "cpcv_test_mean_p10": cp["test_mean_p10"],
            "cpcv_frac_pos": cp["frac_splits_pos"], "cpcv_paths_pos": f"{cp['paths_pos']}/{cp['paths']}",
            "cpcv_path_min": cp["path_mean_min"], "pbo": pb.get("pbo"), "pbo_prob_oos_loss": pb.get("prob_oos_loss"),
            "pbo_degradation": pb.get("degradation_slope"), "n_stable_configs": len(res["stable_configs"]),
            **({f"vs_live_{k}": v for k, v in res["vs_baseline"].items()} if res.get("vs_baseline") else {}),
            "flip": res["flip"], "failed_checks": ",".join(k for k, v in res["checks"].items() if not v)}
