"""研究用的歷史資料快取位置：所有下載資料集中放一處，不同對話／容器之間共用，不用每次重抓。

快取根目錄的優先順序：
1. 環境變數 GRID_DATA_CACHE
2. /mnt/project-files/data_cache（專案共用資料夾，跨對話保留；掛載在網路磁碟上，所以檔案數要克制：
   一個代號、一個週期一個 csv.gz，不要拆成每月、每天的小檔）
3. ./data（本機，沒有共用資料夾時；已被 .gitignore）

各模組用 data_dir("binance")、data_dir("binance_spot") 取得子資料夾（和以前 data/ 底下的子資料夾同名）。

增量更新用「最後檢查的月份」記在根目錄的 _checked.json：K 線月檔只到上個月月底，
所以檔案裡最後一根 K 線已經到上個月，或這個月已經檢查過，就完全不碰網路。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pandas as pd

_LOCK = threading.Lock()
SHARED_ROOT = Path("/mnt/project-files/data_cache")
LOCAL_ROOT = Path("data")


def cache_root() -> Path:
    env = os.environ.get("GRID_DATA_CACHE")
    if env:
        return Path(env)
    if SHARED_ROOT.parent.is_dir():
        return SHARED_ROOT
    return LOCAL_ROOT


def data_dir(*parts: str) -> Path:
    """子資料夾路徑（不建立；寫檔的地方自己 mkdir(parents=True)）。"""
    return cache_root().joinpath(*parts)


def latest_complete_month(now: pd.Timestamp | None = None) -> pd.Period:
    """data.binance.vision 的月檔最新到上個月。"""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return now.tz_localize(None).to_period("M") - 1 if now.tzinfo else now.to_period("M") - 1


def _manifest_path() -> Path:
    return cache_root() / "_checked.json"


def _read_manifest() -> dict:
    try:
        return json.loads(_manifest_path().read_text())
    except (OSError, ValueError):
        return {}


def checked_month(key: str) -> pd.Period | None:
    v = _read_manifest().get(key)
    return pd.Period(v, "M") if v else None


def mark_checked(key: str, month: pd.Period) -> None:
    """記下這個檔案已經對照過網路上到哪個月（寫入前重讀，減少同時寫入時互相蓋掉）。"""
    with _LOCK:  # 同一個程式的多個執行緒一起抓時不能互相蓋掉
        m = _read_manifest()
        m[key] = str(month)
        path = _manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(m, indent=0, sort_keys=True))
        os.replace(tmp, path)
