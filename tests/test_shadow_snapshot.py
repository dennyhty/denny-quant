from shadow_snapshot import group_of, summarize


def test_group_of_boundaries():
    assert group_of(50, 3e7) is None  # 實盤會做的不記
    assert group_of(100, 2.9e7) == "top100_20M"
    assert group_of(10, 1.5e7) == "top100_10M"
    assert group_of(10, 7e6) == "top100_5M"
    assert group_of(10, 4e6) == "top100_lt5M"
    assert group_of(101, 3e7) == "r101_200_30M"
    assert group_of(150, 1e7) == "r101_200_10M"
    assert group_of(200, 5e6) == "r101_200_lt10M"
    assert group_of(150, None) is None


def test_summarize_counts_closed_and_open():
    book = [{"id": "a", "status": "CLOSED", "r": 1.0}, {"id": "b", "status": "CLOSED", "r": -1.0},
            {"id": "c", "status": "OPEN", "r": None}, {"id": "d", "status": "EXPIRED", "r": None},
            {"id": "e", "status": "CLOSED", "r": 3.0}]
    meta = {k: {"group": "top100_10M"} for k in "abcd"} | {"e": {"group": None}}
    g = summarize(book, meta)["top100_10M"]
    assert g["trades"] == 2 and g["open"] == 1 and g["sum_r"] == 0.0 and g["win_rate"] == 0.5
