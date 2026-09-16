# -*- coding: utf-8 -*-
"""#4 ANN 索引测试：权限分区不可穿越、编码器稳定、ANN 忠实度。

重点是**结构性零泄漏**：级别 L 的查询在代码路径上不得触达 level > L 的分区。
这不是"过滤得好不好"的问题，而是"有没有路径"的问题，所以必须用测试钉死。

运行：D:/conda/envs/cformer-gpu/python.exe -m pytest tests/test_ann_index.py -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cformer_v63.ann_index import PermissionPartitionedIndex, l2_normalize  # noqa: E402
from cformer_v63.embedding import (  # noqa: E402
    HashingEncoder,
    build_encoder,
    normalize_text,
)

OBJECTS = [
    {"id": "pub", "title": "作息", "keywords": ["上班", "时间"],
     "levels": {"0": "公司上班时间8:30-17:30。"}},
    {"id": "mid", "title": "考勤", "keywords": ["打卡", "补卡"],
     "levels": {"0": "视频考勤，每天登记。", "1": "补卡由部门经理核验。"}},
    {"id": "secret", "title": "高管薪酬明细", "keywords": ["薪酬", "年薪"],
     "levels": {"2": "高管年薪档位与期权授予记录，仅HR可见。"}},
]


def min_level(obj: dict) -> int:
    """对象**最低**可见级别。

    泄漏的判据是"这个对象在我的级别上完全不可见"，即 min_level > 我的级别。
    不能用 max_level：一个对象完全可以既在 level 0 有公开条款、又在 level 2 有管理细则
    （如"考勤处罚"），此时 level 0 的用户命中它是**正确**的——
    他只会看到 level 0 那段，而看不到 level 2 那段。
    """
    return min(int(k) for k in obj["levels"])


# ---------------------------------------------------------------- 权限语义

def test_visible_text_is_cumulative_by_level() -> None:
    per_level = PermissionPartitionedIndex.visible_text_per_level(OBJECTS[1])
    assert set(per_level) == {0, 1}
    assert "视频考勤" in per_level[0]
    assert "部门经理核验" not in per_level[0]      # 低级别看不到高级别内容
    assert "视频考勤" in per_level[1]              # 高级别包含低级别内容（累计）
    assert "部门经理核验" in per_level[1]


def test_level_zero_query_cannot_reach_higher_partitions() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)

    index.search("高管年薪", level=0, top_k=10)
    assert index.last_searched_levels == [0]        # 只碰了 level 0 分区

    index.search("高管年薪", level=2, top_k=10)
    assert index.last_searched_levels == [0, 1, 2]


def test_low_level_query_never_returns_secret_object() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)

    for level in (0, 1):
        for query in ("高管年薪档位", "期权授予记录", "高管薪酬明细"):
            hit_ids = [oid for oid, _, _ in index.search(query, level=level, top_k=10)]
            assert "secret" not in hit_ids, f"level={level} 泄漏了 secret"
            # 更一般的判据：命中对象的"最低可见级别"必须 ≤ 查询级别
            assert all(min_level(o) <= level for o in OBJECTS if o["id"] in hit_ids)


def test_level_two_object_is_reachable_at_level_two() -> None:
    """反向对照：确保上一条的"没命中"是因为权限，而不是因为它根本检索不到。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    hit_ids = [oid for oid, _, _ in index.search("高管年薪档位", level=2, top_k=10)]
    assert "secret" in hit_ids


def test_exact_search_also_respects_partitions() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    hit_ids = [oid for oid, _, _ in index.exact_search("高管年薪", level=0, top_k=10)]
    assert "secret" not in hit_ids


def test_allowed_entries_counts_only_visible_levels() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    assert index.allowed_entries(0) == 2            # pub + mid(level0)
    assert index.allowed_entries(1) == 3            # + mid(level1)
    assert index.allowed_entries(2) == 4            # + secret


def test_higher_level_queries_see_more_entries() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    assert index.allowed_entries(0) < index.allowed_entries(2)


# ---------------------------------------------------------------- 索引行为

def test_ann_equals_exact_when_below_brute_force_threshold() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=2048).build(OBJECTS)
    for query in ("上班时间", "打卡", "高管年薪"):
        approx = [oid for oid, _, _ in index.search(query, level=2, top_k=3)]
        exact = [oid for oid, _, _ in index.exact_search(query, level=2, top_k=3)]
        assert set(approx) == set(exact)      # 比集合，避免同分项的排序差异造成假失败


def test_forced_clustering_scans_all_when_probing_every_centroid() -> None:
    """强制聚类（阈值 0）：探测全部质心时结果必须与暴力检索一致。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    part = index.partitions[0]
    assert part.centroids is not None
    for query in ("上班时间", "视频考勤"):
        approx = [oid for oid, _, _ in index.search(query, level=0, top_k=3,
                                                   n_probe=len(part.centroids))]
        exact = [oid for oid, _, _ in index.exact_search(query, level=0, top_k=3)]
        assert set(approx) == set(exact)


def test_clustering_handles_tiny_partition_without_nan() -> None:
    """空簇/簇数多于点数时不能产生 NaN（NaN 会污染后续所有相似度比较）。"""
    tiny = [{"id": "a", "title": "t", "keywords": [], "levels": {"2": "只有一条。"}}]
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(tiny)
    part = index.partitions[2]
    assert part.centroids is not None
    assert not np.isnan(part.centroids).any()
    hits = index.search("只有一条", level=2, top_k=5)
    assert hits and hits[0][0] == "a"


def test_search_dedupes_repeated_object_across_levels() -> None:
    """一个对象在多个级别都建了索引项，结果里也只能出现一次。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    ids = [oid for oid, _, _ in index.search("视频考勤 补卡", level=2, top_k=10)]
    assert ids.count("mid") <= 1


def test_top_k_is_respected() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    assert len(index.search("考勤", level=2, top_k=2)) <= 2


def test_empty_index_search_is_safe() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build([])
    assert index.search("任意问题", level=0) == []
    assert index.stats()["n_entries"] == 0


def test_stats_report_encoder_and_levels() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    stats = index.stats()
    assert stats["per_level"] == {0: 2, 1: 1, 2: 1}
    assert "hashing" in stats["encoder"]


# ---------------------------------------------------------------- 下标空间回归

def test_single_probe_on_clustered_index_is_safe() -> None:
    """回归守卫：n_probe < 簇数 时，cand 里的 **entries 下标**会 ≥ len(scores)。

    曾把这两个下标空间混用（拿 entries 下标去索引 scores），
    导致聚类索引在 n_probe=1 时必然 IndexError——而 5 万条语料正是这种配置。
    另一个测试恰好传 n_probe=簇数（cand 成为 0..n-1 的排列），把越界掩盖了。
    """
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    assert index.partitions[0].centroids is not None
    for q in ("上班时间", "打卡", "高管年薪", "请假", "考勤制度"):
        hits = index.search(q, level=2, top_k=5, n_probe=1)      # 不得抛异常
        assert len(hits) <= 5
        for _oid, score, _lvl in hits:
            assert -1.0001 <= score <= 1.0001                    # 不是错位取到的分数


def test_min_score_filters_before_truncation_on_clustered_index() -> None:
    """极高下限应把全部条目滤掉（返回空），而不是让不合格项占满 top_k 名额。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    assert index.search("上班时间", level=2, top_k=5, n_probe=1, min_score=1.01) == []


# ---------------------------------------------------------------- 校准

def test_calibrate_n_probe_reaches_target_and_sets_default() -> None:
    """校准必须达目标召回，并把结果写进 self.n_probe（供后续默认检索使用）。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    queries = ["上班时间", "打卡补卡", "请假审批", "高管年薪"]
    info = index.calibrate_n_probe(queries, level=2, target_recall=0.95)
    assert info["reached"] is True
    assert info["recall"] >= 0.95
    assert index.n_probe == info["n_probe"]


def test_calibrate_reuses_supplied_reference() -> None:
    """传入 reference 时不应再自己重算暴力基准（大规模下那等于白等一倍时间）。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    queries = ["上班时间", "打卡"]
    ref = {q: [oid for oid, _, _ in index.exact_search(q, 2, top_k=5)] for q in queries}
    info = index.calibrate_n_probe(queries, level=2, top_k=5, reference=ref)
    assert info["n_queries"] == 2
    assert info["reached"] is True


def test_calibrate_with_no_queries_is_safe() -> None:
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False)).build(OBJECTS)
    info = index.calibrate_n_probe([], level=0)
    assert info["reached"] is False
    assert info["recall"] is None


def test_calibrate_reports_failure_when_target_unreachable() -> None:
    """质心极少时达不到目标召回，必须**如实报告 reached=False**，
    而不是悄悄返回一个达不到目标的 n_probe 让人以为已经达标。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    info = index.calibrate_n_probe(["上班时间", "打卡", "请假", "考勤制度", "薪酬明细"],
                                  level=2, target_recall=0.999, top_k=1)
    assert "reached" in info
    if not info["reached"]:
        assert info["recall"] < 0.999
        assert info["max_probe"] >= 1


def test_calibration_recall_matches_independent_remeasure() -> None:
    """校准给出的召回，必须与事后**独立重测**完全一致。

    这条不变量曾被评测脚本破坏过：保真度问题串里有重复，而暴力基准是 dict
    （重复 key 被折叠），于是 calibrate 按"含重复的列表"算均值、measure 按
    "去重后的 dict"算均值——两个分母不同，出现"判定达标、打印却不达标"的矛盾输出。
    测试把它钉死：判定所依据的数必须等于重测的数。
    """
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    queries = ["上班时间", "打卡", "请假", "考勤制度", "薪酬明细", "出差报备", "迟到扣款"]
    ref = {q: [oid for oid, _, _ in index.exact_search(q, 2, top_k=5)] for q in queries}

    info = index.calibrate_n_probe(queries, level=2, top_k=5, reference=ref,
                                   target_recall=0.5)
    assert info["reached"] is True

    recalls = []
    for q in queries:
        got = [oid for oid, _, _ in index.search(q, 2, top_k=5, n_probe=info["n_probe"])]
        recalls.append(len(set(got) & set(ref[q])) / max(1, len(ref[q])))
    assert round(sum(recalls) / len(recalls), 4) == info["recall"]


def test_search_is_deterministic_at_fixed_n_probe() -> None:
    """固定 n_probe 下检索必须可复现；否则任何召回数字都没有意义。"""
    index = PermissionPartitionedIndex(build_encoder("hashing", verbose=False),
                                       brute_force_threshold=0).build(OBJECTS)
    a = [oid for oid, _, _ in index.search("上班时间 打卡", 2, top_k=5, n_probe=2)]
    b = [oid for oid, _, _ in index.search("上班时间 打卡", 2, top_k=5, n_probe=2)]
    assert a == b


# ---------------------------------------------------------------- 编码器

def test_hashing_encoder_is_deterministic_across_instances() -> None:
    a = HashingEncoder(dim=64).encode(["旷工几天被劝退"])
    b = HashingEncoder(dim=64).encode(["旷工几天被劝退"])
    assert np.array_equal(a, b)


def test_hashing_encoder_output_is_l2_normalized() -> None:
    vecs = HashingEncoder(dim=64).encode(["考勤制度", "请假流程"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_hashing_encoder_empty_text_yields_zero_vector() -> None:
    """空文本没有内容可编码 → 零向量（而不是 NaN）。

    这是**期望行为**，不是缺陷：零向量与任何条目的余弦相似度都是 0，
    所以"空条款"永远不会被召回，也不会污染其他条目的排序。
    强行归一化反而会得到 NaN（0/0），NaN 参与比较会让整个排序错乱。
    """
    vec = HashingEncoder(dim=64).encode([""])[0]
    assert not np.isnan(vec).any()
    assert np.linalg.norm(vec) == 0.0


def test_hashing_uses_stable_blake2b_not_builtin_hash() -> None:
    """钉死稳定性：用一个字符的文本独立复算期望向量。

    内置 hash() 每进程加随机盐，会让落盘索引在重启后全部失配——
    而且只在重启后才暴露。这里通过完全复算 blake2b 的结果来防止有人改回 hash()。
    """
    dim = 64
    vec = HashingEncoder(dim=dim).encode(["a"])[0]
    digest = hashlib.blake2b(b"a", digest_size=8).digest()
    idx = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] & 1 else -1.0

    assert np.count_nonzero(vec) == 1              # 单字符文本只产生一个非零分量
    assert vec[idx] == pytest.approx(sign, abs=1e-6)


def test_normalize_text_handles_fullwidth_and_spaces() -> None:
    assert normalize_text("Ａ Ｂ\tＣ") == "abc"


def test_build_encoder_hashing_mode_is_explicit() -> None:
    encoder = build_encoder("hashing", verbose=False)
    assert isinstance(encoder, HashingEncoder)
    assert "hashing" in encoder.name


def test_build_encoder_onnx_without_model_dir_raises() -> None:
    """prefer=onnx 却没配模型时必须报错，不能静默退回无语义编码器。"""
    import os
    old = os.environ.pop("GOVLAYER_ONNX_MODEL", None)
    try:
        with pytest.raises(RuntimeError):
            build_encoder("onnx", verbose=False)
    finally:
        if old is not None:
            os.environ["GOVLAYER_ONNX_MODEL"] = old


def test_l2_normalize_guards_zero_vector() -> None:
    out = l2_normalize(np.zeros((2, 4), dtype=np.float32))
    assert not np.isnan(out).any()
