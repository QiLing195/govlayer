# -*- coding: utf-8 -*-
"""各数据集评测查询集的不变量测试（对**每个**数据集都跑）。

为什么值得为"查询集"写测试：
  查询集是全部检索结论的依据。这个会话里它已经出过两次事故——
    · 「措辞远离」组混进了目标关键词（"扣"），悄悄抬高关键词基线；
    · 换数据集却沿用旧查询集，脚本照样打印"建议阈值"。
  所以这些性质必须是**可验证的**，而不是作者的口头声明，且必须**覆盖每个数据集**。

另外两档无关问题（远域 / 同域未覆盖）都会被校验"不含任何条款关键词"——
否则它们不是"无关"，会把标定出的阈值推高，导致真正相关的问法被误挡。

运行：D:/conda/envs/cformer-gpu/python.exe -m pytest tests/test_eval_query_sets.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_query_sets import DATASETS, sets_for  # noqa: E402

# 数据集 id -> 数据文件（新增数据集时在这里登记）
DATASET_FILES: dict[str, str] = {
    "employee_rules": "gov_employee_rules.json",
    "registration": "gov_registration.json",
    "kaohe": "gov_kaohe.json",
    "renshi": "gov_renshi.json",
}
DATA = ROOT / "data"


def _objects(dataset_id: str) -> dict[str, dict]:
    raw = (DATA / DATASET_FILES[dataset_id]).read_text(encoding="utf-8")
    return {o["id"]: o for o in json.loads(raw)["objects"]}


def _keywords(obj: dict) -> list[str]:
    return list(obj.get("keywords") or [])


def _sets(dataset_id: str):
    result = sets_for(dataset_id)
    assert result is not None, f"{dataset_id} 在 eval_query_sets.py 里没有配套查询集"
    return result


def test_every_query_set_dataset_has_a_source_file() -> None:
    """查询集里登记的每个数据集都必须有对应的数据文件，否则下面的测试是空的。"""
    assert set(DATASETS) <= set(DATASET_FILES), (
        f"这些数据集有查询集但没登记数据文件：{set(DATASETS) - set(DATASET_FILES)}")


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_query_sets_are_big_enough(dataset_id: str) -> None:
    """样本量下限：太少时任何命中率都没有统计意义。"""
    near, distant, off, same = _sets(dataset_id)
    assert len(near) >= 7, f"{dataset_id} 贴近组只有 {len(near)} 条"
    assert len(distant) >= 7, f"{dataset_id} 远离组只有 {len(distant)} 条"
    assert len(off) >= 6, f"{dataset_id} 远域无关组只有 {len(off)} 条"
    assert len(same) >= 5, f"{dataset_id} 同域未覆盖组只有 {len(same)} 条"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_every_expected_target_exists(dataset_id: str) -> None:
    """期望命中的对象必须真实存在，否则永远命中不了，会伪装成"检索失败"。"""
    objects = _objects(dataset_id)
    near, distant, _off, _same = _sets(dataset_id)
    bad = [(q, e) for q, e in [*near, *distant] if e not in objects]
    assert not bad, f"{dataset_id} 期望对象不存在：{bad}"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_near_queries_actually_contain_a_target_keyword(dataset_id: str) -> None:
    """贴近组：必须至少含一个目标关键词，否则这一组名不副实。"""
    objects = _objects(dataset_id)
    near, _distant, _off, _same = _sets(dataset_id)
    bad = [(q, e) for q, e in near
           if not any(k in q for k in _keywords(objects[e]))]
    assert not bad, f"{dataset_id} 以下「贴近」问题不含目标关键词：{bad}"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_distant_queries_avoid_all_target_keywords(dataset_id: str) -> None:
    """远离组：不得含目标关键词——这是"语义检索值不值"全部结论的立足点。

    这条抓到过真实问题：原查询「早上来得太晚会被扣多少钱？」含有 penalty 的关键词
    「扣」，于是它并不"远离"，会略微抬高字符重叠检索的成绩。
    """
    objects = _objects(dataset_id)
    _near, distant, _off, _same = _sets(dataset_id)
    bad = []
    for q, expected in distant:
        leaked = [k for k in _keywords(objects[expected]) if k in q]
        if leaked:
            bad.append((q, expected, leaked))
    assert not bad, f"{dataset_id} 以下「远离」问题含目标关键词：{bad}"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
@pytest.mark.parametrize("group_index,group_name", [(2, "远域无关"), (3, "同域未覆盖")])
def test_unrelated_queries_avoid_every_keyword(dataset_id: str, group_index: int,
                                              group_name: str) -> None:
    """两档无关问题都不得含**任何**条款的关键词。

    否则它们不是"无关"——分数会被关键词路径抬高，进而把标定出的阈值推高，
    导致真正相关的问法被误挡。这是阈值标定的前提。
    """
    objects = _objects(dataset_id)
    queries = _sets(dataset_id)[group_index]
    all_kw = {k for obj in objects.values() for k in _keywords(obj)}
    bad = [(q, sorted(k for k in all_kw if k in q))
           for q in queries if any(k in q for k in all_kw)]
    assert not bad, f"{dataset_id} 以下「{group_name}」问题含条款关键词：{bad}"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_two_unrelated_groups_do_not_overlap(dataset_id: str) -> None:
    """远域组与同域组不得重复——它们要回答不同的问题，重复会污染对比。"""
    _near, _distant, off, same = _sets(dataset_id)
    dup = set(off) & set(same)
    assert not dup, f"{dataset_id} 两档无关问题重复：{sorted(dup)}"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_targets_have_content(dataset_id: str) -> None:
    """目标必须有实际条款内容。

    注意这里**不能**要求"目标在最低级别可见"：查询集里刻意包含指向受限条款的问题
    （如 kaohe 的 art15/art16/art33/art42），它们用于**跨级别检索**——
    calibrate_threshold.py 正是以 max_level 检索的。
    只有 eval_semantic_retrieval.py 那种在 level 0 下评测的脚本才要求目标在 level 0 可见。
    """
    objects = _objects(dataset_id)
    near, distant, _off, _same = _sets(dataset_id)
    bad = [(q, e) for q, e in [*near, *distant] if not objects[e].get("levels")]
    assert not bad, f"{dataset_id} 以下目标没有条款内容：{bad}"


@pytest.mark.parametrize("dataset_id", sorted(DATASET_FILES))
def test_near_and_distant_do_not_overlap(dataset_id: str) -> None:
    """两组问题不应重复，否则同一道题被算了两次。"""
    near, distant, _off, _same = _sets(dataset_id)
    assert not ({q for q, _ in near} & {q for q, _ in distant})
