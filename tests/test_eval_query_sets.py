# -*- coding: utf-8 -*-
"""评测查询集的不变量测试。

为什么值得为"查询集"写测试：
  措辞贴近 / 措辞远离 的分组，是这个检索评测**全部结论的依据**。
  如果"措辞远离"组里混进了目标条款的关键词，那这组就不再远离，
  结论会被静默削弱——而且从总分上完全看不出来。

  所以这两条性质必须是**可验证的**，而不是作者的口头声明：
    · 贴近组：每条问题**至少含一个**目标条款关键词（否则不算"贴近"）；
    · 远离组：每条问题**不含任何**目标条款关键词（否则不算"远离"）。
  写成测试后，任何人改动查询集都会被立刻拦住。

运行：D:/conda/envs/cformer-gpu/python.exe -m pytest tests/test_eval_query_sets.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_semantic_retrieval import DISTANT_QUERIES, NEAR_QUERIES  # noqa: E402

DATA = ROOT / "data" / "gov_employee_rules.json"


def _objects() -> dict[str, dict]:
    spec = json.loads(DATA.read_text(encoding="utf-8"))
    return {o["id"]: o for o in spec["objects"]}


def _keywords(obj: dict) -> list[str]:
    return list(obj.get("keywords") or [])


def test_query_sets_are_big_enough_to_mean_something() -> None:
    """样本量下限：查询太少时任何命中率都没有统计意义。"""
    assert len(NEAR_QUERIES) >= 8
    assert len(DISTANT_QUERIES) >= 8


def test_every_expected_target_exists_in_dataset() -> None:
    """期望命中的对象必须真实存在，否则永远命中不了，会伪装成"检索失败"。"""
    objects = _objects()
    for question, expected in [*NEAR_QUERIES, *DISTANT_QUERIES]:
        assert expected in objects, f"「{question}」的期望对象 {expected} 不在数据集里"


def test_near_queries_actually_contain_a_target_keyword() -> None:
    """贴近组：必须至少含一个目标关键词，否则这一组名不副实。"""
    objects = _objects()
    for question, expected in NEAR_QUERIES:
        kws = _keywords(objects[expected])
        assert any(k in question for k in kws), \
            f"「{question}」不含 {expected} 的任何关键词 {kws}，不能算「措辞贴近」"


def test_distant_queries_avoid_all_target_keywords() -> None:
    """远离组：不得含目标条款的任何关键词——这是全部结论的立足点。

    这条曾经抓到过真实问题：原查询「早上来得太晚会被扣多少钱？」含有
    penalty 的关键词「扣」，于是它并不"远离"，会略微抬高字符重叠检索的成绩。
    """
    objects = _objects()
    offenders = []
    for question, expected in DISTANT_QUERIES:
        leaked = [k for k in _keywords(objects[expected]) if k in question]
        if leaked:
            offenders.append((question, expected, leaked))
    assert not offenders, f"以下「措辞远离」问题含有目标关键词：{offenders}"


def test_distant_queries_do_not_reuse_near_query_text() -> None:
    """两组问题不应重复，否则同一道题被算了两次。"""
    near = {q for q, _ in NEAR_QUERIES}
    distant = {q for q, _ in DISTANT_QUERIES}
    assert not (near & distant)


@pytest.mark.parametrize("question,expected", [*NEAR_QUERIES, *DISTANT_QUERIES])
def test_target_is_visible_at_level_zero(question: str, expected: str) -> None:
    """目标条款必须有 level 0 内容——否则在最低权限级别下检索必然命中不了，
    那不是检索质量问题，而是测试设置错误（会污染整个结论）。"""
    objects = _objects()
    levels = objects[expected]["levels"]
    assert 0 in {int(k) for k in levels}, \
        f"{expected} 没有 level 0 内容，{question} 在级别 0 下不可能命中"
