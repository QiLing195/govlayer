# -*- coding: utf-8 -*-
"""混合检索接入测试：RRF 融合、检索路径上报、以及**不许回归的既有行为**。

这次改动把本地语义检索接进 GovLayer 的检索路径，风险不在新功能能不能用，
而在**既有已验证的行为会不会被静默改掉**。因此测试重点有三块：

  1. 融合正确性：语义独有命中要能补进结果、无检索器时行为与原来完全一致；
  2. 权限不许松：（restricted 判定、内容不越权、两种披露立场可切换）；
  3. 顺带钉死本次发现并修复的一个既有 bug（整条不可见时 verdict 误报 "covered"）。

用 `_StubEncoder` 而不是真实模型：单元测试要确定、秒级、且不依赖 24MB 外部文件。
stub 把"同义但不同字面"的词映射到同一维度，因此能构造出**关键词必然miss、
语义必然hit** 的场景——这正是融合逻辑需要被验证的地方。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cformer_v63.governance import GovLayer  # noqa: E402
from cformer_v63.local_semantic import SemanticRetriever  # noqa: E402

OBJECTS = [
    {"id": "work-hours", "title": "作息", "keywords": ["上班时间"],
     "levels": {"0": "公司上班时间8:30-17:30。"}},
    {"id": "leave", "title": "请假", "keywords": ["请假"],
     "levels": {"0": "请假需填请假条；病假需医院证明。"}},
    {"id": "penalty", "title": "考勤处罚", "keywords": ["迟到", "早退"],
     "levels": {"0": "迟到早退每次扣50元。",
                "2": "当月旷工5天或全年累计7天予以劝退。"}},
    # 只有 level 2 内容的条款：用于验证"整条不可见"的判定
    {"id": "secret-salary", "title": "高管薪酬", "keywords": ["薪酬"],
     "levels": {"2": "高管年薪档位仅HR可见。"}},
]
ROLES = {"employee": 0, "manager": 1, "hr": 2}
# 探针规则：问"劝退"时，可见内容里必须有"当月旷工5天"才算覆盖
PROBE = [[["劝退"], ["当月旷工5天"]]]


class _StubEncoder:
    """把同义词映射到同一维度的假编码器（无语义模型，但足以验证融合逻辑）。"""

    name = "stub-semantic"
    dim = 4
    TOPIC = {
        "旷工": 0, "开除": 0, "劝退": 0, "不来上班": 0,
        "请假": 1, "病假": 1, "身体": 1,
        "薪酬": 2, "年薪": 2, "高管": 2,
        "打卡": 3, "刷卡": 3,
    }

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word, dim in self.TOPIC.items():
                if word in text:
                    vecs[i, dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)


class _StubBackend:
    """假的 LLM 语义后端（鸭子类型，字段与 SemanticResult 对齐）。"""

    available = True
    max_candidates = 100

    def __init__(self, relevant=(), covered: bool = True, missing: str = ""):
        self.relevant_ids = list(relevant)
        self.covered = covered
        self.missing = missing
        self.seen: list[list[str]] = []          # 每次 match 时看到的候选 id

    def match(self, question, candidates):
        self.seen.append([c["id"] for c in candidates])
        return self


def _layer(retriever: bool, strict: bool = False, backend=None,
           preference: str = "auto") -> GovLayer:
    objects = json.loads(json.dumps(OBJECTS))          # 深拷贝，避免测试间串味
    ret = SemanticRetriever(objects, _StubEncoder()) if retriever else None
    return GovLayer(objects=objects, roles=dict(ROLES), probe_pairs=PROBE,
                    cases=[], typo_map={}, semantic_backend=backend,
                    semantic_retriever=ret,
                    strict_nonexistence_disclosure=strict,
                    retrieval_preference=preference)


# ---------------------------------------------------------------- 融合正确性

def test_hybrid_equals_keyword_when_no_retriever() -> None:
    """没有语义检索器时，混合检索必须**完全等价**于原来的关键词检索。"""
    layer = _layer(retriever=False)
    for q in ["请假需要什么材料", "上班时间是几点", "迟到早退怎么算"]:
        assert layer.retrieve_hybrid(q, top_n=3) == layer.retrieve(q, top_n=3)


def test_keyword_path_misses_but_hybrid_finds_semantic_hit() -> None:
    """核心场景：问题不含任何关键词（关键词必然 miss），语义路径把它找回来。"""
    layer = _layer(retriever=True)
    question = "一个月不来上班会被开除吗？"

    assert layer.retrieve(question, top_n=5) == []          # 关键词路径一无所获
    assert "penalty" in layer.retrieve_hybrid(question, top_n=5)


def test_hybrid_keeps_keyword_hits() -> None:
    """语义路径不能把关键词已经找到的丢掉（这是纯语义替换会犯的错）。"""
    layer = _layer(retriever=True)
    assert "leave" in layer.retrieve_hybrid("请假需要什么材料", top_n=5)


def test_hybrid_respects_top_n() -> None:
    layer = _layer(retriever=True)
    for n in (1, 2, 3):
        assert len(layer.retrieve_hybrid("请假 迟到 上班时间 高管年薪", top_n=n)) <= n


def test_hybrid_dedupes_object_appearing_in_both_paths() -> None:
    layer = _layer(retriever=True)
    ids = layer.retrieve_hybrid("请假需要什么材料 病假", top_n=10)
    assert len(ids) == len(set(ids))


def test_hybrid_is_deterministic() -> None:
    layer = _layer(retriever=True)
    runs = [layer.retrieve_hybrid("请假 迟到 高管年薪", top_n=5) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_retrieval_mode_reported() -> None:
    assert _layer(retriever=False).answer("请假材料", "employee").retrieval_mode == "keyword"
    assert _layer(retriever=True).answer("请假材料", "employee").retrieval_mode == "hybrid"


def test_denied_hit_that_cannot_cover_is_not_reported_as_restricted() -> None:
    """**诚实性守卫**：命中条款即使把高密级内容**全给它**也答不了这个问题，
    就不能说"制度中有相关规定"——那是假话。

    实测触发案例：问"公司食堂几点开饭？"，系统曾回答
    "制度中有相关规定，但超出你的角色权限范围，请联系 HR 查询"。
    """
    objects = json.loads(json.dumps(OBJECTS))
    ret = SemanticRetriever(objects, _StubEncoder())
    layer = GovLayer(objects=objects, roles=dict(ROLES),
                     probe_pairs=[[["开除"], ["从重处罚细则"]]],   # 全库都没有这句话
                     cases=[], typo_map={}, semantic_retriever=ret)
    ans = layer.answer("一个月不来上班会被开除吗？", "employee")

    assert ans.verdict != "restricted", f"实际 verdict={ans.verdict}"
    assert "有相关规定" not in ans.boundary_note


def test_genuine_permission_truncation_is_still_reported() -> None:
    """反向对照：若高密级内容**确实能**回答该问题，就必须照常报 restricted，
    不能因为上一条守卫而把真正的权限截断也一并禁掉。"""
    ans = _layer(retriever=True).answer("员工连续旷工多久会被劝退？", "employee")
    assert ans.verdict == "restricted"
    assert "有相关规定" in ans.boundary_note
    assert "劝退" not in " ".join(ans.visible_texts.values())


# ---------------------------------------------------------------- 权限不许松

def test_restricted_verdict_preserved_with_hybrid() -> None:
    """**回归守卫**：接入混合检索后，restricted 判定必须原样保留。

    这正是"按权限物理分区的索引不能直接当检索路径"的原因——
    分区索引看不到更高级别的条款，就永远判不出"存在但无权"。
    """
    ans = _layer(retriever=True).answer("员工连续旷工多久会被劝退？", "employee")
    assert ans.verdict == "restricted"
    assert ans.denied_fields == ["penalty"]
    assert "劝退" not in " ".join(ans.visible_texts.values())


def test_fully_invisible_clause_is_restricted_not_covered() -> None:
    """修复的既有 bug：命中条款的内容**全在更高权限级别**时，
    verdict 曾停留在默认 "covered"（前端显示"明文可答"），而实际无任何内容可展示。"""
    ans = _layer(retriever=True).answer("高管年薪档位是多少？", "employee")
    assert ans.verdict == "restricted", f"实际 verdict={ans.verdict}"
    assert "secret-salary" in ans.denied_fields
    assert not any(ans.visible_texts.values())        # 确实一个字都不可见


def test_hr_sees_what_employee_cannot() -> None:
    """反向对照：同一问题 HR 必须能拿到明文，证明上面是权限在起作用。"""
    ans = _layer(retriever=True).answer("员工连续旷工多久会被劝退？", "hr")
    assert ans.verdict == "covered"
    assert "劝退" in " ".join(ans.visible_texts.values())


def test_no_high_level_content_leaks_under_hybrid() -> None:
    """内容越权守卫：任何级别 0 的回答里都不得出现 level 2 的原文。"""
    layer = _layer(retriever=True)
    forbidden = ["劝退", "旷工5天", "年薪档位"]
    for question in ["高管年薪档位是多少？", "员工连续旷工多久会被劝退？",
                     "一个月不来上班会被开除吗？", "薪酬怎么算"]:
        ans = layer.answer(question, "employee")
        blob = " ".join(ans.visible_texts.values()) + " " + ans.boundary_note + " " + \
            json.dumps(ans.precedents, ensure_ascii=False)
        for token in forbidden:
            assert token not in blob, f"「{question}」泄漏了 {token}"


def test_strict_mode_does_not_disclose_existence() -> None:
    """立场 B：连"存在但无权"都不披露 → 不再出现 restricted。"""
    ans = _layer(retriever=True, strict=True).answer("高管年薪档位是多少？", "employee")
    assert ans.verdict != "restricted"
    assert ans.denied_fields == []
    assert "可见的范围" in ans.boundary_note


def test_strict_mode_still_uses_hybrid_within_visible_levels() -> None:
    """严格模式下检索仍然可用（只是限定在本级别可见范围内）。"""
    ans = _layer(retriever=True, strict=True).answer("请假需要什么材料", "employee")
    assert ans.verdict == "covered"
    assert ans.retrieval_mode == "hybrid"


def test_unrelated_question_is_out_of_scope() -> None:
    ans = _layer(retriever=True).answer("公司食堂几点开饭？", "employee")
    assert ans.verdict == "out_of_scope"
    assert ans.denied_fields == []


def test_score_floor_prevents_arbitrary_hits() -> None:
    """**关键守卫**：向量检索永远会返回 top-k，哪怕问题与知识库毫无关系。

    没有相似度下限时，任何问题都会捞回若干条款 → 判定变成 covered，
    "知识空白诚实升级"这个核心卖点被静默废掉。本测试同时证明下限是必要的：
    显式关掉下限后，确实会捞回条目。
    """
    objects = json.loads(json.dumps(OBJECTS))
    ret = SemanticRetriever(objects, _StubEncoder())

    assert ret.search("公司食堂几点开饭？", level=None, top_k=5) == []
    # min_score=-1 → 显式关闭下限（仅评测用）：立刻捞回一堆无关条目
    assert ret.search("公司食堂几点开饭？", level=None, top_k=5, min_score=-1) != []


def test_hybrid_applies_score_floor_through_governance() -> None:
    """下限必须真的贯穿到 Governor 的检索路径，而不只是组件层面有效。"""
    layer = _layer(retriever=True)
    assert layer.retrieve_hybrid("公司食堂几点开饭？", top_n=5) == []


# ---------------------------------------------------------------- 组件边界

def test_retriever_search_level_none_crosses_levels() -> None:
    """level=None（立场 A）能检索到更高密级条款的存在。"""
    objects = json.loads(json.dumps(OBJECTS))
    ret = SemanticRetriever(objects, _StubEncoder())
    assert "secret-salary" in ret.search("高管年薪档位", level=None, top_k=5)
    assert "secret-salary" not in ret.search("高管年薪档位", level=0, top_k=5)


def test_retriever_max_level_from_objects() -> None:
    objects = json.loads(json.dumps(OBJECTS))
    ret = SemanticRetriever(objects, _StubEncoder())
    assert ret.max_level == 2


# ---------------------------------------------------------------- 检索路径开关

def test_llm_preference_uses_llm_retrieval() -> None:
    """auto/llm 模式下 LLM 接管检索（保持既有行为）。"""
    backend = _StubBackend(relevant=["leave"])
    ans = _layer(retriever=True, backend=backend, preference="llm").answer(
        "请假需要什么材料", "employee")
    assert ans.retrieval_mode == "llm"
    assert backend.seen and len(backend.seen[0]) >= 3      # 看到的是全库可见候选


def test_local_preference_keeps_retrieval_out_of_llm() -> None:
    """**隐私关键**：local 模式下检索由本地完成，LLM 只对本地选中的候选做覆盖判定。

    同时验证：LLM 即使返回了别的条款 id，也**不能**改变本地检索结论——
    否则 LLM 又能绕过本地路径把候选捞回来，"检索本地化"就名存实亡。
    """
    backend = _StubBackend(relevant=["work-hours"], covered=True)
    ans = _layer(retriever=True, backend=backend, preference="local").answer(
        "请假需要什么材料", "employee")

    assert ans.retrieval_mode == "hybrid"                  # 检索没走 LLM
    assert backend.seen, "覆盖判定应当仍调用 LLM"
    assert len(backend.seen[0]) <= 3, "LLM 只应看到本地选中的少量候选"
    assert ans.hit_ids == ["leave"], f"LLM 不得改写检索结论，实际 {ans.hit_ids}"


def test_auto_preference_prefers_llm_when_available() -> None:
    backend = _StubBackend(relevant=["leave"])
    ans = _layer(retriever=True, backend=backend, preference="auto").answer(
        "请假需要什么材料", "employee")
    assert ans.retrieval_mode == "llm"


def test_local_preference_without_llm_still_hybrid() -> None:
    """没配 LLM 时 local 与 auto 都退化为本地混合检索（不该报错）。"""
    ans = _layer(retriever=True, backend=None, preference="local").answer(
        "请假需要什么材料", "employee")
    assert ans.retrieval_mode == "hybrid"


def test_invalid_retrieval_preference_raises() -> None:
    """非法取值必须**报错**，不能静默当成 auto——否则隐私设置被写错也无人察觉。"""
    import pytest
    objects = json.loads(json.dumps(OBJECTS))
    with pytest.raises(ValueError):
        GovLayer(objects=objects, roles=dict(ROLES), probe_pairs=PROBE,
                 typo_map={}, retrieval_preference="locall")
