# -*- coding: utf-8 -*-
"""通用知识治理层（GovLayer）：域无关的确定性治理框架。

把 toB POC 验证过的四支柱逻辑（权限 mask / 字段级分级 / 空白识别 / 先例沉淀）
抽成**可配置通用模块**——知识库内容、角色、权限规则全部由 dataset 驱动，
本层不含任何业务域硬编码。因此：
  - toB：换企业制度 dataset，即企业制度问答治理；
  - toC：换"个人知识 + 个人权限"dataset，即个人化定制（同一个 GovLayer）。

Dataset 约定（任何域遵守即可接入）：
  objects: [{id, title, keywords:[...], levels: {0: 公开内容, 1: 内容, ...}}]
           levels 的数字 = 可见该内容所需的最低角色级别（0 最低=全员可见）
  roles:   {role_name: level}（如 {"员工":0, "经理":1, "HR":2}）
  precedent_cases: [{case_id, topic, question, context, ruling,
                     reasoning, approver, date, status, reference_only}]

四能力：
  A. GuardLayer.visible_content(obj, role)   —— 字段级权限：返回 role 级别及以下内容
  B. GovLayer.retrieve(text)                  —— 关键词检索 Top-N（可换语义后端）
  C. GovLayer.answer(text, role)              —— 完整回答：检索→分级内容→覆盖探测→GAP/先例
  D. PrecedentStore                           —— 案例沉淀/检索（隐性知识显性化）
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# 常见错别字/同音字纠正（真实用户输入容错）
# 关键词检索对错字零容忍，但真实用户打错字是常态——先纠正再检索，
# 并在回答里透明提示"已将 X 理解为 Y"（不偷偷改，保持可审计）。
DEFAULT_TYPO_MAP = {
    "矿工": "旷工", "旷功": "旷工",
    "请加": "请假", "清假": "请假",
    "打刻": "打卡", "打咖": "打卡",
    "迟道": "迟到",
    "销加": "销假",
    "出拆": "出差",
    "绿通": "绿色通道", "助学": "绿色通道",
}


# ---------------------------------------------------------------- 数据模型

@dataclass
class GovAnswer:
    question: str
    role: str
    hit_ids: list[str] = field(default_factory=list)
    visible_texts: dict[str, str] = field(default_factory=dict)   # 条款id -> 该角色可见内容
    denied_fields: list[str] = field(default_factory=list)        # 命中但字段超权限的条款
    verdict: str = "covered"      # covered | gap | out_of_scope | restricted
    precedents: list[dict] = field(default_factory=list)
    boundary_note: str = ""
    typo_corrections: list[str] = field(default_factory=list)     # 纠错记录（透明可审计）
    semantic_used: bool = False                                    # 本次是否走了 LLM 语义检索
    retrieval_mode: str = "keyword"   # keyword | hybrid | llm —— 本次实际用的检索路径
    gap_reason: str = ""                                           # 空白原因（LLM 指出缺什么）


class GovLayer:
    """确定性治理层：检索 + 字段级权限 + 覆盖判定 + 先例。域无关。"""

    def __init__(self, objects: list[dict], roles: dict[str, int],
                 probe_pairs: list[tuple[list[str], list[str]]] | None = None,
                 cases: list[dict] | None = None,
                 typo_map: dict[str, str] | None = None,
                 semantic_backend=None,
                 semantic_retriever=None,
                 strict_nonexistence_disclosure: bool = False,
                 semantic_min_score: float | None = None,
                 retrieval_preference: str = "auto") -> None:
        """
        objects: 知识对象列表（约定见模块 docstring）
        roles: 角色名 -> 级别（级别=可见字段上限）
        probe_pairs: 覆盖探测规则（**仅当无语义后端时**作为回退使用）
        cases: 先例案例列表（可选）
        typo_map: 错别字纠正表（默认内置常见词；传 {} 可关闭）
        semantic_backend: LLMSemanticBackend 实例（None = 不做 LLM 语义）
        semantic_retriever: 本地语义检索器（None = 纯关键词）。见 local_semantic.py
        strict_nonexistence_disclosure: 见下
             False（默认，立场 A）：跨权限级别检索，因此能判定"存在但无权"
                 → 回答"有相关规定，请联系 HR"。内容由输出门控保证不越权。
             True（立场 B）：只检索本级别可见内容，结构性不披露存在性
                 → 回答"你可见范围内没有对应内容"，但不再能判定 restricted。
            两者安全性不同、体验不同，是**产品决策**，故做成显式开关。
        """
        self.objects = objects
        self.roles = roles
        self._probe_rules = probe_pairs or []
        self.cases = cases or []
        self.typo_map = DEFAULT_TYPO_MAP if typo_map is None else typo_map
        self.semantic_backend = semantic_backend
        self.semantic_retriever = semantic_retriever
        self.strict_nonexistence_disclosure = strict_nonexistence_disclosure
        # 相似度下限：None = 用检索器自带的默认值（见 local_semantic.DEFAULT_MIN_SCORE）
        self.semantic_min_score = semantic_min_score
        # 检索路径偏好（决定"检索这一半会不会出内网"）：
        #   auto （默认）：有 LLM 后端就优先用 LLM 检索（保持既有行为）
        #   local        ：检索一律走本地混合检索；LLM 只对**本地选出的可见候选**
        #                  做覆盖判定 → 制度原文的检索不再发往第三方
        #   llm          ：强制用 LLM 检索
        if retrieval_preference not in ("auto", "local", "llm"):
            raise ValueError(f"retrieval_preference 只能是 auto/local/llm，收到 "
                             f"{retrieval_preference!r}")
        self.retrieval_preference = retrieval_preference
        self._keyword_index: dict[str, list[str]] = {}  # 关键词 -> 对象id列表
        for obj in objects:
            for kw in obj.get("keywords", []):
                self._keyword_index.setdefault(kw, []).append(obj["id"])

    # ---- 错别字容错（真实用户输入）----
    def correct_typos(self, text: str) -> tuple[str, list[str]]:
        """纠正常见错别字，返回（纠正后文本, 纠错记录）。"""
        corrections = []
        corrected = text
        for wrong, right in self.typo_map.items():
            if wrong in corrected and wrong != right:
                corrected = corrected.replace(wrong, right)
                corrections.append(f"{wrong}→{right}")
        return corrected, corrections

    # ---- 检索：关键词（可替换为语义后端）----
    def retrieve(self, text: str, top_n: int = 3) -> list[str]:
        scored: dict[str, int] = {}
        for kw, ids in self._keyword_index.items():
            if kw in text:
                for obj_id in ids:
                    scored[obj_id] = scored.get(obj_id, 0) + 1
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        return [oid for oid, _ in ranked[:top_n]]

    RRF_K = 60   # Reciprocal Rank Fusion 的平滑常数（业界常用 60）

    def _retrieval_level(self, role: str) -> int | None:
        """返回检索应使用的权限级别。

        None  = 跨级别检索（立场 A）：能判定"存在但无权" → restricted
        级别  = 只检索该级别可见分区（立场 B）：结构性不披露存在性
        """
        return self.role_level(role) if self.strict_nonexistence_disclosure else None

    def retrieve_hybrid(self, text: str, top_n: int = 3,
                        level: int | None = None) -> list[str]:
        """关键词 + 本地语义 双路召回，用 RRF 融合。

        为什么用 RRF 而不是加权求和：关键词得分是"命中关键词个数"（整数、无上界），
        语义得分是余弦相似度（[-1,1]，且被归一化），**两者量纲不可比**。
        强行加权需要先做分数校准，而校准本身会引入新的调参面与不确定性。
        RRF 只用**排名**：score = Σ 1/(k + rank)，k=60。
        它天然容忍两路分数分布不同，是这里最稳的选择。

        单路缺失时自动退化为另一路（语义不可用时等价于原来的纯关键词检索）。
        """
        pool = max(top_n * 4, 10)
        kw_ranked = self.retrieve(text, top_n=pool)
        sem_ranked: list[str] = []
        if self.semantic_retriever is not None:
            # 必须带相似度下限：向量检索永远会返回 top-k，
            # 哪怕问题与知识库毫无关系（相似度全为 0）。没有下限，
            # "知识空白 / 范围外"判定会被静默废掉——系统对任何问题都答得出来。
            sem_ranked = self.semantic_retriever.search(
                text, level=level, top_k=pool, min_score=self.semantic_min_score)

        if not sem_ranked:
            return kw_ranked[:top_n]
        if not kw_ranked:
            return sem_ranked[:top_n]

        scores: dict[str, float] = {}
        for rank, oid in enumerate(kw_ranked):
            scores[oid] = scores.get(oid, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        for rank, oid in enumerate(sem_ranked):
            scores[oid] = scores.get(oid, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        # 同分时按 id 排序，保证结果可复现
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [oid for oid, _ in ranked[:top_n]]

    # ---- 字段级权限 ----
    def role_level(self, role: str) -> int:
        return self.roles.get(role, 0)

    def visible_content(self, obj: dict, role: str) -> str:
        """返回 role 级别及以下字段内容（级别数字=可见所需最低级别）。"""
        level = self.role_level(role)
        parts = []
        for field_level, content in sorted(obj.get("levels", {}).items(),
                                           key=lambda kv: int(kv[0])):
            if int(field_level) <= level:
                parts.append(content)
        return " ".join(parts)

    def max_field_level(self, obj: dict) -> int:
        levels = obj.get("levels", {})
        return max((int(k) for k in levels), default=0)

    # ---- 覆盖探测（空白识别）：条款内容是否覆盖问题诉求 ----
    def _probe_covered(self, question: str, hit_content: str) -> bool:
        for q_kws, answer_probes in self._probe_rules:
            if any(kw in question for kw in q_kws):
                return any(probe in hit_content for probe in answer_probes)
        # 无匹配规则时，默认命中即有内容可答（保守：不误判空白）
        return True

    def _covered_with_full_access(self, question: str, oids: list[str]) -> bool:
        """假设把命中条款的**全部级别**内容都给用户，能否覆盖该问题？

        这是区分"权限截断"与"知识空白"的判据，也是**诚实性守卫**：
          · 能覆盖   → 确实是"有规定但你没权限" → restricted（诚实，用户该去找 HR）
          · 不能覆盖 → 即使高密级内容全给它也答不了 → 是知识空白，**不能报 restricted**

        为什么必须加这一步：原来只要"gap + 有被截断字段"就报 restricted，
        而 gap 来自与语义命中无关的关键词探针，两者一凑就会对**无关问题**
        说出"制度中有相关规定，但超出你的角色权限范围"——一句会被当场戳穿的假话。
        实测触发案例：问"公司食堂几点开饭？"。
        """
        by_id = {o["id"]: o for o in self.objects}
        parts: list[str] = []
        for oid in oids:
            obj = by_id.get(oid)
            if not obj:
                continue
            for _lvl, content in sorted(obj.get("levels", {}).items(),
                                        key=lambda kv: int(kv[0])):
                parts.append(str(content))
        return self._probe_covered(question, " ".join(parts))

    # ---- 先例检索 ----
    def search_precedents(self, question: str, top_n: int = 2) -> list[dict]:
        hits = []
        for case in self.cases:
            topic = case.get("topic", "")
            if any(tok in question and tok in topic + case.get("question", "")
                   for tok in ["出差", "超期", "请假", "迟到", "打卡", "报销", "申诉"]):
                hits.append(case)
        return hits[:top_n]

    # ---- 语义候选：只把【该角色可见的内容】交给 LLM ----
    def _visible_candidates(self, role: str, corrected_question: str) -> list[dict]:
        """构造语义检索候选——权限过滤在前，LLM 只看得到可见内容（零泄漏前提）。"""
        candidates = []
        for obj in self.objects:
            vis = self.visible_content(obj, role)
            if vis:  # 该角色无可见内容的条款不进候选（不交给 LLM）
                candidates.append({"id": obj["id"], "title": obj.get("title", obj["id"]),
                                   "content": vis})
        # 大库先粗筛，控制 prompt 规模（可扩展性）
        backend = self.semantic_backend
        if backend is not None and len(candidates) > backend.max_candidates:
            # 粗筛也走混合检索：候选本来就已按角色过滤，所以这里用该角色的级别，
            # 不会把不可见内容重新捞回来。
            kw = set(self.retrieve_hybrid(corrected_question,
                                          top_n=backend.max_candidates,
                                          level=self.role_level(role)))
            filtered = [c for c in candidates if c["id"] in kw]
            if filtered:
                return filtered
            return candidates[: backend.max_candidates]
        return candidates

    def _candidates_for_ids(self, role: str, ids: list[str]) -> list[dict]:
        """把指定条款 id 组装成候选（内容仍按角色过滤）。

        local 模式专用：LLM 只看到**本地检索已选中的**那些条款，而不是全库可见内容。
        既减少外发内容量，也避免 LLM 绕过本地检索结论重新捞回别的条款。
        """
        by_id = {o["id"]: o for o in self.objects}
        out: list[dict] = []
        for oid in ids:
            obj = by_id.get(oid)
            if obj is None:
                continue
            vis = self.visible_content(obj, role)
            if vis:
                out.append({"id": oid, "title": obj.get("title", oid), "content": vis})
        return out

    # ---- 完整回答 ----
    def answer(self, question: str, role: str) -> GovAnswer:
        ans = GovAnswer(question=question, role=role)
        # 错别字容错：先纠正再检索（透明记录，不偷偷改）
        corrected, corrections = self.correct_typos(question)
        ans.typo_corrections = corrections

        # 1) 检索：auto 模式优先 LLM（它同时做检索与覆盖判定）；
        #    local 模式一律走本地混合检索（关键词 + 本地语义 RRF），
        #    LLM 只对**本地选出的可见候选**做覆盖判定 → 检索不再出内网。
        retrieval_level = self._retrieval_level(role)
        backend_ready = (self.semantic_backend is not None
                         and self.semantic_backend.available
                         and self.retrieval_preference != "local")
        semantic_result = None
        if backend_ready:
            candidates = self._visible_candidates(role, corrected)
            semantic_result = self.semantic_backend.match(corrected, candidates)
        if semantic_result is not None:
            known_ids = {o["id"] for o in self.objects}
            # 过滤模型可能编造的 id（只保留真实条款）
            hit_ids = [i for i in semantic_result.relevant_ids if i in known_ids]
            ans.semantic_used = True
            ans.retrieval_mode = "llm"
        else:
            hit_ids = self.retrieve_hybrid(corrected, top_n=3, level=retrieval_level)
            ans.retrieval_mode = ("hybrid" if self.semantic_retriever is not None
                                  else "keyword")
            # local 模式：LLM 退居"覆盖判定"，且只看本地已选中的可见候选。
            # 注意 relevant_ids 在这里**不使用**——检索结论已经由本地路径给出，
            # 否则 LLM 又能绕过权限把候选捞回来。
            if (self.retrieval_preference == "local" and hit_ids
                    and self.semantic_backend is not None
                    and self.semantic_backend.available):
                coverage = self.semantic_backend.match(
                    corrected, self._candidates_for_ids(role, hit_ids))
                if coverage is not None:
                    semantic_result = coverage
                    ans.semantic_used = True

        ans.hit_ids = hit_ids
        if not hit_ids:
            # 空命中时区分：真空白 vs 相关条款存在但超出该角色权限
            raw_hits = self.retrieve_hybrid(corrected, top_n=3, level=retrieval_level)
            restricted_hits = [
                oid for oid in raw_hits
                if self.max_field_level(next(o for o in self.objects if o["id"] == oid))
                > self.role_level(role)
            ]
            if restricted_hits:
                ans.verdict = "restricted"
                ans.hit_ids = restricted_hits
                ans.denied_fields = restricted_hits
                for oid in restricted_hits:
                    ans.visible_texts[oid] = self.visible_content(
                        next(o for o in self.objects if o["id"] == oid), role)
                ans.boundary_note = ("该问题涉及的信息超出你的角色权限范围，未向你展示；"
                                     "如需了解请联系 HR 查询。")
                if corrections:
                    ans.boundary_note += " 已将输入中的 " + "、".join(corrections) + " 按制度用词理解。"
                return ans
            ans.verdict = "out_of_scope"
            ans.precedents = self.search_precedents(corrected)
            if self.strict_nonexistence_disclosure:
                # 立场 B：不披露"是否存在高密级条款"，所以不能沿用"不在知识范围内"
                # 这种说法——那会变成一句可能为假的断言。
                ans.boundary_note = "在你可见的范围内没有对应内容，如需确认请直接联系 HR。"
            else:
                ans.boundary_note = "问题不在已登记知识范围内，建议升级人工。"
            return ans

        # 2) 字段级权限 + 覆盖判定
        for oid in hit_ids:
            obj = next(o for o in self.objects if o["id"] == oid)
            level = self.role_level(role)
            max_level = self.max_field_level(obj)
            vis = self.visible_content(obj, role)
            ans.visible_texts[oid] = vis
            if max_level > level:
                ans.denied_fields.append(oid)  # 该条款有更高权限字段，已被截断
                if not vis:
                    ans.boundary_note = (
                        f"命中条目 [{obj['id']}] 的内容级别为 {max_level}，"
                        f"你的角色 [{role}] 级别 {level} 无查看权限。"
                    )
                    continue
            # 覆盖判定：语义后端结果优先；否则回退词表探针
            if semantic_result is not None:
                covered = semantic_result.covered
                if not covered and semantic_result.missing:
                    ans.gap_reason = semantic_result.missing
            else:
                covered = self._probe_covered(corrected, vis)
            if not covered:
                ans.verdict = "gap"

        # 修正：若命中的条款**全部**超出该角色权限（visible_texts 全为空），
        # 循环里的 `continue` 会跳过覆盖判定，导致 verdict 停留在默认 "covered"——
        # 前端会打出"明文可答"标签，而实际一个字都没返回，属于**误报**。
        # 这类条款此前在数据集中恰好不存在，所以一直没暴露；接入语义检索后
        # （匹配面比关键词更宽）这条路径会更容易走到，必须先修。
        if (ans.verdict == "covered" and ans.denied_fields
                and not any(ans.visible_texts.values())):
            if self._covered_with_full_access(corrected, ans.denied_fields):
                ans.verdict = "restricted"
                ans.boundary_note = ("制度中有相关规定，但超出你的角色权限范围，"
                                     "未向你展示；如需了解请联系 HR 查询。")
            else:
                # 高密级内容全给它也答不了 → 不是权限问题，是制度里没这条
                ans.verdict = "out_of_scope"
                ans.boundary_note = "问题不在已登记知识范围内，建议升级人工。"

        ans.precedents = self.search_precedents(corrected)
        if ans.verdict == "gap":
            # 关键区分：知识空白 vs 权限截断（真实场景语义完全不同）
            if ans.denied_fields and self._covered_with_full_access(corrected,
                                                                   ans.denied_fields):
                ans.verdict = "restricted"
                ans.boundary_note = ("制度中有相关规定，但超出你的角色权限范围，"
                                     "未向你展示；如需了解请联系 HR 查询。")
            else:
                reason = f"（缺少：{ans.gap_reason}）" if ans.gap_reason else ""
                if ans.precedents:
                    ans.boundary_note = (f"知识未覆盖此问题{reason}，但有过往先例可参考"
                                         "（仅供参考，以实际确认为准）。")
                else:
                    ans.boundary_note = (f"知识未覆盖此问题{reason}，且无过往先例。"
                                         "AI 不编造规则：建议升级人工裁决，裁决后将沉淀为先例。")
        elif ans.denied_fields and not ans.boundary_note:
            ans.boundary_note = "已返回你可视范围内内容；命中条目含更高权限字段，未向你展示。"
        # 纠错透明提示（可审计：让用户知道你理解成了什么）
        if corrections:
            hint = "已将输入中的 " + "、".join(corrections) + " 按制度用词理解。"
            ans.boundary_note = (ans.boundary_note + " " + hint).strip()
        return ans

    # ---- 案例沉淀 ----
    def add_precedent(self, *, topic: str, question: str, context: str,
                      ruling: str, reasoning: str, approver: str,
                      department: str = "") -> dict:
        case = {
            "case_id": f"C-{time.strftime('%Y')}-{100 + len(self.cases):03d}",
            "topic": topic, "question": question, "context": context,
            "ruling": ruling, "reasoning": reasoning,
            "approver": approver, "department": department,
            "date": time.strftime("%Y-%m-%d"), "status": "closed",
            "reference_only": True,
        }
        self.cases.append(case)
        return case


# ---------------------------------------------------------------- 便捷加载

def load_dataset(dataset_path: str | Path) -> dict:
    """从 dataset JSON 加载 objects/roles/probe_rules/cases。"""
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    return {
        "objects": data.get("objects", []),
        "roles": data.get("roles", {"user": 0}),
        "probe_rules": data.get("probe_rules", []),
        "cases": data.get("precedent_cases", []),
    }


def build_govlayer(dataset_path: str | Path, cases_path: str | Path | None = None,
                   semantic_backend=None, semantic_retriever=None,
                   local_semantic: bool = False,
                   strict_nonexistence_disclosure: bool = False,
                   retrieval_preference: str = "auto") -> GovLayer:
    """一行构建：dataset 驱动。toB/toC 只换 dataset 路径。

    semantic_backend 非空 → 启用 LLM 语义检索 + 覆盖判定；
    local_semantic=True   → 接入本地 ONNX 语义检索（与关键词 RRF 混合），
                            不联网、数据不出内网；模型不可用时自动退回关键词；
    retrieval_preference  → "auto"（LLM 优先）/ "local"（检索本地化，LLM 只做覆盖判定）
                            / "llm"；
    strict_nonexistence_disclosure=True → 立场 B（连"存在但无权"也不披露）。
    """
    spec = load_dataset(dataset_path)
    cases = list(spec["cases"])
    if cases_path and Path(cases_path).exists():
        cases.extend(json.loads(Path(cases_path).read_text(encoding="utf-8")))
    if local_semantic and semantic_retriever is None:
        # 延迟导入：没装 numpy/onnxruntime 时不能拖垮只做关键词的调用方
        from cformer_v63.local_semantic import build_local_retriever  # noqa: PLC0415
        semantic_retriever = build_local_retriever(spec["objects"])
    return GovLayer(objects=spec["objects"], roles=spec["roles"],
                    probe_pairs=spec["probe_rules"], cases=cases,
                    semantic_backend=semantic_backend,
                    semantic_retriever=semantic_retriever,
                    strict_nonexistence_disclosure=strict_nonexistence_disclosure,
                    retrieval_preference=retrieval_preference)
