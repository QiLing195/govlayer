# -*- coding: utf-8 -*-
"""C-Former GovLayer Web 服务：把确定性治理层封装成可部署 API + 前端。

设计要点（部署友好）：
  - GovLayer 纯标准库，服务本身不依赖 torch → 镜像小、启动秒级；
  - 治理逻辑（权限/空白/先例）在服务端确定性执行，LLM 只负责"把可见内容写成自然语言"；
  - 可选接入 DeepSeek：**只有权限过滤后的可见内容才进 Prompt**（泄漏在源头掐断）。

接口：
  GET  /healthz                 → 健康检查
  GET  /api/datasets            → 可用知识库 + 角色列表
  GET  /api/tokens              → 演示用身份令牌（生产应下线，改由 SSO 签发）
  POST /api/ask                 → {dataset, question} → 治理后答案 + 依据
                                  身份取自请求头 X-API-Token（请求体 role 一律忽略）
  GET  /                        → 前端演示页

鉴权（#3 角色伪造防护）：
  客户端只能提供 Token；服务端解析出**权限级别**，再按知识库映射到角色名。
  这样同一个 Token 在"企业制度/迎新报到/家庭"等不同库自动落到对应角色。

运行：
  pip install -r requirements-serve.txt
  uvicorn app:app --host 0.0.0.0 --port 8000     # 在 server/ 目录下
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVER_DIR))   # 使同目录 auth.py 可导入

from cformer_v63.governance import GovLayer, load_dataset  # noqa: E402
from cformer_v63.semantic import LLMSemanticBackend  # noqa: E402
# 本地语义检索（#4b）：依赖可选。numpy/onnxruntime 缺失时该模块自身退化为不可用，
# 不会拖垮服务启动——默认服务镜像不含这两个依赖。
from cformer_v63.local_semantic import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    build_local_retriever,
)
from auth import DEMO_TOKENS, dev_mode, resolve_identity, role_for_level  # noqa: E402
from audit import AuditLog, question_field, token_fingerprint  # noqa: E402

DATA_DIR = ROOT / "data"
STATIC_DIR = SERVER_DIR / "static"

app = FastAPI(title="C-Former GovLayer", version="0.2.0")

# 语义后端（#1 检索语义化 + #2 覆盖判定）：有 DEEPSEEK_API_KEY 时启用，
# 否则自动降级为关键词检索 + 词表探针（零依赖可跑）。
SEMANTIC = LLMSemanticBackend()

# 审计日志（#5）：追加式 JSONL，绝不落原始令牌，只存指纹；写盘失败不影响业务。
AUDIT = AuditLog()

# ---- 知识库加载（dataset 驱动：新客户只需加一个 gov_*.json）----
_LAYERS: dict[str, GovLayer] = {}
_DATASETS: dict[str, dict] = {}
_WARNED_UNSET_FLOOR = [False]   # "未标定阈值"只提示一次（本函数按数据集调用多次）


def _load_all_datasets() -> None:
    # 混合检索开关：默认开启，但只有真正配好 GOVLAYER_ONNX_MODEL 才会生效；
    # 置 GOVLAYER_LOCAL_SEMANTIC=0 可强制退回纯关键词（排障用）。
    use_local = os.environ.get("GOVLAYER_LOCAL_SEMANTIC", "1") not in ("0", "false", "False")
    # 相似度下限：未在真实语料上校准，允许用环境变量覆盖（见 local_semantic.py 的说明）
    raw_floor = os.environ.get("GOVLAYER_SEMANTIC_MIN_SCORE", "").strip()
    try:
        min_score = float(raw_floor) if raw_floor else None
    except ValueError:
        print(f"[local-semantic] 忽略非法 GOVLAYER_SEMANTIC_MIN_SCORE={raw_floor!r}")
        min_score = None
    for path in sorted(DATA_DIR.glob("gov_*.json")):
        spec = load_dataset(path)
        dataset_id = path.stem.replace("gov_", "")
        retriever = (build_local_retriever(spec["objects"]) if use_local else None)
        if retriever is not None and min_score is None and not _WARNED_UNSET_FLOOR[0]:
            # 阈值没有单一适用值（实测四份语料 0.499~0.567），必须逐库标定。
            # 用兜底值时必须**说出来**，否则"未标定"会被当成"已调好"。
            _WARNED_UNSET_FLOOR[0] = True
            print("[local-semantic] ⚠️ 未设置 GOVLAYER_SEMANTIC_MIN_SCORE，"
                  f"使用兜底值 {DEFAULT_MIN_SCORE}。")
            print("           实测四份语料的建议值为 0.499 / 0.535 / 0.557 / 0.567 —— "
                  "没有任何单一值适用。")
            print("           每份知识库都应单独标定（calibrate_threshold.py），"
                  "否则会放进跑题问题、进而编出'制度中有相关规定'。")
        pref = os.environ.get("GOVLAYER_RETRIEVAL", "auto").strip().lower() or "auto"
        if pref not in ("auto", "local", "llm"):
            print(f"[local-semantic] 非法 GOVLAYER_RETRIEVAL={pref!r} → 按 auto 处理")
            pref = "auto"
        _LAYERS[dataset_id] = GovLayer(
            objects=spec["objects"], roles=spec["roles"],
            probe_pairs=spec["probe_rules"], cases=spec["cases"],
            semantic_backend=SEMANTIC if SEMANTIC.available else None,
            semantic_retriever=retriever,
            semantic_min_score=min_score,
            retrieval_preference=pref,
        )
        _DATASETS[dataset_id] = {
            "id": dataset_id,
            "file": path.name,
            "roles": spec["roles"],
            "n_objects": len(spec["objects"]),
            "n_cases": len(spec["cases"]),
            "local_semantic": retriever is not None,
            "llm_semantic": SEMANTIC.available,
        }


_load_all_datasets()


# ---- 请求/响应模型 ----
class AskRequest(BaseModel):
    dataset: str
    question: str
    # 已废弃：身份由 X-API-Token 决定，此字段一律忽略（防伪造越权）
    role: str | None = None


class AskResponse(BaseModel):
    question: str
    role: str                    # 服务端解析出的角色（非客户端声明）
    verdict: str                 # covered | restricted | gap | out_of_scope
    answer_text: str             # 治理后返回给用户的内容（LLM 或条款原文）
    visible_sources: list[dict]  # [{id, title, content}] 权限过滤后的依据
    denied_fields: list[str]     # 命中但因权限被截断的条目
    precedents: list[dict]       # 过往先例（仅供参考）
    boundary_note: str           # 边界声明（空白/权限/先例提示）
    typo_corrections: list[str]  # 错别字纠正记录（透明可审计）
    semantic_used: bool          # LLM 是否参与（检索或覆盖判定）；纯本地检索时为 False
    retrieval_mode: str          # keyword | hybrid | llm —— 本次实际生效的检索路径
    identity: str                # 服务端解析出的身份（来自令牌，非客户端声明）
    llm_used: bool


def _llm_answer(question: str, visible_context: str, verdict: str) -> str | None:
    """可选：把**权限过滤后的可见内容**交给 DeepSeek 生成自然语言答案。
    关键：Prompt 只包含可见内容——不可见内容在检索前已被治理层排除。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None
    if verdict == "gap":
        return None  # 空白问题不交给 LLM（避免编造），走治理层的升级话术
    import requests

    system = ("你是企业知识库助手。只能依据下方【已授权资料】回答；"
              "资料不足时必须回答'根据现有资料无法回答'，不得编造任何制度条款。")
    prompt = f"【已授权资料】\n{visible_context}\n\n【员工提问】{question}\n\n【回答】"
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 300},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001 —— LLM 不可用时降级为条款原文
        return None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "datasets": list(_DATASETS)}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    return list(_DATASETS.values())


@app.get("/api/tokens")
def tokens() -> list[dict]:
    """演示用身份令牌列表（生产环境应下线此接口，改由 SSO 签发）。"""
    return [{"token": t, "label": info["label"], "level": info["level"]}
            for t, info in DEMO_TOKENS.items()]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest,
        x_api_token: str | None = Header(default=None, alias="X-API-Token")) -> AskResponse:
    t0 = time.perf_counter()
    rec: dict = {
        "dataset": req.dataset,
        "token_fp": token_fingerprint(x_api_token),   # 只存指纹，绝不存令牌原文
        **question_field(req.question),
    }
    try:
        # #3 安全：先校验身份，再解析数据集。
        # 顺序很重要：若先查 dataset，未认证请求会因库名存在与否得到 404 / 401 两种
        # 结果，这个差异本身就变成一个**未认证的知识库名枚举探针**。
        # 先认证则未认证请求一律 401，不泄漏任何数据集信息。
        role_levels = _LAYERS[req.dataset].roles if req.dataset in _LAYERS else {}
        identity = resolve_identity(x_api_token, req.role, role_levels)
        if identity is None:
            # 未认证请求：只留"有人试过"的信号，**不留问题原文**。
            # 否则任何人无需凭证就能往审计日志里灌任意内容（日志投毒 / 撑爆磁盘），
            # 审计日志本身反而成了攻击面。
            for k in ("question", "question_sha256", "question_redacted"):
                rec.pop(k, None)
            rec["question_withheld"] = "unauthenticated"
            rec.update(event="auth_failed", reason="invalid_or_missing_token", status=401)
            raise HTTPException(status_code=401,
                                detail="缺少或无效的身份令牌（请在 X-API-Token 头提供）")
        rec.update(identity_label=identity.label, level=identity.level)

        layer = _LAYERS.get(req.dataset)
        if layer is None:
            rec.update(event="ask_denied", reason="unknown_dataset", status=404)
            raise HTTPException(status_code=404, detail=f"unknown dataset: {req.dataset}")

        role = role_for_level(layer.roles, identity.level)
        if role is None:
            rec.update(event="ask_denied", reason="level_below_dataset_minimum", status=403)
            raise HTTPException(status_code=403,
                                detail=f"你的权限级别 {identity.level} 无权访问该知识库")
        rec["role"] = role

        ans = layer.answer(req.question, role)

        # 权限过滤后的可见依据（只有这些内容会被交给 LLM）
        sources = []
        for oid, content in ans.visible_texts.items():
            obj = next(o for o in layer.objects if o["id"] == oid)
            sources.append({"id": oid, "title": obj.get("title", oid),
                            "content": content or "（无权限查看此内容）"})

        visible_context = "\n---\n".join(
            f"《{s['title']}》：{s['content']}" for s in sources
            if s["content"] and "无权限" not in s["content"]
        )
        # 回答生成：
        #  - restricted（权限截断）：**不交给 LLM**——否则模型会把"你无权限"说成"资料不足"，
        #    必须由治理层直接给出权限声明（这是本系统的核心语义，不能被生成层稀释）；
        #  - gap（知识空白）：也不交给 LLM（避免编造），走治理层升级话术；
        #  - 其余情况才允许 LLM 基于可见内容生成自然语言。
        llm_text = None
        if ans.verdict not in ("restricted", "gap"):
            llm_text = _llm_answer(req.question, visible_context, ans.verdict)

        if ans.verdict == "restricted":
            answer_text = ans.boundary_note or "该问题涉及的信息超出你的角色权限范围。"
        elif llm_text:
            answer_text = llm_text
        elif ans.verdict == "gap":
            answer_text = ("根据现有制度无法回答此问题。" + (ans.boundary_note or ""))
        elif ans.verdict == "out_of_scope":
            answer_text = "此问题不在已登记制度范围内，建议咨询 HR。"
        else:
            answer_text = visible_context or "根据你的权限，没有可展示的内容。"

        # 审计重点记录"判定"而非"回答文本"：复盘时要回答的是
        # 「谁·问了哪个知识点·为什么被截断」，不是复述答案。
        rec.update(event="ask", status=200, verdict=ans.verdict,
                   hit_ids=list(ans.hit_ids), denied_fields=list(ans.denied_fields),
                   n_sources=len(sources), semantic_used=bool(ans.semantic_used),
                   retrieval_mode=ans.retrieval_mode,
                   llm_used=bool(llm_text),
                   typo_corrections=list(ans.typo_corrections))

        return AskResponse(
            # role 必须是**服务端解析出的角色**：回显 req.role 会把伪造声明当成事实，
            # 前端据此展示身份就会把"员工"显示成"HR"（#3 修复的最后一个漏点）。
            question=req.question, role=role, verdict=ans.verdict,
            answer_text=answer_text,
            visible_sources=sources,
            denied_fields=ans.denied_fields,
            precedents=[{k: c.get(k) for k in ("case_id", "topic", "ruling", "reasoning",
                                               "approver", "date", "reference_only")}
                        for c in ans.precedents],
            boundary_note=ans.boundary_note or "",
            typo_corrections=ans.typo_corrections,
            semantic_used=ans.semantic_used,
            retrieval_mode=ans.retrieval_mode,
            identity=f"{identity.label}（级别 {identity.level} → 角色 {role}）",
            llm_used=bool(llm_text),
        )
    finally:
        # 成功 / 401 / 403 / 404 / 甚至未预期异常（500）都留痕：
        # 审计的价值恰恰在"失败与拒绝"上，只记成功等于没记。
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        AUDIT.record(rec)


# ---- 审计查询（本身也受权限门控：审计日志不是谁都能看）----
def _auditor_level() -> int:
    """能看审计日志所需的级别 = 所有知识库中的最高级别。"""
    levels = [lvl for spec in _DATASETS.values() for lvl in spec["roles"].values()]
    return max(levels) if levels else 0


def _require_auditor(x_api_token: str | None):
    ident = resolve_identity(x_api_token, None, {})
    if ident is None:
        AUDIT.record({"event": "auth_failed", "scope": "audit",
                      "reason": "invalid_or_missing_token", "status": 401,
                      "token_fp": token_fingerprint(x_api_token)})
        raise HTTPException(status_code=401,
                            detail="缺少或无效的身份令牌（请在 X-API-Token 头提供）")
    if ident.level < _auditor_level():
        AUDIT.record({"event": "audit_denied", "scope": "audit",
                      "reason": "level_below_auditor", "level": ident.level,
                      "status": 403, "token_fp": token_fingerprint(x_api_token)})
        raise HTTPException(status_code=403,
                            detail=f"审计日志需要权限级别 ≥ {_auditor_level()}")
    return ident


@app.get("/api/audit")
def audit_tail(limit: int = 50, event: str | None = None,
               x_api_token: str | None = Header(default=None, alias="X-API-Token")) -> dict:
    """最近 N 条审计记录（最新在前）+ 汇总统计。"""
    _require_auditor(x_api_token)
    return {"records": AUDIT.tail(limit, event), "stats": AUDIT.stats()}


@app.get("/api/audit/stats")
def audit_stats(x_api_token: str | None = Header(default=None,
                                                 alias="X-API-Token")) -> dict:
    """审计汇总：按判定/事件/角色统计 + 鉴权失败次数 + 延迟。"""
    _require_auditor(x_api_token)
    return AUDIT.stats()


# ---- 前端 ----
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
