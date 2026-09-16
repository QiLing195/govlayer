# -*- coding: utf-8 -*-
"""本地语义检索器：把 ONNX int8 编码器 + 按权限分区的索引封装成可插拔组件。

为什么要有这一层（而不是直接在 GovLayer 里 import）：
  1. **依赖必须可选**：默认服务镜像只装 fastapi/uvicorn/pydantic/requests，
     不含 numpy/onnxruntime。若在模块顶层硬 import numpy，服务会直接起不来。
     所以这里把 numpy 相关的 import 全部包在 try 里，失败时退化为"不可用"。
  2. **只在真有语义能力时才接入**：哈希编码器（字符 n-gram）没有任何语义能力，
     用它做"混合检索"相对纯关键词**不会有任何提升**，只会徒增复杂度和延迟。
     所以本模块在编码器是 hashing 时**主动返回 None**，让 GovLayer 走原来的关键词路径。
  3. **权限策略必须显式**：见 `search()` 的 `level` 参数与下方说明——
     "能否检索到高密级条款的存在"是一个**产品决策**，不是实现细节。

关于权限的两种立场（GovLayer 的 restricted 判定依赖它）：

  A. 披露"存在但无权"（GovLayer 默认行为）
     员工问"旷工几天劝退"→ 答"制度中有相关规定，但超出你的角色权限范围，请联系 HR"。
     这要求检索**跨权限级别**（否则无法得知"存在"），安全性由**输出时**的
     `visible_content()` 门控保证：内容永远不越权。
     代价：泄露了"存在性"（知道有这么一条规定）。

  B. 连存在都不披露（严格模式）
     同一问题 → 答"无法回答/超出范围"。此时可以按 level 分区检索，
     level > L 的分区在代码路径上不可达，保证更强。
     代价：体验变差（员工不知道该去找 HR）。

  `search(level=None)` = 立场 A（跨级别检索，供 restricted 判定用）；
  `search(level=L)`    = 立场 B（结构性隔离）。

  两者都不是"漏洞"，但**不能混着说**：用了 B 就不能再声称"系统会告诉你去找 HR"。
"""

from __future__ import annotations

import os

# ---- 可选依赖：缺失时整个模块退化为"不可用"，绝不能拖垮服务启动 ----
_LOCAL_IMPORT_ERROR: str | None = None
try:
    from cformer_v63.ann_index import PermissionPartitionedIndex
    from cformer_v63.embedding import build_encoder
except Exception as exc:                                    # noqa: BLE001
    _LOCAL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    PermissionPartitionedIndex = None                       # type: ignore[assignment]
    build_encoder = None                                    # type: ignore[assignment]

LOCAL_SEMANTIC_AVAILABLE = _LOCAL_IMPORT_ERROR is None

# 语义检索的相似度下限。
#
# 这个值的历史很能说明问题：
#   · 最初是**猜的**（0.25）→ 实测太低，问"公司食堂几点开饭？"也捞回 3 条，
#     系统据此说出"制度中有相关规定，但超出你的角色权限范围"——一句会被戳穿的假话；
#   · 后来在**单份语料**（employee_rules）上标成 0.51；
#   · 四语料实测（见 V63_RETRIEVAL_THRESHOLD_REPORT.md）证明**没有单一值适用**：
#       employee_rules 0.499 | registration 0.535 | kaohe 0.557 | renshi 0.567
#
# 取 0.55 作为"四者中位附近"的兜底值——**但它不能替代逐库标定**。
#
# ⚠️ 生产部署必须对每份知识库跑 calibrate_threshold.py，并把结果写进
#    GOVLAYER_SEMANTIC_MIN_SCORE。偏低会放进跑题问题（进而编出"有相关规定"），
#    偏高会漏掉措辞较远的真命中。
# ⚠️ 阈值只能过滤"完全跑题"的问题，**判断"相关但制度没规定"必须靠覆盖判定**：
#    实测 kaohe 上"考核不合格会影响退休待遇吗？"得分 0.6869，深埋在相关组分部内部。
DEFAULT_MIN_SCORE = 0.55


class SemanticRetriever:
    """本地语义检索：一次 build，按需检索。权限语义见模块 docstring。"""

    def __init__(self, objects: list[dict], encoder, *, n_probe: int = 2,
                 cluster_multiplier: float = 1.0,
                 brute_force_threshold: int = 2048,
                 min_score: float | None = DEFAULT_MIN_SCORE) -> None:
        self.encoder = encoder
        self.index = PermissionPartitionedIndex(
            encoder, n_probe=n_probe, cluster_multiplier=cluster_multiplier,
            brute_force_threshold=brute_force_threshold)
        self.index.build(objects)
        self.objects = objects
        self.min_score = min_score
        self.max_level = max(
            (max((int(k) for k in o.get("levels", {})), default=0) for o in objects),
            default=0)

    @property
    def name(self) -> str:
        return getattr(self.encoder, "name", type(self.encoder).__name__)

    def search(self, query: str, level: int | None = None, top_k: int = 3,
               min_score: float | None = None) -> list[str]:
        """返回 object_id 列表（按相关度降序）。

        level=None → 立场 A：跨所有权限级别检索（用于判定"存在但无权"）。
        level=L    → 立场 B：只检索该级别可见的分区，结构性隔离。
        min_score=None → 用实例默认下限（传 -1 可显式关闭过滤，仅评测用）。
        """
        search_level = self.max_level if level is None else level
        floor = self.min_score if min_score is None else min_score
        if floor is not None and floor < 0:
            floor = None
        return [oid for oid, _, _ in self.index.search(query, search_level, top_k=top_k,
                                                      min_score=floor)]

    def __repr__(self) -> str:                                # pragma: no cover
        return f"SemanticRetriever(encoder={self.name!r}, objects={len(self.objects)})"


def build_local_retriever(objects: list[dict], prefer: str = "auto",
                          verbose: bool = True) -> SemanticRetriever | None:
    """构造本地语义检索器；不可用/无语义能力时返回 None（调用方回退关键词）。

    返回 None 的三种情况都会**打印原因**，不静默降级：
      1. 可选依赖没装（numpy/onnxruntime/tokenizers）；
      2. 没配 GOVLAYER_ONNX_MODEL；
      3. 只拿到哈希编码器（无语义能力，接入无收益）。
    """
    if not LOCAL_SEMANTIC_AVAILABLE:
        if verbose:
            print(f"[local-semantic] 不可用（{_LOCAL_IMPORT_ERROR}）→ 检索走关键词路径")
        return None

    model_dir = os.environ.get("GOVLAYER_ONNX_MODEL", "").strip()
    if prefer == "auto" and not model_dir:
        if verbose:
            print("[local-semantic] 未设置 GOVLAYER_ONNX_MODEL → 检索走关键词路径")
        return None

    try:
        encoder = build_encoder(prefer=prefer, verbose=verbose)
    except Exception as exc:                                # noqa: BLE001
        if verbose:
            print(f"[local-semantic] 编码器加载失败（{type(exc).__name__}: {exc}）"
                  f" → 检索走关键词路径")
        return None

    # 哈希编码器无语义能力：接进来只会增加复杂度而不提升召回，明确拒绝
    if getattr(encoder, "name", "").startswith("hashing"):
        if verbose:
            print("[local-semantic] 只拿到无语义的哈希编码器 → 不接入（无收益）")
        return None

    if not objects:
        return None

    retriever = SemanticRetriever(objects, encoder)
    if verbose:
        print(f"[local-semantic] 已接入：{retriever.name}，"
              f"{len(objects)} 条知识，最高级别 {retriever.max_level}")
    return retriever
