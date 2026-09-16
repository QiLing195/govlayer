# -*- coding: utf-8 -*-
"""用真实数据标定语义检索的**相似度下限**（阈值）。

为什么需要这个脚本：
  向量检索永远会返回 top-k，哪怕问题与知识库毫无关系。所以必须有相似度下限，
  否则"知识空白/范围外"判定会被废掉——实测已经发生过：默认值 0.25 太低，
  问"公司食堂几点开饭？"竟被判成 restricted，页面会显示
  "制度中有相关规定，但超出你的角色权限范围" —— 一句会被当场戳穿的假话。

  阈值不能靠猜（0.25 就是猜的）。正确做法是**量出两个分布的间隔**：
    · 相关问题（该命中）的 top-1 分数分布；
    · 无关问题（不该命中）的 top-1 分数分布；
  阈值取在两者之间。如果两个分布重叠，说明**单靠阈值分不开**——
  那是一个必须如实报告的负结果，而不是"再调一调"。

用法：
    set GOVLAYER_ONNX_MODEL=models\\bge-small-zh-v1.5-int8
    python calibrate_threshold.py
    python calibrate_threshold.py --dataset data/gov_home.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cformer_v63.ann_index import PermissionPartitionedIndex  # noqa: E402
from cformer_v63.embedding import build_encoder               # noqa: E402
from eval_semantic_retrieval import DISTANT_QUERIES, NEAR_QUERIES  # noqa: E402

# 与制度**无关**但同属职场语境的问题：制度里确实没有这些规定，
# 正确判定应当是 out_of_scope / gap，绝不能是 restricted 或 covered。
UNRELATED_QUERIES: list[str] = [
    "公司食堂几点开饭？",
    "班车几点发车？",
    "工装什么时候发？",
    "宿舍怎么申请？",
    "年会什么时候办？",
    "电脑坏了找谁修？",
    "停车位怎么分配？",
    "公司有健身房吗？",
    "年终奖怎么算？",
    "有加班费吗？",
    "调岗需要本人同意吗？",
    "竞业协议签几年？",
    "社保按什么基数交？",
    "公司附近有地铁吗？",
    "团建一般去哪？",
]


def probe(index: PermissionPartitionedIndex, query: str, level: int, top_k: int = 3):
    """不分级过滤地取 top-k（含分数），用于看真实分值分布。"""
    return index.search(query, level=level, top_k=top_k, min_score=None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/gov_employee_rules.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    spec = json.loads(dataset_path.read_text(encoding="utf-8"))
    objects = spec["objects"]
    print(f"\n数据集：{args.dataset}（{len(objects)} 条知识）")

    encoder = build_encoder("auto")
    if encoder.name.startswith("hashing"):
        print("❌ 当前是哈希编码器（无语义能力），标定阈值没有意义。")
        print("   请先设置 GOVLAYER_ONNX_MODEL 指向 ONNX 模型目录。")
        return 2

    index = PermissionPartitionedIndex(encoder).build(objects)
    max_level = max((max(int(k) for k in o.get("levels", {})) for o in objects), default=0)
    print(f"编码器：{encoder.name}（dim={encoder.dim}）| 跨级别检索上限 level={max_level}")

    related = [(q, t) for q, t in [*NEAR_QUERIES, *DISTANT_QUERIES]]
    print(f"相关问题 {len(related)} 条 | 无关问题 {len(UNRELATED_QUERIES)} 条\n")

    rel_rows, unrel_rows = [], []
    print(f"{'类型':<6} {'top1':>7} {'top1 命中目标':>12} {'top3 含目标':>11}  问题")
    print("-" * 88)
    for q, target in related:
        hits = probe(index, q, max_level, args.top_k)
        top1 = hits[0][1] if hits else 0.0
        ids = [h[0] for h in hits]
        rel_rows.append((top1, q, target in ids, ids))
        print(f"{'相关':<6} {top1:>7.4f} {str(target == (ids[0] if ids else None)):>12} "
              f"{str(target in ids):>11}  {q}")
    for q in UNRELATED_QUERIES:
        hits = probe(index, q, max_level, args.top_k)
        top1 = hits[0][1] if hits else 0.0
        ids = [h[0] for h in hits]
        unrel_rows.append((top1, q, ids))
        print(f"{'无关':<6} {top1:>7.4f} {'-':>12} {'-':>11}  {q}   → top: {ids}")

    rel_scores = [r[0] for r in rel_rows]
    unrel_scores = [r[0] for r in unrel_rows]
    min_rel, max_unrel = min(rel_scores), max(unrel_scores)

    print("\n" + "=" * 88)
    print(f"相关问题 top1：最低 {min_rel:.4f}  中位 {statistics.median(rel_scores):.4f}  "
          f"最高 {max(rel_scores):.4f}")
    print(f"无关问题 top1：最低 {min(unrel_scores):.4f}  中位 {statistics.median(unrel_scores):.4f}  "
          f"最高 {max_unrel:.4f}")
    print("-" * 88)

    # 阈值代价曲线：直接看"选多少会损失什么"，比一句"可分/不可分"有用
    print(f"{'阈值':>7} | {'相关召回':>10} | {'无关误纳':>10}")
    print("-" * 40)
    grid = sorted({round(x, 3) for x in
                   [0.40, 0.45, 0.48, 0.50, 0.51, 0.52, 0.53, 0.55, 0.60]})
    for t in grid:
        tp = sum(1 for s in rel_scores if s >= t)
        fp = sum(1 for s in unrel_scores if s >= t)
        print(f"{t:>7.3f} | {tp:>4}/{len(rel_scores):<5} | {fp:>4}/{len(unrel_scores):<5}")
    print("-" * 40)

    if min_rel > max_unrel:
        rec = round((min_rel + max_unrel) / 2, 3)
        print(f"✅ 严格可分：无关最高 {max_unrel:.4f} < 相关最低 {min_rel:.4f}")
        print(f"   建议阈值 {rec}（区间 ({max_unrel:.4f}, {min_rel:.4f}] 内均可）")
    else:
        # 严格可分不成立，但仍可能有"零无关误纳"的可用窗口——这比一句"不可分"有用得多
        above = sorted(s for s in rel_scores if s > max_unrel)
        print(f"⚠️ **严格可分不成立**：无关最高 {max_unrel:.4f} ≥ 相关最低 {min_rel:.4f}"
              f"（重叠窗口仅 {min_rel - max_unrel:.4f}）")
        print("   但中位数差距大，实用上仍有可用窗口：")
        if above:
            t_lo, t_hi = max_unrel, above[0]
            rec = round((t_lo + t_hi) / 2, 3)
            kept = sum(1 for s in rel_scores if s >= rec)
            lost = [(round(s, 4), q) for s, q, _hit, _ids in rel_rows if s < rec]
            print(f"   零无关误纳窗口：({t_lo:.4f}, {t_hi:.4f}]")
            print(f"   ✅ 建议阈值 {rec} → 相关召回 {kept}/{len(rel_scores)}"
                  f"（{kept / len(rel_scores):.0%}），无关误纳 0/{len(unrel_scores)}")
            if lost:
                print(f"   代价：漏掉 {len(lost)} 条相关（这些正是措辞最远的问法）：")
                for s, q in lost:
                    print(f"        {s:.4f}  {q}")
        else:
            print("   ❌ 连可用窗口都没有：**单靠相似度阈值无法分开**（负结果，必须如实记录）。")

    # 与阈值无关的更重要结论
    print("\n" + "!" * 88)
    print("⚠️ 比阈值更要紧的一条（请务必连同阈值一起引用）：")
    print("   **相似度 ≠ 覆盖。** 看上面的无关问题 top-3：几乎总是同几个条款，")
    print("   因为它们在**同一领域**内本就语义邻近。阈值只能过滤'完全不相关'，")
    print("   无法判断'相关但制度没规定'——而后者正是 gap（空白升级人工）。")
    print("   所以：**'制度空白识别'不能建立在相似度阈值上，它需要覆盖判定。**")
    print("   落地含义：GOVLAYER_RETRIEVAL=local 仍需 LLM 做覆盖判定；")
    print("   纯本地（无 LLM）时覆盖判定退化为关键词探针，无法诚实地说'制度没规定'。")
    print("!" * 88)

    worst = sorted(unrel_rows, reverse=True)[:3]
    print("\n最'像相关'的无关问题（阈值必须高于这些才会被判为范围外）：")
    for score, q, ids in worst:
        print(f"  {score:.4f}  {q}  → {ids}")

    return 0 if min_rel > max_unrel else 1


if __name__ == "__main__":
    raise SystemExit(main())
