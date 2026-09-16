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

try:
    from cformer_v63.ann_index import PermissionPartitionedIndex  # noqa: E402
    from cformer_v63.embedding import build_encoder               # noqa: E402
    from eval_query_sets import DATASETS, sets_for                # noqa: E402
except ModuleNotFoundError as exc:
    # 裸 traceback 对要交付给别人的工具不合适：说清缺什么、以及该用哪个解释器。
    _missing = getattr(exc, "name", "?")
    print(f"❌ 缺少依赖：{_missing}")
    print("   本脚本需要 numpy（标定时还需要 onnxruntime 与 tokenizers）。")
    print("   常见原因：用了系统 python 而不是装了依赖的那个解释器。请改用：")
    print(r"     D:\conda\envs\cformer-gpu\python.exe calibrate_threshold.py")
    print("   或先安装依赖：python -m pip install -r requirements-ann.txt")
    raise SystemExit(2) from None


def probe(index: PermissionPartitionedIndex, query: str, level: int, top_k: int = 3):
    """不分级过滤地取 top-k（含分数），用于看真实分值分布。"""
    return index.search(query, level=level, top_k=top_k, min_score=None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/gov_employee_rules.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    dataset_id = dataset_path.stem.replace("gov_", "")
    spec = json.loads(dataset_path.read_text(encoding="utf-8"))
    objects = spec["objects"]
    print(f"\n数据集：{args.dataset}（{len(objects)} 条知识）")

    # 查询集必须与该数据集配套。没有就**硬失败**——否则会拿 A 库的问题给 B 库标定，
    # 打印出一个看起来很正常、却毫无意义的阈值，而它还会被写进代码常数。
    query_sets = sets_for(dataset_id)
    if query_sets is None:
        print(f"❌ 数据集 {dataset_id} 还没有配套的评测查询集。")
        print(f"   现有配套查询集的数据集：{', '.join(DATASETS)}")
        print("   请在 eval_query_sets.py 中为它补上 near / distant / unrelated 三组问题，")
        print("   否则标定结果无意义。")
        return 2
    NEAR_QUERIES, DISTANT_QUERIES, OFF_DOMAIN, SAME_DOMAIN = query_sets

    encoder = build_encoder("auto")
    if encoder.name.startswith("hashing"):
        print("❌ 当前是哈希编码器（无语义能力），标定阈值没有意义。")
        print("   请先设置 GOVLAYER_ONNX_MODEL 指向 ONNX 模型目录。")
        return 2

    # 护栏：本脚本的相关/无关查询集是**针对特定数据集写的**。
    # 若 --dataset 指向别的知识库，就会拿 A 库的问题去和 B 库的条款算相似度，
    # 然后**照样打印一个"建议阈值"**——静默输出垃圾，而且它建议的还是代码常数，
    # 比一般错误更危险。所以这里硬失败，而不是"尽力而为"。
    known_ids = {o.get("id") for o in objects}
    missing = sorted({t for _q, t in [*NEAR_QUERIES, *DISTANT_QUERIES]} - known_ids)
    if missing:
        print(f"❌ 数据集 {args.dataset} 里找不到查询集期望的条款：{missing}")
        print("   本脚本的查询集是为 data/gov_employee_rules.json 写的。")
        print("   换数据集时必须先为它写好配套的'相关/无关'查询集，否则标定结果无意义。")
        return 2

    index = PermissionPartitionedIndex(encoder).build(objects)
    max_level = max((max(int(k) for k in o.get("levels", {})) for o in objects), default=0)
    print(f"编码器：{encoder.name}（dim={encoder.dim}）| 跨级别检索上限 level={max_level}")

    related = [(q, t) for q, t in [*NEAR_QUERIES, *DISTANT_QUERIES]]
    print(f"相关问题 {len(related)} 条 | 远域无关 {len(OFF_DOMAIN)} 条 | "
          f"同域未覆盖 {len(SAME_DOMAIN)} 条\n")

    def scan(queries, tag):
        rows = []
        for q, target in queries:
            hits = probe(index, q, max_level, args.top_k)
            top1 = hits[0][1] if hits else 0.0
            ids = [h[0] for h in hits]
            rows.append((top1, q, target, ids))
            mark = ("True" if target == (ids[0] if ids else None) else "False") if target else "-"
            hit3 = (str(target in ids) if target else "-")
            print(f"{tag:<10} {top1:>7.4f} {mark:>12} {hit3:>11}  {q}"
                  + (f"   → top: {ids}" if not target else ""))
        return rows

    print(f"{'类型':<10} {'top1':>7} {'top1 命中目标':>12} {'top3 含目标':>11}  问题")
    print("-" * 96)
    rel_rows = scan(related, "相关")
    off_rows = scan([(q, None) for q in OFF_DOMAIN], "远域无关")
    same_rows = scan([(q, None) for q in SAME_DOMAIN], "同域无关")

    rel_scores = [r[0] for r in rel_rows]
    off_scores = [r[0] for r in off_rows]
    same_scores = [r[0] for r in same_rows]

    def line(label, scores):
        print(f"{label}：最低 {min(scores):.4f}  中位 {statistics.median(scores):.4f}  "
              f"最高 {max(scores):.4f}")

    print("\n" + "=" * 96)
    line("相关问题      ", rel_scores)
    line("远域无关      ", off_scores)
    line("同域未覆盖    ", same_scores)
    print("-" * 96)

    # 三列代价曲线：一眼看出"阈值能不能同时挡住两档无关问题"
    print(f"{'阈值':>7} | {'相关召回':>10} | {'远域误纳':>10} | {'同域误纳':>10}")
    print("-" * 52)
    for t in (0.40, 0.45, 0.48, 0.50, 0.51, 0.52, 0.53, 0.55, 0.60, 0.65, 0.69):
        tp = sum(1 for s in rel_scores if s >= t)
        print(f"{t:>7.3f} | {tp:>4}/{len(rel_scores):<5} | "
              f"{sum(1 for s in off_scores if s >= t):>4}/{len(off_scores):<5} | "
              f"{sum(1 for s in same_scores if s >= t):>4}/{len(same_scores):<5}")
    print("-" * 52)

    def window(label, other_scores):
        max_other = max(other_scores)
        print(f"\n【{label}】最高 {max_other:.4f}")
        if min(rel_scores) > max_other:
            rec = round((min(rel_scores) + max_other) / 2, 3)
            print(f"  ✅ 严格可分：建议阈值 {rec}"
                  f"（区间 ({max_other:.4f}, {min(rel_scores):.4f}]）")
            return
        above = sorted(s for s in rel_scores if s > max_other)
        if not above:
            print("  ❌ 连零误纳窗口都没有 → **单靠阈值无法分开**（负结果，必须如实记录）")
            return
        t_lo, t_hi = max_other, above[0]
        rec = round((t_lo + t_hi) / 2, 3)
        kept = sum(1 for s in rel_scores if s >= rec)
        print(f"  ⚠️ 严格可分不成立（重叠 {min(rel_scores) - max_other:+.4f}）"
              f"，但存在零误纳窗口 ({t_lo:.4f}, {t_hi:.4f}]")
        print(f"     建议阈值 {rec} → 相关召回 {kept}/{len(rel_scores)}"
              f"（{kept / len(rel_scores):.0%}）")
        lost = [(round(s, 4), q) for s, q, _t, _i in rel_rows if s < rec]
        if lost:
            print(f"     代价：漏掉 {len(lost)} 条相关：")
            for s, q in lost:
                print(f"        {s:.4f}  {q}")

    window("远域无关（完全跑题）", off_scores)
    window("同域未覆盖（本领域但没规定）", same_scores)

    print("\n" + "!" * 96)
    print("⚠️ 比阈值更要紧的一条（请务必连同阈值一起引用）：")
    print("   阈值能挡住'完全跑题'的问题，**挡不住'问的是本领域但制度没规定'的问题**——")
    print("   而后者才是真实员工最常问的。要判断它，需要的是**覆盖判定**，不是相似度。")
    print("   落地含义：GOVLAYER_RETRIEVAL=local 仍需 LLM 做覆盖判定；")
    print("   纯本地（无 LLM）时覆盖判定退化为关键词探针，无法诚实地说'制度没规定'。")
    print("!" * 96)

    worst = sorted(same_rows, reverse=True)[:3]
    print("\n最'像相关'的同域未覆盖问题（阈值必须高于这些才能挡住）：")
    for score, q, _t, ids in worst:
        print(f"  {score:.4f}  {q}  → {ids}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
