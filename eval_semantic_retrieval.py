# -*- coding: utf-8 -*-
"""检索质量实测：**措辞贴近** vs **措辞远离** 时，字符重叠检索与语义检索谁找得到。

为什么要单独做这个测量（这是本脚本存在的全部理由）：

  已有评测（eval_ann_index.py 的 7 条手写问题、eval_semantic_gov.py）用的问题
  **都含有条款关键词**（"旷工""打卡""请假"）。在这种题上，纯关键词/字符重叠检索
  本来就能命中，于是"语义检索提升明显"这种结论是测不出来的——
  两边在同一条起跑线上，看不出差别。

  真正决定语义检索值不值的，是**措辞远离**时还能不能找到：
    "一个月不来上班几天会被开除？"  →  制度原文写的是"旷工…予以劝退"
  两句**没有任何共同关键词**，字符重叠检索几乎必然失败，语义检索才可能成功。

  这恰好也是本项目已知的弱点（身份层对"措辞远离训练"的问法泛化有限），
  所以必须分组报告，而不是给一个混在一起的漂亮总分。

设计要点：
  · 用**暴力精确检索**（不走 ANN），把"编码器质量"与"索引近似误差"彻底分开；
  · 语料必须放足够多的**同域干扰条款**（默认 300 条 HR 主题），
    否则库只有 7 条、top-5 几乎全中，测不出任何东西；
  · 输出**逐条命中与排名**，失败的条目全部打印出来，不只看总分。

用法：
    python eval_semantic_retrieval.py                 # 只用字符重叠（零依赖）
    python eval_semantic_retrieval.py --compare       # 字符重叠 vs ONNX 语义，并排对比
    set GOVLAYER_ONNX_MODEL=models\\bge-small-zh-v1.5-int8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cformer_v63.ann_index import PermissionPartitionedIndex  # noqa: E402
from cformer_v63.embedding import build_encoder               # noqa: E402
# 查询集改为从单一来源取（避免多处定义漂移；换数据集时也不用改本脚本）
from eval_query_sets import sets_for                          # noqa: E402

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

# 本脚本默认针对员工制度库；查询集定义见 eval_query_sets.py
NEAR_QUERIES, DISTANT_QUERIES, _OFF_DOMAIN, _SAME_DOMAIN = sets_for("employee_rules")

# ---- 同域干扰条款词表（HR 主题，越像真制度越有区分度）----
TOPICS = ["报销", "加班", "培训", "绩效", "薪酬", "社保", "公积金", "调岗", "晋升",
          "考核", "劳动合同", "试用期", "转正", "离职", "工作交接", "年假", "婚假",
          "产假", "陪产假", "值班", "轮岗", "夜班", "工装", "食堂", "宿舍", "班车",
          "差旅", "招待", "采购", "用印", "保密", "竞业", "兼职", "招聘", "背调",
          "工龄", "津贴", "奖金", "罚款", "申诉", "工会", "体检", "意外险", "工伤"]
ACTIONS = ["需填写申请表并报部门负责人审批", "由人力资源部统一归档备查",
           "按季度汇总后报送总经理", "须提前三个工作日提交书面说明",
           "由直属主管核定并在系统内登记", "参照上年标准执行",
           "需附相关证明材料原件", "由财务部复核后发放",
           "每半年复核一次并公示", "特殊情况需报总经理批准"]


def load_needles() -> list[dict]:
    """真实制度条款（"针"）+ 1 条仅 level 2 的秘密条款。"""
    spec = json.loads((DATA / "gov_employee_rules.json").read_text(encoding="utf-8"))
    objects = list(spec["objects"])
    objects.append({
        "id": "secret-salary",
        "title": "高管薪酬明细",
        "keywords": ["薪酬", "工资", "年薪", "高管"],
        "levels": {"2": "高管薪酬明细仅 HR 与总经理可见，含各岗位年薪档位与期权授予记录。"},
    })
    return objects


def make_distractors(count: int, rng: random.Random) -> list[dict]:
    """生成同域干扰条款：单条是"主题+动作"的准制度句，且**主题词不出现在针里**。"""
    out = []
    for i in range(count):
        topic = TOPICS[i % len(TOPICS)]
        action = ACTIONS[(i * 7 + 3) % len(ACTIONS)]
        text = f"{topic}管理细则：{topic}{action}。"
        out.append({
            "id": f"dist-{i}",
            "title": f"{topic}管理",
            "keywords": [topic],
            # 大部分是最低密级；每 11 条放一条经理级，顺带验证分区在多级语料下也正常
            "levels": {str(1 if i % 11 == 0 else 0): text},
        })
    rng.shuffle(out)
    return out


def evaluate(index: PermissionPartitionedIndex, queries: list[tuple[str, str]],
             level: int = 0, top_k: int = 5) -> dict:
    """针命中 + 命中时的排名 + 逐条明细（失败的必须看得见）。"""
    hits, ranks, detail = 0, [], []
    for question, expected in queries:
        ids = [oid for oid, _, _ in index.exact_search(question, level, top_k=top_k)]
        hit = expected in ids
        rank = ids.index(expected) + 1 if hit else None
        hits += int(hit)
        ranks.append(rank)
        detail.append({"query": question, "expected": expected, "hit": hit,
                       "rank": rank, "top_ids": ids})
    return {
        "n": len(queries),
        "hits": hits,
        "hit_rate": round(hits / max(1, len(queries)), 4),
        "mean_rank_when_hit": (round(sum(r for r in ranks if r) / hits, 2)
                               if hits else None),
        "detail": detail,
    }


def run_one(encoder, corpus: list[dict], top_k: int, level: int) -> dict:
    index = PermissionPartitionedIndex(encoder, n_probe=1)
    index.build(corpus)                       # 故意用暴力精确检索：隔离"编码器"这一个变量
    return {
        "encoder": encoder.name,
        "near": evaluate(index, NEAR_QUERIES, level=level, top_k=top_k),
        "distant": evaluate(index, DISTANT_QUERIES, level=level, top_k=top_k),
    }


def print_block(title: str, result: dict, top_k: int) -> None:
    print(f"\n【{title}】{result['encoder']}")
    for group, label in (("near", "措辞贴近（含关键词）"), ("distant", "措辞远离（刻意避开关键词）")):
        r = result[group]
        print(f"  {label}：命中 {r['hits']}/{r['n']}（{r['hit_rate']:.0%}）"
              f"，命中时平均排名 {r['mean_rank_when_hit']}")
        for d in r["detail"]:
            if not d["hit"]:
                print(f"      ✗ 未进 top-{top_k}：「{d['query']}」→ 期望 {d['expected']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distractors", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--level", type=int, default=0, help="以该权限级别检索（0=最低）")
    parser.add_argument("--compare", action="store_true", help="字符重叠 vs ONNX 语义 并排对比")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    needles = load_needles()
    corpus = needles + make_distractors(args.distractors, rng)
    print(f"\n语料：真实条款 {len(needles)} 条（含 1 条仅 level 2）+ 同域干扰 {args.distractors} 条 "
          f"= {len(corpus)} 条")
    print(f"检索：暴力精确（不走 ANN，隔离编码器这一个变量） | top-{args.top_k} | 级别 {args.level}")
    print(f"查询：措辞贴近 {len(NEAR_QUERIES)} 条 / 措辞远离 {len(DISTANT_QUERIES)} 条")

    lexical = build_encoder("hashing", verbose=False)
    results = [run_one(lexical, corpus, args.top_k, args.level)]

    semantic_encoder = None
    if args.compare:
        try:
            semantic_encoder = build_encoder("onnx", verbose=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"\n⚠️ ONNX 语义编码器不可用（{type(exc).__name__}: {exc}）")
            print("   本次只测出字符重叠基线。要对比语义效果请先配好 GOVLAYER_ONNX_MODEL，")
            print("   见 DEPLOY / V63_ANN_POC.md 与 fetch_embedding_model.py。")
    if semantic_encoder is not None and not semantic_encoder.name.startswith("hashing"):
        results.append(run_one(semantic_encoder, corpus, args.top_k, args.level))

    for r in results:
        print_block("检索结果", r, args.top_k)

    # 对比表
    print("\n" + "=" * 74)
    print(f"{'检索方式':<34} | {'措辞贴近':>10} | {'措辞远离':>10}")
    print("-" * 74)
    for r in results:
        print(f"{r['encoder'][:33]:<34} | "
              f"{r['near']['hits']}/{r['near']['n']:>8} | {r['distant']['hits']}/{r['distant']['n']:>8}")
    print("=" * 74)

    if len(results) < 2:
        print("⚠️ 只有一种编码器，无法得出「语义检索是否有用」的结论。")
        print("   字符重叠基线在**措辞远离**一组上的失败率，就是语义检索要解决的问题。")
    else:
        lex, sem = results[0], results[-1]
        gain = sem["distant"]["hits"] - lex["distant"]["hits"]
        print(f"措辞远离组：语义比字符重叠多命中 {gain} 条"
              f"（{lex['distant']['hits']}/{lex['distant']['n']} → "
              f"{sem['distant']['hits']}/{sem['distant']['n']}）")
        if gain <= 0:
            print("⚠️ 语义编码器**没有**带来提升——这是一个负结果，应当如实记录，")
            print("   不要用「措辞贴近」那组的成绩来替代它。")

    print("\n限制（引用这些数字时必须一并说明）：")
    print("  · 查询由人工编写，措辞远离组仅 %d 条，样本量小；" % len(DISTANT_QUERIES))
    print("  · 「措辞远离」= 刻意避开该条款的关键词表，**不等于零字重叠**：")
    print("    中文里仍可能有个别字（如「休」「需」）同时出现在问题与条款中，")
    print("    所以字符重叠检索未必全错——这也正是必须分组看、不能只看总分的理由；")
    print("  · 干扰条款为模板合成，与真实制度的措辞分布有差距；")
    print("  · 本脚本测的是**纯编码器检索**，不含 GovLayer 的字段级权限与空白判定。")

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "semantic_retrieval_eval.json"
    out.write_text(json.dumps({"corpus_size": len(corpus), "top_k": args.top_k,
                               "level": args.level, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
