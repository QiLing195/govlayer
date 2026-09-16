# -*- coding: utf-8 -*-
"""通用制度文档导入工具：把真实制度原文转成 GovLayer 可用的 dataset。

这是"生产化待补"里那条**知识对象化的自动化工具**——接新客户时把文档喂进来，
产出 `data/gov_*.json`，GovLayer 零改动接入。

为什么要把原文与派生结果分开：
  手写 JSON 会把"原文"与"派生结果"混在一起，出问题无法判断是转录错了还是解析错了。
  分开之后：原文可核对、派生过程可重跑、分层规则是一张**可审的表**。

四个设计决定：

  1. **章名从文档自身解析**，不硬编码——换任何文档都不用改代码。
  2. **关键词 = 领域词表 ∩ 条款原文**，不做 n-gram 自动抽取。
     中文不分词抽 n-gram 会产生"准评价""德表现"这类碎片；词表是可见可审的。
  3. **probe_rules 留空，绝不自动生成**。探针规则是"覆盖判定"的配置，
     而"公司食堂几点开饭"被误判成 restricted 的根因正是随手写的探针规则
     （`几点 → 8:30`）。自动生成覆盖规则是同一个错误的温床。覆盖判定交给 LLM。
  4. **权限分层是显式的、按条款编号的表**，默认全员可见。
     ⚠️ 表里的值是**提议值，必须由业务方确认**——它决定"谁能看到什么"，
     是产品决策，不是技术细节。改一行即可调整。

⚠️ 关于这两份文档的一个重要事实：
  它们都是**公开发布的规范性文件**，绝大部分内容本就应该公开。
  因此它们适合验证**检索质量与规模**，**不适合验证权限隔离**——
  在公开文件上人为设层级，验证"零泄漏"只会得到一个假的通过。
  真实权限边界只能由客户提供。

用法：
    python build_gov_regulation.py --name all        # 生成全部登记的制度
    python build_gov_regulation.py --name kaohe      # 只生成一个
    python build_gov_regulation.py --check --name all # 只校验原文结构
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_CN_DIGITS = "零一二三四五六七八九"

# ---------------------------------------------------------------------------
# 领域词表（每份文档一份；全部取自该文档自身的高价值术语）
# ---------------------------------------------------------------------------

VOCAB_KAOHE: list[str] = [
    "年度考核", "聘期考核", "平时考核", "专项考核",
    "考核委员会", "考核工作领导小组", "考核方案", "年度考核表",
    "总结述职", "民主测评", "绩效评价", "服务对象满意度", "第三方评价",
    "公示", "5个工作日", "复核", "申诉",
    "优秀档次", "合格档次", "基本合格", "不合格档次", "不确定档次", "只写评语",
    "档次", "比例", "20%", "25%", "15%",
    "薪级工资", "绩效工资", "岗位等级", "职员等级", "职称", "任职年限",
    "聘用合同", "续订", "解除聘用", "不再续聘",
    "试用期", "初次就业", "病假", "事假", "产假",
    "挂职", "援派", "驻外", "外派", "双岗位双考核",
    "管理岗位", "专业技术岗位", "工勤技能", "科研人员", "机关工勤人员",
    "干部人事档案", "档案",
    "党纪政务处分", "组织处理", "诫勉", "立案审查调查", "涉嫌犯罪",
    "徇私舞弊", "打击报复", "弄虚作假",
    "政治素质", "德才表现", "工作实绩", "廉洁从业", "岗位职责",
    # 补：初版词表漏掉导致这些条款"只能靠语义检索"，见 --name all 的覆盖率告警
    "党管干部", "德才兼备", "分级分类", "公益服务", "信息化",
    "具体办法", "负责解释", "施行",
]

VOCAB_RENSHI: list[str] = [
    "岗位管理制度", "岗位类别", "岗位等级", "岗位设置方案", "备案",
    "公开招聘", "竞聘上岗", "招聘方案", "招聘信息", "资格审查",
    "考试", "考察", "体检", "公示", "聘用手续",
    "聘用合同", "合同期限", "3年", "试用期", "12个月",
    "10年", "法定退休年龄", "聘用至退休",
    "连续旷工", "15个工作日", "30个工作日", "解除聘用合同", "提前30日书面通知",
    "开除处分", "人事关系终止",
    "平时考核", "年度考核", "聘期考核", "优秀", "合格", "基本合格", "不合格",
    "工作绩效", "服务对象",
    "培训计划", "分级分类培训", "岗前培训", "在岗培训", "转岗培训", "专项培训", "培训经费",
    "奖励", "嘉奖", "记功", "记大功", "荣誉称号", "精神奖励", "物质奖励",
    "处分", "警告", "记过", "降低岗位等级", "撤职", "开除",
    "6个月", "12个月", "24个月", "解除处分",
    "基本工资", "绩效工资", "津贴补贴", "工资增长机制", "工资制度",
    "福利待遇", "工时制度", "休假制度", "社会保险", "退休",
    "人事争议", "劳动争议调解仲裁法", "复核", "申诉", "回避", "近亲属",
    "投诉", "举报", "监察机关",
    "责令限期改正", "赔礼道歉", "恢复名誉", "赔偿",
    "滥用职权", "玩忽职守", "徇私舞弊", "刑事责任",
    # 补：初版词表漏掉导致这些条款"只能靠语义检索"，见 --name all 的覆盖率告警
    "合法权益", "党管干部", "主管部门", "职工代表大会",
    "任职条件", "工作标准", "正常增长机制", "交流",
]

# ---------------------------------------------------------------------------
# 制度登记表
# ---------------------------------------------------------------------------

DOCS: dict[str, dict] = {
    "kaohe": {
        "raw": "data/raw/shiye_kaohe_guiding_2023.txt",
        "out": "data/gov_kaohe.json",
        "description": "事业单位工作人员考核规定（人社部发〔2023〕6号，50 条）",
        "n_articles": 50,
        "roles": {"staff": 0, "manager": 1, "hr": 2},
        "vocab": VOCAB_KAOHE,
        # 分层原则：0 = 与个人权益直接相关的实体规定（全员）；
        #           1 = 组织实施的内部口径；2 = 组织人事/纪检监察的内部处置。
        "levels": {
            15: 1,   # 优秀档次名额比例核定口径（20% / 25% / 15%）
            16: 1,   # 考核委员会或考核工作领导小组的组成
            33: 2,   # 结论性材料存入本人干部人事档案
            35: 2,   # 发现问题后的处理、处分、追究刑事责任
            42: 2,   # 涉嫌违纪违法被立案审查调查期间的处理
            43: 2,   # 受党纪政务处分、组织处理、诫勉时的档次确定
        },
    },
    "renshi": {
        "raw": "data/raw/shiye_renshi_tiaoli.txt",
        "out": "data/gov_renshi.json",
        "description": "事业单位人事管理条例（43 条，泛 HR 主题）",
        "n_articles": 43,
        "roles": {"staff": 0, "manager": 1, "hr": 2},
        "vocab": VOCAB_RENSHI,
        "levels": {
            3: 1,    # 三级人事管理部门的职责分工（内部管理口径）
            7: 1,    # 岗位设置方案报备案（内部流程）
            30: 2,   # 给予处分的证据与程序要求
            31: 2,   # 处分期满解除处分
            39: 2,   # 履职回避要求
            40: 2,   # 投诉举报的调查处理
            41: 2,   # 对单位的责令改正与对责任人员的处分
            42: 2,   # 赔礼道歉、恢复名誉、赔偿
            43: 2,   # 滥用职权、玩忽职守、徇私舞弊的法律责任
        },
    },
}

MAX_KEYWORDS = 14


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def cn_to_int(text: str) -> int:
    """中文数字 → int，覆盖 一..九十九。"""
    if text == "十":
        return 10
    if "十" in text:
        tens_s, _, ones_s = text.partition("十")
        tens = _CN_DIGITS.index(tens_s) if tens_s else 1
        ones = _CN_DIGITS.index(ones_s) if ones_s else 0
        return tens * 10 + ones
    return _CN_DIGITS.index(text)


def parse(raw: str) -> tuple[dict[int, str], dict[int, str]]:
    """解析出 {条号: 正文} 与 {条号: 章名}。章名取自文档自身。"""
    articles: dict[int, str] = {}
    chapter_of: dict[int, str] = {}
    chapter_title = ""
    current_num: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_num is not None and buffer:
            articles[current_num] = "\n".join(buffer).strip()
            chapter_of[current_num] = chapter_title

    chapter_re = re.compile(r"^第([一二三四五六七八九十]+)章[\s\u3000]*(.*)$")
    article_re = re.compile(r"^第([一二三四五六七八九十]+)条[\s\u3000]*(.*)$")

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m_ch = chapter_re.match(line)
        if m_ch:
            flush()
            current_num, buffer = None, []
            chapter_title = re.sub(r"[\s\u3000]+", "", m_ch.group(2))
            continue
        m_art = article_re.match(line)
        if m_art:
            flush()
            current_num = cn_to_int(m_art.group(1))
            buffer = [m_art.group(2).strip()]
            continue
        if current_num is not None:
            buffer.append(line)
    flush()
    return articles, chapter_of


def verify(articles: dict[int, str], expected: int) -> list[str]:
    """结构完整性校验。返回问题列表（空 = 通过）。

    注意：本函数只能保证"条数齐全、没有空条"，
    **不能保证逐字无误**——原文是人工转录的。发现出入请直接改原文再重跑。
    """
    problems: list[str] = []
    missing = sorted(set(range(1, expected + 1)) - set(articles))
    extra = sorted(set(articles) - set(range(1, expected + 1)))
    if missing:
        problems.append(f"缺少条款：{missing}")
    if extra:
        problems.append(f"多出条款（超出预期 {expected} 条）：{extra}")
    for num, text in sorted(articles.items()):
        if len(text) < 10:
            problems.append(f"第{num}条内容过短（{len(text)} 字）：{text[:40]!r}")
    return problems


def keywords_for(text: str, vocab: list[str]) -> list[str]:
    """单条文本命中的领域词（供覆盖率告警用；正式生成走 build_keywords）。"""
    return [t for t in _match_terms(text, vocab)]


def _match_terms(text: str, vocab: list[str]) -> list[str]:
    """长词优先匹配，命中后占位替换，避免短词在长词内部重复计数。"""
    ordered = sorted(set(vocab), key=len, reverse=True)
    found: list[str] = []
    remaining = text
    for term in ordered:
        if term in remaining:
            found.append(term)
            remaining = remaining.replace(term, "\u0000")
    return found


def build_keywords(articles: dict[int, str], vocab: list[str],
                   max_kw: int = MAX_KEYWORDS) -> dict[int, list[str]]:
    """挑选每条的关键词：**本文内出现次数降序** → 文档频率升序 → 长度降序。

    这个排序改了三次，每次都是因为判据太单一：
      · 只按长度降序：把"降低岗位等级"排进来，却挤掉"处分"这种用户真正会打的短词
        （实测导致第29条关键词里没有"处分"）；
      · 只按文档频率升序（越独特越靠前）：把"考核工作领导小组"这种罕见长词排到前面，
        却挤掉本文中心词"年度考核"——而用户恰恰这么问（实测导致第17条失效）。
      · 现在：**先看这个词在这条里重复了多少次**。重复次数就是"它是不是本文主题词"的
        最直接信号，然后再用独特性与长度做次级排序。

    仍需注意：这是启发式。真正的最优关键词只有业务方知道（员工实际会怎么问），
    所以输出始终要人工过一遍。
    """
    ordered = sorted(set(vocab), key=len, reverse=True)
    df = {t: sum(1 for txt in articles.values() if t in txt) for t in ordered}
    result: dict[int, list[str]] = {}
    for num, text in articles.items():
        found = _match_terms(text, vocab)
        found.sort(key=lambda t: (-text.count(t), df[t], -len(t), ordered.index(t)))
        result[num] = found[:max_kw]
    return result


def title_for(num: int, chapter: str, text: str) -> str:
    head = re.split(r"[。；，]", text, maxsplit=1)[0][:20]
    return f"第{num}条（{chapter}）{head}" if chapter else f"第{num}条 {head}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_one(name: str, check_only: bool) -> int:
    spec_in = DOCS[name]
    raw_path = ROOT / spec_in["raw"]
    if not raw_path.exists():
        print(f"❌ 找不到原文：{raw_path}")
        return 2

    raw = raw_path.read_text(encoding="utf-8")
    articles, chapter_of = parse(raw)
    print(f"\n[{name}] {raw_path.name}（{len(raw)} 字）")
    print(f"  解析出 {len(articles)} 条，章节："
          f"{sorted(set(chapter_of.values()), key=lambda c: list(chapter_of.values()).index(c))}")

    problems = verify(articles, spec_in["n_articles"])
    if problems:
        print("  ❌ 结构校验未通过：")
        for p in problems:
            print(f"     · {p}")
        return 1
    print(f"  ✅ 结构校验通过：第一条~第{spec_in['n_articles']}条齐全，无空条")

    kw_map = build_keywords(articles, spec_in["vocab"])
    no_kw = [n for n, kws in kw_map.items() if not kws]
    if no_kw:
        print(f"  ⚠️ {len(no_kw)} 条未匹配到领域词表（只能靠语义检索命中）：{no_kw}")

    if check_only:
        return 0

    objects = []
    for num in sorted(articles):
        text = articles[num]
        level = spec_in["levels"].get(num, 0)
        objects.append({
            "id": f"art{num}",
            "title": title_for(num, chapter_of[num], text),
            "keywords": kw_map[num],
            "levels": {str(level): text},
        })

    out_path = ROOT / spec_in["out"]
    out_path.write_text(json.dumps({
        "dataset": name,
        "description": spec_in["description"],
        "objects": objects,
        "roles": spec_in["roles"],
        # 刻意留空：见模块 docstring 第 3 条
        "probe_rules": [],
        "precedent_cases": [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    per_level: dict[int, list[int]] = {}
    for num in sorted(articles):
        per_level.setdefault(spec_in["levels"].get(num, 0), []).append(num)
    print(f"  已写出 {out_path.relative_to(ROOT)}（{len(objects)} 条）")
    print("  分层：" + " | ".join(
        f"level {lvl} {len(per_level[lvl])} 条" for lvl in sorted(per_level)))
    if spec_in["levels"]:
        print("  ⚠️ 以下分层是**提议值，需业务方确认**（改 DOCS 里的 levels 一行即可）：")
        for num in sorted(spec_in["levels"]):
            lvl = spec_in["levels"][num]
            print(f"     level {lvl}  第{num}条  {articles[num][:32]}…")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="all", choices=[*DOCS, "all"])
    parser.add_argument("--check", action="store_true", help="只校验原文结构，不写文件")
    args = parser.parse_args()

    names = list(DOCS) if args.name == "all" else [args.name]
    worst = 0
    for name in names:
        worst = max(worst, build_one(name, args.check))
        if worst:
            break
    if worst == 0 and not args.check:
        print("\n下一步：为每个新数据集补 near/distant/unrelated 查询集"
              "（eval_query_sets.py），然后跑 calibrate_threshold.py 标定阈值。")
        print("未补查询集的数据集上跑标定会**硬失败**——这是刻意的。")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
