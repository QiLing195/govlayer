# C-Former

> **让企业 AI 从"敢说"变成"说得对、不越权、可追责"。**
> C-Former 是"共享对象世界 + 受控推理层"的检索与身份治理系统——它不是又一个知识库方案，而是给任何 RAG/LLM 装上**确定性治理层**的中间件。

## 为什么存在

企业 AI 落地失败，不是因为模型不够聪明，而是因为**不敢信**：

| 怕什么 | C-Former 怎么解决 |
|---|---|
| 怕泄密——员工问到别人的机密 | **检索前权限 mask**：不可见内容原理上不可达（实测 0 泄漏） |
| 怕乱编——AI 胡诌制度条文 | **制度空白识别**：明文才答，空白诚实升级 HR，绝不编造 |
| 怕无据可查——答错了谁负责 | **全链路留痕**：谁问的、权限判定、检索依据、升级记录全可追溯（`/api/audit`，只存令牌指纹不存原文） |
| 怕"规则在难搞的人手里" | **先例沉淀闭环**：每次人工裁决记录成案例，隐性规则逐步显性化 |

## 核心成果（全部真实数据实测）

| 能力 | 指标 | 报告 |
|---|---|---|
| **GovLayer 通用治理框架**（域无关内核） | 检索 + 字段级权限 + 空白识别 + 先例沉淀；toB 制度 / 流程 / **toC 家庭**三域一套代码跑通 | [`TOB_POC_REPORT.md`](TOB_POC_REPORT.md) |
| **RAG 权限闸门**（检索前过滤） | 敏感问题：无闸门泄漏 67–83% → 闸门后 **0 泄漏** | [`RAG_FUSION_POC.md`](RAG_FUSION_POC.md) |
| 身份解析（精确层 + 神经层混合） | name/alias 精确命中 **100%**；神经层 heldout **99%** | [`V62_OBSERVER_REPORT.md`](V62_OBSERVER_REPORT.md) |
| 理解层（QueryUnderstanding） | 33 条盲测（口语改写/网页语境/库外对象）**100%** | [`V63_RECURSION_REPORT.md`](V63_RECURSION_REPORT.md) |
| 递归层（确定性关系图） | AI/电影/国家**三域** 全 **100%** | 同上 |
| **ANN 规模化检索**（按权限分区） | 1k/10k/50k 全配置**跨级泄漏 0**；5 万条扫 14.9% 达忠实度 0.987（15.7×） | [`V63_ANN_POC.md`](V63_ANN_POC.md) |
| **语义检索**（本地 ONNX int8，数据不出内网） | 措辞远离关键词时：关键词检索 **11%** → 语义 **78%**；措辞贴近时均 100%（关键词排名更优 → 结论是**混合检索**）；阈值已实测标定为 0.51 | [`V63_SEMANTIC_RETRIEVAL_POC.md`](V63_SEMANTIC_RETRIEVAL_POC.md) |

**诚实声明**（项目一贯纪律，负结果完整存档）：零样本跨域迁移不成立（实测 5.2% ≈ 随机）；TTT 查询编码为负结果；ANN 忠实度在**合成干扰语料**上测得、会高估 IVF；**实测发现"相似度 ≠ 覆盖"——同域无关问题的相似度天然接近相关问题，故纯本地（无 LLM）配置无法诚实判定"制度没规定"**；相似度阈值仅在单一语料上标定、换语料须重跑 `calibrate_threshold.py`；措辞远离查询集仅 **9 条**、样本量偏小——这些边界都有报告与数据支撑，不粉饰。

## 架构一览

```
用户问题（任意角色）
  → GovLayer（域无关治理内核）
      ├─ 理解层：意图 + 锚定 + 库外拦截
      ├─ 精确层：对象名/别名 100% 精确匹配
      ├─ 神经层：描述性指代兜底（heldout 99%）
      ├─ 递归层：latest/predecessor 确定性推理
      ├─ 权限 mask + 字段级角色分级
      ├─ 空白识别（明文才答，空白升级）
      └─ 先例沉淀（裁决留痕 → 隐性规则显性化）
  → 审计：全链路可重放
```

**私人化定制 = 换一个 dataset**：企业制度 / 学校流程 / 家庭个人知识，只需提供 `{objects, roles, probe_rules, cases}` JSON，GovLayer 零改动接入。

## 快速开始

```powershell
# 1. 安装（Python 3.10+ / PyTorch 2.x）
pip install -e .[dev]

# 2. 全量测试（85 passed）
#    注意：pyproject.toml 里 testpaths=["tests"]，放在仓库根目录的 test_*.py 会被静默跳过
python -m pytest tests/ -q

# 3. 身份解析训练与评测（真实 AI 模型 273 对象）
python train_eval_real.py --steps 600 --seeds 1 2 3 --d-model 256

# 4. V6.3 递归层（确定性，秒级）——三域全 100%
python train_eval_v63.py
python train_eval_v63.py --data data/movies_dataset.json
python train_eval_v63.py --data data/countries_recursion.json

# 5. GovLayer 通用治理（一套框架三域：企业/流程/toC家庭）
python build_gov_datasets.py && python build_gov_home.py
python demo_govlayer.py

# 6. Web 服务（可部署：FastAPI + 单页前端；镜像不含 torch，启动秒级）
pip install -r requirements-serve.txt
uvicorn --app-dir server app:app --host 127.0.0.1 --port 8000
#   身份由 X-API-Token 决定（请求体 role 一律忽略），token 鉴权见 server/auth.py
python test_auth_security.py      # 8 项角色伪造防护单测（脚本式，直接运行）
python -m pytest tests/test_audit_log.py -q   # 12 项审计日志测试（含"绝不落令牌原文"）
python verify_live_auth.py        # 19 项在线鉴权/越权/审计验证（需服务已启动）

# 7. ANN 规模化检索（可选：只需 numpy/onnxruntime/tokenizers，不含 torch）
#    实测：5 万条扫 14.9% 达忠实度 0.987；≤1 万条建议直接用暴力精确检索
python -m pip install -r requirements-ann.txt
python eval_ann_index.py --sizes 10000 50000 --cluster-multiplier 4

# 8. 语义检索实测（本地 ONNX int8，数据不出内网）
#    结论：措辞远离关键词时 关键词检索 11% → 语义 78%；措辞贴近时两者均 100%
python eval_semantic_retrieval.py                    # 零依赖基线（无需模型）
python fetch_embedding_model.py --endpoint https://hf-mirror.com
$env:GOVLAYER_ONNX_MODEL="models\bge-small-zh-v1.5-int8"
python eval_semantic_retrieval.py --compare
```

> 部署细节、Docker 用法、3 分钟现场演示脚本见 [`DEPLOY.md`](DEPLOY.md)。

## 目录导航

```text
cformer_v59/   治理层：EvidenceVerifier + CandidateLedger
cformer_v60/   共享 Token Transformer（身份编码）
cformer_v63/   GovLayer + 理解层 + 精确层 + 递归层（核心）
cformer_real/  真实数据管线
data/          AI/国家/电影/制度(gov_)/流程(gov_)/家庭(gov_home) 数据集
server/        Web 服务：FastAPI app.py + 令牌鉴权 auth.py + 单页前端 static/
scripts        根目录：build_*.py 数据构建 · train_eval_*.py 训练 · eval_*.py 评测 · demo_govlayer.py 治理演示
TOB_POC_REPORT.md   企业 AI 治理落地完整报告（toB 入口）
RAG_FUSION_POC.md   RAG × C-Former 融合 POC
DEPLOY.md           部署说明（本地/Docker、接口表、3 分钟演示脚本）
V63_ANN_POC.md      ANN 规模化检索 POC（权限分区零泄漏 / 校准 / 负结果）
V63_SEMANTIC_RETRIEVAL_POC.md  语义检索 POC（措辞贴近 vs 远离，分组实测）
```

## 工程

- 版本号映射：内部 V6.x ↔ 测试版 0.6.x
- 大文件（检查点、结果 JSON）在 `artifacts/`，不入库
- 提交信息中文一句话，里程碑打 tag
