# 部署说明（C-Former GovLayer）

把确定性治理层部署为一个可访问的 Web 服务：**多角色提问 → 权限隔离 → 空白诚实升级 → 依据可追溯**。

## 方式一：本地直接运行（开发/演示最快）

```bash
pip install -r requirements-serve.txt
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

可选启用 LLM 自然语言回答（只把**权限过滤后**的可见内容喂给模型）：

```bash
export DEEPSEEK_API_KEY=sk-xxx      # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 方式二：Docker 部署（交付/私有化）

```bash
docker compose up --build -d
# 浏览器打开 http://<服务器IP>:8000
docker compose logs -f        # 看日志
docker compose down           # 停止
```

镜像特点：**不含 torch**（治理层纯标准库）→ 体积小、启动秒级、可在低配服务器/内网离线运行。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/datasets` | 列出可用知识库与角色 |
| GET | `/api/tokens` | 演示用身份令牌（**生产必须下线**，改由 SSO 签发） |
| POST | `/api/ask` | `{dataset, question}` + 请求头 `X-API-Token` → 答案 + 依据 + 权限判定 |
| GET | `/api/audit` | 审计记录（最新在前）+ 汇总统计——**需最高级别令牌** |
| GET | `/api/audit/stats` | 仅审计汇总（判定/事件/角色分布、鉴权失败次数、延迟） |
| GET | `/healthz` | 健康检查（供容器探针） |
| GET | `/` | 前端演示页 |

> `dataset` 用文件名去掉 `gov_` 前缀后的部分：`gov_employee_rules.json` → `employee_rules`。

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Token: demo-l0-employee" \
  -d '{"dataset":"employee_rules","question":"旷工超过多少天会被劝退？"}'
```

把 `X-API-Token` 换成 `demo-l2-hr` 再问同一句，可见内容层级不同——这是治理层的核心能力。

**身份只能由令牌决定**：请求体里写 `"role":"hr"` 会被服务端忽略（#3 安全修复）。
无令牌时生产模式返回 **401**；设 `GOVLAYER_DEV_MODE=1` 可匿名演示，但只能拿到**最低级别**。

## 演示脚本（面试/客户现场 3 分钟）

1. 打开页面，知识库选 `employee_rules`，身份选 **普通员工（级别 0）**；
2. 问："旷工超过多少天会被劝退？" → 只见公开条款（扣 50 元），**劝退标准被权限截断**；
3. 身份切到 **HR（级别 2）**，问同一问题 → 完整看到"当月旷工 5 天/全年 7 天劝退"；
4. 问："出差超期了，原定 2 天结果待了 5 天怎么处理？" → 判定 **制度空白**，不给编造答案，
   给出**过往先例**（仅供参考）；
5. 换知识库 `home`（toC 示例），身份 **孩子（级别 0）** 问"家里储蓄账户有多少钱" → **明确无权限**；
   身份切 **财务管家（级别 2）** → 可见。
6. 收尾（可选）：在浏览器控制台伪造 `{"role":"hr"}` 重发请求，服务端仍按最低级别处理——
   这是"角色不可伪造"的现场证明。

## 接新客户（零代码）

新增一个知识库只需在 `data/` 放一个 `gov_<name>.json`：

```json
{
  "objects": [{"id":"leave","title":"请假","keywords":["请假","病假"],
               "levels": {"0":"公开内容…","1":"经理级内容…","2":"HR级内容…"}}],
  "roles": {"employee":0,"manager":1,"hr":2},
  "probe_rules": [[["超期","延长"],["变更","延长"]]],
  "precedent_cases": [{"case_id":"C-001","topic":"出差超期","question":"…",
                       "ruling":"…","reasoning":"…","approver":"HR","date":"2026-03-12"}]
}
```

重启服务即生效——这就是"私人化定制只换数据集"的落地形态。

## 审计日志（#5）

每次提问都会落一条追加式 JSONL 记录——**成功、被拒（401/403）、甚至异常都记**，
因为审计的价值恰恰在"谁被挡在门外"上：

```bash
# 用 HR 令牌查最近 20 条 + 汇总
curl -H "X-API-Token: demo-l2-hr" "http://127.0.0.1:8000/api/audit?limit=20"
# 用员工令牌查 → 403（审计日志本身也受权限门控）
curl -H "X-API-Token: demo-l0-employee" "http://127.0.0.1:8000/api/audit"
```

记录字段：`ts / event / dataset / question / token_fp / identity_label / level / role /
verdict / hit_ids / denied_fields / semantic_used / llm_used / latency_ms`。

设计要点（也是客户会追问的点）：

- **绝不落令牌原文**，只存 SHA-256 前 12 位指纹（`token_fp`）——日志是内部人可读的，
  写进令牌就等于一次日志泄漏 = 一次凭证泄漏；
- 默认存问题原文（审计要能复盘），`GOVLAYER_AUDIT_REDACT=1` 可改为只存哈希+长度；
- **未认证请求不落问题原文**（只记 `question_withheld`）：否则任何人无需凭证就能往日志里
  灌任意内容（日志投毒 / 撑爆磁盘），审计日志自己变成攻击面；
- **写盘失败不影响业务**：磁盘满不会让服务不可用，失败计入 `stats.write_errors`；
- 启动时从文件回填，所以**重启后统计与历史仍在**（这是"持久化"的可见证据）；
- 路径由 `GOVLAYER_AUDIT_LOG` 指定，默认 `audit/audit.jsonl`；Docker 已把 `./audit` 挂成卷。

前端演示：页面上点 **审计日志** 按钮——用普通员工身份点会看到 403 提示，
切到 HR 再看，就是完整记录与统计。

## 混合检索（#4b：本地语义 + 关键词）

默认走**关键词检索**；配好本地 ONNX 模型后自动切换为**关键词 + 本地语义 RRF 混合检索**
（实测：措辞远离关键词时，命中率 11% → 78%，见 `V63_SEMANTIC_RETRIEVAL_POC.md`）。

```bash
python -m pip install -r requirements-ann.txt          # numpy/onnxruntime/tokenizers（不含 torch）
python fetch_embedding_model.py --endpoint https://hf-mirror.com
export GOVLAYER_ONNX_MODEL=models/bge-small-zh-v1.5-int8   # PowerShell: $env:...
uvicorn --app-dir server app:app --host 0.0.0.0 --port 8000
```

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `GOVLAYER_ONNX_MODEL` | 未设置 | 模型目录；**不设就纯关键词**（行为与接入前完全一致） |
| `GOVLAYER_LOCAL_SEMANTIC` | `1` | 置 `0` 强制退回纯关键词（排障用） |
| `GOVLAYER_SEMANTIC_MIN_SCORE` | `0.51` | 相似度下限；**已实测标定，换语料/换模型必须重跑 `calibrate_threshold.py`** |
| `GOVLAYER_RETRIEVAL` | `auto` | `auto`=LLM 优先；`local`=**检索本地化**；`llm`=强制 LLM |

三个必须知道的行为点：

1. **相似度下限不能省，而且必须实测标定**。向量检索永远会返回 top-k，哪怕问题与知识库
   毫无关系。默认值曾是猜的 `0.25`，实测证明太低：问"公司食堂几点开饭？"也会捞回 3 条，
   系统据此说出"制度中有相关规定，但超出你的角色权限范围"——一句会被当场戳穿的假话。
   用 `calibrate_threshold.py` 实测（`data/gov_employee_rules.json` + bge-small-zh-v1.5-int8）：

   ```
   相关问题 top1：最低 0.4971  中位 0.6135
   无关问题 top1：最高 0.5026  中位 0.4507
   ```

   **严格可分不成立**（尾部重叠仅 0.0055），但存在"零无关误纳"窗口 `(0.5026, 0.5235]`，
   窗口内相关召回 15/17（88%）。现取 `0.51`。

   ⚠️ **但比阈值更要紧的一条**：**相似度 ≠ 覆盖**。同域无关问题（"有加班费吗？""调岗需要
   本人同意吗？"）与考勤条款本就语义邻近，分数天然接近相关问题——无关问题的 top-3 几乎
   总是同几个条款。**阈值只能过滤"完全不相关"，判断"相关但制度没规定"（= `gap`）必须靠
   覆盖判定。** 落地含义：
   - `GOVLAYER_RETRIEVAL=local`（检索本地化）**仍需 LLM 做覆盖判定**；
   - **纯本地（连 LLM 都没有）时，覆盖判定退化为 6 条关键词探针**，
     默认"无匹配规则即视为覆盖"，因此**无法诚实地说出"制度没规定"**。
     这个配置可以演示权限隔离与检索，但**不要用它宣称"知识空白识别"**。
2. **混合而非替换**。实测"措辞贴近"时关键词排名更准（1.0 vs 1.25），
   所以关键词路径保留、语义路径补充召回，用 RRF 融合（两路分数量纲不可比，
   只看排名）。纯语义会拿精确术语的准确性去换口语能力。
3. **权限语义有两条路线，默认选 A**：
   - A（默认）：跨权限级别检索，因此能判定"存在但无权" → 回答"有相关规定，请联系 HR"。
     安全性由**输出时**的字段级门控保证（内容永不越权），代价是披露了"存在性"。
   - B（`strict_nonexistence_disclosure=True`）：只检索本级别可见内容，
     结构性不披露存在性，但**不再能给出 restricted 提示**，体验变差。
   两者安全性不同、体验不同，是产品决策，不是实现细节——**不能混着宣传**。

前端会在回答上方标出本次实际走的路径：`关键词检索` / `本地混合检索` / `LLM 语义检索`。

4. **`GOVLAYER_RETRIEVAL=auto` 时本地编码器几乎不会被用到**——因为 LLM 后端一旦可用
   就优先接管检索。若你的目标是"**检索这一半不出内网**"，必须显式设
   `GOVLAYER_RETRIEVAL=local`：此时检索完全由本地混合检索完成，LLM 退居
   **覆盖判定**，且只看到本地已选中的那几条可见候选（外发内容量也随之下降）。
   这一点很容易被忽略——配了模型却仍在把全库可见内容发出去，是"看起来环保、实际没变"。

## 生产化待补（诚实清单）

- **身份对接**：令牌 → 权限级别 → 角色名的链路已就位（`server/auth.py`），
  生产化只需把 `DEMO_TOKENS` 换成 SSO / LDAP / OAuth 校验——**接口不变**；
  同时必须**下线 `/api/tokens`**（它会列出演示令牌）；
- **审计加固**：日志轮转/归档、防篡改（当前是同机追加写，能改）、对接 SIEM；多进程部署时
  需换成共享 sink（当前 JSONL + 进程内锁，适合单进程/单容器）；
- **语义检索**：相似度下限需在真实语料上标定；措辞远离查询集仅 9 条、
  置信区间过宽；编码器未与其他模型对比；模型为社区转换版，**交付前应校验哈希**；
- 知识对象化的自动化工具（当前制度条款需人工+LLM 半自动整理）；
- ANN 规模化检索（已实现并实测，但**尚未接入服务**——当前检索是暴力精确扫描）；
- 覆盖判定仍依赖 LLM（"检索"这一半已可本地化，"判定制度有没有覆盖"这一半没有）；
- 并发与压测、监控告警、K8s 编排。
