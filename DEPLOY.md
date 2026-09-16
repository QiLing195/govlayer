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

## 生产化待补（诚实清单）

- **身份对接**：令牌 → 权限级别 → 角色名的链路已就位（`server/auth.py`），
  生产化只需把 `DEMO_TOKENS` 换成 SSO / LDAP / OAuth 校验——**接口不变**；
  同时必须**下线 `/api/tokens`**（它会列出演示令牌）；
- **审计加固**：日志轮转/归档、防篡改（当前是同机追加写，能改）、对接 SIEM；多进程部署时
  需换成共享 sink（当前 JSONL + 进程内锁，适合单进程/单容器）；
- 知识对象化的自动化工具（当前制度条款需人工+LLM 半自动整理）；
- 检索升级（当前关键词锚定 + LLM 语义；大库需接向量检索 + ANN）；
- 并发与压测、监控告警、K8s 编排。
