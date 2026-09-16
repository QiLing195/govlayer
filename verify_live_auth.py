"""对运行中的 GovLayer 服务做端到端鉴权 / 防越权验证（只依赖标准库）。

用法：
    1) 另开一个终端启动服务：
       D:\\conda\\envs\\cformer-gpu\\python.exe -m uvicorn --app-dir server app:app --host 127.0.0.1 --port 8001
    2) 运行本脚本：
       D:\\conda\\envs\\cformer-gpu\\python.exe verify_live_auth.py

可用环境变量 GOVLAYER_BASE 覆盖服务地址（默认 http://127.0.0.1:8001）。
退出码 0 = 全部通过；1 = 有失败项；2 = 连不上服务。
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("GOVLAYER_BASE", "http://127.0.0.1:8001")
# 注意：dataset id 是文件名去掉 gov_ 前缀后的部分（gov_employee_rules.json → employee_rules）。
# 下面不写死，改为从 /api/datasets 自动挑选，避免改名后整脚本 404。
PREFERRED_DATASET = "employee_rules"
DATASET = PREFERRED_DATASET
QUESTION = "员工连续旷工多久会被劝退？"

# 只出现在"经理/HR 级"条款里的敏感措辞，用于跨级泄漏检测
SENSITIVE = ["劝退", "旷工5天", "旷工五天", "累计7天", "累计七天", "5天或全年", "七天"]

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((bool(ok), name, detail))


def _request(method: str, path: str, payload: dict | None = None, token: str | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-API-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}
    except urllib.error.URLError as exc:
        print(f"\n[无法连接] {BASE} -> {exc.reason}")
        print("请先启动服务，或用 GOVLAYER_BASE 指定正确地址。")
        sys.exit(2)


def leaked(blob: str) -> list[str]:
    return [w for w in SENSITIVE if w in (blob or "")]


def blob_of(ans: dict) -> str:
    """把一个回答里所有面向用户的文本汇成一个串，用于泄漏扫描。"""
    return "\n".join([
        str(ans.get("answer_text") or ""),
        str(ans.get("boundary_note") or ""),
        json.dumps(ans.get("visible_sources") or [], ensure_ascii=False),
        json.dumps(ans.get("precedents") or [], ensure_ascii=False),
        str(ans.get("identity") or ""),
    ])


def parse_level(ans: dict) -> int | None:
    """identity 是形如 '普通员工 / 学生 / 孩子（级别 0 → 角色 employee）' 的字符串。"""
    m = re.search(r"级别\s*(\d+)", str(ans.get("identity") or ""))
    return int(m.group(1)) if m else None


def pick_dataset(items: list[dict]) -> tuple[str, dict | None]:
    """从 /api/datasets 里挑出员工制度库（优先 PREFERRED_DATASET，其次名字含 employee）。"""
    ids = [d.get("id", "") for d in items]
    chosen = PREFERRED_DATASET if PREFERRED_DATASET in ids else next(
        (i for i in ids if "employee" in i), ids[0] if ids else PREFERRED_DATASET)
    return chosen, next((d for d in items if d.get("id") == chosen), None)


def main() -> int:
    global DATASET

    # 0. 服务存活
    status, health = _request("GET", "/healthz")
    check(status == 200 and health.get("status") == "ok",
          "服务健康检查 /healthz", f"HTTP {status} {health}")

    # 0b. 选定数据集（用服务端真实 id，避免写死过期）
    status, items = _request("GET", "/api/datasets")
    if status != 200 or not isinstance(items, list) or not items:
        check(False, "知识库目录 /api/datasets 可用", f"HTTP {status} body={items}")
        return _report()
    DATASET, spec = pick_dataset(items)
    roles = (spec or {}).get("roles") or {}
    check(bool(roles),
          f"选定数据集 {DATASET} 并读取角色表", f"roles={roles}")

    # 1. 令牌目录（返回的是 **列表**，不是对象）
    status, tokens = _request("GET", "/api/tokens")
    names = [t.get("token") for t in tokens] if status == 200 and isinstance(tokens, list) else []
    check(status == 200 and len(names) >= 3,
          "令牌目录 /api/tokens 至少 3 个级别", f"HTTP {status} tokens={names}")

    # 2. 无令牌：生产模式应 401；dev 模式应降级为最低级别
    status, body = _request("POST", "/api/ask", {"dataset": DATASET, "question": QUESTION})
    dev_mode = status == 200 and parse_level(body) == 0
    check(status == 401 or dev_mode,
          "无令牌：生产模式拒绝 / 开发模式降级为最低级别",
          f"HTTP {status} identity={body.get('identity')!r}")

    # 3. 伪造令牌
    status, _ = _request("POST", "/api/ask",
                         {"dataset": DATASET, "question": QUESTION}, token="not-a-real-token")
    check(status == 401, "无效令牌被拒绝", f"HTTP {status}")

    # 3b. 未认证 + 不存在的库名：应先 401，而不是用 404 泄漏库名是否存在
    status, _ = _request("POST", "/api/ask",
                         {"dataset": "no_such_dataset_xyz", "question": QUESTION})
    check(status == 401, "未认证请求不因库名泄漏 401/404 差异", f"HTTP {status}")

    # 4. 员工令牌 + 请求体伪造 role=hr  ->  必须不越权
    status, emp = _request("POST", "/api/ask",
                           {"dataset": DATASET, "question": QUESTION, "role": "hr"},
                           token="demo-l0-employee")
    emp_leak = leaked(blob_of(emp))
    check(status == 200 and not emp_leak,
          "员工令牌 + 伪造 role=hr：无敏感条款泄漏",
          f"HTTP {status} verdict={emp.get('verdict')} role={emp.get('role')!r} "
          f"denied={emp.get('denied_fields')} 泄漏={emp_leak}")
    check(parse_level(emp) == 0,
          "员工令牌解析出的级别为 0（请求体 role 被忽略）",
          f"identity={emp.get('identity')!r} role={emp.get('role')!r}")
    check(emp.get("role") != "hr",
          "响应 role 不采信客户端声明（回显漏洞已修）",
          f"role={emp.get('role')!r}")

    # 5. HR 令牌 -> 同一问题应能拿到明文（对照组成立）
    status, hr = _request("POST", "/api/ask",
                          {"dataset": DATASET, "question": QUESTION},
                          token="demo-l2-hr")
    hr_hits = leaked(blob_of(hr))
    check(status == 200 and bool(hr_hits),
          "HR 令牌：同一问题可读到受限条款（对照组成立）",
          f"HTTP {status} verdict={hr.get('verdict')} 命中={hr_hits}")
    check(parse_level(hr) == 2, "HR 令牌解析出的级别为 2",
          f"identity={hr.get('identity')!r}")

    # 6. 经理令牌：级别 1，应比员工多、比 HR 不多
    status, mgr = _request("POST", "/api/ask",
                           {"dataset": DATASET, "question": QUESTION},
                           token="demo-l1-manager")
    check(status == 200 and parse_level(mgr) == 1,
          "经理令牌解析出的级别为 1", f"HTTP {status} identity={mgr.get('identity')!r}")

    # 7. 语义后端是否真的在工作
    check(bool(hr.get("semantic_used")),
          "语义检索已启用（DEEPSEEK_API_KEY 生效）",
          f"semantic_used={hr.get('semantic_used')} llm_used={hr.get('llm_used')}")

    # 8. 审计日志（#5）：自身受门控 + 留痕 + 不落令牌原文
    status, denied = _request("GET", "/api/audit", token="demo-l0-employee")
    check(status == 403, "员工令牌查审计日志被拒绝（审计自身也受门控）",
          f"HTTP {status} detail={denied.get('detail')!r}")

    status, audit = _request("GET", "/api/audit?limit=50", token="demo-l2-hr")
    records = (audit.get("records") or []) if status == 200 else []
    check(status == 200 and bool(records), "HR 令牌可读到审计记录",
          f"HTTP {status} 记录数={len(records)}")

    # 最关键的一条：日志是内部人可读的，若把令牌写进去，一次日志泄漏=一次凭证泄漏
    audit_blob = json.dumps(audit, ensure_ascii=False)
    token_leak = [t for t in ("demo-l0-employee", "demo-l1-manager", "demo-l2-hr")
                  if t in audit_blob]
    check(not token_leak, "审计记录不含令牌原文（只存 SHA-256 指纹）",
          f"泄漏={token_leak}")

    forged = [r for r in records if r.get("event") == "ask" and r.get("level") == 0]
    check(any(r.get("verdict") == "restricted" for r in forged),
          "刚才那次伪造越权已留痕（level=0 且 verdict=restricted）",
          f"level=0 的提问记录 {len(forged)} 条，判定="
          f"{[r.get('verdict') for r in forged][:5]}")

    stats = audit.get("stats") or {}
    check(stats.get("auth_failures", 0) >= 1 and stats.get("total", 0) >= 1,
          "审计统计含鉴权失败计数（越权探测可被审计发现）",
          f"total={stats.get('total')} auth_failures={stats.get('auth_failures')} "
          f"by_verdict={stats.get('by_verdict')}")

    af = [r for r in records if r.get("event") == "auth_failed"]
    check(bool(af) and all("question" not in r for r in af),
          "未认证请求不落问题原文（防匿名日志投毒/撑爆磁盘）",
          f"auth_failed {len(af)} 条，其中含 question 的 "
          f"{sum(1 for r in af if 'question' in r)} 条")

    # 汇总
    return _report()


def _report() -> int:
    print("\n" + "=" * 78)
    print(f"GovLayer 在线鉴权验证 @ {BASE}  （数据集：{DATASET}）")
    print("=" * 78)
    for ok, name, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"        {detail}")
    failed = [r for r in results if not r[0]]
    print("-" * 78)
    print(f"共 {len(results)} 项，通过 {len(results) - len(failed)} 项，失败 {len(failed)} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
