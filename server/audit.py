# -*- coding: utf-8 -*-
"""审计日志：把每一次「谁·在什么时候·问了什么·为什么给这个答案」落成追加式记录。

为什么审计是治理层的必需件，而不是附属功能：
  #3 让身份不可伪造之后，客户的下一个问题必然是「那你留痕了吗」——
    · 出事了要能复盘：谁问过、返回了什么判定、截断了哪些字段；
    · 越权探测要能发现：失败的鉴权尝试本身是最有价值的安全信号；
    · 合规要能出报表：按判定类型/身份/时间统计。

设计取舍（都写在这里，便于评审）：
  1. **追加式 JSONL**：一行一条，崩溃时最多丢最后一行；不需要数据库、便于交付与离线。
  2. **绝不落原始令牌**：只存 token 的 SHA-256 前 12 位指纹。日志本身是"内部人可读"的，
     如果把 Token 写进去，一次日志泄漏就等于一次凭证泄漏。
  3. **问题默认原文存储**（审计的意义就在于可复盘），但支持
     `GOVLAYER_AUDIT_REDACT=1` 只存哈希+长度——面向强隐私客户的开关。
  4. **best-effort**：审计写盘失败**不能**让业务请求失败（否则磁盘满会变成服务不可用），
     失败只记到内存环形缓冲并计数，由 /api/audit/stats 暴露 write_errors。
  5. 内存环形缓冲同时用于服务 /api/audit 查询——避免每次读整文件；
     启动时从磁盘回填，因此**重启后统计仍在**（这正是"持久化"的可见证据）。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

AUDIT_PATH_ENV = "GOVLAYER_AUDIT_LOG"
AUDIT_REDACT_ENV = "GOVLAYER_AUDIT_REDACT"     # =1 时不存问题原文，只存哈希
RING_SIZE = 2000                                # 内存缓冲条数（同时是启动回填上限）
MAX_LIMIT = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def token_fingerprint(token: str | None) -> str:
    """令牌指纹：只用于把同一个令牌的多次请求关联起来，不可反推原令牌。"""
    if not token:
        return "none"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def question_field(question: str) -> dict:
    """按隐私开关决定存原文还是只存哈希+长度。"""
    if os.environ.get(AUDIT_REDACT_ENV, "") in ("1", "true", "True"):
        return {
            "question_redacted": True,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "question_len": len(question),
        }
    return {"question": question}


class AuditLog:
    """线程安全的追加式审计日志（JSONL + 内存环形缓冲）。"""

    def __init__(self, path: str | os.PathLike | None = None):
        default = Path(__file__).resolve().parent.parent / "audit" / "audit.jsonl"
        env_path = os.environ.get(AUDIT_PATH_ENV, "").strip()
        self.path = Path(path or env_path or default)
        self._lock = threading.Lock()
        self._ring: deque[dict] = deque(maxlen=RING_SIZE)
        self.write_errors = 0
        self.written = 0
        self.loaded_from_disk = 0
        self._load_existing()

    # ---- 启动回填：让重启后的统计连续 ----
    def _load_existing(self) -> None:
        try:
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as fh:
                tail = deque(fh, maxlen=RING_SIZE)
            for line in tail:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._ring.append(json.loads(line))
                    self.loaded_from_disk += 1
                except json.JSONDecodeError:
                    continue  # 容忍半行/坏行（崩溃残留）
        except OSError:
            pass  # 读不到就当空历史，不影响服务

    # ---- 写入 ----
    def record(self, rec: dict) -> dict:
        entry = {"ts": utc_now(), **rec}
        entry.setdefault("event", "error")
        with self._lock:
            self._ring.append(entry)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.written += 1
            except OSError:
                # 磁盘/权限问题不能影响业务：只计数，由 stats 暴露
                self.write_errors += 1
        return entry

    # ---- 查询 ----
    def tail(self, limit: int = 50, event: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), MAX_LIMIT))
        with self._lock:
            items = list(self._ring)
        if event:
            items = [e for e in items if e.get("event") == event]
        return items[-limit:][::-1]  # 最新的在前

    def stats(self) -> dict:
        with self._lock:
            items = list(self._ring)
            written, errors = self.written, self.write_errors
        by_verdict = Counter(e["verdict"] for e in items if e.get("verdict"))
        by_event = Counter(e.get("event", "error") for e in items)
        by_role = Counter(e["role"] for e in items if e.get("role"))
        lat = [e["latency_ms"] for e in items if isinstance(e.get("latency_ms"), (int, float))]
        return {
            "total": len(items),
            # "已落盘"= 启动时从文件读回的 + 本次进程写入的。
            # 若只算本次写入，重启后会出现"共 1 条·已落盘 0 条"的假象，像是数据丢了。
            "written_to_disk": self.loaded_from_disk + written,
            "loaded_from_disk": self.loaded_from_disk,
            "written_this_session": written,
            "write_errors": errors,
            "log_path": str(self.path),
            "by_event": dict(by_event),
            "by_verdict": dict(by_verdict),
            "by_role": dict(by_role),
            "auth_failures": by_event.get("auth_failed", 0),
            "unique_token_fingerprints": len({
                e.get("token_fp") for e in items if e.get("token_fp") not in (None, "none")
            }),
            "latency_ms_avg": round(sum(lat) / len(lat), 1) if lat else None,
            "latency_ms_max": max(lat) if lat else None,
            "first_ts": items[0].get("ts") if items else None,
            "last_ts": items[-1].get("ts") if items else None,
            "redact_questions": os.environ.get(AUDIT_REDACT_ENV, "") in ("1", "true", "True"),
        }
