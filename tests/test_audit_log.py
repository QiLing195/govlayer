# -*- coding: utf-8 -*-
"""#5 审计日志测试：持久化、可查询、不泄令牌、写盘失败不致命。

放在 tests/ 下是因为 pyproject.toml 里 `testpaths = ["tests"]`——
放在仓库根目录的 test_*.py 会被 pytest **静默跳过**（不报错、不收集）。

运行：D:/conda/envs/cformer-gpu/python.exe -m pytest tests/test_audit_log.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# server/ 不是安装过的包（cformer_v60 之类是），所以需要显式加进 sys.path
SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER_DIR))

from audit import MAX_LIMIT, AuditLog, question_field, token_fingerprint  # noqa: E402


def test_record_persists_valid_jsonl(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record({"event": "ask", "verdict": "covered", "role": "hr"})
    log.record({"event": "auth_failed", "status": 401})

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert entry["ts"].endswith("Z")          # UTC 时间戳
        assert "event" in entry


def test_token_fingerprint_never_stores_raw_token(tmp_path: Path) -> None:
    secret = "demo-l2-hr"
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record({"event": "ask", "token_fp": token_fingerprint(secret)})

    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert secret not in raw                       # 令牌原文绝不落盘
    assert token_fingerprint(secret) in raw
    assert token_fingerprint(secret) != secret
    assert len(token_fingerprint(secret)) == 12


def test_fingerprint_of_missing_token_is_placeholder() -> None:
    assert token_fingerprint(None) == "none"
    assert token_fingerprint("") == "none"


def test_tail_returns_newest_first_and_respects_limit(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.record({"event": "ask", "seq": i})

    latest = log.tail(2)
    assert [e["seq"] for e in latest] == [4, 3]
    assert len(log.tail(MAX_LIMIT + 5000)) <= MAX_LIMIT


def test_tail_can_filter_by_event(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record({"event": "ask", "seq": 1})
    log.record({"event": "auth_failed", "seq": 2})
    log.record({"event": "ask", "seq": 3})

    assert [e["seq"] for e in log.tail(10, event="auth_failed")] == [2]


def test_stats_aggregates_verdicts_events_and_auth_failures(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record({"event": "ask", "verdict": "covered", "role": "hr", "latency_ms": 10.0})
    log.record({"event": "ask", "verdict": "restricted", "role": "employee",
                "latency_ms": 30.0})
    log.record({"event": "auth_failed", "status": 401})

    s = log.stats()
    assert s["total"] == 3
    assert s["by_verdict"] == {"covered": 1, "restricted": 1}
    assert s["by_role"] == {"hr": 1, "employee": 1}
    assert s["auth_failures"] == 1
    assert s["latency_ms_avg"] == 20.0
    assert s["latency_ms_max"] == 30.0
    assert s["written_to_disk"] == 3
    assert s["write_errors"] == 0


def test_stats_on_empty_log_is_safe(tmp_path: Path) -> None:
    s = AuditLog(tmp_path / "audit.jsonl").stats()
    assert s["total"] == 0
    assert s["latency_ms_avg"] is None
    assert s["first_ts"] is None


def test_reload_after_restart_keeps_history(tmp_path: Path) -> None:
    """持久化的可见证据：新实例（= 重启）仍能读到历史。"""
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path)
    first.record({"event": "ask", "verdict": "covered", "token_fp": token_fingerprint("t")})

    second = AuditLog(path)
    s = second.stats()
    assert s["total"] == 1
    assert s["by_verdict"] == {"covered": 1}
    assert s["first_ts"] is not None
    # 重启后"已落盘"必须仍然是 1，不能显示成 0（否则看起来像丢了数据）
    assert s["loaded_from_disk"] == 1
    assert s["written_to_disk"] == 1
    assert s["written_this_session"] == 0


def test_corrupt_lines_are_tolerated_on_reload(tmp_path: Path) -> None:
    """崩溃可能留下半行 JSON，不能让整个服务起不来。"""
    path = tmp_path / "audit.jsonl"
    path.write_text('{"event": "ask", "seq": 1}\n{not json at all\n\n', encoding="utf-8")

    log = AuditLog(path)
    assert log.stats()["total"] == 1            # 只有合法行进入缓冲


def test_write_failure_is_nonfatal(tmp_path: Path) -> None:
    """审计写盘失败绝不能让业务请求失败（磁盘满不应导致服务不可用）。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    log = AuditLog(blocker / "audit.jsonl")     # 父路径是文件 → mkdir 必失败

    entry = log.record({"event": "ask", "verdict": "covered"})   # 不应抛异常
    assert entry["verdict"] == "covered"
    assert log.write_errors >= 1
    assert log.stats()["write_errors"] >= 1
    assert log.stats()["total"] == 1            # 内存里仍然可查


def test_redact_mode_hides_question_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOVLAYER_AUDIT_REDACT", "1")
    field = question_field("旷工几天被劝退？")

    assert "question" not in field
    assert field["question_redacted"] is True
    assert field["question_len"] == len("旷工几天被劝退？")
    assert "劝退" not in json.dumps(field, ensure_ascii=False)


def test_question_stored_verbatim_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOVLAYER_AUDIT_REDACT", raising=False)
    assert question_field("旷工几天被劝退？") == {"question": "旷工几天被劝退？"}
