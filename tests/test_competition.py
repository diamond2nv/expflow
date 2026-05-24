#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.competition module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _clear_comp_logger_cache():
    """Reset singleton cache before each test."""
    from expflow_pde.competition.comp_log import CompLogger

    CompLogger._instances = {}


# ── comp_log tests ──────────────────────────────────────────────────────


def test_comp_logger_creates_handlers(tmp_path: Path):
    """CompLogger produces all 5 log files."""
    from expflow_pde.competition.comp_log import CompLogger

    log = CompLogger("test_handlers", operator="FNO", tag="v1", log_dir=tmp_path)
    log.info("hello")
    log.metric("loss", 0.01)
    log.record_time("train", 100.0)
    log.agent_note("hypothesis")
    log.flush()

    assert (tmp_path / "fast.log").exists()
    assert (tmp_path / "agent.log").exists()
    assert (tmp_path / "all.log").exists()
    assert (tmp_path / "metric.jsonl").exists()
    assert (tmp_path / "time.jsonl").exists()


def test_get_logger_singleton(tmp_path: Path):
    """get_logger returns the same instance for the same key."""
    from expflow_pde.competition.comp_log import get_logger

    a = get_logger("test", operator="FNO", tag="v1", log_dir=tmp_path)
    b = get_logger("test", operator="FNO", tag="v1", log_dir=tmp_path)
    assert a is b


def test_metric_writes_jsonl(tmp_path: Path):
    """metric() writes valid JSONL to metric.jsonl."""
    from expflow_pde.competition.comp_log import CompLogger

    # Clear singleton cache to avoid cross-test pollution
    CompLogger._instances = {}

    log = CompLogger("test_metric", operator="FNO", tag="v2", log_dir=tmp_path)
    log.metric("loss", 0.0123, extra={"epoch": 5})
    log.flush()

    with open(tmp_path / "metric.jsonl", "r") as f:
        lines = f.read().strip().split("\n")

    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert entry["metric"] == "loss"
    assert entry["value"] == 0.0123
    assert entry["epoch"] == 5


def test_compute_time_scores():
    """Time score computation per competition rules."""
    from expflow_pde.competition.comp_log import compute_time_scores

    # Fast training + fast inference
    scores = compute_time_scores(3000, 30)
    assert scores["task1_train_score"] == 35  # 50min < 60min
    assert scores["task1_inference_score"] >= 30.0  # 0.5min => 30.0 pts
    assert scores["task1_inference_safe"] is True

    # Slow training
    scores2 = compute_time_scores(8000, 30)
    assert scores2["task1_train_score"] == 20  # 133min => 20pts

    # Inference timeout
    scores3 = compute_time_scores(3000, 130)
    assert scores3["task1_inference_score"] == 0.0
    assert scores3["task1_inference_safe"] is False


# ── merge tests ─────────────────────────────────────────────────────────


def _write_fast_log(path: Path) -> None:
    """Write a sample fast.log."""
    lines = [
        "2026-05-23T10:00:01.000Z -  INFO - [train:306           ] - [FNO:v1] - [PID:123] - Epoch 1/80 | loss=0.003400 | lr=1.00e-03 | GPU=2048/4096MB",
        "2026-05-23T10:01:00.000Z -  INFO - [train:320           ] - [FNO:v1] - [PID:123] -   [VAL] Epoch 10: MSE=0.120000, Rel-MSE=0.080000, Seg=67.30/100",
        "2026-05-23T10:55:00.000Z -  INFO - [train:350           ] - [FNO:v1] - [PID:123] - Training completed: 3540.0s (59.0min)",
        "2026-05-23T10:55:01.000Z -  INFO - [train:363           ] - [FNO:v1] - [PID:123] -   Seg1=85.50 Seg2=82.30 Seg3=55.20 Total=70.10/100",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_agent_log(path: Path) -> None:
    """Write a sample agent.log."""
    lines = [
        "2026-05-23T10:00:00.000Z -  INFO - [agent:10            ] - [FNO:v1] - [PID:456] - [AGENT] Starting experiment with FNO n_modes=12",
        "2026-05-23T10:30:00.000Z -  INFO - [agent:20            ] - [FNO:v1] - [PID:456] - [TC] write({\"filePath\": \"train.py\"})",
        "2026-05-23T10:30:05.000Z -  INFO - [agent:25            ] - [FNO:v1] - [PID:456] - [AGENT] Generated train.py with sliding window pairs",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_llm_jsonl(path: Path) -> None:
    """Write a sample litellm callback JSONL with timestamps within the fast.log span."""
    # Keep LLM timestamps within the 2026-05-23T10:00:00 to 10:55:01 window
    entries = [
        {
            "timestamp": "2026-05-23T10:00:02.000+00:00",
            "elapsed_seconds": 8.5,
            "model": "deepseek-chat",
            "response": "I will create a training script for FNO...",
            "tool_calls": "write_file({\"path\": \"train.py\"})",
            "session_id": "expflow-task1-deadbeef",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_merge_three_sources(tmp_path: Path):
    """Merge from fast.log + agent.log + llm-*.jsonl → valid competition log."""
    from expflow_pde.competition.merge import merge_logs, validate_log

    _write_fast_log(tmp_path / "fast.log")
    _write_agent_log(tmp_path / "agent.log")
    _write_llm_jsonl(tmp_path / "llm-session1.jsonl")

    output = tmp_path / "task1_logs.log"
    n = merge_logs(
        fast_log=tmp_path / "fast.log",
        agent_log=tmp_path / "agent.log",
        llm_files=[tmp_path / "llm-session1.jsonl"],
        output=output,
        task="task1",
    )

    assert n > 0
    assert output.exists()
    errors = validate_log(output)
    assert len(errors) == 0, f"Validation errors: {errors}"


def test_validate_rejects_non_json(tmp_path: Path):
    """validate_log rejects non-JSON lines."""
    from expflow_pde.competition.merge import validate_log

    path = tmp_path / "bad.log"
    path.write_text("this is not json\n")
    errors = validate_log(path)
    assert len(errors) > 0


def test_validate_rejects_missing_fields(tmp_path: Path):
    """validate_log rejects entries missing timestamp/elapsed_seconds."""
    from expflow_pde.competition.merge import validate_log

    path = tmp_path / "bad.log"
    path.write_text('{"response": "hello"}\n')
    errors = validate_log(path)
    assert len(errors) > 0
    assert any("missing" in e.lower() for e in errors)


def test_validate_rejects_backward_timestamps(tmp_path: Path):
    """validate_log rejects non-monotonic timestamps."""
    from expflow_pde.competition.merge import validate_log

    path = tmp_path / "bad.log"
    path.write_text(
        '{"timestamp": "2026-05-23T11:00:00+00:00", "elapsed_seconds": 1}\n'
        '{"timestamp": "2026-05-23T10:00:00+00:00", "elapsed_seconds": 1}\n'
    )
    errors = validate_log(path)
    assert len(errors) > 0
    assert any("backward" in e.lower() for e in errors)


def test_merge_empty_sources(tmp_path: Path):
    """merge_logs handles empty sources gracefully."""
    from expflow_pde.competition.merge import merge_logs

    n = merge_logs(
        fast_log=tmp_path / "nonexistent.log",
        agent_log=tmp_path / "nonexistent.log",
        llm_files=[],
        output=tmp_path / "task1_logs.log",
    )
    assert n == 0


# ── CompetitionSession tests (no proxy — integration markers) ───────────


def test_session_init(tmp_path: Path):
    """CompetitionSession.__init__ creates log directory."""
    from expflow_pde.competition import CompetitionSession

    log_dir = tmp_path / "comp_logs"
    s = CompetitionSession(task="task1", tag="test", log_dir=str(log_dir))
    assert s.task == "task1"
    assert s.tag == "test"
    assert log_dir.exists()


def test_session_status_not_started(tmp_path: Path):
    """status() reports not started."""
    from expflow_pde.competition import CompetitionSession

    s = CompetitionSession(log_dir=str(tmp_path))
    st = s.status()
    assert st["started"] is False


@pytest.mark.integration
def test_session_start_stop_merge(tmp_path: Path):
    """Full lifecycle: start (no proxy), stop, merge, validate."""
    from expflow_pde.competition import get_comp_logger

    log_dir = tmp_path / "comp_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write some fast.log content first
    log = get_comp_logger(
        "train_test", operator="FNO", tag="v1", log_dir=log_dir
    )
    log.info("Epoch 1/10 | loss=0.01 | GPU=1024/4096MB")
    log.metric("train_loss", 0.01, extra={"epoch": 1})
    log.record_time("training", 60.0, task="task1")
    log.flush()

    # Now merge
    from expflow_pde.competition.merge import merge_logs, validate_log

    output = log_dir / "task1_logs.log"
    merge_logs(
        fast_log=log_dir / "fast.log",
        agent_log=log_dir / "agent.log",
        llm_files=[],
        output=output,
    )
    assert output.exists()
    errors = validate_log(output)
    assert len(errors) == 0


# ── _ingest_server tests ────────────────────────────────────────────────


def test_extract_entry_from_standard_payload():
    """_extract_entry correctly parses StandardLoggingPayload."""
    from expflow_pde.competition._ingest_server import _extract_entry

    payload = {
        "model": "deepseek-chat",
        "startTime": 1000.5,
        "endTime": 1005.3,
        "status": "success",
        "response": {
            "choices": [{
                "message": {
                    "content": "I will create train.py",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "train.py"}',
                            }
                        }
                    ],
                }
            }]
        },
        "metadata": {
            "requester_metadata": {"session_id": "expflow-task1-123"},
        },
    }

    entry = _extract_entry(payload)
    assert entry["model"] == "deepseek-chat"
    assert entry["elapsed_seconds"] == 4.8
    assert entry["response"] == "I will create train.py"
    assert "write_file" in entry["tool_calls"]
    assert entry["session_id"] == "expflow-task1-123"
    assert entry["status"] == "success"


def test_extract_entry_no_response():
    """_extract_entry handles missing response fields."""
    from expflow_pde.competition._ingest_server import _extract_entry

    entry = _extract_entry({})
    assert entry["model"] == ""
    assert entry["elapsed_seconds"] == 0.0
    assert entry["response"] == ""
    assert entry["tool_calls"] is None


def test_extract_entry_empty_choices():
    """_extract_entry handles empty choices list."""
    from expflow_pde.competition._ingest_server import _extract_entry

    payload = {
        "model": "gpt-4",
        "startTime": 1.0,
        "endTime": 2.0,
        "response": {"choices": []},
        "status": "success",
    }
    entry = _extract_entry(payload)
    assert entry["elapsed_seconds"] == 1.0
    assert entry["response"] == ""


# ── Litellm config generation tests ──────────────────────────────────────


def test_generate_litellm_config(tmp_path: Path):
    """_generate_litellm_config produces valid no-database config with generic_api."""
    from expflow_pde.competition.session import CompetitionSession

    s = CompetitionSession(task="task1", tag="e2e", proxy_port=4000,
                           ingest_port=8099, log_dir=str(tmp_path))
    s._config_path = tmp_path / "litellm_config.yaml"
    s.master_key = "sk-test-456"
    s.target_api_key = "fake-key"
    s.target_url = "https://api.deepseek.com/v1"

    s._generate_litellm_config()

    assert s._config_path.exists()
    cfg = yaml.safe_load(open(s._config_path))

    # Verify structure
    assert "model_list" in cfg
    assert cfg["model_list"][0]["model_name"] == "deepseek-chat"
    assert cfg["model_list"][0]["litellm_params"]["model"] == "deepseek/deepseek-chat"

    assert "litellm_settings" in cfg
    cb = cfg["litellm_settings"]["callback_settings"]["jsonl_logger"]
    assert cb["callback_type"] == "generic_api"
    assert "8099/ingest" in cb["endpoint"]
    assert cb["log_format"] == "ndjson"
    assert cb["batch_size"] == 1

    assert "general_settings" in cfg
    assert cfg["general_settings"]["master_key"] == "sk-test-456"

    # No database_url — no prisma dependency
    assert "database_url" not in cfg.get("general_settings", {})
    assert "database_url" not in str(cfg)


@pytest.mark.integration
def test_session_start_stop_smoke(tmp_path: Path):
    """Smoke test: start session (proxy + ingest), stop, merge, validate.

    Only runs when DEEPSEEK_API_KEY is set in environment.
    Requires litellm and network access.
    """
    import os

    from expflow_pde.competition import CompetitionSession

    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")

    s = CompetitionSession(
        task="task1", tag="smoke",
        proxy_port=14002, ingest_port=18099,
        log_dir=str(tmp_path),
    )

    try:
        meta = s.start()
        assert meta["session_id"].startswith("expflow-task1")
        assert meta["proxy_port"] == 14002

        # Verify both services respond
        import urllib.request
        for port, label in [(14002, "proxy"), (18099, "ingest")]:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200, f"{label} health check failed"

        result = s.stop(merge=True)
        assert result["status"] == "stopped"

        # Even without real API calls, merge should not crash
        if "merge" in result:
            assert "output" in result["merge"]

    finally:
        if s._started:
            s.stop(merge=False)
