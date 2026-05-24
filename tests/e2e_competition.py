#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end test: litellm proxy + competition callback + merge.

Run with hfpapers-crawler venv (has litellm[proxy] installed):
    /path/to/hfpapers-crawler/venv/bin/python tests/e2e_competition.py
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Add expflow to path so callback import works
_EXPFLOW_ROOT = os.path.expanduser("~/Gitlab/Agentic4Sci/expflow")
sys.path.insert(0, _EXPFLOW_ROOT)

# Load API key from .env
_ENV_FILE = os.path.expanduser("~/.hermes/.env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.strip().strip('"').strip("'")
            if key not in os.environ or not os.environ[key]:
                os.environ[key] = val

os.environ["COMPETITION_LOG_DIR"] = str(
    Path(tempfile.gettempdir()) / "comp_e2e_test"
)

PROXY_PORT = 14000  # Non-standard port to avoid conflicts


def _build_config(log_dir: str) -> dict:
    """Build litellm proxy config as a dict."""
    return {
        "general_settings": {
            "master_key": "sk-master-e2e-test",
            "database_url": f"sqlite:///{log_dir}/litellm.db",
        },
        "litellm_settings": {
            "callbacks": ["jsonl_logger"],
            "callback_settings": {
                "jsonl_logger": {
                    "callback_type": "generic_api",
                    "endpoint": "",
                    "log_format": "ndjson",
                    "batch_size": 1,
                    "flush_interval": 1,
                }
            },
        },
        "model_list": [
            {
                "model_name": "deepseek-chat",
                "litellm_params": {
                    "model": "deepseek/deepseek-chat",
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            }
        ],
        "router_settings": {
            "routing_strategy": "usage-based",
        },
    }


def test_e2e_competition_litellm():
    """Full E2E: start proxy -> make call -> check JSONL -> merge -> validate."""
    log_dir = Path(os.environ["COMPETITION_LOG_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build config
    config = _build_config(str(log_dir))
    config_path = log_dir / "litellm_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. Start litellm proxy
    print(f"[1/5] Starting litellm proxy on port {PROXY_PORT}...")
    log_file = log_dir / "proxy.log"

    # Use litellm CLI binary from same venv as current python
    litellm_bin = os.path.join(os.path.dirname(sys.executable), "litellm")
    if not os.path.isfile(litellm_bin):
        litellm_bin = "litellm"  # fallback
    cmd = [
        litellm_bin,
        "--config", str(config_path),
        "--port", str(PROXY_PORT),
        "--host", "0.0.0.0",
    ]
    with open(log_file, "w") as f_log:
        env = os.environ.copy()
        env["PYTHONPATH"] = _EXPFLOW_ROOT + ":" + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            cmd, stdout=f_log, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
            cwd=_EXPFLOW_ROOT,  # litellm resolves callbacks relative to CWD
        )

    # Wait for proxy to be ready
    deadline = time.time() + 30
    proxy_ready = False
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"http://localhost:{PROXY_PORT}/health",
                headers={"Authorization": "Bearer sk-master-e2e-test"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    proxy_ready = True
                    break
        except Exception:
            time.sleep(2)

    assert proxy_ready, (
        f"Proxy did not start within 30s. "
        f"Check {log_file} for errors."
    )
    print(f"   [OK] Proxy ready on port {PROXY_PORT}")

    # 3. Make a real API call through the proxy
    print("[2/5] Making test API call through proxy...")
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Say 'hello' in exactly one word."}
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
        "metadata": {
            "user_api_key": "sk-expflow-task1-e2etest",
            "session_id": "expflow-task1-e2etest-abcdef",
            "task": "task1",
        },
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{PROXY_PORT}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-expflow-task1-e2etest",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            content = result["choices"][0]["message"]["content"]
            print(f"   [OK] API response: '{content}'")
            assert len(content) > 0, "Empty response"
    except Exception as exc:
        print(f"   [FAIL] API call failed: {exc}")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise

    # Wait a moment for callback to flush
    time.sleep(2)

    # 4. Check that callback wrote competition JSONL
    print("[3/5] Checking competition JSONL output...")
    llm_files = sorted(log_dir.glob("llm-*.jsonl"))
    assert len(llm_files) > 0, (
        f"No llm-*.jsonl files found in {log_dir}. "
        f"Callback may not have fired."
    )

    found_entries = 0
    for lf in llm_files:
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Verify required fields
                assert "timestamp" in entry, f"Missing timestamp: {entry}"
                assert "elapsed_seconds" in entry, (
                    f"Missing elapsed_seconds: {entry}"
                )
                found_entries += 1

    assert found_entries >= 1, (
        f"No valid entries in {llm_files}"
    )
    print(f"   [OK] Found {found_entries} competition log entries")

    # 5. Merge with dummy fast.log -> task1_logs.log
    print("[4/5] Testing merge pipeline...")
    from expflow_pde.competition.merge import merge_logs, validate_log

    # Write a sample fast.log
    fast_log = log_dir / "fast.log"
    fast_log.write_text(
        "2026-05-23T10:00:01.000Z -  INFO - "
        "[train:306] - [FNO:v1] - [PID:123] - "
        "Epoch 1/80 | loss=0.003400 | GPU=2048/4096MB\n"
    )

    output = log_dir / "task1_logs.log"
    n = merge_logs(
        fast_log=fast_log,
        agent_log=log_dir / "agent.log",
        llm_files=llm_files,
        output=output,
        task="task1",
    )
    assert n > 0, f"No entries after merge"
    print(f"   [OK] Merged {n} entries -> {output}")

    # 6. Validate
    print("[5/5] Validating competition log...")
    errors = validate_log(output)
    for e in errors:
        print(f"   [WARN] {e}")
    print(f"   [{'OK' if not errors else 'FAIL'}] "
          f"Validation: {len(errors)} errors")

    # 7. Cleanup
    print("\nCleaning up...")
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    time.sleep(1)

    if not errors:
        print("E2E TEST PASSED")
    else:
        print("E2E TEST COMPLETED (non-critical validation issues)")
        print(f"Log dir preserved at: {log_dir}")

    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        sys.exit(test_e2e_competition_litellm())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
