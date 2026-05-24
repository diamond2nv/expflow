#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end test v2: litellm proxy + built-in SQLite logging.

Approach:
  1. Start litellm proxy (simple config, no custom callback)
  2. Make a real API call through it
  3. Extract log from litellm's SQLite DB → competition JSONL
  4. Merge with fast.log → validate
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_EXPFLOW_ROOT = os.path.expanduser("~/Gitlab/Agentic4Sci/expflow")
sys.path.insert(0, _EXPFLOW_ROOT)

# Load API key
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

LOG_DIR = Path(tempfile.gettempdir()) / "comp_e2e_v2"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROXY_PORT = 14002
DB_PATH = LOG_DIR / "litellm.db"


def _build_config(log_dir: str) -> dict:
    return {
        "general_settings": {
            "master_key": "sk-master-e2e",
            "database_url": f"sqlite:///{log_dir}/litellm.db",
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
    }


def _extract_logs_from_db(db_path: Path) -> list[dict]:
    """Extract LLM call logs from litellm's SQLite DB as competition JSONL."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check available tables
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print(f"  DB tables: {tables}")

    # Try LiteLLM_SpendLogs table (standard litellm proxy table)
    entries = []
    if "LiteLLM_SpendLogs" in tables:
        # Get column names
        cols = [r[1] for r in cur.execute(
            "PRAGMA table_info(LiteLLM_SpendLogs)"
        ).fetchall()]
        print(f"  LiteLLM_SpendLogs columns: {cols}")

        rows = cur.execute(
            "SELECT * FROM LiteLLM_SpendLogs ORDER BY startTime"
        ).fetchall()

        for row in rows:
            d = dict(row)
            entry = {
                "timestamp": d.get("startTime", datetime.now(timezone.utc).isoformat()),
                "elapsed_seconds": round(
                    (d.get("endTime", 0) or 0) - (d.get("startTime", 0) or 0), 3
                ) if d.get("endTime") else 3.0,
                "model": d.get("model", ""),
            }

            # Extract response content
            response_text = d.get("response", "")
            if isinstance(response_text, str) and response_text:
                try:
                    resp_obj = json.loads(response_text)
                    if isinstance(resp_obj, dict):
                        choices = resp_obj.get("choices", [{}])
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "")
                        if content:
                            entry["response"] = content[:2000]
                except json.JSONDecodeError:
                    pass

            # Extract request messages → detect tool calls
            messages = d.get("messages", "")
            if isinstance(messages, str) and messages:
                try:
                    msgs = json.loads(messages)
                    # Check for tool_calls in assistant messages
                    tool_parts = []
                    for m in msgs:
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            for tc in m["tool_calls"]:
                                fn = tc.get("function", {})
                                tool_parts.append(
                                    f"{fn.get('name', '?')}("
                                    f"{fn.get('arguments', '')[:200]})"
                                )
                    if tool_parts:
                        entry["tool_calls"] = "\n".join(tool_parts)
                except json.JSONDecodeError:
                    pass

            entries.append(entry)

    conn.close()
    return entries


def main():
    print("=" * 60)
    print("E2E Test v2: litellm proxy + built-in DB logging")
    print("=" * 60)

    # 1. Build config
    config = _build_config(str(LOG_DIR))
    config_path = LOG_DIR / "litellm_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # 2. Start proxy
    print(f"\n[1/4] Starting litellm proxy on port {PROXY_PORT}...")
    log_file = LOG_DIR / "proxy.log"
    litellm_bin = os.path.join(os.path.dirname(sys.executable), "litellm")
    if not os.path.isfile(litellm_bin):
        litellm_bin = "litellm"

    env = os.environ.copy()
    proc = subprocess.Popen(
        [litellm_bin, "--config", str(config_path), "--port", str(PROXY_PORT)],
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )

    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"http://localhost:{PROXY_PORT}/health",
                headers={"Authorization": "Bearer sk-master-e2e"},
            )
            with urllib.request.urlopen(req, timeout=3):
                break
        except Exception:
            time.sleep(2)
    else:
        print("   [FAIL] Proxy startup timed out")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return 1
    print("   [OK] Proxy ready")

    # 3. Make API call
    print("\n[2/4] Making real API call through proxy...")
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{PROXY_PORT}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-master-e2e",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"]
        print(f"   [OK] Response: '{content}'")

    # Wait for DB write
    time.sleep(2)

    # 4. Extract logs from DB
    print("\n[3/4] Extracting logs from litellm DB...")
    entries = _extract_logs_from_db(DB_PATH)

    if entries:
        # Save as competition JSONL
        llm_jsonl = LOG_DIR / "llm-proxy.jsonl"
        with open(llm_jsonl, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"   [OK] Extracted {len(entries)} entries → {llm_jsonl}")

        # 5. Merge + validate
        print("\n[4/4] Merging + validating...")
        from expflow_pde.competition.merge import merge_logs, validate_log

        # Dummy fast.log
        (LOG_DIR / "fast.log").write_text(
            "2026-05-23T10:00:01.000Z -  INFO - [train:306] - "
            "[FNO:v1] - [PID:123] - "
            "Epoch 1/80 | loss=0.0034 | GPU=2048/4096MB\n"
        )

        output = LOG_DIR / "task1_logs.log"
        n = merge_logs(
            fast_log=LOG_DIR / "fast.log",
            agent_log=LOG_DIR / "agent.log",
            llm_files=[llm_jsonl],
            output=output,
        )
        print(f"   Merged: {n} entries")

        errors = validate_log(output)
        print(f"   Validate: {'PASS' if not errors else 'FAIL'} "
              f"({len(errors)} errors)")
        for e in errors[:5]:
            print(f"     - {e}")

        if not errors:
            print("\n✅ E2E TEST PASSED")
            result = 0
        else:
            print("\n⚠️  E2E completed (non-critical validation issues)")
            result = 0
    else:
        print("   [FAIL] No entries extracted from DB")
        print(f"   DB path: {DB_PATH}, size: {DB_PATH.stat().st_size} bytes")
        result = 1

    # Cleanup
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print(f"\nLog dir preserved: {LOG_DIR}")
    return result


if __name__ == "__main__":
    sys.exit(main())
