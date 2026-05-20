#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taskctl.py — Background task monitor CLI.

Zero-token, crontab-driven monitoring for long-running processes.
Register a PID-based task with expected duration; a periodic check
detects completion or timeout, sends notifications, and optionally
triggers chain commands for the reverse pipeline pattern.

Usage:
  # Register a task
  python3 taskctl.py add --id my_task --pid 12345 --ctx "description" --duration 3600
  python3 taskctl.py list
  python3 taskctl.py remove my_task
  python3 taskctl.py status
  python3 taskctl.py check         # Single check (called by crontab)
  python3 taskctl.py clear         # Clean up expired completed tasks

System crontab (add via crontab -e):
  */15 * * * * cd ~/.hermes/task_monitor && python3 taskctl.py check >/dev/null 2>&1
"""

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── Constants ──────────────────────────────────────────────────────────
BASE_DIR = Path.home() / ".hermes" / "task_monitor"
TASKS_FILE = BASE_DIR / "tasks.json"
LOG_FILE = BASE_DIR / "taskctl.log"
CONF_FILE = BASE_DIR / "taskctl.conf"

QQ_SEND_SCRIPT = (
    Path(__file__).parent / "qq_send.py"
    if (Path(__file__).parent / "qq_send.py").exists()
    else Path.home() / ".hermes" / "task_monitor" / "qq_send.py"
)

MAX_TASKS = 50
TIMEOUT_MULTIPLIER = 1.5

# ─── Logging (5MB rotation, 3 backups) ──────────────────────────────────
BASE_DIR.mkdir(parents=True, exist_ok=True)
_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
_log = logging.getLogger("taskctl")
_log.setLevel(logging.INFO)
_log.addHandler(_handler)
_log.addHandler(logging.StreamHandler(sys.stderr))


# ─── Config ─────────────────────────────────────────────────────────────
def _load_config() -> dict:
    """Load configuration from taskctl.conf (simple KEY=VALUE format)."""
    config: dict = {}
    if CONF_FILE.exists():
        for line in CONF_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    # Environment variable takes precedence
    if os.environ.get("QQ_TARGET_USER"):
        config["QQ_TARGET_USER"] = os.environ["QQ_TARGET_USER"]
    return config


# ─── Core Data Operations ───────────────────────────────────────────────
def load_tasks() -> list[dict]:
    """Load task list from tasks.json."""
    if not TASKS_FILE.exists():
        return []
    try:
        data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        return data.get("tasks", [])
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return []


def save_tasks(tasks: list[dict]) -> list[dict]:
    """Save task list to tasks.json (FIFO, max MAX_TASKS)."""
    trimmed = tasks[-MAX_TASKS:] if len(tasks) > MAX_TASKS else tasks
    TASKS_FILE.write_text(
        json.dumps(
            {"tasks": trimmed, "updated_at": time.time()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return trimmed


# ─── Process Detection ──────────────────────────────────────────────────
def check_process(pid: int) -> bool:
    """Check if a process is alive (signal 0 probe)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ─── QQ Notification ────────────────────────────────────────────────────
def send_notification(message: str) -> bool:
    """Send a notification via the QQ REST API script.

    Returns True if the message was sent successfully.
    Falls back to print if qq_send.py is unavailable.
    """
    config = _load_config()
    target_user = config.get("QQ_TARGET_USER", "")

    if not QQ_SEND_SCRIPT.exists():
        _log.warning("qq_send.py not found, falling back to console")
        _log.info("Notification: %s", message[:200])
        return False

    cmd = [
        sys.executable,
        str(QQ_SEND_SCRIPT),
        message,
    ]
    if target_user:
        cmd.extend(["--user", target_user])

    for attempt in range(3):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                _log.info("Notification sent (attempt %d)", attempt + 1)
                return True
            _log.warning(
                "Notification #%d failed: %s",
                attempt + 1,
                result.stderr[:200],
            )
        except subprocess.TimeoutExpired:
            _log.warning("Notification #%d timeout", attempt + 1)
        except Exception as exc:
            _log.warning("Notification #%d error: %s", attempt + 1, exc)
        if attempt < 2:
            time.sleep(5 * (2**attempt))

    _log.error("All notification attempts failed")
    return False


# ─── Gateway Health Check (optional side-effect) ────────────────────────
_last_gw_check: float = 0


def _random_trigger(probability: float = 0.2) -> bool:
    """Simple pseudo-random trigger — fire ~every 1/probability calls."""
    h = hashlib.md5(str(time.time()).encode()).hexdigest()
    return int(h[:4], 16) / 65535 < probability


# ─── Subcommand Implementations ─────────────────────────────────────────
def cmd_add(args: argparse.Namespace) -> None:
    """Register a new task."""
    task = {
        "id": args.id,
        "pid": args.pid,
        "context": args.ctx,
        "start_time": time.time(),
        "expected_duration_s": args.duration,
        "expected_end": time.time() + args.duration,
        "status": "running",
        "notified_complete": False,
        "notified_timeout": False,
        "on_success_chain": args.on_success,
        "on_fail_chain": args.on_fail,
    }
    tasks = load_tasks()
    for i, t in enumerate(tasks):
        if t["id"] == args.id:
            tasks[i] = task
            break
    else:
        tasks.append(task)
    save_tasks(tasks)
    _log.info("Registered task: %s (pid=%d, duration=%ds)", args.id, args.pid, args.duration)
    print(f"[OK] Registered: {args.id} (pid={args.pid})")


def cmd_check(args: argparse.Namespace) -> None:
    """Check all running tasks."""
    tasks = load_tasks()
    changed = False
    now = time.time()

    for task in tasks:
        if task.get("status") != "running":
            continue

        pid = task["pid"]
        alive = check_process(pid)
        expected = task["expected_duration_s"]
        elapsed = now - task["start_time"]
        timeout_threshold = task["start_time"] + expected * TIMEOUT_MULTIPLIER
        is_overdue = now > timeout_threshold

        if not alive:
            task["status"] = "completed"
            task["end_time"] = now
            changed = True
            _log.info("Completed: %s (%.0fs)", task["id"], elapsed)

            if not task.get("notified_complete"):
                msg = (
                    "[OK] Task Complete\n"
                    f"  ID: {task['id']}\n"
                    f"  Context: {task['context'][:80]}\n"
                    f"  Duration: {elapsed:.0f}s\n"
                    f"  Time: {datetime.now().strftime('%H:%M')}"
                )
                send_notification(msg)
                task["notified_complete"] = True

            if task.get("on_success_chain"):
                _log.info("Starting success chain: %s", task["on_success_chain"])
                subprocess.Popen(
                    task["on_success_chain"],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        elif is_overdue:
            task["status"] = "timeout"
            changed = True
            _log.warning(
                "Timeout: %s (%.0fs, expected=%ds)",
                task["id"], elapsed, expected,
            )

            if not task.get("notified_timeout"):
                msg = (
                    "[WARN] Task Timeout\n"
                    f"  ID: {task['id']}\n"
                    f"  Context: {task['context'][:80]}\n"
                    f"  Elapsed: {elapsed:.0f}s\n"
                    f"  Expected: {expected}s\n"
                    f"  Time: {datetime.now().strftime('%H:%M')}"
                )
                send_notification(msg)
                task["notified_timeout"] = True

            if task.get("on_fail_chain"):
                _log.info("Starting fail chain: %s", task["on_fail_chain"])
                subprocess.Popen(
                    task["on_fail_chain"],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    if changed:
        save_tasks(tasks)


def cmd_list(args: argparse.Namespace) -> None:
    """List all tasks with status."""
    tasks = load_tasks()
    if not tasks:
        print("[NF] No tasks")
        return

    running = [t for t in tasks if t["status"] == "running"]
    done = [t for t in tasks if t["status"] != "running"]
    print(f"[OK] Total {len(tasks)} | Running {len(running)} | Done {len(done)}")
    print()

    if running:
        print("-- Running --")
        for t in running:
            elapsed = time.time() - t["start_time"]
            pct = min(100, elapsed / t["expected_duration_s"] * 100)
            print(f"  [OK] {t['id']}")
            print(f"    {t['context'][:80]}")
            print(
                f"    pid={t['pid']}  "
                f"{elapsed:.0f}s / {t['expected_duration_s']}s ({pct:.0f}%)"
            )

    if done:
        print("-- Done --")
        for t in done[-10:]:
            duration = (
                t.get("end_time", t["start_time"]) - t["start_time"]
            )
            icon_map = {"completed": "[OK]", "timeout": "[WARN]"}
            icon = icon_map.get(t["status"], "[ERR]")
            print(
                f"  {icon} {t['id']}  {duration:.0f}s  "
                f"({t['context'][:50]})"
            )


def cmd_remove(args: argparse.Namespace) -> None:
    """Remove a task by ID."""
    tasks = load_tasks()
    before = len(tasks)
    tasks = [t for t in tasks if t["id"] != args.id]
    if len(tasks) < before:
        save_tasks(tasks)
        print(f"[OK] Removed: {args.id}")
        _log.info("Removed task: %s", args.id)
    else:
        print(f"[WARN] Task not found: {args.id}")


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear old completed tasks (keep running + last 10 done)."""
    tasks = load_tasks()
    before = len(tasks)
    active = [t for t in tasks if t["status"] == "running"]
    recent_done = [t for t in tasks if t["status"] != "running"][-10:]
    save_tasks(active + recent_done)
    cleared = before - len(active + recent_done)
    _log.info("Cleared %d stale tasks", cleared)
    print(
        f"[OK] Cleared. Remaining: {len(active)} running "
        f"+ {len(recent_done)} recent done"
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Quick status overview."""
    tasks = load_tasks()
    running = sum(1 for t in tasks if t["status"] == "running")
    completed = sum(1 for t in tasks if t["status"] == "completed")
    timeout_count = sum(1 for t in tasks if t["status"] == "timeout")
    total = len(tasks)
    print(f"[OK] {running} [WARN] {timeout_count} [NF] {completed} total {total}")


# ─── Main Entry ─────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="taskctl — Background task monitor"
    )
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Register a new task")
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--pid", type=int, required=True)
    p_add.add_argument("--ctx", default="")
    p_add.add_argument("--duration", type=int, default=3600)
    p_add.add_argument(
        "--on-success",
        default=None,
        help="Shell command to run on successful completion (reverse pipeline)",
    )
    p_add.add_argument(
        "--on-fail",
        default=None,
        help="Shell command to run on timeout",
    )

    # check
    sub.add_parser("check", help="Check all task statuses")

    # list
    sub.add_parser("list", help="List all tasks")

    # remove
    p_rm = sub.add_parser("remove", help="Remove a task")
    p_rm.add_argument("id")

    # clear
    sub.add_parser("clear", help="Clear expired completed tasks")

    # status
    sub.add_parser("status", help="Quick status overview")

    args = parser.parse_args()

    cmds = {
        "add": cmd_add,
        "check": cmd_check,
        "list": cmd_list,
        "remove": cmd_remove,
        "clear": cmd_clear,
        "status": cmd_status,
    }

    if args.command not in cmds:
        parser.print_help()
        sys.exit(1)

    cmds[args.command](args)


if __name__ == "__main__":
    main()
