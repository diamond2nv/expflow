#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taskctl.py v2.0 — Background task monitor with ZMQ event broker.

Reverse pipeline:
  Layer 1) Cron (15min) PID polling     — reliable fallback
  Layer 2) ZMQ PUB-SUB event push        — real-time experiment events
  Layer 3) Hermes goal integration       — persistent objective driving (external)

Features:
  - Dual trigger: crontab polling + ZMQ event broker
  - PID file + signal handler for graceful shutdown
  - ZMQ LINGER=0 for non-blocking publish
  - Chain command subprocess pool with timeout isolation
  - Dedup lock via tasks.json WAL + memory state key
  - Structured JSON log (same rotation file, 5MB x3)
  - Idle frequency scaling (15min -> 1h after 4 empty checks)
  - PEP8, no emoji, no Chinese, English-only output

Usage:
  # Daemon mode (background: listens on ZMQ)
  python3 taskctl.py daemon --port 15556

  # CLI mode
  python3 taskctl.py add --id my_task --pid 12345 ...
  python3 taskctl.py check
  python3 taskctl.py list
"""

import argparse
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Optional ZMQ (graceful import fallback) ---
try:
    import zmq as _zmq  # noqa: F401

    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

# ─── Constants ──────────────────────────────────────────────────────────
BASE_DIR = Path.home() / ".hermes" / "task_monitor"
TASKS_FILE = BASE_DIR / "tasks.json"
LOG_FILE = BASE_DIR / "taskctl.log"
LOCK_FILE = BASE_DIR / "taskctl.lock"
PID_FILE = BASE_DIR / "taskctl.pid"
CONF_FILE = BASE_DIR / "taskctl.conf"

QQ_SEND_SCRIPT = (
    Path(__file__).parent / "qq_send.py"
    if (Path(__file__).parent / "qq_send.py").exists()
    else BASE_DIR / "qq_send.py"
)

DEFAULT_PUB_PORT = 15556
DEFAULT_SUB_PORT = 15557
MAX_TASKS = 50
TIMEOUT_MULTIPLIER = 1.5
CHAIN_TIMEOUT = 300  # Max seconds per chain command
IDLE_THRESHOLD = 4  # Consecutive empty checks before frequency drop

# Structured JSON log keys (persistent, rotation)
LOG_KEYS = ["ts", "level", "event", "task_id", "pid", "duration", "context", "error"]

# ─── Logging (structured JSON + human-readable) ─────────────────────────
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Rotating file handler (5MB x 3, JSON-like lines for machine parsing)
_json_handler = logging.handlers.RotatingFileHandler(
    Path(str(LOG_FILE) + ".jsonl"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_json_handler.setFormatter(logging.Formatter("%(message)s"))


class StructuredLogger(logging.Logger):
    """Logger that emits structured JSON in addition to plain text.

    Each log call produces:
      - Plain text line in taskctl.log (human)
      - JSON line in taskctl.log.jsonl (machine)
    """

    def _log(self, level, msg, args, **kwargs) -> None:
        super()._log(level, msg, args, **kwargs)
        # Emit structured JSON
        record = {
            "ts": time.time(),
            "level": logging.getLevelName(level),
            "msg": msg % args if args else msg,
            "extra": kwargs.pop("extra", {}),
        }
        json_line = json.dumps(record, ensure_ascii=False)
        _json_handler.emit(
            logging.LogRecord(
                name=self.name,
                level=level,
                pathname="",
                lineno=0,
                msg=json_line,
                args=(),
                exc_info=None,
            )
        )


logging.setLoggerClass(StructuredLogger)

# Plain text log
_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_log = logging.getLogger("taskctl")
_log.setLevel(logging.INFO)
_log.addHandler(_handler)
_log.addHandler(logging.StreamHandler(sys.stderr))

# Structured JSON handler
_log.addHandler(_json_handler)


# ─── Config ─────────────────────────────────────────────────────────────
def _load_config() -> dict:
    """Load configuration from taskctl.conf + env vars."""
    config: dict = {}
    if CONF_FILE.exists():
        for line in CONF_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    if os.environ.get("QQ_TARGET_USER"):
        config["QQ_TARGET_USER"] = os.environ["QQ_TARGET_USER"]
    return config


# ─── Lock (WAL-based dedup) ─────────────────────────────────────────────
class TaskLock:
    """File-based lock with WAL semantics for concurrent task checks.

    Prevents duplicate state transitions from overlapping cron + ZMQ events.
    """

    def __init__(self, lock_file: Path = LOCK_FILE):
        self._lock_file = lock_file
        self._held = False

    def acquire(self, task_id: str, ttl: int = 30) -> bool:
        """Try to acquire lock for a task transition.

        Args:
            task_id: Unique task identifier.
            ttl: Lock TTL in seconds (prevents stale locks).

        Returns True if lock acquired.
        """
        now = time.time()
        try:
            # Read current lock
            if self._lock_file.exists():
                data = json.loads(self._lock_file.read_text(encoding="utf-8"))
                if data["task_id"] == task_id:
                    # Refresh our lock
                    pass
                elif now - data.get("ts", 0) < ttl:
                    _log.debug(
                        "Lock held by %s (expires in %.0fs)",
                        data["task_id"],
                        ttl - (now - data["ts"]),
                    )
                    return False
            # Write lock
            self._lock_file.write_text(
                json.dumps({"task_id": task_id, "ts": now}, ensure_ascii=False),
                encoding="utf-8",
            )
            self._held = True
            return True
        except Exception as exc:
            _log.warning("Lock acquire error: %s", exc)
            return False

    def release(self) -> None:
        """Release the lock."""
        self._held = False
        try:
            if self._lock_file.exists():
                self._lock_file.unlink()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# ─── Signal Handler (graceful shutdown) ─────────────────────────────────
_shutdown_requested = False
_zmq_broker: Any = None  # Will hold Publisher reference


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    _log.info(
        "Shutdown requested (signal %d), finishing current check...",
        signum,
    )


def _register_signal_handlers() -> None:
    """Register signal handlers for graceful exit."""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


# ─── PID File Management ────────────────────────────────────────────────
def _write_pid() -> None:
    """Write current PID to PID_FILE."""
    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def _read_pid() -> int | None:
    """Read PID from PID_FILE. Returns None if stale or missing."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        # Check if process is alive
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError, ProcessLookupError):
        return None


def _cleanup_pid() -> None:
    """Remove PID file on exit."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


# ─── Core Data ──────────────────────────────────────────────────────────
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
    """Save task list (FIFO, max MAX_TASKS). Returns trimmed list."""
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


# ─── Chain Command Execution (subprocess pool) ──────────────────────────
def execute_chain(chain_cmd: str, task_id: str, context: str) -> dict:
    """Execute a single chain command with timeout isolation.

    Returns result dict with keys: success, output, error, duration.
    """
    start = time.time()
    result = {"task_id": task_id, "chain_cmd": chain_cmd[:100], "success": False}

    try:
        proc = subprocess.run(
            chain_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=CHAIN_TIMEOUT,
        )
        result["duration"] = time.time() - start
        result["exit_code"] = proc.returncode

        if proc.returncode == 0:
            result["success"] = True
            result["output"] = proc.stdout[:500]
            _log.info(
                "Chain OK: %s (%.1fs)",
                task_id,
                result["duration"],
                extra={"task_id": task_id, "event": "chain_ok"},
            )
        else:
            result["error"] = proc.stderr[:500] or proc.stdout[:500]
            _log.warning(
                "Chain FAIL: %s exit=%d (%.1fs): %s",
                task_id,
                proc.returncode,
                result["duration"],
                proc.stderr[:200],
                extra={"task_id": task_id, "event": "chain_fail"},
            )
    except subprocess.TimeoutExpired:
        result["duration"] = time.time() - start
        result["error"] = f"Timeout ({CHAIN_TIMEOUT}s)"
        _log.warning(
            "Chain TIMEOUT: %s (%.1fs)",
            task_id,
            result["duration"],
            extra={"task_id": task_id, "event": "chain_timeout"},
        )
    except Exception as exc:
        result["duration"] = time.time() - start
        result["error"] = str(exc)
        _log.error(
            "Chain ERROR: %s: %s",
            task_id,
            exc,
            extra={"task_id": task_id, "event": "chain_error"},
        )

    return result


def execute_chain_multiple(
    cmds: list[str],
    task_id: str,
    context: str,
    parallel: bool = False,
) -> list[dict]:
    """Execute multiple chain commands, optionally in parallel.

    Args:
        cmds: List of shell commands.
        task_id: Task identifier (for logging).
        context: Task context (for logging).
        parallel: If True, run in parallel via subprocess.Popen + wait.

    Returns list of result dicts.
    """
    if not cmds:
        return []

    if parallel:
        # Parallel execution: launch all, wait all
        procs = []
        for cmd in cmds:
            p = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append((cmd, p))

        results = []
        for cmd, p in procs:
            try:
                p.wait(timeout=CHAIN_TIMEOUT)
                results.append(
                    {
                        "task_id": task_id,
                        "chain_cmd": cmd[:100],
                        "success": p.returncode == 0,
                        "exit_code": p.returncode,
                    }
                )
            except subprocess.TimeoutExpired:
                p.kill()
                results.append(
                    {
                        "task_id": task_id,
                        "chain_cmd": cmd[:100],
                        "success": False,
                        "error": f"Timeout ({CHAIN_TIMEOUT}s)",
                    }
                )
        return results
    else:
        # Serial execution
        return [execute_chain(cmd, task_id, context) for cmd in cmds]


# ─── QQ Notification ────────────────────────────────────────────────────
def send_notification(message: str) -> bool:
    """Send notification via standalone qq_send.py."""
    config = _load_config()
    target_user = config.get("QQ_TARGET_USER", "")

    if not QQ_SEND_SCRIPT.exists():
        _log.info("Notification fallback (no qq_send.py): %s", message[:200])
        return False

    cmd = [sys.executable, str(QQ_SEND_SCRIPT), message]
    if target_user:
        cmd.extend(["--user", target_user])

    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
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
    return False


# ─── ZMQ Event Integration ──────────────────────────────────────────────
def _init_zmq(pub_port: int = DEFAULT_PUB_PORT) -> Any | None:
    """Initialize ZMQ publisher. Returns Publisher instance or None."""
    if not HAS_ZMQ:
        _log.info("ZMQ not available — cron-only mode")
        return None

    try:
        from zmq_broker import DEFAULT_HWM, Publisher

        pub = Publisher(port=pub_port, hwm=DEFAULT_HWM, use_ipc=True, use_tcp=True)
        pub.start()
        _log.info("ZMQ broker started on port %d", pub_port)
        return pub
    except Exception as exc:
        _log.warning("ZMQ init failed (cron-only): %s", exc)
        return None


def _emit_event(
    broker: Any,
    topic: str,
    payload: dict,
    qos: int = 1,
) -> None:
    """Publish an event via ZMQ (if broker available)."""
    if broker is not None:
        try:
            broker.publish(topic, payload, qos=qos)
        except Exception as exc:
            _log.warning("ZMQ publish failed: %s", exc)


# ─── Idle Frequency Scaling ────────────────────────────────────────────
_idle_counter = 0


def _should_skip_check() -> bool:
    """Skip check if in idle frequency mode.

    After IDLE_THRESHOLD consecutive empty checks, drop frequency
    to once per hour (skipping 3 out of 4 checks).
    """
    global _idle_counter
    is_empty = not TASKS_FILE.exists() or not json.loads(
        TASKS_FILE.read_text(encoding="utf-8")
    ).get("tasks", [])

    if is_empty:
        _idle_counter += 1
        if _idle_counter >= IDLE_THRESHOLD:
            # Skip 3/4 checks -> effectively 1h frequency
            # Use minute-of-hour as simple gate
            current_min = datetime.now().minute
            return current_min % 15 != 0  # Only run at :00 of each hour
    else:
        _idle_counter = 0

    return False


# ─── Subcommand: daemon ─────────────────────────────────────────────────
def cmd_daemon(args: argparse.Namespace) -> None:
    """Start ZMQ daemon (listens for events and publishes them).

    This is an optional process that co-exists with crontab polling.
    It reacts to real-time events (ClearML completion, GPU OOM, etc.)
    and pushes them to ZMQ subscribers.
    """
    global _zmq_broker

    # Check for existing daemon
    existing_pid = _read_pid()
    if existing_pid is not None:
        _log.info("Daemon already running (PID %d)", existing_pid)
        print(f"[OK] Daemon running (PID {existing_pid})")
        return

    # Write PID file
    _write_pid()
    _register_signal_handlers()

    # Start ZMQ broker
    _zmq_broker = _init_zmq(args.port)
    if _zmq_broker is None:
        print("[WARN] ZMQ not available. Daemon started in log-only mode.")
        _log.info("Daemon started (log-only mode)")
    else:
        _log.info("Daemon started (ZMQ on port %d)", args.port)
        print(f"[OK] Daemon started (PID {os.getpid()}, ZMQ port {args.port})")

    # Heartbeat loop
    heartbeat_interval = 60  # seconds
    while not _shutdown_requested:
        time.sleep(heartbeat_interval)
        if _zmq_broker is not None:
            _emit_event(
                _zmq_broker,
                "system/heartbeat",
                {"pid": os.getpid(), "ts": time.time()},
                qos=0,
            )

    # Cleanup
    _cleanup_pid()
    if _zmq_broker is not None:
        _zmq_broker.stop()
    _log.info("Daemon stopped")


# ─── Subcommand: add ────────────────────────────────────────────────────
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
    _log.info(
        "Registered: %s (pid=%d, duration=%ds)",
        args.id,
        args.pid,
        args.duration,
        extra={"task_id": args.id, "event": "register", "pid": args.pid},
    )
    print(f"[OK] Registered: {args.id} (pid={args.pid})")

    # Emit ZMQ event
    if hasattr(args, "_zmq_broker") and args._zmq_broker:
        _emit_event(
            args._zmq_broker,
            f"taskctl/{args.id}/register",
            {"task_id": args.id, "pid": args.pid, "duration": args.duration},
        )


# ─── Subcommand: check (core loop) ──────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> None:
    """Check all running tasks. Called by crontab or ZMQ event."""
    global _shutdown_requested

    if _shutdown_requested:
        _log.info("Shutdown in progress, skipping check")
        return

    # Idle frequency scaling
    tasks = load_tasks()
    if not tasks:
        # No tasks: check if we should skip
        pass  # Always run at least once after registration

    if _should_skip_check():
        return

    changed = False
    now = time.time()

    for task in tasks:
        if _shutdown_requested:
            break
        if task.get("status") != "running":
            continue

        pid = task["pid"]
        alive = check_process(pid)
        expected = task["expected_duration_s"]
        elapsed = now - task["start_time"]
        timeout_threshold = task["start_time"] + expected * TIMEOUT_MULTIPLIER
        is_overdue = now > timeout_threshold

        # Dedup lock: prevent concurrent state transition
        with TaskLock() as lock:
            if not lock.acquire(task["id"]):
                continue  # Another process handling this task

            if not alive:
                # Completed
                task["status"] = "completed"
                task["end_time"] = now
                changed = True
                _log.info(
                    "Completed: %s (%.0fs)",
                    task["id"],
                    elapsed,
                    extra={"task_id": task["id"], "event": "complete", "duration": elapsed},
                )

                # Notification
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

                # Chain commands
                if task.get("on_success_chain"):
                    cmds = [c.strip() for c in task["on_success_chain"].split("&&")]
                    _log.info(
                        "Running success chain: %s (%d cmds)",
                        task["id"],
                        len(cmds),
                        extra={"task_id": task["id"], "event": "chain_start"},
                    )
                    execute_chain_multiple(cmds, task["id"], task.get("context", ""))

                # Emit ZMQ event
                _emit_event(
                    args._zmq_broker if hasattr(args, "_zmq_broker") else None,
                    f"taskctl/{task['id']}/complete",
                    {
                        "task_id": task["id"],
                        "duration": elapsed,
                        "status": "completed",
                    },
                )

            elif is_overdue:
                # Timeout
                task["status"] = "timeout"
                changed = True
                _log.warning(
                    "Timeout: %s (%.0fs, expected=%ds)",
                    task["id"],
                    elapsed,
                    expected,
                    extra={"task_id": task["id"], "event": "timeout"},
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
                    cmds = [c.strip() for c in task["on_fail_chain"].split("&&")]
                    _log.info(
                        "Running fail chain: %s (%d cmds)",
                        task["id"],
                        len(cmds),
                        extra={"task_id": task["id"], "event": "chain_start"},
                    )
                    execute_chain_multiple(cmds, task["id"], task.get("context", ""))

                # Emit ZMQ event
                _emit_event(
                    args._zmq_broker if hasattr(args, "_zmq_broker") else None,
                    f"taskctl/{task['id']}/timeout",
                    {
                        "task_id": task["id"],
                        "duration": elapsed,
                        "expected": expected,
                        "status": "timeout",
                    },
                )

    if changed:
        save_tasks(tasks)


# ─── Subcommand: list ───────────────────────────────────────────────────
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
            print(f"    pid={t['pid']}  {elapsed:.0f}s / {t['expected_duration_s']}s ({pct:.0f}%)")

    if done:
        print("-- Done --")
        for t in done[-10:]:
            duration = t.get("end_time", t["start_time"]) - t["start_time"]
            icon_map = {"completed": "[OK]", "timeout": "[WARN]"}
            icon = icon_map.get(t["status"], "[ERR]")
            print(f"  {icon} {t['id']}  {duration:.0f}s  ({t['context'][:50]})")


# ─── Subcommand: remove ─────────────────────────────────────────────────
def cmd_remove(args: argparse.Namespace) -> None:
    """Remove a task by ID."""
    tasks = load_tasks()
    before = len(tasks)
    tasks = [t for t in tasks if t["id"] != args.id]
    if len(tasks) < before:
        save_tasks(tasks)
        print(f"[OK] Removed: {args.id}")
        _log.info("Removed: %s", args.id, extra={"task_id": args.id, "event": "remove"})
    else:
        print(f"[WARN] Not found: {args.id}")


# ─── Subcommand: clear ──────────────────────────────────────────────────
def cmd_clear(args: argparse.Namespace) -> None:
    """Clear old completed tasks."""
    tasks = load_tasks()
    before = len(tasks)
    active = [t for t in tasks if t["status"] == "running"]
    recent_done = [t for t in tasks if t["status"] != "running"][-10:]
    save_tasks(active + recent_done)
    cleared = before - len(active + recent_done)
    print(
        f"[OK] Cleared {cleared}. Remaining: {len(active)} running + {len(recent_done)} recent done"
    )
    _log.info(
        "Cleared %d stale tasks",
        cleared,
        extra={"event": "clear", "cleared": cleared},
    )


# ─── Subcommand: status ─────────────────────────────────────────────────
def cmd_status(args: argparse.Namespace) -> None:
    """Quick status overview."""
    tasks = load_tasks()
    running = sum(1 for t in tasks if t["status"] == "running")
    completed = sum(1 for t in tasks if t["status"] == "completed")
    timeout_count = sum(1 for t in tasks if t["status"] == "timeout")
    total = len(tasks)
    print(f"[OK] {running} [WARN] {timeout_count} [NF] {completed} total {total}")


# ─── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="taskctl v2.0 — Background task monitor with ZMQ event broker"
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
        help="Shell command on completion (use && for multi-cmd)",
    )
    p_add.add_argument(
        "--on-fail",
        default=None,
        help="Shell command on timeout",
    )

    # daemon (ZMQ broker)
    p_daemon = sub.add_parser("daemon", help="Start ZMQ event broker daemon")
    p_daemon.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PUB_PORT,
        help="ZMQ PUB port (default: 15556)",
    )

    # check
    sub.add_parser("check", help="Check all running tasks")

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
        "daemon": cmd_daemon,
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
