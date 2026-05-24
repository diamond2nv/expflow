#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition session lifecycle manager — litellm proxy + JSONL ingest server.

Orchestrates:
  - litellm proxy start/stop (no database_url, no prisma)
  - Local JSONL ingest server start/stop (generic_api callback)
  - nvidia-smi + uname recording
  - Log merge + validation on teardown

Architecture:
    Hermes/OpenCode ──→ litellm proxy (:4000) ──→ DeepSeek API
                             │
                             │ generic_api callback (ndjson, batch_size=1)
                             ▼
                     ingest server (:8099) ──→ llm-YYYYMMDD.jsonl

Zero external services: no Langfuse, no prisma, no database.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .comp_log import _LOG_DIR, get_logger

if TYPE_CHECKING:
    from types import FrameType

# Name prefix for session IDs and log filenames
SESSION_PREFIX = "expflow"

# Default ports — keep them separated
DEFAULT_PROXY_PORT = 4000
DEFAULT_INGEST_PORT = 8099

# Path to the bundled ingest server module
_INGEST_SERVER = os.path.join(os.path.dirname(__file__), "_ingest_server.py")


class CompetitionSession:
    """Manage the lifecycle of a competition logging session.

    Typical workflow:
        session = CompetitionSession(task='task1', tag='v1')
        session.start()
        # ... agent work (training + LLM calls via litellm proxy) ...
        session.stop()

    After stop(), the merged task1_logs.log is written to
    ~/.hermes/competition_logs/{task}_{tag}/.
    """

    def __init__(
        self,
        task: str = "task1",
        tag: str = "",
        proxy_port: int = DEFAULT_PROXY_PORT,
        ingest_port: int = DEFAULT_INGEST_PORT,
        target_url: str = "",
        log_dir: str | None = None,
    ) -> None:
        """Initialize a competition session.

        Args:
            task: Task identifier ('task1' or 'task2').
            tag: Human-readable experiment tag.
            proxy_port: Port for litellm proxy.
            ingest_port: Port for JSONL ingest server.
            target_url: Upstream API base URL (default: DEEPSEEK_BASE_URL env).
            log_dir: Override default log directory.
        """
        self.task = task
        self.tag = tag
        self.proxy_port = proxy_port
        self.ingest_port = ingest_port
        self.session_id = f"{SESSION_PREFIX}-{task}-{int(time.time())}"
        self.target_url = target_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.target_api_key = os.environ.get(
            "DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")
        )
        self.master_key = f"sk-comp-{task}-{int(time.time())}"

        self._log_dir = Path(log_dir) if log_dir else _LOG_DIR
        self._log_dir = self._log_dir / f"{task}_{tag}" if tag else self._log_dir / task
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._comp_log = get_logger(
            f"session_{self.task}", operator="SESSION", tag=tag,
            log_dir=str(self._log_dir),
        )
        self._started = False
        self._session_start_ts: str | None = None
        self._proxy_pid: int | None = None
        self._ingest_pid: int | None = None
        self._config_path = self._log_dir / "litellm_config.yaml"

        # Profile isolation
        self._profile_name = f"comp-{task}-{tag}" if tag else f"comp-{task}"
        self._original_profile = os.environ.get("HERMES_PROFILE", "")
        self._profile_created = False

    # ── Public API ─────────────────────────────────────────────────────

    def start(self) -> dict:
        """Start the competition session.

        Returns:
            dict with session metadata: session_id, master_key, proxy_port,
            ingest_port, log_dir.
        """
        if self._started:
            return {"session_id": self.session_id, "status": "already_running"}

        # 1. Record system environment
        self._record_system_env()

        # 2. Register signal handlers for graceful shutdown
        self._install_signal_handlers()

        # 3. Start ingest server FIRST (so callback endpoint is live)
        self._start_ingest_server()

        # 4. Generate litellm config + start proxy
        self._start_proxy()

        # 5. Create isolated Hermes profile
        self._create_profile()

        # 6. Mark as started — record precise UTC timestamp for first-entry elapsed computation
        self._started = True
        self._session_start_ts = datetime.now(timezone.utc).isoformat()
        self._comp_log.info(
            f"Competition session started: {self.session_id} "
            f"task={self.task} tag={self.tag} "
            f"proxy_port={self.proxy_port} ingest_port={self.ingest_port}"
        )
        self._comp_log.agent_note(
            f"[SESSION_START] session_id={self.session_id} "
            f"proxy_port={self.proxy_port} ingest_port={self.ingest_port}"
        )

        return {
            "session_id": self.session_id,
            "master_key": self.master_key,
            "proxy_port": self.proxy_port,
            "ingest_port": self.ingest_port,
            "log_dir": str(self._log_dir),
            "hermes_profile": self._profile_name,
            "profile_created": self._profile_created,
        }

    def stop(self, merge: bool = True) -> dict:
        """Stop the competition session.

        Args:
            merge: If True, merge logs and validate on teardown.

        Returns:
            dict with stop status and optional merge report.
        """
        if not self._started:
            return {"status": "not_running"}

        self._stop_ingest_server()
        self._stop_proxy()
        self._started = False

        # Remove temp config
        if self._config_path.exists():
            self._config_path.unlink()

        # Clean up profile (keep by default, user can delete manually)
        self._comp_log.info(
            f"To switch back to original profile: "
            f"hermes -p {self._original_profile or 'default'}"
        )
        self._comp_log.info(
            f"To delete competition profile: "
            f"hermes profile delete {self._profile_name}"
        )

        result = {"status": "stopped", "session_id": self.session_id}

        if merge:
            merge_result = self._merge_and_validate()
            result["merge"] = merge_result

        self._comp_log.info(
            f"Competition session stopped: {self.session_id}"
        )
        return result

    def status(self) -> dict:
        """Return current session status."""
        return {
            "started": self._started,
            "session_id": self.session_id,
            "task": self.task,
            "tag": self.tag,
            "proxy_port": self.proxy_port,
            "ingest_port": self.ingest_port,
            "log_dir": str(self._log_dir),
            "proxy_alive": self._proxy_alive(),
            "ingest_alive": self._ingest_alive(),
        }

    # ── Profile isolation ───────────────────────────────────────────────

    def _create_profile(self) -> None:
        """Create isolated Hermes profile for this competition session.

        Creates a profile named comp-{task}-{tag} via hermes profile CLI,
        then configures its providers.deepseek section to point to the
        litellm proxy on localhost. If profile creation fails (e.g., YAML
        parse error in main config), falls back to direct file manipulation.

        After this call, the user can run:
            hermes -p {profile_name}
        or set:
            export HERMES_PROFILE={profile_name}
        to route LLM calls through the competition proxy.
        """
        profile_dir = Path.home() / ".hermes" / "profiles" / self._profile_name

        # Check if already exists
        if profile_dir.exists():
            self._profile_created = True
            self._comp_log.info(
                f"Profile already exists: {self._profile_name}"
            )
            # Still update the config to match current proxy
            self._setup_profile_config(profile_dir)
            return

        # Try hermes CLI first
        try:
            r = subprocess.run(
                ["hermes", "profile", "create", self._profile_name, "--clone"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                self._profile_created = True
                self._comp_log.info(
                    f"Created profile: {self._profile_name} (via hermes CLI)"
                )
                self._setup_profile_config(profile_dir)
                return
            else:
                self._comp_log.warning(
                    f"hermes profile create failed: {r.stderr.strip()}"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            self._comp_log.warning(
                f"hermes CLI unavailable for profile creation: {exc}"
            )

        # Fallback: manual profile creation
        self._create_profile_manual(profile_dir)

    def _setup_profile_config(self, profile_dir: Path) -> None:
        """Configure proxy base_url and api_key in the profile config.

        Uses hermes config set CLI for reliability, falls back to direct
        YAML patching.
        """
        proxy_base = f"http://localhost:{self.proxy_port}"

        # Try hermes config set
        try:
            for key, val in [
                ("providers.deepseek.base_url", proxy_base),
                ("providers.deepseek.api_key", self.master_key),
            ]:
                subprocess.run(
                    ["hermes", "config", "set", "-p", self._profile_name, key, val],
                    capture_output=True, text=True, timeout=15,
                    env={**os.environ, "HERMES_PROFILE": self._profile_name},
                )
            self._comp_log.info(
                f"Profile {self._profile_name}: base_url={proxy_base}"
            )
            return
        except Exception as exc:
            self._comp_log.warning(
                f"hermes config set failed, using direct YAML patch: {exc}"
            )

        # Fallback: direct YAML write
        config_path = profile_dir / "config.yaml"
        if config_path.exists():
            try:
                cfg = yaml.safe_load(open(config_path))
                cfg.setdefault("providers", {})
                cfg["providers"]["deepseek"] = {
                    "base_url": proxy_base,
                    "api_key": self.master_key,
                }
                # Also update model section
                cfg.setdefault("model", {})
                cfg["model"]["base_url"] = proxy_base
                cfg["model"]["api_key"] = self.master_key
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                self._comp_log.info(
                    f"Profile {self._profile_name} config written directly"
                )
            except Exception as exc:
                self._comp_log.error(
                    f"Failed to write profile config: {exc}"
                )

    def _create_profile_manual(self, profile_dir: Path) -> None:
        """Manually create a profile directory with config from main config."""
        profile_dir.mkdir(parents=True, exist_ok=True)

        main_config = Path.home() / ".hermes" / "config.yaml"
        if main_config.exists():
            import shutil
            shutil.copy(main_config, profile_dir / "config.yaml")

        self._setup_profile_config(profile_dir)
        self._profile_created = True
        self._comp_log.info(
            f"Created profile manually: {self._profile_name}"
        )

    def _delete_profile(self) -> None:
        """Delete the competition profile on cleanup."""
        if not self._profile_created:
            return
        try:
            subprocess.run(
                ["hermes", "profile", "delete", self._profile_name],
                capture_output=True, text=True, timeout=15,
                input="y\n",
            )
            self._comp_log.info(f"Deleted profile: {self._profile_name}")
        except Exception as exc:
            self._comp_log.warning(
                f"Could not delete profile {self._profile_name}: {exc}"
            )

    # ── System environment ─────────────────────────────────────────────

    def _record_system_env(self) -> None:
        """Record nvidia-smi output and uname to agent log."""
        self._comp_log.agent_note("[SYSTEM] Recording host environment...")

        # nvidia-smi
        try:
            r = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                self._comp_log.agent_note(
                    f"[SYSTEM] nvidia-smi:\n{r.stdout}"
                )
            else:
                self._comp_log.agent_note(
                    f"[SYSTEM] nvidia-smi failed (rc={r.returncode})"
                )
        except FileNotFoundError:
            self._comp_log.agent_note("[SYSTEM] nvidia-smi not found (CPU-only?)")
        except Exception as exc:
            self._comp_log.agent_note(f"[SYSTEM] nvidia-smi error: {exc}")

        # uname + host info
        import platform
        import socket
        self._comp_log.agent_note(
            f"[SYSTEM] hostname={socket.gethostname()}, "
            f"OS={platform.system()} {platform.release()}, "
            f"Python={platform.python_version()}, "
            f"CPU_count={os.cpu_count()}"
        )

        # GPU count via nvidia-smi -L
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                gpu_lines = [line for line in r.stdout.strip().split("\n") if line]
                self._comp_log.agent_note(
                    f"[SYSTEM] GPU count: {len(gpu_lines)}, "
                    f"models: {', '.join(gpu_lines)}"
                )
        except Exception:
            pass

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers to stop proxy on exit."""

        def _handler(signum: int, frame: FrameType | None) -> None:
            if self._started:
                self._stop_ingest_server()
                self._stop_proxy()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except ValueError:
                pass  # Not in main thread

    # ── Litellm proxy ──────────────────────────────────────────────────

    def _generate_litellm_config(self) -> None:
        """Generate litellm_config.yaml with generic_api callback.

        Key design choices:
          - No database_url → no prisma dependency
          - generic_api callback → logs to local ingest server
          - batch_size=1, flush_interval=1 → real-time JSONL writes
          - ndjson format → one JSON record per line (competition JSONL)
        """
        # Parse target_url to extract provider + model
        # Default: deepseek/deepseek-chat
        model_params = {
            "model": "deepseek/deepseek-chat",
            "api_key": self.target_api_key,
        }
        if self.target_url and self.target_url != "https://api.deepseek.com/v1":
            model_params["api_base"] = self.target_url

        config = {
            "model_list": [
                {
                    "model_name": "deepseek-chat",
                    "litellm_params": model_params,
                }
            ],
            "litellm_settings": {
                "callbacks": ["jsonl_logger"],
                "callback_settings": {
                    "jsonl_logger": {
                        "callback_type": "generic_api",
                        "endpoint": f"http://127.0.0.1:{self.ingest_port}/ingest",
                        "log_format": "ndjson",
                        "batch_size": 1,
                        "flush_interval": 1,
                    }
                },
            },
            "general_settings": {
                "master_key": self.master_key,
            },
        }

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _start_proxy(self) -> None:
        """Generate litellm config, then start proxy as background subprocess.

        Uses litellm CLI with --config pointing to the generated YAML.
        No database_url set, so no prisma dependency.
        """
        self._generate_litellm_config()

        log_file = self._log_dir / "litellm_proxy.log"
        cmd = [
            sys.executable, "-m", "litellm",
            "--config", str(self._config_path),
            "--port", str(self.proxy_port),
            "--host", "127.0.0.1",
        ]

        env = os.environ.copy()
        # Ensure DEEPSEEK_API_KEY is in env for config interpolation
        if self.target_api_key:
            env["DEEPSEEK_API_KEY"] = self.target_api_key

        with open(log_file, "a") as f_log:
            proc = subprocess.Popen(
                cmd,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )

        self._proxy_pid = proc.pid
        self._comp_log.info(
            f"Starting litellm proxy on port {self.proxy_port} "
            f"(PID={proc.pid})"
        )
        self._wait_for_port(self.proxy_port, timeout=30, label="litellm proxy")

    def _stop_proxy(self) -> None:
        """Stop litellm proxy subprocess."""
        if self._proxy_pid is None:
            self._proxy_alive()  # Refresh
        if self._proxy_pid is None:
            return

        try:
            os.kill(self._proxy_pid, signal.SIGTERM)
            try:
                os.waitpid(self._proxy_pid, os.WNOHANG)
            except ChildProcessError:
                pass
        except ProcessLookupError:
            pass
        finally:
            self._proxy_pid = None
            self._comp_log.info("Litellm proxy stopped")

    def _proxy_alive(self) -> bool:
        """Check if litellm proxy subprocess is still running."""
        if self._proxy_pid is None:
            # Check by port
            try:
                r = subprocess.run(
                    ["lsof", "-ti", f":{self.proxy_port}"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.stdout.strip():
                    self._proxy_pid = int(r.stdout.strip().split("\n")[0])
                    return True
            except Exception:
                pass
            return False
        try:
            os.kill(self._proxy_pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False

    # ── Ingest server ──────────────────────────────────────────────────

    def _start_ingest_server(self) -> None:
        """Start the local JSONL ingest server as background subprocess.

        Listens on localhost:{ingest_port} for generic_api callbacks
        from litellm. Writes extracted competition entries to
        llm-YYYYMMDD.jsonl in the session log directory.
        """
        log_file = self._log_dir / "ingest_server.log"
        cmd = [
            sys.executable, _INGEST_SERVER,
            "--port", str(self.ingest_port),
            "--log-dir", str(self._log_dir),
        ]

        with open(log_file, "a") as f_log:
            proc = subprocess.Popen(
                cmd,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        self._ingest_pid = proc.pid
        self._comp_log.info(
            f"Starting ingest server on port {self.ingest_port} "
            f"(PID={proc.pid})"
        )
        self._wait_for_port(self.ingest_port, timeout=10, label="ingest server")

    def _stop_ingest_server(self) -> None:
        """Stop the ingest server subprocess."""
        if self._ingest_pid is None:
            self._ingest_alive()  # Refresh
        if self._ingest_pid is None:
            return

        try:
            os.kill(self._ingest_pid, signal.SIGTERM)
            try:
                os.waitpid(self._ingest_pid, os.WNOHANG)
            except ChildProcessError:
                pass
        except ProcessLookupError:
            pass
        finally:
            self._ingest_pid = None
            self._comp_log.info("Ingest server stopped")

    def _ingest_alive(self) -> bool:
        """Check if ingest server subprocess is still running."""
        if self._ingest_pid is None:
            try:
                r = subprocess.run(
                    ["lsof", "-ti", f":{self.ingest_port}"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.stdout.strip():
                    self._ingest_pid = int(r.stdout.strip().split("\n")[0])
                    return True
            except Exception:
                pass
            return False
        try:
            os.kill(self._ingest_pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False

    # ── Port readiness ─────────────────────────────────────────────────

    def _wait_for_port(self, port: int, timeout: int = 30, label: str = "") -> None:
        """Wait for a service to start accepting connections on a port.

        Uses health-check HTTP GET on /health endpoint.
        """
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{port}/health"

        while time.time() < deadline:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                pass
            time.sleep(0.5)

        self._comp_log.warning(
            f"{label or f'port {port}'} health check timed out after {timeout}s"
        )

    # ── Merge + validate ───────────────────────────────────────────────

    def _check_ingest_errors(self) -> int:
        """Query ingest server /health for cumulative error count.

        Returns:
            int: error count from ingest server, or -1 if unreachable.
        """
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self.ingest_port}/health"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read().decode())
                return int(body.get("error_count", -1))
        except Exception:
            return -1

    def _merge_and_validate(self) -> dict:
        """Merge fast.log + agent.log + llm-*.jsonl -> task1_logs.log and validate."""
        from .merge import merge_logs, validate_log

        output_path = self._log_dir / f"{self.task}_logs.log"

        # Check ingest error count before merge (callback dropout detection)
        ingest_errors = self._check_ingest_errors()
        if ingest_errors > 0:
            self._comp_log.warning(
                f"Ingest server reports {ingest_errors} errors during session — "
                "some LLM calls may not have been logged"
            )

        # Find LLM log files
        llm_files = sorted(self._log_dir.glob("llm-*.jsonl"))

        merge_logs(
            fast_log=self._log_dir / "fast.log",
            agent_log=self._log_dir / "agent.log",
            llm_files=llm_files,
            output=output_path,
            task=self.task,
            session_start_ts=getattr(self, '_session_start_ts', None),
        )

        # Validate
        errors = validate_log(output_path)
        valid = len(errors) == 0
        if ingest_errors > 0:
            warn_msg = f"Ingest errors: {ingest_errors}"
            errors.append(warn_msg)

        return {
            "output": str(output_path),
            "entries": sum(1 for _ in open(output_path, "r")),
            "valid": valid,
            "errors": errors[:20] if errors else [],
            "span_hours": self._compute_span(output_path),
            "ingest_errors": max(ingest_errors, 0),
        }

    def _compute_span(self, log_path: Path) -> float:
        """Compute time span (hours) between first and last log entries."""
        timestamps = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = json.loads(line)["timestamp"]
                    timestamps.append(datetime.fromisoformat(ts))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        if len(timestamps) < 2:
            return 0.0
        return (timestamps[-1] - timestamps[0]).total_seconds() / 3600
