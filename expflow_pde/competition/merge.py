#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge fast.log + agent.log + llm-*.jsonl → competition-compliant JSONL.

Produces task1_logs.log (or task2_logs.log) with each line:
    {"timestamp": "...", "elapsed_seconds": N, "response": "...",
     "tool_calls": "..."}

Also provides validate_log() that checks monotonicity, 12h span, and JSON
validity per official competition rules.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Regex for comp_log formatted lines:
#   UTC - LEVEL - [module:line] - [operator:tag] - [PID:NNN] - message
_LOG_LINE_RE = re.compile(
    r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})Z?\s*-\s*\w+\s*-\s*'
    r'\[.*?\]\s*-\s*\[(.*?):(.*?)\]\s*-\s*\[PID:(\d+)\]\s*-\s*(.*)'
)

# Default limits — override via config.yaml:
#   competition.log.max_span_hours: 12.0   (0 = disabled)
#   competition.log.max_elapsed_seconds: 60.0
_MAX_SPAN_HOURS_DEFAULT = 12.0
_MAX_ELAPSED_DEFAULT = 60.0


def _resolve_max_elapsed() -> float:
    """Read max_elapsed_seconds from config.yaml, fallback to default."""
    try:
        from expflow_pde.config import get as cfg_get
        val = cfg_get("competition.log.max_elapsed_seconds")
        if val is not None and float(val) > 0:
            return float(val)
    except Exception:
        pass
    return _MAX_ELAPSED_DEFAULT


def _resolve_max_span_hours() -> float:
    """Read max_span_hours from config.yaml, fallback to default."""
    try:
        from expflow_pde.config import get as cfg_get
        val = cfg_get("competition.log.max_span_hours")
        if val is not None and float(val) > 0:
            return float(val)
    except Exception:
        pass
    return _MAX_SPAN_HOURS_DEFAULT


def _ts(ts_str: str) -> str:
    """Normalize to ISO 8601 +00:00."""
    if ts_str.endswith("Z"):
        return ts_str.replace("Z", "+00:00")
    if "+" not in ts_str[10:]:
        return ts_str + "+00:00"
    return ts_str


def _elapsed(entries: list[dict], idx: int, session_start_ts: str | None = None) -> float:
    """Compute elapsed_seconds between consecutive entries.

    For the first entry (idx=0), uses the gap from session_start_ts to the
    first entry's timestamp. Falls back to 3.0 if no session_start_ts is given.
    """
    if idx == 0:
        if session_start_ts and entries:
            try:
                t0 = datetime.fromisoformat(session_start_ts)
                t1 = datetime.fromisoformat(entries[0]["timestamp"])
                gap = (t1 - t0).total_seconds()
                if gap < 0:
                    gap = 3.0  # fallback on clock skew
                return min(round(gap, 3), _resolve_max_elapsed())
            except (KeyError, ValueError, TypeError):
                pass
        return 3.0
    try:
        t0 = datetime.fromisoformat(entries[idx - 1]["timestamp"])
        t1 = datetime.fromisoformat(entries[idx]["timestamp"])
        gap = (t1 - t0).total_seconds()
        return min(round(gap, 3), _resolve_max_elapsed())
    except (KeyError, ValueError):
        return 3.0


def _parse_fast_log(path: Path) -> list[dict]:
    """Parse fast.log → list of {ts, resp, tc} entries."""
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LOG_LINE_RE.match(line)
            if not m:
                continue
            ts_str, operator, tag, pid, msg = m.groups()
            ts = _ts(ts_str)

            # Parse epoch lines: "Epoch N/M | loss=X | ...GPU=X/YMB"
            ep_m = re.search(
                r'Epoch\s+(\d+)/(\d+)\s*\|\s*loss=([\d.e+\-]+)', msg
            )
            if ep_m:
                ep_num = ep_m.group(1)
                loss_val = ep_m.group(3)
                gpu_m = re.search(r'GPU=([\d.]+)/([\d.]+)MB', msg)
                if gpu_m:
                    gpu_str = f"GPU={gpu_m.group(1)}/{gpu_m.group(2)}MB"
                else:
                    gpu_str = "GPU=N/A"
                entries.append({
                    "ts": ts,
                    "resp": f"Epoch {ep_num}: loss={loss_val}, {gpu_str}.",
                    "tc": None,
                    "source": "fast",
                })
                continue

            # Parse validation lines: "[VAL] Epoch N: MSE=X, Rel-MSE=Y, Seg=Z"
            val_m = re.search(
                r'\[VAL\].*Epoch\s+(\d+):\s*MSE=([\d.]+)\s*[,;]\s*'
                r'Rel-MSE=([\d.]+)\s*[,;]\s*Seg=([\d.]+)', msg
            )
            if val_m:
                ep, mse, rel, seg = val_m.groups()
                entries.append({
                    "ts": ts,
                    "resp": (
                        f"Epoch {ep} VAL: MSE={mse}, "
                        f"Rel-MSE={rel}, Seg={seg}/100."
                    ),
                    "tc": None,
                    "source": "fast",
                })
                continue

            # Parse training completion
            if "Training completed" in msg:
                tm = re.search(
                    r'completed:\s*([\d.]+)s\s*\(([\d.]+)min\)', msg
                )
                if tm:
                    entries.append({
                        "ts": ts,
                        "resp": (
                            f"Training completed: {tm.group(1)}s "
                            f"({tm.group(2)} min)."
                        ),
                        "tc": None,
                        "source": "fast",
                    })
                else:
                    entries.append({
                        "ts": ts,
                        "resp": "Training completed.",
                        "tc": None,
                        "source": "fast",
                    })
                continue

            # Parse final segment scores
            seg_m = re.search(
                r'Seg1=([\d.]+)\s+Seg2=([\d.]+)\s+Seg3=([\d.]+)\s+'
                r'Total=([\d.]+)', msg
            )
            if seg_m:
                s1, s2, s3, tot = seg_m.groups()
                entries.append({
                    "ts": ts,
                    "resp": f"Seg1={s1} Seg2={s2} Seg3={s3} Total={tot}/100.",
                    "tc": None,
                    "source": "fast",
                })
                continue

    return entries


def _parse_agent_log(path: Path) -> list[dict]:
    """Parse agent.log → list of {ts, resp, tc} entries."""
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LOG_LINE_RE.match(line)
            if not m:
                continue
            ts_str, operator, tag, pid, msg = m.groups()
            ts = _ts(ts_str)

            # Check for [TC] marker (tool call indicator)
            tc_match = re.match(r'\[TC\]\s+(.*)', msg)
            if tc_match:
                entries.append({
                    "ts": ts, "tc": tc_match.group(1),
                    "resp": None, "source": "agent",
                })
            else:
                is_system = msg.startswith("[SYSTEM]")
                entries.append({
                    "ts": ts, "tc": None,
                    "resp": msg[:500], "source": "agent",
                    "is_system": is_system,
                })
    return entries


def _parse_llm_jsonl(paths: list[Path]) -> list[dict]:
    """Parse litellm callback JSONL → list of {ts, resp, tc} entries."""
    entries: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _ts(d.get("timestamp", ""))
                resp = d.get("response")
                tc = d.get("tool_calls")
                entries.append({
                    "ts": ts,
                    "resp": resp,
                    "tc": tc,
                    "source": "llm",
                })
    return entries


def merge_logs(
    fast_log: Path,
    agent_log: Path,
    llm_files: list[Path],
    output: Path,
    task: str = "task1",
    session_start_ts: str | None = None,
) -> int:
    """Merge all log sources into a competition-compliant JSONL file.

    Args:
        fast_log: Path to fast.log (training metrics).
        agent_log: Path to agent.log (agent reasoning).
        llm_files: List of paths to llm-*.jsonl (litellm callback output).
        output: Output path for task1_logs.log.
        task: Task identifier.
        session_start_ts: ISO timestamp of session start (for computing
            first-entry elapsed_seconds from real gap instead of hardcoded 3.0).

    Returns:
        Number of entries written.
    """
    # Parse all sources
    fast = _parse_fast_log(fast_log)
    agent = _parse_agent_log(agent_log)
    llm = _parse_llm_jsonl(llm_files)

    # Combine all entries
    all_entries = fast + agent + llm

    if not all_entries:
        print("WARNING: No log entries found from any source")
        return 0

    # Sort chronologically
    all_entries.sort(key=lambda e: e["ts"])

    # Deduplicate: only merge when same source, same timestamp, and similar content
    # First-30-chars prefix check caused false dedup across sources.
    merged = []
    for e in all_entries:
        if merged:
            prev = merged[-1]
            # Only dedup entries from the SAME source
            if e.get("source") == prev.get("source"):
                prev_ts = prev["ts"].rstrip("0").rstrip(".")
                cur_ts = e["ts"].rstrip("0").rstrip(".")
                prev_full = prev.get("resp") or ""
                cur_full = e.get("resp") or ""
                # Same microsecond-trimmed timestamp AND exact same full response
                if cur_ts == prev_ts and cur_full == prev_full:
                    # Keep whichever has more info (non-empty preferred)
                    if len(cur_full) > len(prev_full):
                        merged[-1] = e
                    continue
        merged.append(e)

    # Build competition format
    comp = []
    for i, e in enumerate(merged):
        d: dict[str, Any] = {
            "timestamp": e["ts"],
            "elapsed_seconds": _elapsed(merged, i, session_start_ts=session_start_ts),
        }
        if e.get("resp"):
            d["response"] = str(e["resp"])
        if e.get("tc"):
            d["tool_calls"] = str(e["tc"])
        # Tag system records so downstream parsers can identify them
        if e.get("is_system"):
            d["metadata"] = {"is_system": True, "source": e.get("source", "")}
        comp.append(d)

    # Write
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for e in comp:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Summary
    fast_n = sum(1 for e in merged if e["source"] == "fast")
    agent_n = sum(1 for e in merged if e["source"] == "agent")
    llm_n = sum(1 for e in merged if e["source"] == "llm")
    print(
        f"Merged: fast({fast_n}) + agent({agent_n}) + llm({llm_n}) "
        f"→ {len(comp)} entries → {output}"
    )

    return len(comp)


def validate_log(path: Path, max_span_hours: float | None = None) -> list[str]:
    """Validate a competition log file.

    Checks:
      1. Every line is valid JSON.
      2. Every line has 'timestamp' and 'elapsed_seconds'.
      3. Timestamps are monotonically increasing.
      4. Total span does not exceed max_span_hours (12h default).
      5. Warns if entry count is suspiciously low (< 10 entries).
      6. Reports gaps > 10 minutes between consecutive entries (not an error, but warns).

    Args:
        path: Path to the task1_logs.log file.
        max_span_hours: Maximum allowed time span (default 12).

    Returns:
        List of error strings (empty = passed).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return [f"Log file not found: {path}"]

    # Resolve max_span_hours from config if not passed explicitly
    if max_span_hours is None:
        max_span_hours = _resolve_max_span_hours()

    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return ["Log file is empty"]

    prev_ts: str | None = None
    first_ts: str | None = None
    last_ts: str | None = None
    gap_count = 0

    for i, line in enumerate(lines, 1):
        # 1. Valid JSON
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {i}: invalid JSON — {exc}")
            continue

        # 2. Required fields
        for k in ("timestamp", "elapsed_seconds"):
            if k not in d:
                errors.append(f"Line {i}: missing '{k}'")

        # 3. Monotonic timestamps
        cur = d.get("timestamp", "")
        if first_ts is None:
            first_ts = cur
        last_ts = cur

        if prev_ts and cur < prev_ts:
            errors.append(f"Line {i}: timestamp went backward ({cur} < {prev_ts})")

        # 6. Gap detection (>10 min between consecutive entries)
        if prev_ts and cur > prev_ts:
            try:
                t_prev = datetime.fromisoformat(prev_ts)
                t_cur = datetime.fromisoformat(cur)
                gap_min = (t_cur - t_prev).total_seconds() / 60.0
                if gap_min > 10.0:
                    gap_count += 1
                    warnings.append(
                        f"Line {i}: {gap_min:.1f}min gap between entries "
                        f"({prev_ts} -> {cur})"
                    )
            except (ValueError, TypeError):
                pass

        prev_ts = cur

    # 4. Span check
    if first_ts and last_ts:
        try:
            t0 = datetime.fromisoformat(first_ts)
            t1 = datetime.fromisoformat(last_ts)
            span_h = (t1 - t0).total_seconds() / 3600
            if span_h > max_span_hours:
                errors.append(
                    f"Log span {span_h:.1f}h exceeds {max_span_hours}h limit"
                )
        except ValueError as exc:
            errors.append(f"Timestamp parse error: {exc}")

    # 5. Entry count sanity
    if len(lines) < 10:
        warnings.append(
            f"Suspiciously few entries: {len(lines)} (expected >= 10 for a real session)"
        )

    # Print result
    total_elapsed = 0.0
    span_h = 0.0
    try:
        if first_ts and last_ts:
            span_h = (
                (datetime.fromisoformat(last_ts) - datetime.fromisoformat(first_ts)).total_seconds() / 3600  # type: ignore[arg-type]
            )
        total_elapsed = sum(
            json.loads(line)["elapsed_seconds"] for line in lines
        )
    except (ValueError, TypeError, KeyError):
        pass

    if not errors:
        status = "PASS"
        extra = ""
        if gap_count > 0:
            extra = f", {gap_count} gaps >10min"
        if warnings:
            extra += "; " + "; ".join(warnings)
        print(
            f"Validate {status}: {len(lines)} entries, "
            f"span={span_h:.1f}h, "
            f"total_elapsed={total_elapsed:.1f}s"
            f"{extra}"
        )
    else:
        print(
            f"Validate FAIL ({len(errors)} errors): {len(lines)} entries, "
            f"span={span_h:.1f}h"
        )

    # Return errors only; warnings are printed but not fatal
    return errors
