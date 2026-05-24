"""Generic code-log consistency validator for competition submissions.

Verifies that every file in a code/ directory has a matching write()
entry in the competition log. This is the competition website's
automated check — simulate it locally before uploading.

Usage:
    from expflow_pde.competition.validate import validate_code_log, validate_log_format

    errors = validate_code_log("submission/code", "submission/task1_logs.log")
    format_errors = validate_log_format("submission/task1_logs.log")
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def extract_identifiers(code_text: str) -> list[str]:
    """Extract class/function names from source code for grep matching."""
    identifiers: list[str] = []
    for m in re.finditer(r"class\s+(\w+)", code_text):
        identifiers.append(m.group(1))
    for m in re.finditer(r"def\s+(\w+)", code_text):
        identifiers.append(m.group(1))
    # Dedupe while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for name in identifiers:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def find_write_entry(
    log_entries: list[dict[str, Any]],
    flat_name: str,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    """Find the write() entry in log that has filePath matching flat_name.

    Args:
        log_entries: Parsed JSONL entries from the competition log.
        flat_name: Flat filename (e.g. 'model.py').

    Returns:
        (entry_index, entry_dict, tool_calls_string) or (None, None, None).
    """
    for i, entry in enumerate(log_entries):
        tc = entry.get("tool_calls", "") or ""
        if not tc or "write" not in tc:
            continue
        # Match: "filePath": "flat_name"
        pattern = rf'"filePath"\s*:\s*"{re.escape(flat_name)}"'
        if re.search(pattern, tc):
            return i, entry, tc
    return None, None, None


def validate_code_log(
    code_dir: str | Path,
    log_path: str | Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Validate that every code file has a matching write() entry in the log.

    Args:
        code_dir: Directory containing code files.
        log_path: Path to competition log JSONL file.
        verbose: Print detailed per-file PASS/FAIL.

    Returns:
        Dict with: passed, failed, total, errors (list).
    """
    code_dir = Path(code_dir)
    log_path = Path(log_path)

    errors: list[str] = []

    if not code_dir.exists():
        return {"passed": 0, "failed": 0, "total": 0, "errors": [f"code/ directory not found: {code_dir}"]}
    if not log_path.exists():
        return {"passed": 0, "failed": 0, "total": 0, "errors": [f"Log file not found: {log_path}"]}

    code_files = sorted(
        f for f in os.listdir(code_dir) if (code_dir / f).is_file()
    )

    log_entries: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    log_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if verbose:
        print(f"code/ files: {len(code_files)}")
        print(f"log entries: {len(log_entries)}")

    passed = 0
    failed = 0

    for flat_name in code_files:
        code_path = code_dir / flat_name
        code_text = code_path.read_text(encoding="utf-8")

        entry_idx, entry, tc = find_write_entry(log_entries, flat_name)

        if entry_idx is None:
            msg = f"FAIL: code/{flat_name} — no write() entry with matching filePath in log"
            if verbose:
                print(f"  {msg}")
            errors.append(msg)
            failed += 1
            continue

        if not tc:
            msg = f"FAIL: code/{flat_name} — write() at log line {entry_idx} has no tool_calls content"
            if verbose:
                print(f"  {msg}")
            errors.append(msg)
            failed += 1
            continue

        # Try to extract content from write() call
        content_match = re.search(r'"content"\s*:\s*(".*?")(?:\s*[,}])', tc, re.DOTALL)
        logged_content = None

        if content_match:
            try:
                logged_content = json.loads(content_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        if logged_content is not None and logged_content.strip() == code_text.strip():
            if verbose:
                print(f"  PASS: code/{flat_name} — full content match (log line {entry_idx})")
            passed += 1
            continue

        # Fallback: check identifiers
        idents = extract_identifiers(code_text)
        found = sum(1 for name in idents[:10] if name in tc)
        missing = [name for name in idents[:10] if name not in tc]

        if found >= 2:
            if verbose:
                print(f"  PASS: code/{flat_name} — {found}/10 identifiers match ({entry_idx})")
            passed += 1
        else:
            code_lines = len(code_text.splitlines())
            log_char_len = len(logged_content) if logged_content else 0
            msg = (
                f"FAIL: code/{flat_name} — only {found} identifiers found in log "
                f"(need >=2). Content mismatch (code={code_lines}L, "
                f"log={log_char_len}chars). Missing: {missing[:3]}"
            )
            if verbose:
                print(f"  {msg}")
            errors.append(msg)
            failed += 1

    result: dict[str, Any] = {
        "passed": passed,
        "failed": failed,
        "total": len(code_files),
        "errors": errors,
    }
    return result


def validate_log_format(
    log_path: str | Path,
    max_span_hours: float | None = None,
) -> list[str]:
    """Validate competition log format.

    Checks each entry has required fields and monotonic timestamps.
    Delegates to merge.validate_log for the heavy lifting.

    Args:
        log_path: Path to competition log JSONL file.
        max_span_hours: Max allowed span (None = from config / default).

    Returns:
        List of error strings (empty = passed).
    """
    from .merge import validate_log as _validate_log

    return _validate_log(Path(log_path), max_span_hours=max_span_hours)


def validate_submission(
    submission_dir: str | Path,
    problem_id: str = "task1",  # Task name for file naming (task1/task2/task3)
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full submission validation: log format + code-log match.

    Checks:
    1. Required files exist (pred.hdf5, time.csv, logs.log, code/)
    2. Log format is valid JSONL with proper fields
    3. Every code/ file has a matching write() in the log

    Args:
        submission_dir: Submission directory (contains pred.hdf5, time.csv, etc.).
        problem_id: Problem identifier for file naming.
        verbose: Print per-file details.

    Returns:
        Dict with all_pass, checks (list of per-check results).
    """
    submission_dir = Path(submission_dir)
    log_file = submission_dir / f"{problem_id}_logs.log"
    code_dir = submission_dir / "code"
    pred_file = submission_dir / f"{problem_id}_pred.hdf5"
    time_file = submission_dir / f"{problem_id}_time.csv"

    checks: list[dict[str, Any]] = []

    # Check 1: Pred file exists
    checks.append({
        "name": "pred_file",
        "label": f"{problem_id}_pred.hdf5",
        "passed": pred_file.exists(),
        "detail": str(pred_file) if pred_file.exists() else "not found",
    })

    # Check 2: Time file exists
    checks.append({
        "name": "time_file",
        "label": f"{problem_id}_time.csv",
        "passed": time_file.exists(),
        "detail": str(time_file) if time_file.exists() else "not found",
    })

    # Check 3: Log format valid
    log_errors = validate_log_format(log_file) if log_file.exists() else ["log file not found"]
    checks.append({
        "name": "log_format",
        "label": "log JSONL format",
        "passed": len(log_errors) == 0,
        "detail": f"{len(log_errors)} errors" if log_errors else "valid",
        "errors": log_errors[:5] if log_errors else [],
    })

    # Check 4: Code-log consistency
    code_result = validate_code_log(code_dir, log_file, verbose=verbose) if log_file.exists() else {
        "passed": 0, "failed": 0, "total": 0, "errors": ["log not available"],
    }
    checks.append({
        "name": "code_log_match",
        "label": "code → log traceability",
        "passed": code_result["failed"] == 0,
        "detail": f"{code_result['passed']}/{code_result['total']} files matched",
        "errors": code_result["errors"][:10],
    })

    all_pass = all(c["passed"] for c in checks)

    return {
        "all_pass": all_pass,
        "checks": checks,
        "submission_dir": str(submission_dir),
    }
