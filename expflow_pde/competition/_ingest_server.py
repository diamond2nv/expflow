#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local JSONL ingest server for litellm generic_api callback.

Receives StandardLoggingPayload via POST /log from litellm proxy,
extracts competition-relevant fields (model, response, tool_calls,
timing), and writes to llm-{date}.jsonl in the session log directory.

Runs as a subprocess managed by CompetitionSession. No external
dependencies beyond FastAPI + uvicorn + stdlib.

Architecture:
    litellm proxy (configurable port) --generic_api--> ingest server (configurable port)
                                                   |
                                                   v
                                          llm-YYYYMMDD.jsonl

Health endpoint exposes cumulative error count so CompetitionSession
can detect callback dropout during session stop.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

# Configured via command-line args at startup
LOG_DIR: Path | None = None
_ERROR_COUNT: int = 0


def _ensure_llm_file() -> Path:
    """Create or return today's LLM JSONL file path.

    Returns:
        Path to llm-YYYYMMDD.jsonl in LOG_DIR.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = LOG_DIR / f"llm-{date_str}.jsonl"  # type: ignore[operator]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _extract_entry(payload: dict) -> dict:
    """Extract competition JSONL fields from StandardLoggingPayload.

    The StandardLoggingPayload spec (docs.litellm.ai/docs/proxy/logging_spec)
    defines: id, model, messages, response, startTime, endTime, status,
    metadata, usage, response_cost, model_parameters, hidden_params, etc.

    We extract:
      - timestamp: ISO 8601 UTC
      - elapsed_seconds: endTime - startTime
      - model: the model name
      - response: assistant message content
      - tool_calls: JSON-serialized tool_calls array (or null)
      - session_id: from metadata.requester_metadata.session_id (or "")
      - status: "success" or "failure"

    Args:
        payload: StandardLoggingPayload dict from litellm.

    Returns:
        Dict ready for JSONL serialization.
    """
    start = payload.get("startTime", 0)
    end = payload.get("endTime", 0)
    elapsed = round(float(end) - float(start), 3) if start and end else 0.0

    # Extract assistant message from response object
    response_obj = payload.get("response", {})
    response_text = ""
    tool_calls = None

    if isinstance(response_obj, dict):
        choices = response_obj.get("choices", [])
        if choices and isinstance(choices, list) and isinstance(choices[0], dict):
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                response_text = msg.get("content") or ""
                tcs = msg.get("tool_calls") or []
                if tcs:
                    tool_calls = json.dumps(tcs, ensure_ascii=False)

    # Extract session_id from metadata
    metadata = payload.get("metadata", {})
    session_id = ""
    if isinstance(metadata, dict):
        rm = metadata.get("requester_metadata", {})
        if isinstance(rm, dict):
            session_id = rm.get("session_id", "")

    ts = datetime.now(timezone.utc).isoformat()

    # Cap tool_calls at same limit as response to prevent oversized lines
    max_field_len = 2000
    return {
        "timestamp": ts,
        "elapsed_seconds": elapsed,
        "model": payload.get("model", ""),
        "response": response_text[:max_field_len] if response_text else "",
        "tool_calls": (tool_calls[:max_field_len] + "...(truncated)") if tool_calls and len(tool_calls) > max_field_len else tool_calls,
        "session_id": session_id,
        "status": payload.get("status", ""),
    }


@app.post("/ingest")
async def ingest(request: Request):
    """Receive StandardLoggingPayload from litellm generic_api callback.

    The litellm proxy POSTs each logged event as an individual NDJSON
    record (batch_size=1, flush_interval=1). We deserialize, extract
    competition fields, and append to the current llm-*.jsonl file.
    Returns cumulative error count so upstream can detect persistent failures.
    """
    global _ERROR_COUNT
    try:
        payload = await request.json()
    except Exception:
        _ERROR_COUNT += 1
        return {"status": "error", "reason": "invalid json", "error_count": _ERROR_COUNT}

    entry = _extract_entry(payload)

    path = _ensure_llm_file()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        _ERROR_COUNT += 1
        return {"status": "error", "reason": "write_failed", "error_count": _ERROR_COUNT}

    return {"status": "ok", "file": str(path), "error_count": _ERROR_COUNT}


@app.get("/health")
def health():
    """Health check endpoint — returns cumulative error count for dropout detection."""
    return {"status": "ok", "error_count": _ERROR_COUNT, "log_dir": str(LOG_DIR) if LOG_DIR else "unset"}


def main():
    """CLI entry point: start the ingest server."""
    global LOG_DIR

    parser = argparse.ArgumentParser(
        description="LLM JSONL ingest server for competition logging"
    )
    parser.add_argument(
        "--port", type=int, default=8099, help="Listen port (default: 8099)"
    )
    parser.add_argument(
        "--log-dir", required=True, help="Session log directory for llm-*.jsonl output"
    )
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        app, host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
