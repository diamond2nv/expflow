#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow competition CLI command group.

Subcommands:
    init    — Start a competition session (proxy + logging)
    stop    — Stop and merge logs
    status  — Show current session status
    merge   — Merge logs post-hoc
    validate — Validate existing log file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="competition",
    help="Competition logging session management",
    no_args_is_help=True,
)


@app.command("init")
def init_session(
    task: str = typer.Option("task1", help="Task identifier (task1/task2)"),
    tag: str = typer.Option("", help="Experiment tag for log subdirectory"),
    proxy_port: int = typer.Option(
        4000, help="Litellm proxy listen port"
    ),
    ingest_port: int = typer.Option(
        8099, help="JSONL ingest server listen port"
    ),
    target_url: str = typer.Option(
        "",
        help="Upstream API base URL (default: DEEPSEEK_BASE_URL env)",
    ),
    output: Optional[Path] = typer.Option(
        None, help="Log output directory"
    ),
) -> None:
    """Initialize a competition session.

    Starts litellm proxy (no database needed) + local JSONL ingest server,
    records nvidia-smi, and prepares the log directory for three-stream
    logging (fast + agent + llm).

    The proxy uses litellm's generic_api callback to stream every LLM
    request/response as NDJSON to the ingest server, which writes
    competition-compliant llm-YYYYMMDD.jsonl in real time.
    """
    from expflow_pde.competition import CompetitionSession

    session = CompetitionSession(
        task=task,
        tag=tag,
        proxy_port=proxy_port,
        ingest_port=ingest_port,
        target_url=target_url,
        log_dir=str(output) if output else None,
    )

    metadata = session.start()
    if metadata.get("status") == "already_running":
        typer.echo(
            "Session already running. Use 'stop' first."
        )
        raise typer.Exit(1)

    typer.echo("=" * 60)
    typer.echo("Competition session started (litellm proxy + ingest server)")
    typer.echo("=" * 60)
    typer.echo(f"  Session ID:   {metadata['session_id']}")
    typer.echo(f"  Proxy Port:   {metadata['proxy_port']}")
    typer.echo(f"  Ingest Port:  {metadata['ingest_port']}")
    typer.echo(f"  Master Key:   {metadata['master_key']}")
    typer.echo(f"  Profile:      {metadata['hermes_profile']} "
               f"({'created' if metadata.get('profile_created') else 'FAILED'})")
    typer.echo(f"  Log Dir:      {metadata['log_dir']}")
    typer.echo()
    typer.echo("Hermes profile has been created with proxy routing configured.")
    typer.echo("Start Hermes with this profile:")
    typer.echo(f"  hermes -p {metadata['hermes_profile']}")
    typer.echo()
    typer.echo("Or for OpenCode:")
    typer.echo(
        f"  OPENAI_BASE_URL=http://localhost:{metadata['proxy_port']} "
        f"OPENAI_API_KEY={metadata['master_key']} opencode"
    )


@app.command("stop")
def stop_session(
    no_merge: bool = typer.Option(
        False, "--no-merge", help="Skip log merge + validation"
    ),
    delete_profile: bool = typer.Option(
        False, "--delete-profile", help="Delete competition Hermes profile"
    ),
    task: str = typer.Option("task1", help="Task identifier"),
    tag: str = typer.Option("", help="Experiment tag"),
) -> None:
    """Stop the competition session.

    Shuts down litellm proxy and ingest server, merges all log sources,
    and validates the output against competition rules (JSON validity,
    12h span, monotonic timestamps).

    Competition profile is kept by default. Use --delete-profile to
    remove it.
    """
    from expflow_pde.competition import CompetitionSession

    session = CompetitionSession(task=task, tag=tag)
    result = session.stop(merge=not no_merge)

    if delete_profile:
        session._delete_profile()
        typer.echo(f"Deleted profile: {session._profile_name}")

    typer.echo(f"Session stopped: {result['session_id']}")

    if "merge" in result:
        mr = result["merge"]
        typer.echo(f"  Output: {mr['output']}")
        typer.echo(f"  Entries: {mr['entries']}")
        typer.echo(f"  Valid: {'PASS' if mr['valid'] else 'FAIL'}")
        typer.echo(f"  Span: {mr['span_hours']:.1f}h")
        if mr["errors"]:
            typer.echo(f"  Errors: {len(mr['errors'])}")
            for e in mr["errors"][:10]:
                typer.echo(f"    - {e}")


@app.command("status")
def status_command(
    task: str = typer.Option("task1", help="Task identifier"),
    tag: str = typer.Option("", help="Experiment tag"),
    json_output: bool = typer.Option(
        False, "--json", help="JSON output"
    ),
) -> None:
    """Show current competition session status."""
    from expflow_pde.competition import CompetitionSession

    session = CompetitionSession(task=task, tag=tag)
    s = session.status()

    if json_output:
        typer.echo(json.dumps(s, indent=2, default=str))
        return

    typer.echo("=" * 40)
    typer.echo("Competition Session Status")
    typer.echo("=" * 40)
    typer.echo(f"  Started:    {s['started']}")
    typer.echo(f"  Session ID: {s['session_id']}")
    typer.echo(f"  Task:       {s['task']}")
    typer.echo(f"  Tag:        {s['tag']}")
    typer.echo(f"  Proxy Port: {s['proxy_port']}")
    typer.echo(f"  Ingest Port: {s['ingest_port']}")
    typer.echo(f"  Proxy:      {'RUNNING' if s['proxy_alive'] else 'STOPPED'}")
    typer.echo(f"  Ingest:     {'RUNNING' if s['ingest_alive'] else 'STOPPED'}")
    typer.echo(f"  Log Dir:    {s['log_dir']}")

    # Show log file counts
    log_dir = Path(s["log_dir"])
    if log_dir.exists():
        fast_n = sum(1 for _ in open(log_dir / "fast.log", "r")) \
            if (log_dir / "fast.log").exists() else 0
        agent_n = sum(1 for _ in open(log_dir / "agent.log", "r")) \
            if (log_dir / "agent.log").exists() else 0
        llm_files = sorted(log_dir.glob("llm-*.jsonl"))
        llm_n = sum(
            sum(1 for _ in open(f, "r")) for f in llm_files
        )
        typer.echo(
            f"  Log lines:  fast={fast_n}, agent={agent_n}, "
            f"llm={llm_n} ({len(llm_files)} files)"
        )


@app.command("merge")
def merge_command(
    task: str = typer.Option("task1", help="Task identifier"),
    tag: str = typer.Option("", help="Experiment tag"),
    output: Optional[Path] = typer.Option(
        None, help="Output path (default: task1_logs.log in log dir)"
    ),
) -> None:
    """Merge logs post-hoc (without stopping proxy)."""
    from expflow_pde.competition.comp_log import _LOG_DIR
    from expflow_pde.competition.merge import merge_logs, validate_log

    log_dir = _LOG_DIR / f"{task}_{tag}" if tag else _LOG_DIR
    if not log_dir.exists():
        typer.echo(f"Log directory not found: {log_dir}")
        raise typer.Exit(1)

    output_path = output or (log_dir / f"{task}_logs.log")

    llm_files = sorted(log_dir.glob("llm-*.jsonl"))
    n = merge_logs(
        fast_log=log_dir / "fast.log",
        agent_log=log_dir / "agent.log",
        llm_files=llm_files,
        output=output_path,
        task=task,
    )

    if n:
        errors = validate_log(output_path)
        if errors:
            typer.echo(f"Validation FAILED: {len(errors)} errors")
            for e in errors[:10]:
                typer.echo(f"  - {e}")
            raise typer.Exit(1)
        else:
            typer.echo(f"Validation PASSED: {n} entries")


@app.command("validate")
def validate_command(
    path: Path = typer.Argument(
        ..., help="Path to task1_logs.log"
    ),
) -> None:
    """Validate an existing competition log file."""
    from expflow_pde.competition.merge import validate_log

    errors = validate_log(path)
    if errors:
        typer.echo(f"Validation FAILED: {len(errors)} errors")
        for e in errors[:20]:
            typer.echo(f"  - {e}")
        raise typer.Exit(1)
    else:
        typer.echo("Validation PASSED")


if __name__ == "__main__":
    app()
