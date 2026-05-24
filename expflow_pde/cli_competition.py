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
    problem_id: str = typer.Option(
        "", "--problem-id", "-p",
        help="Problem identifier from competition website (e.g. 'task1')"
    ),
    task: str = typer.Option(
        "", "--task",
        help="Deprecated alias for --problem-id (use --problem-id instead)"
    ),
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

    resolved_problem = problem_id or task or "task1"

    session = CompetitionSession(
        task=resolved_problem,
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
    path: Optional[Path] = typer.Argument(
        None, help="Path to problem_logs.log, or submission dir for full check"
    ),
    submission_dir: Optional[Path] = typer.Option(
        None, "--submission", "-s",
        help="Submission directory (runs full check: log + code-log match)"
    ),
    problem_id: str = typer.Option(
        "task1", "--problem-id", "-p",
        help="Problem identifier (used for file naming)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="JSON output"
    ),
) -> None:
    """Validate a competition log file or full submission directory.

    Two modes:
      1. Single log file: check format, monotonicity, span
      2. Submission directory (--submission): full check including
         required files (pred.hdf5, time.csv) and code-log traceability

    Examples:
        # Validate a single log file
        expflow competition validate task1_logs.log

        # Validate a full submission directory
        expflow competition validate --submission ./submission --problem-id task1
    """
    if submission_dir:
        # Full submission check
        from expflow_pde.competition.validate import validate_submission

        result = validate_submission(
            submission_dir=submission_dir,
            problem_id=problem_id,
            verbose=True,
        )

        if json_output:
            import json
            typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            typer.echo("=== Submission Validation ===")
            for check in result.get("checks", []):
                status = "PASS" if check["passed"] else "FAIL"
                typer.echo(f"  [{status}] {check['label']}: {check['detail']}")
                for err in check.get("errors", []):
                    typer.echo(f"      - {err}")
            typer.echo(f"\n  Overall: {'PASS' if result['all_pass'] else 'FAIL'}")

    else:
        # Single log file check
        from expflow_pde.competition.merge import validate_log

        if path is None:
            typer.echo("Provide a log file path or use --submission for full check.")
            raise typer.Exit(1)

        errors = validate_log(path)
        if errors:
            typer.echo(f"Validation FAILED: {len(errors)} errors")
            for e in errors[:20]:
                typer.echo(f"  - {e}")
            raise typer.Exit(1)
        else:
            typer.echo("Validation PASSED")


# ── Mask sub-command ─────────────────────────────────────


@app.command("mask")
def mask_cmd(
    action: str = typer.Argument(
        ..., help="Action: audit (scan only) | apply (create cleansed copy)"
    ),
    wiki_dir: str = typer.Option(
        "~/wiki", "--wiki-dir", help="Wiki directory to scan"
    ),
    skills_dir: str = typer.Option(
        "~/.hermes/skills", "--skills-dir", help="Skills directory to scan"
    ),
    output_dir: str = typer.Option(
        "~/.competition", "--output-dir", help="Output dir for --apply"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="JSON output for programmatic consumption"
    ),
) -> None:
    """Scan or cleanse wiki/skills for competition-specific content.

    Competition rules forbid pre-existing problem-specific code, data paths,
    and equation formulas in the agent's knowledge base. Use this command to
    audit (default) or create a cleansed working copy (--apply).

    Rules mask: PDE equation names, data file paths, scoring formats,
    submission file names, competition-proven hyperparameters, and
    competition strategy references.
    """
    from pathlib import Path

    from expflow_pde.competition.mask.rules import ALL_RULES
    from expflow_pde.competition.mask.scanner import apply_mask, scan_directory

    wiki_path = Path(wiki_dir).expanduser()
    skills_path = Path(skills_dir).expanduser()

    if action == "audit":
        wiki_results = scan_directory(wiki_path, ALL_RULES)
        skills_results = scan_directory(skills_path, ALL_RULES)
        total = len(wiki_results) + len(skills_results)
        total_violations = sum(
            r["violation_count"] for r in wiki_results + skills_results
        )

        if json_output:
            import json
            typer.echo(json.dumps({
                "wiki_files_with_violations": len(wiki_results),
                "skills_files_with_violations": len(skills_results),
                "total_violations": total_violations,
                "wiki_violations": wiki_results,
                "skills_violations": skills_results,
            }, indent=2, ensure_ascii=False))
        else:
            typer.echo(f"=== Mask Audit: {total_violations} violations in {total} files ===")
            for label, results in [("wiki", wiki_results), ("skills", skills_results)]:
                for r in results:
                    if r.get("violations"):
                        typer.echo(f"  [{label}] {r['path']}: {r['violation_count']} violation(s)")
                        for v in r["violations"][:5]:
                            typer.echo(f"    - {v}")

    elif action == "apply":
        out_path = Path(output_dir).expanduser()
        wiki_out = out_path / "wiki"
        skills_out = out_path / "skills"

        wiki_manifest = apply_mask(wiki_path, wiki_out, ALL_RULES)
        skills_manifest = apply_mask(skills_path, skills_out, ALL_RULES)

        if json_output:
            import json
            typer.echo(json.dumps({
                "wiki": wiki_manifest,
                "skills": skills_manifest,
            }, indent=2, ensure_ascii=False))
        else:
            typer.echo(
                f"Mask applied: wiki ({wiki_manifest['files_masked']} files masked), "
                f"skills ({skills_manifest['files_masked']} files masked)"
            )
            typer.echo(f"Output: {out_path}/")
    else:
        typer.echo(f"Unknown action: {action}. Use 'audit' or 'apply'.")
        raise typer.Exit(1)


# ── Bootstrap sub-command ─────────────────────────────────────


@app.command("bootstrap")
def bootstrap_cmd(
    rules_doc: Optional[str] = typer.Option(
        None, "--rules-doc", "-r",
        help="Path to competition rules document (PDF or markdown)"
    ),
    wiki_dir: str = typer.Option(
        "~/wiki", "--wiki-dir", help="Wiki directory to audit"
    ),
    skills_dir: str = typer.Option(
        "~/.hermes/skills", "--skills-dir", help="Skills directory to audit"
    ),
    output_config: str = typer.Option(
        "~/.competition/config.yaml", "--output-config",
        help="Output path for generated config"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="JSON output"
    ),
) -> None:
    """Self-guided competition setup — verify environment and generate config.

    Checks:
    1. Run mask audit on wiki/ and skills/ for competition-specific content
    2. Extract competition parameters from rules document (if provided)
    3. Generate competition config.yaml with discovered parameters
    4. Report clean/dirty status

    After a clean bootstrap, run:
      expflow competition mask apply
    to create a cleansed working copy.
    """
    from expflow_pde.competition.bootstrap import bootstrap_session

    result = bootstrap_session(
        rules_doc_path=rules_doc,
        wiki_dir=wiki_dir,
        skills_dir=skills_dir,
        output_config_path=output_config,
    )

    if json_output:
        import json
        typer.echo(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        typer.echo("=== Competition Bootstrap Report ===")
        typer.echo(f"  Environment clean: {'YES' if result.get('clean') else 'VIOLATIONS FOUND'}")
        if result.get("violation_count"):
            typer.echo(f"  Violations: {result['violation_count']} "
                       f"(wiki: {result.get('wiki_issues', 0)}, "
                       f"skills: {result.get('skills_issues', 0)})")
        typer.echo(f"  Config generated: {result.get('config_path', 'N/A')}")
        if result.get("extracted_params"):
            typer.echo(f"  Extracted params: {result['extracted_params']}")


# ── Collect sub-command ─────────────────────────────────────


@app.command("collect")
def collect_cmd(
    pipeline_id: str = typer.Argument(
        ..., help="clearml pipeline controller task ID"
    ),
    output_dir: str = typer.Option(
        "~/.competition/artifacts", "--output-dir", "-o",
        help="Output directory for downloaded artifacts"
    ),
    step: Optional[list[str]] = typer.Option(
        None, "--step", help="Only collect from specific step names (can repeat)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="JSON output"
    ),
) -> None:
    """Download artifacts from a completed clearml pipeline.

    After a training+eval pipeline completes on a remote clearml agent,
    the eval step should have uploaded result files (pred.hdf5, time.csv)
    as clearml artifacts. This command downloads them to the local machine.

    Example:
        expflow competition collect PIPELINE_ID --step eval --output-dir ./submission
    """
    from expflow_pde.competition.collect import collect_pipeline_artifacts

    manifest = collect_pipeline_artifacts(
        pipeline_id=pipeline_id,
        output_dir=output_dir,
        step_names=step,
    )

    if json_output:
        import json
        typer.echo(json.dumps(manifest, indent=2, default=str, ensure_ascii=False))
    else:
        typer.echo(f"=== Artifact Collection: {pipeline_id} ===")
        if "error" in manifest:
            typer.echo(f"  ERROR: {manifest['error']}")
            raise typer.Exit(1)
        typer.echo(f"  Total artifacts: {manifest['total_artifacts']}")
        for step in manifest.get("steps", []):
            typer.echo(f"  Step: {step['step_name']} ({step['task_id'][:12]}...)")
            for art in step.get("artifacts", []):
                if "error" in art:
                    typer.echo(f"    ❌ {art['name']}: {art['error']}")
                else:
                    typer.echo(f"    ✅ {art['name']} -> {art['local_path']}")


if __name__ == "__main__":
    app()
