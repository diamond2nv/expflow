#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow analyze CLI — PDE competition task intelligence and strategic advising."""

from typing import Optional

import json
import typer

analyze_app = typer.Typer(
    name="analyze",
    help="PDE competition task intelligence, equation analysis, and strategic advising",
    no_args_is_help=True,
)


def get_analyze_app() -> typer.Typer:
    return analyze_app


# ── Shared helpers ──


def _difficulty_icon(d: str) -> str:
    icons = {"easy": "🟢", "medium": "🟡", "hard": "🔴", "very_hard": "🔥"}
    return icons.get(d, "⚪")


def _status_icon(s: str) -> str:
    icons = {"in_progress": "🔴", "not_started": "⚪", "completed": "🟢"}
    return icons.get(s, "⚪")


# ── Commands ──


@analyze_app.command("task")
def analyze_task_cmd(
    task_id: str = typer.Argument(..., help="Task ID: task1, task2, or task3"),
) -> None:
    """Show detailed analysis for a competition task."""
    from expflow_pde.analyze import analyze_task

    result = analyze_task(task_id)
    if result is None:
        print(f"Unknown task: {task_id}")
        raise typer.Exit(code=1)

    diff_icon = _difficulty_icon(result.get("difficulty", ""))
    status_icon = _status_icon(result.get("status", ""))

    print(f"  {result['label']}")
    print(f"  {'─' * 50}")
    print(f"  难度:   {diff_icon} {result['difficulty'].upper()}")
    print(f"  状态:   {status_icon} {result['status'].replace('_', ' ').title()}")
    print(f"  满分:   {result['max_score']} pts")
    if result.get("current_total"):
        print(f"  当前:   {result['current_total']}/{result['max_score']} pts")
    if result.get("estimated_ceiling"):
        print(f"  预估上限: {result['estimated_ceiling']} pts")
        print(f"  剩余空间: {result['remaining_headroom']} pts")

    # Score estimate
    se = result.get("score_estimate", {})
    if se:
        opt = se.get("optimistic", "?")
        exp = se.get("expected", "?")
        con = se.get("conservative", "?")
        conf = se.get("confidence", "?")
        print(f"  预估得分: 乐观={opt}  期望={exp}  保守={con}  (置信度:{conf})")

    # Equations
    eqs = result.get("equations", [])
    if eqs:
        print("\n  PDE 方程:")
        for e in eqs:
            print(f"    {e['full_name']} ({e['dim']}D)")
            print(f"    LaTeX: {e['latex']}")
            if e.get("samples"):
                print(f"      {e['samples']} samples")

    # Score composition
    sc = result.get("score_composition", {})
    if sc:
        print("\n  评分构成:")
        for k, v in sc.items():
            if isinstance(v, dict):
                curr = v.get("current_estimate", "")
                max_v = v.get("max", "")
                note = v.get("note", "")
                if curr:
                    print(f"    {k}: {curr}/{max_v}")
                else:
                    print(f"    {k}: max={max_v}")
                if note:
                    print(f"      {note}")
            elif isinstance(v, str):
                print(f"    {k}: {v}")

    # Bottlenecks
    bottlenecks = result.get("key_bottlenecks", [])
    if bottlenecks:
        print("\n  关键瓶颈:")
        for b in bottlenecks:
            print(f"     ❌ {b}")

    # Proven strategies
    strategies = result.get("proven_strategies", [])
    if strategies:
        print("\n  已验证策略:")
        for s in strategies:
            print(f"     ✅ {s}")

    # Next steps
    next_steps = result.get("next_steps", [])
    if next_steps:
        print("\n  下一步:")
        for n in next_steps:
            print(f"     ▶  {n}")


@analyze_app.command("equations")
def equations_cmd(
    equation: Optional[str] = typer.Argument(
        None, help="Optional: show details for a specific equation"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Filter: task1, task2, task3, or 'competition'"
    ),
) -> None:
    """List all PDE equations or show details for one."""
    from expflow_pde.analyze import get_equation_analysis, list_all_equations_summary

    if equation:
        # Show single equation detail
        result = get_equation_analysis(equation)
        if result is None:
            print(f"Unknown equation: {equation}")
            raise typer.Exit(code=1)

        print(f"  {result['full_name']}")
        print(f"  {'─' * 50}")
        print(f"  LaTeX: {result['latex']}")
        print(
            f"  Dim:    {result['dim']}D  {'Time-dependent' if result['time_dependent'] else 'Steady-state'}"
        )
        print(f"  Viscosity: {result['viscosity_params']}")
        print(f"  Data:   {result['data_samples']} samples")
        print(f"  Solver: {result['solver']}")
        print(
            f"  Tasks:  {', '.join(result['assigned_tasks']) if result['assigned_tasks'] else 'None'}"
        )
        print(f"  Description: {result['description']}")

        comp_info = result.get("competition_info", {})
        if comp_info:
            print("\n  竞赛信息:")
            for k, v in comp_info.items():
                if isinstance(v, dict):
                    continue
                print(f"    {k}: {v}")

        metrics = result.get("metrics", [])
        if metrics:
            print(f"\n  相关指标: {', '.join(metrics)}")
        return

    # List all equations
    all_eqs = list_all_equations_summary()

    # Filter by task
    if task:
        if task == "competition":
            all_eqs = [e for e in all_eqs if e["competition_task"] != "-"]
        else:
            all_eqs = [e for e in all_eqs if e["competition_task"] == task]

    if not all_eqs:
        print("No equations found.")
        return

    print(f"{'Equation':<30} {'Dim':<6} {'Task':<10} {'Difficulty':<14} {'Time':<6}")
    print(f"{'─' * 66}")
    for e in all_eqs:
        diff_icon = _difficulty_icon(e["difficulty"])
        time_flag = "⏱" if e["time_dependent"] else "◯"
        task_label = e["competition_task"] if e["competition_task"] != "-" else "benchmark"
        print(
            f"{e['name']:<30} {e['dim']:<6} {task_label:<10} {diff_icon} {e['difficulty']:<12} {time_flag:<6}"
        )


@analyze_app.command("status")
def status_cmd() -> None:
    """Show overall competition status across all tasks."""
    from expflow_pde.analyze import list_task_summaries

    summaries = list_task_summaries()
    if not summaries:
        print("No task data.")
        return

    print(f"  {'Task':<8} {'Score':<18} {'Difficulty':<14} {'Status':<14} Priority")
    print(f"  {'─' * 68}")
    for s in summaries:
        diff_icon = _difficulty_icon(s["difficulty"])
        status_icon = _status_icon(s["status"])
        score_str = (
            f"{s['current_total']}/{s['max_score']}"
            if s.get("current_total") is not None
            else f"-/{s['max_score']}"
        )
        print(
            f"  {s['task_id']:<8} {score_str:<18} {diff_icon} {s['difficulty']:<12} {status_icon} {s['status'].replace('_', ' ').title():<12} {s['priority']:<8}"
        )

    # Total
    total_max = sum(s["max_score"] for s in summaries)
    total_current = sum(s.get("current_total") or 0 for s in summaries)
    print(f"\n  总分: {total_current}/{total_max}  ({total_max - total_current} pts remaining)")


@analyze_app.command("advise")
def advise_cmd() -> None:
    """Get strategic recommendation on research focus and schedule."""
    from expflow_pde.analyze import get_strategic_recommendation

    r = get_strategic_recommendation()

    print(f"  战略建议 — 距离截止还有 {r['remaining_days']} 天")
    print(f"  截止时间: {r['competition_deadline']} (每天 {r['submissions_per_day']} 次提交)")
    print()
    print(f"  🥇 首要聚焦: {r['primary_focus']}")
    print(f"     {r['primary_rationale']}")
    print()
    print(f"  🥈 次要聚焦: {r['secondary_focus']}")
    print(f"     {r['secondary_rationale']}")
    print()
    print(f"  🥉 第三聚焦: {r['tertiary_focus']}")
    print(f"     {r['tertiary_rationale']}")
    print()
    print("  建议日程:")
    schedule = r.get("suggested_schedule", {})
    for day, plan in schedule.items():
        print(f"    {day.replace('_', ' → ')}: {plan}")


@analyze_app.command("help")
def analyze_help_cmd() -> None:
    """Show quick reference for analyze commands."""
    print("  用法: expflow analyze <command> [options]")
    print()
    print("  task <id>         详细分析任务 (task1/task2/task3)")
    print("  equations [name]  列出所有PDE方程 / 查看单个方程详情")
    print("  equations --task <t>  按任务过滤方程 (task1/2/3/competition)")
    print("  losses            列出所有可用损失函数及其参数")
    print("  status            竞赛全局状态概览")
    print("  advise            战略建议 (聚焦哪个任务、日程安排)")


@analyze_app.command("losses")
def losses_cmd(
    name: Optional[str] = typer.Argument(
        None, help="Show details for a specific loss function"
    ),
) -> None:
    """List all available PDE loss functions or show details for one."""

    loss_info = {
        "l1_rel": {
            "class": "LprelLoss(p=1)",
            "desc": "Relative L1 norm loss: ||x-y||_1 / ||y||_1",
            "params": "p=1 (fixed), size_mean=True|False|None",
            "use_case": "核心 — Rel-L1, 对异常值鲁棒",
        },
        "l2_rel": {
            "class": "LprelLoss(p=2)",
            "desc": "Relative L2 norm loss: ||x-y||_2 / ||y||_2",
            "params": "p=2 (fixed), size_mean=True|False|None",
            "use_case": "核心 — 竞赛 Rel-MSE 的平方根形式, 默认推荐",
        },
        "h1_1d": {
            "class": "H1relLoss_1D(beta, alpha)",
            "desc": "H1 Sobolev loss for 1D via FFT: alpha*I + beta*k^2",
            "params": "beta=1.0, alpha=1.0, size_mean=True|False|None",
            "use_case": "Burgers FNO — 抑制高频震荡, 理想的光滑解损失",
        },
        "h1_2d": {
            "class": "H1relLoss(beta, alpha)",
            "desc": "H1 Sobolev loss for 2D via FFT: alpha + beta*(kx^2+ky^2)",
            "params": "beta=1.0, alpha=1.0, size_mean=True|False|None",
            "use_case": "2D PDE — Navier-Stokes, Darcy 等",
        },
        "mse_rel": {
            "class": "MSELoss_rel",
            "desc": "Relative MSE: MSE(x,y) / MSE(0,y)",
            "params": "size_mean=True|False|None",
            "use_case": "简单有效的归一化 MSE",
        },
        "smoothl1_rel": {
            "class": "SmoothL1Loss_rel",
            "desc": "Relative Smooth L1: SmoothL1(x,y) / SmoothL1(0,y)",
            "params": "size_mean=True|False|None",
            "use_case": "对大误差鲁棒, 适合噪声数据",
        },
        "l2_abs": {
            "class": "lpLoss(p=2)",
            "desc": "Absolute L2 norm: ||x-y||_2",
            "params": "p=2 (fixed), size_mean=True|False|None",
            "use_case": "参考基线, 无归一化",
        },
        "mse_abs": {
            "class": "nn.MSELoss",
            "desc": "Standard torch MSE loss",
            "params": "无 (使用默认 reduction='mean')",
            "use_case": "参考基线 — 当前 PDEBench 默认损失",
        },
    }

    if name:
        info = loss_info.get(name)
        if info is None:
            print(f"Unknown loss: {name}")
            print(f"Available: {', '.join(loss_info.keys())}")
            return
        print(f"  {name}")
        print(f"  {'─' * 50}")
        print(f"  Class:  {info['class']}")
        print(f"  Desc:   {info['desc']}")
        print(f"  Params: {info['params']}")
        print(f"  Usage:  {info['use_case']}")
        return

    # List all
    print(f"{'Name':<16} {'Class':<30} {'Description':<50}")
    print(f"{'─' * 96}")
    for key, info in loss_info.items():
        print(f"{key:<16} {info['class']:<30} {info['desc']:<50}")
    print()
    print("  Details: expflow analyze losses <name>")
    print("  Usage:   expflow optuna run script.py --loss l2_rel")


@analyze_app.command("diagnose")
def diagnose_cmd(
    task_id: str | None = typer.Option(None, "--task", "-t",
        help="ClearML task ID"),
    json_path: str | None = typer.Option(None, "--json", "-j",
        help="Path to local eval JSON file"),
) -> None:
    """Analyze experiment and identify degradation patterns."""
    if not task_id and not json_path:
        print("ERROR: Provide --task <id> or --json <path>")
        raise typer.Exit(code=1)

    from expflow_pde.analyze import diagnose_experiment

    result = diagnose_experiment(task_id=task_id, json_path=json_path)
    if result is None:
        src = f"task_id={task_id}" if task_id else f"json={json_path}"
        print(f"Cannot load experiment ({src})")
        raise typer.Exit(code=1)

    print(f"  Seg1: {result['seg1']:>6.2f}  | Seg2: {result['seg2']:>6.2f}  "
          f"| Seg3: {result['seg3']:>6.2f}  | Total: {result['total']:>6.2f}")
    print(f"  MSE:  {result['total_mse']:.6f}")
    print(f"  Pattern: {result['degradation_pattern']}")
    print(f"  Diagnosis:")
    for d in result['diagnosis']:
        print(f"    - {d}")


@analyze_app.command("suggest")
def suggest_cmd(
    task_id: str | None = typer.Option(None, "--task", "-t",
        help="ClearML task ID"),
    json_path: str | None = typer.Option(None, "--json", "-j",
        help="Path to eval JSON file"),
) -> None:
    """Analyze experiment and suggest next hyperparameters."""
    from expflow_pde.analyze import diagnose_experiment, suggest_next_params

    hp: dict = {}

    diagnosis = diagnose_experiment(task_id=task_id, json_path=json_path)
    if diagnosis is None:
        src = f"task_id={task_id}" if task_id else f"json={json_path}"
        print(f"Cannot load experiment ({src})")
        raise typer.Exit(code=1)

    suggestion = suggest_next_params(diagnosis, current_hparams=hp)

    print(f"\n  Diagnosis:")
    print(f"    Pattern: {diagnosis['degradation_pattern']}")
    for d in diagnosis['diagnosis']:
        print(f"    - {d}")

    params = suggestion.get("suggested_params", {})
    rationale = suggestion.get("rationale", [])

    print(f"\n  Suggested params:")
    for k, v in params.items():
        if k == "tag":
            continue
        print(f"    --{k}={v}")

    print(f"\n  Rationale:")
    for r in rationale:
        print(f"    - {r}")


@analyze_app.command("deep")
def deep_cmd(
    task_id: str = typer.Argument(..., help="ClearML task ID"),
    wiki: bool = typer.Option(True, "--wiki/--no-wiki",
        help="Include wiki context in analysis"),
) -> None:
    """Deep analysis with reasoning model (requires deepseek-v4-pro).

    Runs the rule engine first (diagnose), then reads wiki context
    and experiment history to provide a comprehensive analysis.

    Uses the 'model.analysis' config from ~/.hermes/config.yaml.
    """
    from expflow_pde.analyze import diagnose_experiment, get_task_meta

    # Step 1: Rule engine (0 token)
    print(f"  [1/3] Rule engine diagnosis...")
    diagnosis = diagnose_experiment(task_id=task_id)
    if diagnosis is None:
        print(f"  Cannot load experiment: {task_id}")
        raise typer.Exit(code=1)

    print(f"    Pattern: {diagnosis['degradation_pattern']}")
    for d in diagnosis['diagnosis']:
        print(f"    - {d}")

    # Step 2: Context
    print(f"\n  [2/3] Loading context...")
    task_meta = get_task_meta()

    # Print what would be included in deep analysis
    print(f"    Task metadata loaded: {list(task_meta.keys())}")
    print(f"    Wiki pages available: ~/wiki/entities/, ~/wiki/concepts/")

    # Step 3: Instruction for the reasoning model
    print(f"\n  [3/3] Deep analysis prompt (ready for deepseek-v4-pro):")
    print(f"    {'=' * 55}")
    print(f"    Analyze experiment {task_id} with reasoning:")
    print(f"    - Seg scores: {diagnosis.get('seg1')}, {diagnosis.get('seg2')}, "
          f"{diagnosis.get('seg3')}")
    print(f"    - Pattern: {diagnosis['degradation_pattern']}")
    print(f"    - Current task knowledge: {json.dumps(task_meta.get('task1', {}).get('proven_strategies', []), indent=2)}")
    print(f"    {'=' * 55}")
    print(f"\n  To run with deepseek-v4-pro:")
    print(f"    hermes \"Deep analysis of experiment {task_id}\"")
