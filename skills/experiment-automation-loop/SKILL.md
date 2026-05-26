---
name: experiment-automation-loop
description: >
  Hermes Agent + expflow experiment automation loop. Recovers experiment
  state on session restart, drives /goal optimization cycles with
  L0 rule engine + L1 LLM reasoning, and detects newly completed
  experiments for auto-analysis.
category: mlops
author: Li Shen
version: 1.0.0
tags: [experiment, loop, automation, clearml, expflow, goal]
---

# Experiment Automation Loop

Hermes + expflow automated experiment loop. Designed for competition
scenarios where you submit training experiments via clearml, analyze
results, and iterate toward a target score.

**Prerequisites:**
- Hermes Agent (with clearml SDK installed)
- expflow (pip install expflow-pde)
- clearml-server + clearml-agent configured (for remote GPU execution)

## Architecture

```
Hermes session start -> Session recovery -> Detect in-flight experiments
     |
     v
/goal "reach seg_total 140" -> Loop (up to 20 auto-continuation turns)
     |
     +-- expflow analyze diagnose     (L0 rule engine, 0 token cost)
     +-- expflow analyze suggest       (L0 rule suggestion)
     +-- expflow iterate run           (one-click submit next iteration)
     +-- cron: every 30min check completed experiments for deep analysis
```

Two-tier analysis:
- **L0 rules** (0 tokens): deterministic pattern detection + param suggestion
- **L1 LLM reasoning** (deepseek-v4-pro or equivalent): triggered on complex
  patterns (compound mid+long term degradation, architecture ceiling) or
  when 2+ consecutive iterations show no score improvement

## When This Skill Activates

1. **Session start** — recover in-flight experiment state (fixes state loss
   after Hermes restart)
2. **User enters /goal ...** — drive experiment optimization loop
3. **User asks about experiment status** — check completed/in-flight experiments
4. **Cron job trigger** — periodic deep analysis of newly completed experiments

## Session Start Recovery (Mandatory)

Run at the start of every new session:

```bash
# Restore clearml in-flight experiment tracking
expflow clearml tasks --status running,in_progress --limit 10

# Check last 5 completed experiments in the last 24h
expflow clearml tasks --status completed --hours 24 --limit 5 --sort-by created

# Check local dispatch DB (SQLite, persistent across restarts)
expflow run status --limit 5
```

Interpretation:
- **running experiments found** -> record their task IDs (via memory or note),
  mark as "in-flight" so you don't resubmit
- **completed but unanalyzed experiments** -> report to user, offer to analyze
- **dispatch DB empty but clearml has tasks** -> dispatch DB may be out of sync,
  run `expflow analyze sync` to pull scores from clearml

## /goal Loop Protocol

When the user enters `/goal reach <target_score> [on Task <N>]`:

Each iteration:
```
Step 1: Check current state
  - expflow clearml tasks --status running
  - If running -> wait (set cron check) before proceeding

Step 2: Analyze most recent completed experiment
  - expflow analyze diagnose --task <task_id>
  - Read output: pattern(short_term/mid_term/long_term/ceiling/
    compound_mid_long/stable/distribution_shift/error)

Step 3: Deep reasoning (conditional)
  - Only when: pattern is compound_mid_long, ceiling, distribution_shift,
    or 2+ consecutive iterations with no improvement
  - Use a frontier reasoning model (deepseek-v4-pro, claude-sonnet-4, etc.)
  - Read wiki pages for relevant background knowledge if available
  - Save analysis to evaluation_results/analysis_<task_id>.md

Step 4: Suggest + submit next iteration
  - expflow analyze suggest --task <task_id>
  - expflow iterate run --task <task_id> --queue default

Step 5: Update /goal state
  - Record current score, iteration count, next check time
```

## Deep Analysis Triggers

Not every completed experiment needs LLM reasoning. Follow these rules:

| Condition | Action |
|-----------|--------|
| `pattern = stable` | Skip deep analysis, iterate directly |
| `pattern = short_term/mid_term/long_term` | Rule suggestion is sufficient, no LLM |
| `pattern = compound_mid_long` | Compound problem -> trigger deep analysis |
| `pattern = ceiling` | Architecture bottleneck -> trigger deep analysis |
| `pattern = distribution_shift` | IC mismatch problem -> trigger deep analysis |
| `pattern = error` | clearml connection failure -> alert user, don't iterate |
| 2+ consecutive iterations with no seg improvement | Trigger deep analysis |
| User specifies `--deep` | Always trigger deep analysis |

Deep analysis output: `evaluation_results/analysis_<task_id>.md`

## Completion Detection (Cron or Manual)

```bash
# Check experiments completed in the last hour
expflow clearml tasks --status completed --hours 1 --limit 20 --sort-by created
```

For each newly completed experiment:
1. Check if already analyzed (exists `evaluation_results/analysis_<task_id>.md`)
2. If unanalyzed -> run `expflow analyze diagnose --task <task_id>`
3. Based on pattern -> decide whether to trigger deep reasoning
4. If score exceeds previous_best -> notify user

## Quick Reference

```bash
# === Session recovery ===
expflow clearml tasks --status running --limit 10
expflow run status --limit 5
ls -t evaluation_results/analysis_* 2>/dev/null | head -3

# === Daily operations ===
expflow clearml tasks --status completed --hours 24
expflow analyze diagnose --task <id>
expflow analyze suggest --task <id>
expflow iterate run --task <id> --queue default
expflow analyze sync
expflow clearml queue-status default

# === Hypothesis tracking ===
expflow hypothesis record --title "..." --context "..."
expflow hypothesis list
expflow hypothesis rejected
```

## Known Pitfalls

- **dispatch DB may lag**: `expflow run status` reads local SQLite, while
  `expflow clearml tasks` reads clearml server. Prefer clearml for current state.
- **Cron job vs LLM cost**: The cron job runs every 30min but only triggers
  LLM reasoning on ~20% of completions (only compound/ceiling/shift patterns).
  Most passes cost 0 tokens (rule engine only).
- **Remote GPU experiments**: Hermes cannot see PID on remote machines. Use
  clearml task status to determine in-flight state, not PID monitoring.
- **Duplicate submission risk**: After restart, Hermes may not know if a
  previously submitted experiment has started. Check `expflow clearml tasks
  --status running` first. If any running, wait. Only submit if none found.
- **analysis_*.md dedup**: If the analysis file already exists for a task ID,
  that experiment has already been analyzed. Skip it.

## Cron Job Setup

If you want automated deep analysis, create a Hermes cron job:

```bash
hermes cron create \
  --name "experiment-deep-analysis" \
  --schedule "*/30 * * * *" \
  --model provider=openai,model=o3-mini \
  --prompt "Check for recently completed clearml experiments in your project. For each unanalyzed task, run diagnose, and if pattern is compound_mid_long or ceiling, save an analysis report."
```

Adjust the model provider/model to match your setup.

## Configuration

No YAML configuration required. This skill assumes:
- `expflow` is installed and in PATH
- clearml credentials are configured
- `evaluation_results/` directory exists (for analysis reports)

Optional: define Hermes cron job for automated analysis.
