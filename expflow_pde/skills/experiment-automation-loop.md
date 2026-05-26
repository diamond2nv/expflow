---
name: experiment-automation-loop
description: Generated from skills/experiment-automation-loop/SKILL.md (package reference copy)
---

     1|---
     2|name: experiment-automation-loop
     3|description: >
     4|  Hermes Agent + expflow experiment automation loop. Recovers experiment
     5|  state on session restart, drives /goal optimization cycles with
     6|  L0 rule engine + L1 LLM reasoning, and detects newly completed
     7|  experiments for auto-analysis.
     8|category: mlops
     9|author: Li Shen
    10|version: 1.0.0
    11|tags: [experiment, loop, automation, clearml, expflow, goal]
    12|---
    13|
    14|# Experiment Automation Loop
    15|
    16|Hermes + expflow automated experiment loop. Designed for competition
    17|scenarios where you submit training experiments via clearml, analyze
    18|results, and iterate toward a target score.
    19|
    20|**Prerequisites:**
    21|- Hermes Agent (with clearml SDK installed)
    22|- expflow (pip install expflow-pde)
    23|- clearml-server + clearml-agent configured (for remote GPU execution)
    24|
    25|## Architecture
    26|
    27|```
    28|Hermes session start -> Session recovery -> Detect in-flight experiments
    29|     |
    30|     v
    31|/goal "reach seg_total 140" -> Loop (up to 20 auto-continuation turns)
    32|     |
    33|     +-- expflow analyze diagnose     (L0 rule engine, 0 token cost)
    34|     +-- expflow analyze suggest       (L0 rule suggestion)
    35|     +-- expflow iterate run           (one-click submit next iteration)
    36|     +-- cron: every 30min check completed experiments for deep analysis
    37|```
    38|
    39|Two-tier analysis:
    40|- **L0 rules** (0 tokens): deterministic pattern detection + param suggestion
    41|- **L1 LLM reasoning** (deepseek-v4-pro or equivalent): triggered on complex
    42|  patterns (compound mid+long term degradation, architecture ceiling) or
    43|  when 2+ consecutive iterations show no score improvement
    44|
    45|## When This Skill Activates
    46|
    47|1. **Session start** — recover in-flight experiment state (fixes state loss
    48|   after Hermes restart)
    49|2. **User enters /goal ...** — drive experiment optimization loop
    50|3. **User asks about experiment status** — check completed/in-flight experiments
    51|4. **Cron job trigger** — periodic deep analysis of newly completed experiments
    52|
    53|## Session Start Recovery (Mandatory)
    54|
    55|Run at the start of every new session:
    56|
    57|```bash
    58|# Restore clearml in-flight experiment tracking
    59|expflow clearml tasks --status running,in_progress --limit 10
    60|
    61|# Check last 5 completed experiments in the last 24h
    62|expflow clearml tasks --status completed --hours 24 --limit 5 --sort-by created
    63|
    64|# Check local dispatch DB (SQLite, persistent across restarts)
    65|expflow run status --limit 5
    66|```
    67|
    68|Interpretation:
    69|- **running experiments found** -> record their task IDs (via memory or note),
    70|  mark as "in-flight" so you don't resubmit
    71|- **completed but unanalyzed experiments** -> report to user, offer to analyze
    72|- **dispatch DB empty but clearml has tasks** -> dispatch DB may be out of sync,
    73|  run `expflow analyze sync` to pull scores from clearml
    74|
    75|## /goal Loop Protocol
    76|
    77|When the user enters `/goal reach <target_score> [on Task <N>]`:
    78|
    79|Each iteration:
    80|```
    81|Step 1: Check current state
    82|  - expflow clearml tasks --status running
    83|  - If running -> wait (set cron check) before proceeding
    84|
    85|Step 2: Analyze most recent completed experiment
    86|  - expflow analyze diagnose --task <task_id>
    87|  - Read output: pattern(short_term/mid_term/long_term/ceiling/
    88|    compound_mid_long/stable/distribution_shift/error)
    89|
    90|Step 3: Deep reasoning (conditional)
    91|  - Only when: pattern is compound_mid_long, ceiling, distribution_shift,
    92|    or 2+ consecutive iterations with no improvement
    93|  - Use a frontier reasoning model (deepseek-v4-pro, claude-sonnet-4, etc.)
    94|  - Read wiki pages for relevant background knowledge if available
    95|  - Save analysis to evaluation_results/analysis_<task_id>.md
    96|
    97|Step 4: Suggest + submit next iteration
    98|  - expflow analyze suggest --task <task_id>
    99|  - expflow iterate run --task <task_id> --queue default
   100|
   101|Step 5: Update /goal state
   102|  - Record current score, iteration count, next check time
   103|```
   104|
   105|## Deep Analysis Triggers
   106|
   107|Not every completed experiment needs LLM reasoning. Follow these rules:
   108|
   109|| Condition | Action |
   110||-----------|--------|
   111|| `pattern = stable` | Skip deep analysis, iterate directly |
   112|| `pattern = short_term/mid_term/long_term` | Rule suggestion is sufficient, no LLM |
   113|| `pattern = compound_mid_long` | Compound problem -> trigger deep analysis |
   114|| `pattern = ceiling` | Architecture bottleneck -> trigger deep analysis |
   115|| `pattern = distribution_shift` | IC mismatch problem -> trigger deep analysis |
   116|| `pattern = error` | clearml connection failure -> alert user, don't iterate |
   117|| 2+ consecutive iterations with no seg improvement | Trigger deep analysis |
   118|| User specifies `--deep` | Always trigger deep analysis |
   119|
   120|Deep analysis output: `evaluation_results/analysis_<task_id>.md`
   121|
   122|## Completion Detection (Cron or Manual)
   123|
   124|```bash
   125|# Check experiments completed in the last hour
   126|expflow clearml tasks --status completed --hours 1 --limit 20 --sort-by created
   127|```
   128|
   129|For each newly completed experiment:
   130|1. Check if already analyzed (exists `evaluation_results/analysis_<task_id>.md`)
   131|2. If unanalyzed -> run `expflow analyze diagnose --task <task_id>`
   132|3. Based on pattern -> decide whether to trigger deep reasoning
   133|4. If score exceeds previous_best -> notify user
   134|
   135|## Quick Reference
   136|
   137|```bash
   138|# === Session recovery ===
   139|expflow clearml tasks --status running --limit 10
   140|expflow run status --limit 5
   141|ls -t evaluation_results/analysis_* 2>/dev/null | head -3
   142|
   143|# === Daily operations ===
   144|expflow clearml tasks --status completed --hours 24
   145|expflow analyze diagnose --task <id>
   146|expflow analyze suggest --task <id>
   147|expflow iterate run --task <id> --queue default
   148|expflow analyze sync
   149|expflow clearml queue-status default
   150|
   151|# === Hypothesis tracking ===
   152|expflow hypothesis record --title "..." --context "..."
   153|expflow hypothesis list
   154|expflow hypothesis rejected
   155|```
   156|
   157|## Known Pitfalls
   158|
   159|- **dispatch DB may lag**: `expflow run status` reads local SQLite, while
   160|  `expflow clearml tasks` reads clearml server. Prefer clearml for current state.
   161|- **Cron job vs LLM cost**: The cron job runs every 30min but only triggers
   162|  LLM reasoning on ~20% of completions (only compound/ceiling/shift patterns).
   163|  Most passes cost 0 tokens (rule engine only).
   164|- **Remote GPU experiments**: Hermes cannot see PID on remote machines. Use
   165|  clearml task status to determine in-flight state, not PID monitoring.
   166|- **Duplicate submission risk**: After restart, Hermes may not know if a
   167|  previously submitted experiment has started. Check `expflow clearml tasks
   168|  --status running` first. If any running, wait. Only submit if none found.
   169|- **analysis_*.md dedup**: If the analysis file already exists for a task ID,
   170|  that experiment has already been analyzed. Skip it.
   171|
   172|## Cron Job Setup
   173|
   174|If you want automated deep analysis, create a Hermes cron job:
   175|
   176|```bash
   177|hermes cron create \
   178|  --name "experiment-deep-analysis" \
   179|  --schedule "*/30 * * * *" \
   180|  --model provider=openai,model=o3-mini \
   181|  --prompt "Check for recently completed clearml experiments in your project. For each unanalyzed task, run diagnose, and if pattern is compound_mid_long or ceiling, save an analysis report."
   182|```
   183|
   184|Adjust the model provider/model to match your setup.
   185|
   186|## Configuration
   187|
   188|No YAML configuration required. This skill assumes:
   189|- `expflow` is installed and in PATH
   190|- clearml credentials are configured
   191|- `evaluation_results/` directory exists (for analysis reports)
   192|
   193|Optional: define Hermes cron job for automated analysis.
   194|
