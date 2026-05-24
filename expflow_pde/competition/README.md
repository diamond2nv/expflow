# Competition Logging — Beginner's Guide

The `expflow competition` subcommands produce competition-compliant JSONL logs
for the PDEBench competition (Task 1 / Task 2 / Task 3). Three streams are
merged into one `{problem_id}_logs.log`:

```
fast.log    — Training metrics (epoch loss, validation scores)
agent.log   — Agent reasoning and tool calls
llm-*.jsonl — LLM API requests captured by litellm proxy → ingest server
```

## Quick Start

```bash
# 0. (Optional) Audit and mask competition-specific content from wiki/skills
expflow competition mask audit --wiki ~/wiki --skills ~/.hermes/skills
expflow competition mask apply --wiki ~/wiki --skills ~/.hermes/skills

# 1. (Optional) Bootstrap — verify clean environment + generate config
expflow competition bootstrap

# 2. Set your API key (required for the proxy)
export DEEPSEEK_API_KEY="sk-..."
# OR set in .env file:
echo "DEEPSEEK_API_KEY=sk-..." >> .env

# 3. Start a session (launches proxy + ingest server)
expflow competition init --problem-id task1 --tag myrun

# 4. Work as normal — all LLM calls are automatically logged
hermes -p comp-task1-myrun

# 5. Stop and merge
expflow competition stop

# 6. The merged log is at ~/.hermes/competition_logs/{problem_id}/{tag}/{problem_id}_logs.log
```

## Multi-Task Workflow

Each task has its own train/eval scripts, data, and submission format.
The competition subsystem is **task-agnostic** — just change `--problem-id`:

```bash
# Task 1 (Burgers nu=0.001)
expflow competition init --problem-id task1 --tag burger-run
# ... agent trains train_task1.py, evaluates eval_task1.py ...
expflow competition stop

# Task 2 (Multi-nu Burgers)
expflow competition init --problem-id task2 --tag multi-nu-v2
# ... agent trains train_task2.py, evaluates eval_task2.py ...
expflow competition stop

# Task 3 (Kuramoto-Sivashinsky)
expflow competition init --problem-id task3 --tag ks-v1
# ... agent trains train_task3.py, evaluates eval_task3.py ...
expflow competition stop
```

After mask + bootstrap, the agent's knowledge base is **competition-clean**
(no equation names, data paths, or scoring formulas), and the config.yaml
tells the agent which task-specific parameters to use.

## Configuration — What Goes Where

| Setting | Where to Set | Example | Purpose |
|---------|-------------|---------|---------|
| `DEEPSEEK_API_KEY` | `.env` or env var | `sk-abc...` | Your upstream API key. **Never commit this.** |
| `DEEPSEEK_BASE_URL` | `.env` or env var | `https://api.deepseek.com/v1` | Custom API endpoint (optional). |
| `COMPETITION_LOG_DIR` | `.env` or env var | `~/competition_logs` | Override log directory (default: `~/.hermes/competition_logs/`). |
| proxy port | `config.yaml` → `competition.proxy_port` | `4000` | litellm proxy listen port (default: `4000`). |
| ingest port | `config.yaml` → `competition.ingest_port` | `8099` | JSONL ingest server port (default: `8099`). |
| max span | `config.yaml` → `competition.log.max_span_hours` | `12.0` | Max log time span in hours (0 = disabled) |
| max elapsed | `config.yaml` → `competition.log.max_elapsed_seconds` | `60` | Per-entry elapsed_seconds cap |
| problem id | CLI flag on `competition init` | `--problem-id task1` | Problem identifier from competition website |
| per-task config | `config.yaml` → `competition.problems.{id}` | see below | Per-task overrides for log paths, caps |

### Per-Problem Configuration

Add per-task overrides under `competition.problems`:

```yaml
competition:
  default_problem: task1
  problems:
    task1:
      max_span_hours: 12
      max_elapsed_seconds: 60
      log_file: "task1_logs.log"
      time_file: "task1_time.csv"
      pred_file: "task1_pred.hdf5"
    task2:
      max_span_hours: 12
      max_elapsed_seconds: 60
      log_file: "task2_logs.log"
      time_file: "task2_time.csv"
      pred_file: "task2_pred.hdf5"
    task3:
      max_span_hours: 24
      max_elapsed_seconds: 90
      log_file: "task3_logs.log"
      time_file: "task3_time.csv"
      pred_file: "task3_pred.hdf5"
```

The `bootstrap` command auto-generates this template with the extracted problem.

### Sensitive values (.env only, never in config.yaml)

- `DEEPSEEK_API_KEY` — always put in `.env` or export as environment variable
- `DEEPSEEK_BASE_URL` — if pointing to a private proxy with auth

### Non-sensitive defaults (config.yaml)

```yaml
# config.yaml (at project root or CWD)
competition:
  proxy_port: 4000       # litellm proxy
  ingest_port: 8099      # JSONL ingest server
  log_dir: ~/.hermes/competition_logs  # override directory
  log:
    max_span_hours: 12.0           # Max log span (0 = disabled)
    max_elapsed_seconds: 60.0      # Per-entry elapsed cap
    allow_binary_check: true
```

## What Each Component Does

### `_ingest_server.py` (port 8099 by default)
Receives POST requests from litellm's `generic_api` callback. Extracts response, tool_calls, model, timing. Writes to `llm-YYYYMMDD.jsonl`. Exposes `/health` with cumulative `error_count` — if errors > 0 on stop, you get a warning.

### `comp_log.py` — CompLogger
Writes 5 log files per session:
- `fast.log` — training progress (epochs, loss, val scores)
- `agent.log` — agent reasoning + tool calls
- `all.log` — everything (debug level)
- `metric.jsonl` — structured metrics in JSONL
- `time.jsonl` — structured phase timing

### `merge.py` — merge_logs() + validate_log()
Reads all 3 streams, sorts chronologically, deduplicates (same source + same full content), and writes competition-format JSONL. Validation checks:
- ✅ Each line is valid JSON
- ✅ Has `timestamp` + `elapsed_seconds`
- ✅ Monotonic timestamps
- ✅ ≤ 12h total span
- ⚠️ Warns on gaps > 10 minutes
- ⚠️ Warns on suspiciously few entries (< 10)

### `session.py` — CompetitionSession
Lifecycle manager. Orchestrates start (ingest first → proxy second → profile creation), stop (ingest kill → proxy kill → merge → validate), and signal handling.

## The Data Flow (Quick Reference)

```
Hermes/OpenCode → litellm proxy (:4000) → DeepSeek API
                       │
                       │ generic_api callback
                       ▼
               ingest server (:8099)
                       │
                       ▼
               llm-YYYYMMDD.jsonl

    + CompLogger fast.log + agent.log
                       │
                       ▼ event: expflow competition stop
                     merge
                       │
                       ▼
               {problem_id}_logs.log
                       │
                       ▼
                validate_log()
```

## Port Conflicts

If port 4000 or 8099 is already in use:

```bash
expflow competition init --proxy-port 4001 --ingest-port 8100
```

Or set in config.yaml:

```yaml
competition:
  proxy_port: 4001
  ingest_port: 8100
```

## Reading the Merged Log

Each line is JSON with fields:

```json
{
  "timestamp": "2026-05-23T10:00:01.000Z",
  "elapsed_seconds": 3.0,
  "response": "loss=0.0034 | GPU=2048/4096MB",
  "tool_calls": null
}
```

- `response` — LLM reply text or training metric line (capped at 2000 chars)
- `tool_calls` — tool call name + args (capped at 2000 chars)
- `elapsed_seconds` — time since previous entry (first entry = time from `session_start`)
- `metadata.is_system` — true for system environment records (nvidia-smi, host info)

## What NOT to Do

- ❌ Put `DEEPSEEK_API_KEY` in `config.yaml` — it gets committed to git
- ❌ Run `competition init` without `DEEPSEEK_API_KEY` set — proxy starts but can't connect
- ❌ Forget to `competition stop` — proxy keeps running, blocking the port
- ❌ Start a second session without stopping the first — "already running" error
