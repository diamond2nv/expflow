# Competition Logging — Beginner's Guide

The `expflow competition` subcommands produce competition-compliant JSONL logs for the PDEBench competition. Three streams are merged into one `task1_logs.log`:

```
fast.log    — Training metrics (epoch loss, validation scores)
agent.log   — Agent reasoning and tool calls
llm-*.jsonl — LLM API requests captured by litellm proxy → ingest server
```

## Quick Start

```bash
# 1. Set your API key (required for the proxy)
export DEEPSEEK_API_KEY="sk-..."
# OR set in .env file:
echo "DEEPSEEK_API_KEY=sk-..." >> .env

# 2. Start a session (launches proxy + ingest server)
expflow competition init --task task1 --tag myrun

# 3. Work as normal — all LLM calls are automatically logged
hermes -p comp-task1-myrun

# 4. Stop and merge
expflow competition stop

# 5. The merged log is at ~/.hermes/competition_logs/task1/myrun/task1_logs.log
```

## Configuration — What Goes Where

| Setting | Where to Set | Example | Purpose |
|---------|-------------|---------|---------|
| `DEEPSEEK_API_KEY` | `.env` or env var | `sk-abc...` | Your upstream API key. **Never commit this.** |
| `DEEPSEEK_BASE_URL` | `.env` or env var | `https://api.deepseek.com/v1` | Custom API endpoint (optional). |
| `COMPETITION_LOG_DIR` | `.env` or env var | `~/competition_logs` | Override log directory (default: `~/.hermes/competition_logs/`). |
| proxy port | `config.yaml` → `competition.proxy_port` | `4000` | litellm proxy listen port (default: `4000`). |
| ingest port | `config.yaml` → `competition.ingest_port` | `8099` | JSONL ingest server port (default: `8099`). |
| task, tag | CLI flags on `competition init` | `--task task1 --tag v1` | Identifying labels for the session. |

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
               task1_logs.log
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
