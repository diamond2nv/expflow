---
name: l2-repair-executor
description: Generated from skills/l2-repair-executor/SKILL.md (package reference copy)
---

# L2 Repair Executor

## When to Use

Automatic trigger: Hermes receives a repair result from `expflow pipeline` or
`expflow dummy` where `result["level"] == "L2"`.

Manual trigger: A JSON file exists at a known `--repair-output` path
(e.g. `/tmp/l2_repair.json`) containing a structured L2 repair context.

## How It Works

The Hermes skill reads the structured L2 result dict and spawns a
`delegate_task` subagent to analyze the failure and produce a fix plan.

### Input Structure (from `repair.py:_try_l2()`)

The L2 result dict contains these machine-readable fields:

```python
{
  "level": "L2",
  "experiment_id": "exp:snow_...",
  "exit_code": 1,
  "exc_type": "torch.cuda.OutOfMemoryError",
  "exc_message": "CUDA out of memory. Tried to allocate 2.45 GiB.",
  "files_to_check": ["/opt/train.py"],
  "wiki_paths": ["~/wiki/troubleshooting/gpu-memory.md"],
  "tb_snippet": ["Traceback (most recent call last):", ...],
  "subagent_prompt": "...",
  "subagent_schema": {
    "goal": "Analyze experiment failure and produce a fix plan",
    "role": "leaf",
    "toolsets": ["terminal", "file", "skills"]
  }
}
```

### Execution Steps

When Hermes sees a repair result with `level == "L2"`:

1. Read the L2 context (from pipeline result or `--repair-output` JSON file)
2. Collect wiki context for each `wiki_path`
3. Fetch experiment record from `DispatchDB`
4. Spawn `delegate_task` subagent with the combined context + subagent_prompt
5. Subagent returns a fix plan: `file: X, old: Y, new: Z`
6. Hermes verifies file exists, reads old_string, applies `patch()`
7. Run syntax check
8. Record to dispatch_db audit_log: `event_type="repair_l2"`

### Cron Job

For production pipelines where Hermes may not be present at submit time:

```bash
# Watch ~/.expflow/repair_pending/ for unprocessed L2 repair files
# Process each file via this skill
# Move processed files to ~/.expflow/repair_done/
```

## Pitfalls

- Subagent is `leaf` role — returns a plan, cannot execute code
- Unknown errors with no traceback likely return `no_plan` → needs_human
- Container file paths may need local equivalent lookup
- Each `--repair-output` file must be processed exactly once
