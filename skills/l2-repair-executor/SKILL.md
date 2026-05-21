---
name: l2-repair-executor
description: >
  Execute L2 reflection repair for expflow RepairStage.
  Consumes the structured L2 result dict from --repair-output,
  spawns a delegate_task subagent with traceback + wiki context,
  and produces a fix plan for Hermes to apply via patch().
category: mlops
author: Li Shen
version: 0.1.0
metadata:
  hermes:
    tags: [mlops, repair, reflection, subagent, expflow]
    homepage: https://github.com/diamond2nv/expflow
    related_skills: [expflow-pipeline-hpo, dispatch-repair, dummy-experiment-game]
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
  "subagent_prompt": "...",       # pre-rendered prompt
  "subagent_schema": {            # Hermes can use this directly
    "goal": "Analyze experiment failure and produce a fix plan",
    "role": "leaf",
    "toolsets": ["terminal", "file", "skills"]
  }
}
```

### Execution Steps

When you see a repair result with `level == "L2"`:

```python
# 1. Read the L2 context
l2 = repair_result  # or load from --repair-output JSON file

# 2. Collect wiki context for each wiki_path
wiki_context = ""
for path in l2.get("wiki_paths", []):
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        with open(expanded) as f:
            wiki_context += f.read()[:3000] + "\n\n"

# 3. Fetch experiment from dispatch_db
from expflow_pde.dispatch_db import DispatchDB
db = DispatchDB()
exp = db.get_experiment(l2["experiment_id"])
exp_context = json.dumps(exp, indent=2) if exp else "(no record)"

# 4. Combine into a context block for the subagent
context_block = f"""
{'-'*60}
L2 REPAIR CONTEXT
{'='*60}

## Experiment Record
{exp_context}

## Failure Details
Exception: {l2.get('exc_type', '?')}
Exit code: {l2.get('exit_code', 1)}
Message: {l2.get('exc_message', '')}

## Traceback
{chr(10).join(l2.get('tb_snippet', []))}

## Files Involved
{chr(10).join(f'  - {f}' for f in l2.get('files_to_check', []))}

## Wiki Context
{wiki_context}

{'='*60}
END L2 CONTEXT
{'-'*60}
"""

# 5. Spawn the delegate_task subagent
#    The subagent receives the context and the subagent_prompt
subagent_result = delegate_task(
    goal=l2["subagent_schema"]["goal"],
    context=f"{context_block}\n\n{l2['subagent_prompt']}",
    toolsets=l2["subagent_schema"]["toolsets"],
    role=l2["subagent_schema"]["role"],
)
```

### Processing the Subagent Plan

The subagent returns a fix plan in this format:

```
file: /opt/train.py
old:   batch_size = 64
new:   batch_size = 32

file: /opt/config.yaml
old:   gradient_checkpoint: false
new:   gradient_checkpoint: true
```

Hermes should:

1. **Verify each file exists** — `os.path.isfile(path)`
2. **Verify old_string exists** — read_file and check
3. **Apply patch** — `patch(path, old_string=old, new_string=new)`
4. **Run syntax check** — ruff format / python -c "compile(...)"
5. **If all patches applied**: write audit_log entry to dispatch_db
6. **If subagent returns `no_plan:` instead**: mark as needs_human

### Audit Trail

After successful L2 repair, record it in dispatch_db:

```python
from expflow_pde.dispatch_db import DispatchDB
db = DispatchDB()
with db._write_tx() as conn:
    db._write_audit(conn, l2["experiment_id"], "repair_l2", {
        "exc_type": l2["exc_type"],
        "files_patched": <list of files>,
        "subagent_summary": subagent_result[:500],
    })
```

## Example End-to-End

```bash
# Hermes runs expflow pipeline with L2 reflection
# (all in one Hermes session)
expflow pipeline submit train_task1.py --queue default \
    --wait --repair --repair-reflection \
    --repair-output /tmp/l2_repair.json

# If the pipeline fails and level == "L2":
#   1. Hermes sees the CLI output "L2 output: /tmp/l2_repair.json"
#   2. Hermes reads the file
#   3. Hermes loads this skill (l2-repair-executor)
#   4. Hermes spawns delegate_task with the context
#   5. Subagent returns a fix plan
#   6. Hermes applies patch() and verifies
#   7. Hermes records to dispatch_db audit_log
```

## Cron Job (Standalone Watcher)

For production pipelines where Hermes may not be present at submit time,
a cron job polls a directory for unprocessed L2 repair requests:

```bash
expflow cron create \
    --name "l2-repair-watcher" \
    --schedule "every 5m" \
    --prompt "Check ~/.expflow/repair_pending/*.json for unprocessed
              L2 repair requests. For each file, load the
              l2-repair-executor skill and follow its instructions.
              After processing, move the file to
              ~/.expflow/repair_done/<filename>.processed." \
    --skills ["l2-repair-executor", "dispatch-repair"] \
    --enabled_toolsets ["terminal", "file", "skills"] \
    --deliver "local"
```

## Pitfalls

- **Subagent cannot execute code** — it's a `leaf` role. It returns a plan.
  Hermes must apply the patch itself.
- **Unknown errors**: If `exc_type == "Unknown"` and no files_to_check,
  the subagent will likely return `no_plan:`. Mark as needs_human.
- **File paths from traceback**: These may be container paths
  (e.g. `/opt/train.py` inside a clearml docker). Hermes should check
  if the local equivalent exists (e.g. `~/Gitlab/.../PDEBench/utils/train.py`).
- **Wiki paths may not exist**: `_exc_type_to_wiki()` returns best-effort
  paths. If the file doesn't exist, spawn subagent without wiki context.
- **Multiple repair requests**: Each `--repair-output` file should be
  processed exactly once. The cron watcher moves processed files to
  `repair_done/` to prevent re-processing.

## Related

- [expflow-pipeline-hpo — Pipeline modes with --repair flag](expflow-pipeline-hpo.md)
- [dispatch-repair — DispatchDB + RepairStage](dispatch-repair.md)
- [dummy-experiment-game — Test L2 with --inject unknown_error](dummy-experiment-game.md)
