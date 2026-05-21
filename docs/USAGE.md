# expflow-pde Usage Guide

## Installation

```bash
# Core CLI (no external SDKs needed)
pip install expflow-pde

# With all SDK integrations
pip install "expflow-pde[all]"

# Individual extras
pip install "expflow-pde[clearml]"   # Task/queue/dataset management
pip install "expflow-pde[optuna]"    # Hyperparameter optimization
pip install "expflow-pde[langfuse]"  # LLM observability traces
pip install "expflow-pde[mcp]"       # MCP server + all SDKs
pip install "expflow-pde[pipeline]"  # Pipeline mode (needs clearml)
```

### Local Development

```bash
git clone https://github.com/diamond2nv/expflow.git
cd expflow
python -m venv venv
source venv/bin/activate
pip install -e ".[all,dev]"
```

## Configuration

First run `expflow init` to configure:

```bash
expflow init                       # Interactive wizard
expflow init --quick                # Quick mode (defaults)
```

Or manually create `config.yaml` in your project root:

```yaml
# ~/my_project/config.yaml
clearml:
  api_server: http://localhost:8008
  web_server: http://localhost:8080
  files_server: http://localhost:8081

langfuse:
  host: http://localhost:3000
  public_key: "pk-..."
  secret_key: "sk-..."
```

For sensitive values (API keys), use `.env`:

```env
LANGFLUSE_PUBLIC_KEY=pk-xxx
LANGFLUSE_SECRET_KEY=sk-xxx
```

Config search order: `CWD/config.yaml` → parent dirs → `.env`.

## CLI Commands

### Top-Level Commands (No SDK Dependencies)

```bash
expflow --help                           # Show help
expflow version                          # Show version
expflow version --verbose                # Show version + build info
expflow info                             # Show system info + SDK versions
expflow config                           # Show current config
expflow init                             # Interactive configuration
```

### ClearML Integration (`expflow clearml`)

**Requires**: `pip install "expflow-pde[clearml]"`

```bash
# Task management
expflow clearml tasks                    # List all tasks
expflow clearml task abc123              # Get task details
expflow clearml enqueue abc123           # Enqueue task
expflow clearml dequeue abc123           # Dequeue task
expflow clearml queues                   # List queues
expflow clearml workers                  # List workers (with GPU info)
expflow clearml compare-scores           # Compare experiment scores
expflow clearml compare-scores \
    --project PDEBench --tags task1 \
    --sort-by seg_total --gate pde_mean:lt:18.09

# Dataset management
expflow clearml dataset-list             # List datasets
expflow clearml dataset-register data/   # Register dataset
expflow clearml dataset-upload data/     # Upload dataset
expflow clearml dataset-download abc123  # Download dataset

# Pipeline management
expflow clearml pipeline-list            # List pipelines
expflow clearml pipeline-create          # Create pipeline
expflow clearml pipeline-start           # Start pipeline

# Scheduler
expflow clearml scheduler-create         # Create scheduler
expflow clearml scheduler-list           # List schedulers
expflow clearml scheduler-start          # Start scheduler
```

### Optuna Integration (`expflow optuna`)

**Requires**: `pip install "expflow-pde[optuna]"`

```bash
# Study management
expflow optuna create-study my_study     # Create study
expflow optuna studies                   # List studies
expflow optuna study my_study            # Get study details
expflow optuna delete-study my_study     # Delete study

# HPO Run (three modes)
expflow optuna run train_task1.py \
    --trials 20                          # Local mode (default)

expflow optuna run train_task1.py \
    --trials 50 --parallel 4 \
    --distributed --queue default        # Distributed mode

expflow optuna run train_task1.py \
    --trials 50 --optimizer -O           # ClearML HyperParameterOptimizer

# Trial interaction
expflow optuna ask my_study              # Ask for next trial
expflow optuna tell my_study trial_id    # Report result
expflow optuna plot my_study             # Plot study
```

### Langfuse Integration (`expflow langfuse`)

**Requires**: `pip install "expflow-pde[langfuse]"`

```bash
expflow langfuse traces                  # List traces
expflow langfuse trace lf_abc123         # Get trace details
expflow langfuse trace-cost lf_abc123    # Get trace cost
expflow langfuse sessions                # List sessions
expflow langfuse session my_session      # Get session details
expflow langfuse metrics                 # Get session metrics
```

### Experiment Dispatch (`expflow run`)

**No SDK dependencies** — in-memory experiment registry.

```bash
expflow run submit train.py              # Submit experiment
expflow run list                         # List experiments
expflow run status abc123                # Get experiment status
expflow run cancel abc123                # Cancel (PIN-guarded)
expflow run cancel abc123 --force        # Cancel (skip PIN)
```

### Pipeline (`expflow pipeline`)

**Requires**: `pip install "expflow-pde[pipeline]"` or `"expflow-pde[clearml]"`

```bash
# Fast mode (train → eval, skip HPO)
expflow pipeline submit train_task1.py \
    --queue default \
    --train-param lr=0.001 --train-param epochs=80 \
    --eval-script eval_task1.py

# Full mode (HPO → train → eval)
expflow pipeline submit-full train_task1.py \
    --queue default \
    --trials 50 --parallel 4 \
    --eval-script eval_task1.py \
    --metric seg_total --direction maximize

# Flexible skip
expflow pipeline submit-full train_task1.py --skip hpo --skip eval
expflow pipeline submit-full train_task1.py --skip train --skip eval
```

### Audit (`expflow audit`)

**No SDK dependencies** for core validation. `--task-id` mode needs clearml.

```bash
# Validate experiment against competition rules
expflow audit validate <exp_id> \
    --competition-rules --task-id abc123

# Check dataset compliance
expflow audit check-dataset <path>

# Generate report
expflow audit report <exp_id>
```

### System (`expflow system`)

```bash
expflow system status                    # Health checks for all components
expflow system board                     # Launch TensorBoard
```

### PIN Protection (`expflow pin`)

**No SDK dependencies**. Protects destructive operations.

```bash
expflow pin init 1234                    # Set PIN (SHA-256 stored)
expflow pin check                        # Verify PIN interactively
expflow pin clear                        # Remove PIN (requires current PIN)
expflow pin clear --force                # Remove PIN (skip verification)
expflow pin status                       # Check if PIN is active
```

### Competition Analysis (`expflow analyze`)

**No SDK dependencies**.

```bash
# Strategic advising (primary entry point)
expflow analyze advise

# Per-task analysis
expflow analyze task task1               # Task 1 details
expflow analyze task task3               # Task 3 (Kuramoto-Sivashinsky)

# Equation reference
expflow analyze equations                # All equations
expflow analyze equations --task competition  # Competition equations only
expflow analyze equations kuramoto_sivashinsky  # Single equation

# Competition overview
expflow analyze status
```

### MCP Server

```bash
expflow mcp                              # Start MCP server (stdio)
```

Register in Hermes Agent `~/.hermes/config.yaml` for agent integration:

```yaml
mcp:
  servers:
    expflow:
      command: "expflow"
      args: ["mcp"]
```

## Pipeline Modes

### Full Pipeline

```
HPO (Optuna) ──► Train (best params) ──► Eval (generate submission)
   │                    │                        │
   ▼                    ▼                        ▼
 clearml trials    clearml task              clearml task
```

Use when: exploration phase of competition. You need to find best hyperparams.

### Fast Pipeline

```
Train (fixed params) ──► Eval (generate submission)
       │                        │
       ▼                        ▼
   clearml task             clearml task
```

Use when: competition sprint. You already know the best params.

### Pipeline Flags

| Flag | Applies To | Description |
|------|:----------:|-------------|
| `--queue <name>` | all | clearml-agent queue for GPU dispatch |
| `--skip hpo` | full | Skip HPO step |
| `--skip eval` | all | Skip evaluation step |
| `--train-param key=val` | all | Extra args for training script |
| `--eval-param key=val` | all | Extra args for eval script |
| `--trials N` | full | Number of HPO trials |
| `--parallel M` | full | Max concurrent trials |

## MCP Tools

When the MCP server is running, Hermes Agent has access to 18+ tools:

| Tool | Description |
|------|-------------|
| `exp_list_tasks` | List ClearML tasks |
| `exp_enqueue_task` | Enqueue a task |
| `exp_dequeue_task` | Dequeue a task |
| `exp_list_queues` | List queues |
| `exp_list_workers` | List workers |
| `exp_compare_scores` | Compare experiment scores |
| `exp_dataset_list` | List datasets |
| `exp_dataset_upload` | Upload dataset |
| `exp_trace_experiment` | Create Langfuse trace |
| `exp_submit_experiment` | Submit experiment |
| `exp_get_status` | Get system status |

## Script Requirements

Training and evaluation scripts must follow these conventions for expflow compatibility:

```python
# 1. Accept hyperparams as --key=value CLI arguments
# 2. Report metrics in a way expflow can capture:
#    - For local mode: print "METRIC:<name>=<value>" to stdout
#    - For distributed mode: clearml Task.current_task().report_scalar(...)
# 3. Accept standard flags: --epochs, --lr, --batch_size, --tag

# Example stdout for HPO capture:
# METRIC:seg_total=57.09
```

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [DEVELOPMENT.md](DEVELOPMENT.md) — Developer guide
- [DATA_LAYER.md](DATA_LAYER.md) — ClearML data layer
- [COMPETITION.md](COMPETITION.md) — Competition integration
- [DUMMY_GAME.md](DUMMY_GAME.md) — Experiment simulator (no GPU needed)

---

## Dummy Experiment Game

The **Dummy Experiment Game** is a zero-dependency simulation of the expflow experiment lifecycle. It replaces real GPU training with a synthetic seg-score model, so you can test the entire diagnose → suggest → submit → fail → repair → iterate loop without any infrastructure.

```bash
# Start a game, run a step, inject a failure
expflow dummy start --task task1
expflow dummy step --params '{"n_modes": 20}'
expflow dummy step --inject cuda_oom

# Inspect the experiment tree created by the game
expflow dispatch tree $(expflow dummy status | grep root_id | cut -d'"' -f4)

# Run a fully automated loop
expflow dummy auto --max-steps 10 --repair
```

### Available Failure Patterns

| Pattern | Repair Level | Description |
|---------|:-----------:|-------------|
| `git_not_found` | L0 (rule) | Git clone fails with "project not found" |
| `module_not_found` | L0 (rule) | Missing Python dependency |
| `cuda_oom` | L1 (traceback) | CUDA out-of-memory error |
| `data_not_found` | L1 (traceback) | Missing data file |
| `unknown_error` | L2 (reflection) | Opaque error, needs deep analysis |

### Use Cases

- **Integration testing**: Verify the repair pipeline responds correctly to each failure class
- **Onboarding**: See how expflow works without installing GPU toolchain
- **CI/CD**: Run the full automate loop in CI to catch regressions in diagnose/suggest/repair

See [DUMMY_GAME.md](DUMMY_GAME.md) for full documentation.
