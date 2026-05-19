# Data Layer: ClearML Fileserver

> **Design Decision**: Replace DVC with ClearML Server's built-in Fileserver
> as the single data backend for expflow experiments.
>
> **Verification**: After comprehensive study of 576+ ClearML documentation files,
> the design direction is confirmed. ClearML's Dataset class (`clearml-data`)
> provides version management, lineage tracking, differential storage,
> local caching, and metadata annotation — fully aligned with expflow's
> data layer requirements.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        expflow Server                              │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐    ┌────────────────────┐   │
│  │ PDEBench      │   │ expflow CLI  │    │ clearml-agent      │   │
│  │ Training      │──→│ (orchestrate)│──→ │ (GPU dispatch)     │   │
│  │ Scripts       │   │              │    │                    │   │
│  └──────────────┘   └──────────────┘    └────────┬───────────┘   │
│                                                   │                │
│  ┌─────────────────────────────────────────────────┴──────────┐   │
│  │           Docker Network: clearml-server                     │   │
│  │                                                              │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │   │
│  │  │ apiserver    │   │ webserver    │   │ fileserver   │    │   │
│  │  │ (8008)       │   │ (8082 → 80)  │   │ (8081)       │    │   │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │   │
│  │         │                  │                    │            │   │
│  │  ┌──────┴───────┐ ┌──────┴───────┐ ┌──────────┴──────┐    │   │
│  │  │ mongo        │ │ elasticsearch│ │ /mnt/fileserver │    │   │
│  │  │ (metadata)   │ │ (search idx) │ │ (file storage)  │    │   │
│  │  └──────────────┘ └──────────────┘ └─────────────────┘    │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **ClearML Fileserver is the single data backend** — all experiment data, model weights, and datasets go through clearml SDK's Dataset/Model API to Fileserver.
2. **No DVC** — clearml's Dataset class (`clearml-data`) already has built-in versioning, lineage tracking, metadata annotation, and differential storage.
3. **File-server based** (not standalone MinIO) — when clearml.conf's `api.api_server` points to clearml-apiserver, all data is automatically stored through the API to Fileserver.

## clearml-data Capability Analysis

### Completeness Assessment

| Requirement | clearml-data Support | Details |
|------------|:-------------------:|---------|
| Versioning | ✅ Native | `Dataset.create(version='1.0')`, semantic auto-increment |
| Lineage | ✅ Native | `parent_datasets` parameter, `Dataset.get(id).parent` |
| Differential Storage | ✅ Native | Child versions store only changes from parent |
| Local Cache | ✅ Native | Auto-cached to `~/.clearml/cache/` |
| Parallel Upload/Download | ✅ Native | `max_workers` parameter, defaults to logical core count |
| Metadata Annotation | ✅ Native | `set_metadata()` / `dataset_tags` |
| File Verification | ✅ Native | `clearml-data verify` (hash and filesize modes) |
| Sharded Download | ✅ Native | `part` / `num_parts` multi-node sharding |
| Offline Mode | ✅ Native | `CLEARML_OFFLINE_MODE=1` creates local zip for batch upload |
| Multiple Backends | ✅ Native | Fileserver (default), S3/GS/Azure/shared directory |
| S3 Compatibility | ✅ Native | `output_uri='s3://host:port/bucket'` (port required) |

**Conclusion**: clearml-data is fully capable as the data layer. expflow needs no
additional data transport implementation — all file transfer, versioning, caching,
and verification are provided by the clearml SDK.

### Core Workflow

```
────── CLI ──────                          ────── SDK ──────

# Create + add files                      dataset = Dataset.create(
clearml-data create \                       dataset_name='name',
  --project PDEBench \                      dataset_project='PDEBench',
  --name 1D_Burgers_v1 \                   parent_datasets=[PARENT_ID],
  --parents <PARENT_ID>                    version='1.0',
                                           )

clearml-data add --files data/            dataset.add_files(path='data/')

clearml-data close                        dataset.upload()
                                          dataset.finalize()
                                          print(dataset.id)
                                          print(dataset.url)

# Download (any version)                  dataset_v2 = Dataset.get(
clearml-data get --id <ID> \                dataset_id=ID,
  --local-copy ~/expflow/data/              alias='burgers_v2',
                                           )
                                           dataset_v2.get_local_copy(
                                             ~/expflow/data/
                                           )
```

## Pipeline Data Flow

### Training Script Data Access

```python
from clearml import Dataset

# Dataset v2.0: Nu=0.001 Burgers — 10,000 training samples
ds = Dataset.get(dataset_id="abc123")

# Get local copy (auto-cached, differential)
local_path = ds.get_local_copy(
    local_cache_dir="~/.clearml/cache/",
)

# Read with expflow pipeline
from expflow_pde.pipeline import ExperimentPipeline
ep = ExperimentPipeline()
result = ep.train_hpo_val_submit(
    train_script="train_task1.py",
    dataset_id="abc123",
    n_trials=50,
)
```

### Experiment Artifact Flow

```
Training Script ───► Model Checkpoint ───► Fileserver (via Model.upload_model())
                   └──► Metrics ──────────► Task.report_scalar()
                                             (visible in Web UI)

Eval Script     ───► pred.hdf5 ──────────► Fileserver (via Dataset.create())
                                            (submission artifact)

expflow audit   ───► Metrics fetch ──────► compare-scores with gating
                    (from clearml Task)    (via Task.get_last_scalar_metrics())
```

## Dataset Naming Convention

To stay compatible with `expflow clearml compare-scores` and the metrics registry,
use consistent naming:

```yaml
# Naming: <dataset_name>_v<major>.<minor>
# Example:
project: PDEBench
datasets:
  1D_Burgers_Nu0.001: v2.0    # Train: 10k samples
  1D_Burgers_Nu0.001_val: v1.0 # Val: 100 samples
  KS_Nu1.0-1.5: v1.0           # Train: 2000 samples
```

## Config Reference

### Minimal clearml.conf

```bash
api:
  api_server: http://localhost:8008
  web_server: http://localhost:8080
  files_server: http://localhost:8081
```

### Environment Variables

```bash
CLEARML_API_HOST=http://localhost:8008
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_FILES_HOST=http://localhost:8081
CLEARML_OFFLINE_MODE=0
```

### Detecting Running Remotely

```python
from clearml.config import running_remotely
if running_remotely():
    # Dataset is already mounted by clearml-agent
    local_path = Dataset.get(dataset_id=...).get_local_copy()
```

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — Overall system architecture
- [USAGE.md](USAGE.md) — CLI reference for dataset commands
- [DEVELOPMENT.md](DEVELOPMENT.md) — Testing dataset operations
