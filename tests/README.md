# expflow Tests

## Test Environment

| Environment | Python | torch | clearml | Status |
|-------------|--------|-------|---------|--------|
| `physicsnemo-cpu` (conda) | 3.11 | ✅ 2.11.0 | ✅ | **Full pass** (464/465) |
| Hermes venv (default) | 3.11 | ❌ | ❌ | Skips torch-dependent tests |

### physicsnemo-cpu (recommended)

```bash
cd ~/Gitlab/Agentic4Sci/expflow
~/miniconda3/bin/conda run -n physicsnemo-cpu python -m pytest tests/ -v
```

This is the **canonical test environment**. It has torch, clearml, and all PDEBench
dependencies installed. 1 pre-existing failure due to version string mismatch
(will be fixed when `tests/test_entry_point.py` hardcoded version is updated).

### Hermes venv (default python fallback)

When running outside conda (plain python via `python -m pytest`), torch-dependent
modules fail at import time. Skip them explicitly:

```bash
python -m pytest tests/ \
    --ignore=tests/test_losses.py \
    --ignore=tests/test_metrics.py \
    -v --no-header -q
```

## Known Pre-existing Failures

### single: `test_entry_point_missing_clearml_shows_module_not_found`

**Cause:** Version hardcoded as `0.5.0` in test assertions.
**Fix:** Update `assert ver_result.stdout.strip() == "expflow v0.6.0"` whenever
the package version changes.

This test installs expflow into a fresh venv, verifies `expflow version` output,
then checks that clearml-dependent commands show helpful `ModuleNotFoundError`
messages rather than crashing. The version assertion is a smoke check that the
package installed correctly — it's tightly coupled to the current version string.

All other tests (464/465) pass in physicsnemo-cpu environment.

## Test File Dependencies

| Test file | External deps | Environment required |
|-----------|--------------|---------------------|
| `test_entry_point.py` | subprocess, new venv, pip install | Any (builds temp venv) |
| `test_audit.py` | `metrics.py` → torch | physicsnemo-cpu |
| `test_compare.py` | clearml SDK | physicsnemo-cpu |
| `test_losses.py` | torch, expflow_pde.losses | physicsnemo-cpu |
| `test_metrics.py` | torch, expflow_pde.metrics | physicsnemo-cpu |
| `test_dispatch_db.py` | stdlib only | Any |
| `test_repair.py`, `test_repair_rules.py` | stdlib only | Any |
| `test_cli.py` | Typer, pytest-mock | Any |
