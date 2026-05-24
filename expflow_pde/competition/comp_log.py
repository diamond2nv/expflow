#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDEBench Competition Unified Logging System — expflow edition.

Dual-stream logging:
  Fast stream (high frequency): training progress, MSE/Rel-MSE, epoch time
  Smart stream (low frequency): Agent reasoning, experiment hypothesis, analysis

Format: UTC time - LOG LEVEL - [module:line] - [operator:tag] - message

Adapted from PDEBench pdebench/utils/comp_log.py.
"""

import builtins
import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---------------------------------------------------------


def _get_log_dir():
    """Resolve log directory: COMPETITION_LOG_DIR env or ~/.hermes/competition_logs."""
    env_dir = os.environ.get("COMPETITION_LOG_DIR")
    if env_dir:
        log_dir = Path(env_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    fallback = Path.home() / ".hermes" / "competition_logs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


_LOG_DIR = _get_log_dir()
_LOG_LEVEL = logging.DEBUG
_LOG_MAX_BYTES = 10 * 1024 * 1024       # 10 MB rotation
_LOG_BACKUP_COUNT = 5
_AGENT_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB (agent logs are larger)
_AGENT_LOG_BACKUP_COUNT = 3

# ─── Custom formatter ─────────────────────────────────────────────────────


class CompetitionFormatter(logging.Formatter):
    """UTC - LEVEL - [module:line] - [operator:tag] - [PID:NNN] - message."""

    def __init__(self, operator='', tag=''):
        super().__init__()
        self._operator = operator
        self._tag = tag

    def format(self, record):
        utc_now = datetime.now(timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        module = f"{record.module}:{record.lineno}"
        optag = f"{self._operator}:{self._tag}" if self._operator else "-"
        pid = os.getpid()
        msg = record.getMessage()
        # Collapse multi-line to single line
        msg_single = msg.replace('\n', '\\n')[:2000]
        return (
            f"{utc_now} - {record.levelname:>5s} - "
            f"[{module:20s}] - [{optag}] - "
            f"[PID:{pid}] - {msg_single}"
        )


class MetricFilter(logging.Filter):
    """Only pass records with is_metric flag."""

    def filter(self, record):
        return getattr(record, 'is_metric', False)


class TimeFilter(logging.Filter):
    """Only pass records with is_time flag."""

    def filter(self, record):
        return getattr(record, 'is_time', False)


# ─── Logger factory ───────────────────────────────────────────────────────


class CompLogger:
    """Competition unified logger — writes to five handlers simultaneously.

    Handlers:
      fast.log   — High-frequency training metrics (no metric/time/agent filters)
      agent.log  — Agent reasoning and analysis
      all.log    — Everything, debug level
      metric.jsonl — Machine-readable JSON Lines metrics
      time.jsonl — Structured phase timing records
    """

    _instances = {}

    def __new__(cls, name, operator='', tag='', log_dir=None):
        key = f"{name}:{operator}:{tag}"
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

    def __init__(self, name, operator='', tag='', log_dir=None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.name = name
        self.operator = operator
        self.tag = tag
        self.log_dir = Path(log_dir) if log_dir else _LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f'comp.{name}')
        self.logger.setLevel(_LOG_LEVEL)

        if self.logger.handlers:
            return

        formatter = CompetitionFormatter(operator=operator, tag=tag)
        metric_formatter = logging.Formatter('%(message)s')

        # Handler 1: fast.log (INFO+, exclude metric/time/agent)
        fast_path = self.log_dir / 'fast.log'
        fast_handler = logging.handlers.RotatingFileHandler(
            fast_path, maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT, encoding='utf-8'
        )
        fast_handler.setLevel(logging.INFO)
        fast_handler.setFormatter(formatter)
        fast_handler.addFilter(lambda r: (
            not getattr(r, 'is_metric', False)
            and not getattr(r, 'is_time', False)
            and '[AGENT]' not in str(r.msg)
        ))
        self.logger.addHandler(fast_handler)

        # Handler 2: agent.log (INFO+, exclude metric/time)
        agent_path = self.log_dir / 'agent.log'
        agent_handler = logging.handlers.RotatingFileHandler(
            agent_path, maxBytes=_AGENT_LOG_MAX_BYTES,
            backupCount=_AGENT_LOG_BACKUP_COUNT, encoding='utf-8'
        )
        agent_handler.setLevel(logging.INFO)
        agent_handler.setFormatter(formatter)
        agent_handler.addFilter(lambda r: (
            not getattr(r, 'is_metric', False)
            and not getattr(r, 'is_time', False)
        ))
        self.logger.addHandler(agent_handler)

        # Handler 3: all.log (DEBUG+, everything)
        all_path = self.log_dir / 'all.log'
        all_handler = logging.handlers.RotatingFileHandler(
            all_path, maxBytes=_LOG_MAX_BYTES * 2,
            backupCount=_LOG_BACKUP_COUNT, encoding='utf-8'
        )
        all_handler.setLevel(logging.DEBUG)
        all_handler.setFormatter(formatter)
        self.logger.addHandler(all_handler)

        # Handler 4: metric.jsonl (JSON Lines, metric only)
        metric_path = self.log_dir / 'metric.jsonl'
        metric_handler = logging.handlers.RotatingFileHandler(
            metric_path, maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT, encoding='utf-8'
        )
        metric_handler.setLevel(logging.INFO)
        metric_handler.setFormatter(metric_formatter)
        metric_handler.addFilter(MetricFilter())
        self.logger.addHandler(metric_handler)

        # Handler 5: time.jsonl (structured timing records)
        time_path = self.log_dir / 'time.jsonl'
        time_handler = logging.handlers.RotatingFileHandler(
            time_path, maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT, encoding='utf-8'
        )
        time_handler.setLevel(logging.INFO)
        time_handler.setFormatter(metric_formatter)
        time_handler.addFilter(TimeFilter())
        self.logger.addHandler(time_handler)

    # ── Convenience methods ──

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def metric(self, key, value, extra=None):
        """Record a machine-readable metric (to metric.jsonl).

        Args:
            key: Metric name (e.g. 'seg1_score', 'train_mse').
            value: Numeric value.
            extra: Optional dict (e.g. {'epoch': 5}).
        """
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO,
            '', 0, '', (), None
        )
        record.is_metric = True
        data = {
            'metric': key,
            'value': round(float(value), 6)
            if isinstance(value, (int, float)) else value,
            'operator': self.operator,
            'tag': self.tag,
            'utc': datetime.now(timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%S.000Z'),
        }
        if extra:
            data.update(extra)
        record.msg = json.dumps(data, ensure_ascii=False)
        self.logger.handle(record)

    def agent_note(self, msg):
        """Record agent reasoning (highlighted in agent.log)."""
        note = f"[AGENT] {msg}"
        self.logger.info(note)

    def flush(self):
        for h in self.logger.handlers:
            h.flush()

    def record_time(self, phase, seconds, task='task1', extra=None):
        """Record phase elapsed time (to time.jsonl).

        Args:
            phase: Phase name (e.g. 'train', 'inference', 'total').
            seconds: Elapsed seconds.
            task: Task identifier (e.g. 'task1', 'task2').
            extra: Optional extra dict.
        """
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO,
            '', 0, '', (), None
        )
        record.is_time = True
        data = {
            'task': task,
            'phase': phase,
            'seconds': round(float(seconds), 3),
            'operator': self.operator,
            'tag': self.tag,
            'utc': datetime.now(timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%S.000Z'),
        }
        if extra:
            data.update(extra)
        record.msg = json.dumps(data, ensure_ascii=False)
        self.logger.handle(record)


# ─── Global factory ───────────────────────────────────────────────────────

_loggers = {}


def get_logger(name, operator='', tag='', log_dir=None):
    """Get or create a competition logger.

    Args:
        name: Module name (e.g. 'fno_finetune', 'train_deeponet').
        operator: Operator name (e.g. 'FNO', 'DeepONet').
        tag: Experiment tag (e.g. 'pw0.0_ep80').
        log_dir: Log directory (default: ~/.hermes/competition_logs/).

    Returns:
        CompLogger instance.
    """
    key = f"{name}:{operator}:{tag}"
    if key not in _loggers:
        _loggers[key] = CompLogger(
            name, operator=operator, tag=tag, log_dir=log_dir)
    return _loggers[key]


# ─── Migration decorator ─────────────────────────────────────────────────


def print_to_log(log_func):
    """Decorator: redirect print() calls to log_func.info for migration."""
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_print = builtins.print
            builtins.print = lambda *a, **k: log_func(
                ' '.join(str(x) for x in a))
            try:
                return func(*args, **kwargs)
            finally:
                builtins.print = original_print
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════
# Time score estimation (from time.jsonl → competition score estimate)
# ══════════════════════════════════════════════════════════════════════════


def compute_time_scores(task1_train_s, task1_infer_s, task2_infer_s=None):
    """Compute competition time score estimates per official rules.

    Task 1:
      - Training time: <=60min=35, <=120min=25, <=300min=20, <=500min=10, >500=0
      - Inference: 0min=40pts, linear down to 2min=0pts
      - Inference >=2min => 0 pts for that task

    Task 2:
      - Training not scored, but total must be <=12h
      - Inference >=2min => 0 pts total

    Parameters
    ----------
    task1_train_s : float  Training + agent thinking total (seconds).
    task1_infer_s : float  Inference total (seconds).
    task2_infer_s : float  Task2 inference time (seconds, optional).

    Returns
    -------
    dict
    """
    scores = {}

    # Task 1 training score
    t1_m = task1_train_s / 60
    if t1_m <= 60:
        scores['task1_train_score'] = 35
    elif t1_m <= 120:
        scores['task1_train_score'] = 25
    elif t1_m <= 300:
        scores['task1_train_score'] = 20
    elif t1_m <= 500:
        scores['task1_train_score'] = 10
    else:
        scores['task1_train_score'] = 0

    # Task 1 inference score
    t1_inf_m = task1_infer_s / 60
    if t1_inf_m >= 2.0:
        scores['task1_inference_score'] = 0.0
    elif t1_inf_m <= 0:
        scores['task1_inference_score'] = 40.0
    else:
        scores['task1_inference_score'] = 40.0 * (1 - t1_inf_m / 2.0)
    scores['task1_inference_safe'] = task1_infer_s < 120
    scores['task1_inference_seconds'] = round(task1_infer_s, 1)

    # Task 1 raw time score (max 75, excl. segmentation)
    scores['task1_time_score_estimate'] = (
        scores['task1_train_score'] + scores['task1_inference_score']
    )

    # Task 2
    if task2_infer_s is not None:
        scores['task2_inference_safe'] = task2_infer_s < 120
        scores['task2_inference_seconds'] = round(task2_infer_s, 1)

    scores['_task1_train_min'] = round(t1_m, 1)
    scores['_task1_inference_min'] = round(t1_inf_m, 1)

    return scores


def export_time_scores(log_dir=None, output_path=None):
    """Export competition time score estimate from time.jsonl.

    Parameters
    ----------
    log_dir : str  Log directory (default: ~/.hermes/competition_logs/).
    output_path : str  Output CSV path (default: print to stdout).

    Returns
    -------
    dict  Score estimates grouped by task.
    """
    from collections import defaultdict

    log_dir = Path(log_dir) if log_dir else _LOG_DIR
    time_file = log_dir / 'time.jsonl'

    if not time_file.exists():
        print(f"[WARN] time.jsonl not found at {time_file}")
        return {}

    # Read all time records
    records = defaultdict(list)
    with open(time_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                task = d.get('task', 'task1')
                records[task].append(d)
            except json.JSONDecodeError:
                continue

    results = {}
    lines_output = []
    lines_output.append("=== Competition time score estimate ===")
    lines_output.append(
        f"{'task':>6s} | {'phase':20s} | {'seconds':>8s} | "
        f"{'score':>8s} | notes"
    )
    lines_output.append("-" * 60)

    for task in ['task1', 'task2']:
        if task not in records:
            continue

        train_t = 0
        infer_t = 0
        for r in records[task]:
            if r.get('phase') in ('train', 'training'):
                train_t = r['seconds']
            elif r.get('phase') in ('inference', 'infer'):
                infer_t = r['seconds']
            elif r.get('phase') == 'total':
                if r.get('seconds') > train_t + infer_t:
                    train_t = r['seconds']

        if task == 'task1':
            sc = compute_time_scores(train_t, infer_t)
            results[task] = sc
            lines_output.append(
                f"{task:>6s} | {'train+agent':20s} | {train_t:>8.0f} | "
                f"{sc['task1_train_score']:>3d}/35 | "
                f"{sc['_task1_train_min']:.1f}min"
            )
            lines_output.append(
                f"{'':>6s} | {'inference':20s} | {infer_t:>8.0f} | "
                f"{sc['task1_inference_score']:.0f}/40 | "
                f"{sc['_task1_inference_min']:.1f}min"
            )
            lines_output.append(
                f"{'':>6s} | {'total':20s} | {'':>8s} | "
                f"{sc['task1_time_score_estimate']:.0f}/75 | "
                f"{'OK' if sc['task1_inference_safe'] else 'WARN: >2min'}"
            )
        elif task == 'task2':
            sc = compute_time_scores(0, infer_t, task2_infer_s=infer_t)
            results[task] = sc
            safe = sc.get('task2_inference_safe', False)
            lines_output.append(
                f"{task:>6s} | {'inference':20s} | {infer_t:>8.0f} | "
                f"{'n/a':>8s} | "
                f"{'OK <=2min' if safe else 'WARN: >2min, score=0'}"
            )
            lines_output.append(
                f"{'':>6s} | {'note':20s} | {'':>8s} | {'':>8s} | "
                f"Training not scored; inference timeout => score 0"
            )

    output = '\n'.join(lines_output)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"Exported: {output_path}")
    else:
        print(output)

    return results
