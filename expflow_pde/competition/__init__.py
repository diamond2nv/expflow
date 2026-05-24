"""expflow competition logging system.

Three-stream architecture:
  fast.jsonl     — Training metrics (epoch/batch/loss/time) via CompLogger
  llm-{id}.jsonl — LLM API calls captured by litellm proxy custom callback
  system record  — nvidia-smi + uname at session start

Merge into competition-compliant task1_logs.log on stop.

Usage:
    from expflow_pde.competition import get_comp_logger, CompetitionSession

    log = get_comp_logger('train_task1', operator='FNO', tag='v1')
    log.info("Training started...")
    log.metric("train_loss", 0.0034, extra={'epoch': 5})

    session = CompetitionSession(task='task1', tag='v1')
    session.start()   # launch litellm proxy
    # ... agent work ...
    session.stop()    # merge logs + validate

Mask (competition cleansing):
    from expflow_pde.competition.mask import ALL_RULES, scan_directory, apply_mask

    violations = scan_directory(Path("~/wiki"), ALL_RULES)
    manifest = apply_mask(Path("~/wiki"), Path("~/.competition/wiki"), ALL_RULES)

Bootstrap:
    from expflow_pde.competition.bootstrap import bootstrap_session
    result = bootstrap_session()
"""

from __future__ import annotations

from .comp_log import CompLogger, compute_time_scores, export_time_scores
from .comp_log import get_logger as get_comp_logger
from .session import CompetitionSession

__all__ = [
    "CompLogger",
    "get_comp_logger",
    "compute_time_scores",
    "export_time_scores",
    "CompetitionSession",
]
