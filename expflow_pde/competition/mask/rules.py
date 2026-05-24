"""Mask rules for competition-cleansing — pattern-based content masking.

Each rule defines regex patterns that match competition-specific content
(PDE equations, data paths, scoring formats, etc.) and a replacement
string. Used by scanner.py to audit or cleanse wiki/skills directories.

Covers: Task 1 (Burgers nu=0.001), Task 2 (Multi-nu Burgers), Task 3 (K-S).
PCMF / physics-informed residual / spectral filtering are NOT masked
(they are cross-task general methods).

Usage:
    from expflow_pde.competition.mask.rules import ALL_RULES
    for rule in ALL_RULES:
        masked, violations = rule.apply(content)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MaskRule:
    """A single masking rule.

    Attributes:
        name: Human-readable label (e.g. 'pde_equation').
        patterns: List of regex strings to match.
        replacement: What to replace matches with.
        severity: 'high' (code blocker) | 'medium' (reference) | 'low' (cosmetic).
        description: Explanation of what this rule masks.
    """

    name: str
    patterns: list[str] = field(default_factory=list)
    replacement: str = "[COMPETITION_SPECIFIC]"
    severity: str = "high"
    description: str = ""

    def apply(self, text: str) -> tuple[str, list[str]]:
        """Apply this rule to text.

        Args:
            text: Input text (markdown, code, etc.).

        Returns:
            (masked_text, violations_found) where violations is a list
            of human-readable strings describing each match.
        """
        violations: list[str] = []
        result = text
        for pat in self.patterns:
            try:
                matches = re.findall(pat, result, re.IGNORECASE)
                if matches:
                    for m in matches:
                        short = str(m)[:80]
                        violations.append(f"{self.name}: {short}")
                    result = re.sub(pat, self.replacement, result, flags=re.IGNORECASE)
            except re.error:
                # Skip malformed patterns
                pass
        return result, violations


# ── Rule definitions ───────────────────────────────────────────

EQUATION_RULES = MaskRule(
    name="pde_equation",
    patterns=[
        # Named equations — all 3 tasks
        r"Burgers['\u2019]?\s*equation",
        r"Kuramoto[\u2013\u2014-]Sivashinsky",
        r"K[S\u2013]?\s*equation",
        r"Navier[\u2013\u2014-]Stokes",
        # PDE operator forms
        r"u_t\s*\+?\s*u\s*\*?\s*u_x\s*=",
        r"u_t[+\s].*?u_xxxx",
        # nu values — task-specific
        r"nu\s*=\s*0\.001",
        r"nu\s*[:=]\s*0\.0?0?1\b",
        r"default_nu\s*=\s*0\.0?1?",
        r"nu\s*=\s*[\d\.]+\s*(?:default|fallback)",
        # Lambda2 conditioning (Task 3 specific)
        r"lambda2\s*(?:predictor|encoder|clip)",
        r"lambda2_pred",
        r"lambda2\s*=\s*1\.25",
        r"lambda2_val\s*=\s*1\.25",
        # Equation-specific constants
        r"TIME_STEPS\s*=\s*20",  # Task 3 uses 20 (vs Task 1/2's 10)
        r"N_PREDICT\s*=\s*38",  # Task 3: 380 steps
        r"TOTAL_STEPS\s*=\s*40",
        # NuEncoder (Task 2 specific)
        r"NuEncoder",
        r"nu_encoder",
        # Grid/channel patterns (Task 3)
        r"grid_base",
        r"in_channels\s*=\s*TIME_STEPS\s*\+\s*2",
    ],
    replacement="[PDE_EQUATION]",
    severity="high",
    description="PDE equation names, formulas, and equation-specific parameters (nu, lambda2, grid)",
)

DATA_PATH_RULES = MaskRule(
    name="data_path",
    patterns=[
        # Common base paths
        r"(?:data_new2|data_old|data_new3)[/\w_/-]+",
        r"sample_submission[/\w]*",
        r"train_val_test_init",
        # Task 1 paths
        r"task1_test\.hdf5",
        r"task1_val\.hdf5",
        r"task1_pred\.hdf5",
        r"task1_time\.csv",
        # Task 2 paths (.h5 suffix!)
        r"task2_test\.h5",
        r"task2_val\.h5",
        r"task2_pred\.hdf5",
        r"task2_time\.csv",
        r"task2_train_part\d",
        # Task 3 paths (.hdf5 suffix)
        r"task3_test\.hdf5",
        r"task3_val\.hdf5",
        r"task3_pred_.*\.hdf5",
        r"task3_time\.csv",
        r"TASK3_TRAIN",
        r"TASK3_VAL",
        r"TASK3_TEST",
        r"x-coordinate",
        # Generic wildcard for any task*_ path
        r"task\d+_[a-z]+_part\d",
    ],
    replacement="[COMPETITION_DATA_PATH]",
    severity="high",
    description="Competition dataset file paths and HDF5 keys",
)

SCORING_RULES = MaskRule(
    name="scoring_format",
    patterns=[
        # Segmented scoring
        r"Seg[1-3]\s*[(].*?[)]\s*=\s*\d+[\.\d]*",
        r"Seg\s*Total[\s:=]+\d+[\.\d]*",
        r"\bseg_total\b",
        # Log file names — all 3 tasks
        r"task1_logs\.log",
        r"task2_logs\.log",
        r"task3_logs\.log",
        # Competition scoring module
        r"compute_task3_segmented_scores",
        r"compute_segmented_scores",
        r"compute_competition_score",
        # Score caps
        r"ReL[-\s]*MSE",
        r"Lorentzian",
        r"Frechet",
        r"frechet_distance",
        # Threshold values
        r"\bweight\s*25%\b",
        r"\bweight\s*50%\b",
    ],
    replacement="[SCORE_METRIC_OR_FILE]",
    severity="medium",
    description="Scoring output format, log file names, and seg thresholds",
)

SUBMISSION_RULES = MaskRule(
    name="submission_format",
    patterns=[
        # CSV headers
        r"train_time,inference_time",
        r"segmented_score",
        # Output conventions
        r"train_time\s*=\s*\d+\.\d+",
        r"inference_time\s*=\s*\d+\.\d+",
        r"infer_time\s*=\s*\d+\.\d+",
    ],
    replacement="[COMPETITION_FORMAT]",
    severity="medium",
    description="Competition submission file format keys",
)

TRAIN_PARAM_RULES = MaskRule(
    name="train_param",
    patterns=[
        # Task-specific training configs
        r"TIME_STEPS\s*=\s*10",       # Task 1/2
        r"TIME_STEPS\s*=\s*20",       # Task 3
        r"INITIAL_STEP\s*=\s*TIME_STEPS",
        r"N_PREDICT\s*=\s*380",
        r"TOTAL_STEPS\s*=\s*400",
        # Competition default hparams
        r"sub_step\s*[:=]\s*\d+",
        r"num_sub_steps\s*[:=]\s*\d+",
        r"n_train\s*[:=]\s*500\b",
        r"batch_size\s*[:=]\s*16\b",
        r"\bn_modes\s*[:=]\s*(?:12|16|24|32|64)\b",
        r"hidden_channels\s*[:=]\s*(?:20|32|64|128)\b",
        r"n_layers\s*[:=]\s*4\b",
        r"n_windows\s*[:=]\s*30000\b",
        # Training loop
        r"eval_every\s*[:=]\s*10\b",
        r"eval_every\s*[:=]\s*50\b",
        r"M\s*[:=]\s*10\b",  # Task 3 multi-step
        r"tag\s*[:=]\s*['\"]task[123]",  # tag = 'task1_v1'
        r"default_nu\s*[:=]\s*0\.01",
        # Task tags
        r"['\"]task\d+'",
        r"['\"]task\d+_\w+['\"]",
        # ClearML task naming
        r"project_name=['\"]PDEBench['\"]",
        r"task\d+v\d",
    ],
    replacement="[TRAIN_HYPERPARAMETER]",
    severity="medium",
    description="Competition-optimized training parameters and task tags",
)

STRATEGY_RULES = MaskRule(
    name="strategy",
    patterns=[
        r"competition[\s-]*compliant",
        r"pdebench[\s_]competition",
        r"scoring[\s-]*formula",
        r"deadline[\s:]*202[56]",
        r"PDEBench\s+tags\s+v0\.[\d.]+",
        r"PDEBench[\s_]+Task\d+",
    ],
    replacement="[COMPETITION_STRATEGY]",
    severity="high",
    description="Competition strategy references",
)

# All active rules
ALL_RULES: list[MaskRule] = [
    EQUATION_RULES,
    DATA_PATH_RULES,
    SCORING_RULES,
    SUBMISSION_RULES,
    TRAIN_PARAM_RULES,
    STRATEGY_RULES,
]


def get_rules_by_severity(severity: str | None = None) -> list[MaskRule]:
    """Filter rules by severity level. Returns all if severity is None."""
    if severity is None:
        return list(ALL_RULES)
    return [r for r in ALL_RULES if r.severity == severity]


def describe_rules() -> list[dict[str, Any]]:
    """Return a list of rule metadata dicts (for CLI --help / JSON)."""
    return [
        {
            "name": r.name,
            "severity": r.severity,
            "pattern_count": len(r.patterns),
            "description": r.description,
            "replacement": r.replacement,
        }
        for r in ALL_RULES
    ]
