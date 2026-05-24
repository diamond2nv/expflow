"""Mask rules for competition-cleansing — pattern-based content masking.

Each rule defines regex patterns that match competition-specific content
(PDE equations, data paths, scoring formats, etc.) and a replacement
string. Used by scanner.py to audit or cleanse wiki/skills directories.

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
        r"Burgers['\u2019]?\s*equation",
        r"u_t\s*\+?\s*u\s*\*?\s*u_x\s*=",
        r"Kuramoto[\u2013\u2014-]Sivashinsky",
        r"K[S\u2013]?\s*equation",
        r"Navier[\u2013\u2014-]Stokes",
        r"nu\s*=\s*0\.001",
    ],
    replacement="[PDE_EQUATION]",
    severity="high",
    description="PDE equation names and formulas",
)

DATA_PATH_RULES = MaskRule(
    name="data_path",
    patterns=[
        r"(?:data_new2|data_old)[/\w_/-]+",
        r"task1_test\.hdf5",
        r"task1_val\.hdf5",
        r"task1_pred\.hdf5",
        r"task1_time\.csv",
        r"task2_pred\.hdf5",
        r"sample_submission[/\w]*",
        r"train_val_test_init",
    ],
    replacement="[COMPETITION_DATA_PATH]",
    severity="high",
    description="Competition dataset file paths",
)

SCORING_RULES = MaskRule(
    name="scoring_format",
    patterns=[
        r"Seg1=\d+[\.\d]*",
        r"Seg\s*Total[\s:=]+\d+[\.\d]*",
        r"\bseg_total\b",
        r"task1_logs\.log",
    ],
    replacement="[SCORE_METRIC_OR_FILE]",
    severity="medium",
    description="Scoring output format and log file names",
)

SUBMISSION_RULES = MaskRule(
    name="submission_format",
    patterns=[
        r"train_time,inference_time",
        r"segmented_score",
    ],
    replacement="[COMPETITION_FORMAT]",
    severity="medium",
    description="Competition submission file format keys",
)

TRAIN_PARAM_RULES = MaskRule(
    name="train_param",
    patterns=[
        r"sub_step\s*[:=]\s*\d+",
        r"num_sub_steps\s*[:=]\s*\d+",
        r"n_train\s*[:=]\s*500\b",
        r"batch_size\s*[:=]\s*16\b",
        r"\bn_modes\s*[:=]\s*(?:16|24|32)\b",
    ],
    replacement="[TRAIN_HYPERPARAMETER]",
    severity="medium",
    description="Competition-optimized training parameters",
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
