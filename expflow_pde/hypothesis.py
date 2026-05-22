#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hypothesis registry — track experiment hypotheses and their outcomes.

Each hypothesis records what was tried, why, and whether it succeeded.
Negative results are recorded explicitly, preventing re-exploration of
rejected directions.

Data stored in ~/.expflow/hypotheses.yaml as a list of hypothesis dicts.

Usage:
    from expflow_pde.hypothesis import (
        HypothesisRegistry,
        record_hypothesis,
        list_hypotheses,
        close_hypothesis,
        show_hypothesis,
    )
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any


def _get_hypotheses_path() -> str:
    home = os.environ.get("EXPFLOW_HOME", os.path.expanduser("~/.expflow"))
    return os.path.join(home, "hypotheses.yaml")


def _load_hypotheses() -> list[dict[str, Any]]:
    path = _get_hypotheses_path()
    if not os.path.exists(path):
        return []
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_hypotheses(hypotheses: list[dict[str, Any]]) -> None:
    import yaml

    path = _get_hypotheses_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(hypotheses, f, default_flow_style=False, allow_unicode=True)


def _next_id(hypotheses: list[dict[str, Any]]) -> str:
    now = datetime.now()
    base = f"hyp_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
    existing = {h["id"] for h in hypotheses if "id" in h}
    candidate = base
    counter = 1
    while candidate in existing:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


class HypothesisRegistry:
    """CRUD for experiment hypotheses persisted to YAML."""

    @staticmethod
    def record(
        hypothesis: str,
        rationale: str,
        suggested_params: dict[str, Any] | None = None,
        origin_task_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a new hypothesis before running an experiment.

        Args:
            hypothesis: The hypothesis statement (e.g. 'Increase n_modes
                from 16 to 24 will improve Seg3').
            rationale: Why this hypothesis makes sense.
            suggested_params: The parameter changes proposed.
            origin_task_id: clearml task ID that inspired this hypothesis.

        Returns:
            The recorded hypothesis dict with assigned id.
        """
        hyps = _load_hypotheses()
        entry: dict[str, Any] = {
            "id": _next_id(hyps),
            "created": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "rationale": rationale,
            "status": "proposed",
        }
        if suggested_params:
            entry["suggested_params"] = suggested_params
        if origin_task_id:
            entry["origin_task_id"] = origin_task_id
        hyps.append(entry)
        _save_hypotheses(hyps)
        return dict(entry)

    @staticmethod
    def list(
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all hypotheses, optionally filtered by status.

        Args:
            status: 'proposed', 'accepted', 'rejected', 'inconclusive',
                or None for all.

        Returns:
            List of hypothesis dicts.
        """
        hyps = _load_hypotheses()
        if status:
            return [h for h in hyps if h.get("status") == status]
        return hyps

    @staticmethod
    def close(
        hypothesis_id: str,
        status: str,
        evidence: str,
        evidence_task_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Close a hypothesis with outcome evidence.

        Args:
            hypothesis_id: The hypothesis id to close.
            status: 'accepted', 'rejected', or 'inconclusive'.
            evidence: Description of what the experiment showed.
            evidence_task_id: Optional clearml task ID that provided evidence.

        Returns:
            Updated hypothesis dict, or None if not found.
        """
        valid_statuses = ("accepted", "rejected", "inconclusive")
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")

        hyps = _load_hypotheses()
        for h in hyps:
            if h.get("id") == hypothesis_id:
                h["status"] = status
                h["closed"] = datetime.now().isoformat()
                h["evidence"] = evidence
                if evidence_task_id:
                    h["evidence_task_id"] = evidence_task_id
                _save_hypotheses(hyps)
                return dict(h)
        return None

    @staticmethod
    def show(hypothesis_id: str) -> dict[str, Any] | None:
        """Show a single hypothesis by id."""
        hyps = _load_hypotheses()
        for h in hyps:
            if h.get("id") == hypothesis_id:
                return dict(h)
        return None

    @staticmethod
    def get_rejected_directions() -> list[dict[str, Any]]:
        """Return all rejected hypotheses (negative results).

        Useful for deep analysis to avoid suggesting directions
        that have already been proven ineffective.
        """
        hyps = _load_hypotheses()
        return [h for h in hyps if h.get("status") == "rejected"]

    @staticmethod
    def get_open_hypotheses() -> list[dict[str, Any]]:
        """Return all proposed (unresolved) hypotheses."""
        hyps = _load_hypotheses()
        return [h for h in hyps if h.get("status") == "proposed"]


# ── Simple API (no class) ──


def record_hypothesis(
    hypothesis: str,
    rationale: str,
    suggested_params: dict[str, Any] | None = None,
    origin_task_id: str | None = None,
) -> dict[str, Any]:
    """Record a new hypothesis (convenience wrapper)."""
    return HypothesisRegistry.record(
        hypothesis=hypothesis,
        rationale=rationale,
        suggested_params=suggested_params,
        origin_task_id=origin_task_id,
    )


def list_hypotheses(
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List all hypotheses (convenience wrapper)."""
    return HypothesisRegistry.list(status=status)


def close_hypothesis(
    hypothesis_id: str,
    status: str,
    evidence: str,
    evidence_task_id: str | None = None,
) -> dict[str, Any] | None:
    """Close a hypothesis with outcome evidence."""
    return HypothesisRegistry.close(
        hypothesis_id=hypothesis_id,
        status=status,
        evidence=evidence,
        evidence_task_id=evidence_task_id,
    )


def show_hypothesis(hypothesis_id: str) -> dict[str, Any] | None:
    """Show a single hypothesis by id."""
    return HypothesisRegistry.show(hypothesis_id)


def get_rejected_directions() -> list[dict[str, Any]]:
    """Return all rejected hypotheses."""
    return HypothesisRegistry.get_rejected_directions()


def get_open_hypotheses() -> list[dict[str, Any]]:
    """Return all unresolved hypotheses."""
    return HypothesisRegistry.get_open_hypotheses()
