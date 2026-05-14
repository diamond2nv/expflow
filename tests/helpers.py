#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for expflow.clearml — Task listing, queue management, dataset compliance.

All tests use mocked clearml SDK. No real clearml server needed.
"""

from unittest.mock import MagicMock

# ── Helpers ──


def _make_mock_task(
    task_id: str = "abc123",
    name: str = "test_task",
    project: str = "test_project",
    status: str = "completed",
    tags: list[str] | None = None,
) -> MagicMock:
    """Create a mock clearml Task with dict-like id/project access."""
    mock = MagicMock()
    mock.id = task_id
    mock.name = name
    mock.project = project
    mock.status = status
    mock.get_tags.return_value = tags or []
    mock.last_iteration = 100
    return mock


def _make_mock_queue(
    queue_id: str = "queue1",
    name: str = "default",
) -> MagicMock:
    """Create a mock clearml Queue."""
    mock = MagicMock()
    mock.id = queue_id
    mock.name = name
    return mock
