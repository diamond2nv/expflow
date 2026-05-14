#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for expflow.langfuse — trace query, cost, sessions.

All tests use mocked langfuse SDK. No real langfuse server needed.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Mock helpers ──


def _make_mock_trace(
    trace_id: str = "trace_1",
    name: str = "inference",
    user_id: str | None = "user1",
    tags: list[str] | None = None,
    session_id: str | None = None,
    cost: float = 0.05,
) -> MagicMock:
    t = MagicMock(name=f"Trace({trace_id})")
    t.id = trace_id
    t.name = name
    t.user_id = user_id
    t.tags = tags or []
    t.session_id = session_id
    t.timestamp = "2026-05-14T00:00:00Z"
    t.cost = cost
    t.total_cost = cost
    t.input = None
    t.output = None
    t.latency = 1.5
    t.usage = {"input": 500, "output": 200}
    return t


def _make_mock_session(
    session_id: str = "session_1",
    user_id: str | None = "user1",
) -> MagicMock:
    s = MagicMock(name=f"Session({session_id})")
    s.id = session_id
    s.user_id = user_id
    s.created_at = "2026-05-14T00:00:00Z"
    return s


# ── Fixture: mock langfuse.client ──


@pytest.fixture(autouse=True)
def mock_langfuse_client() -> MagicMock:
    """Mock langfuse.Langfuse and .api namespace."""
    # Mock the Langfuse client
    client_cls = MagicMock(name="LangfuseClass")
    client_instance = MagicMock(name="LangfuseInstance")
    client_cls.return_value = client_instance

    # Mock .api.trace / .api.session / .api.metrics
    api = MagicMock(name="api")
    client_instance.api = api
    api.trace = MagicMock(name="trace_api")
    api.session = MagicMock(name="session_api")
    api.metrics = MagicMock(name="metrics_api")

    pkg = MagicMock(name="langfuse_pkg")
    pkg.Langfuse = client_cls

    for mod in ["expflow.langfuse", "langfuse"]:
        if mod in sys.modules:
            del sys.modules[mod]

    with patch.dict("sys.modules", {"langfuse": pkg}):
        yield pkg

    for mod in ["expflow.langfuse", "langfuse"]:
        if mod in sys.modules:
            del sys.modules[mod]


# ══════════════════════════════════════════════════════════════
# list_traces
# ══════════════════════════════════════════════════════════════


class TestListTraces:
    """list_traces() — list traces with optional filters."""

    def test_list_traces_no_filter(self, mock_langfuse_client):
        """list_traces returns serialized list."""
        mock_langfuse_client.Langfuse.return_value.api.trace.list.return_value = [
            _make_mock_trace("t1", "run_a", "user1", tags=["test"]),
            _make_mock_trace("t2", "run_b", "user2", tags=[]),
        ]

        from expflow.langfuse import list_traces

        result = list_traces()

        assert len(result) == 2
        assert result[0]["id"] == "t1"
        assert result[0]["name"] == "run_a"
        assert result[0]["tags"] == ["test"]
        assert result[1]["user_id"] == "user2"

    def test_list_traces_empty(self, mock_langfuse_client):
        """Empty traces returns empty list."""
        mock_langfuse_client.Langfuse.return_value.api.trace.list.return_value = []

        from expflow.langfuse import list_traces

        assert list_traces() == []

    def test_list_traces_with_filters(self, mock_langfuse_client):
        """Filters are passed through."""
        mock_langfuse_client.Langfuse.return_value.api.trace.list.return_value = []

        from expflow.langfuse import list_traces

        list_traces(limit=50, user_id="u1", tags=["prod"], session_id="s1")

        mock_langfuse_client.Langfuse.return_value.api.trace.list.assert_called_with(
            limit=50,
            user_id="u1",
            tags=["prod"],
            session_id="s1",
        )


# ══════════════════════════════════════════════════════════════
# get_trace
# ══════════════════════════════════════════════════════════════


class TestGetTrace:
    """get_trace() — get single trace."""

    def test_get_trace_returns_serialized(self, mock_langfuse_client):
        mock_langfuse_client.Langfuse.return_value.api.trace.get.return_value = _make_mock_trace(
            "t1", "infer", "u1", cost=0.12
        )

        from expflow.langfuse import get_trace

        result = get_trace("t1")

        assert result["id"] == "t1"
        assert result["name"] == "infer"
        assert result["user_id"] == "u1"

    def test_get_trace_passes_id(self, mock_langfuse_client):
        from expflow.langfuse import get_trace

        get_trace("t1")
        mock_langfuse_client.Langfuse.return_value.api.trace.get.assert_called_with("t1")


# ══════════════════════════════════════════════════════════════
# get_trace_cost
# ══════════════════════════════════════════════════════════════


class TestGetTraceCost:
    """get_trace_cost() — aggregate cost for a trace."""

    def test_get_trace_cost_returns_cost(self, mock_langfuse_client):
        trace = _make_mock_trace("t1", cost=0.05)
        mock_langfuse_client.Langfuse.return_value.api.trace.get.return_value = trace

        from expflow.langfuse import get_trace_cost

        result = get_trace_cost("t1")

        assert result["trace_id"] == "t1"
        assert result["total_cost"] == 0.05
        assert "usage" in result


# ══════════════════════════════════════════════════════════════
# Sessions
# ══════════════════════════════════════════════════════════════


class TestSessions:
    """list_sessions() and get_session()."""

    def test_list_sessions(self, mock_langfuse_client):
        mock_langfuse_client.Langfuse.return_value.api.session.list.return_value = [
            _make_mock_session("s1", "u1"),
            _make_mock_session("s2", "u2"),
        ]

        from expflow.langfuse import list_sessions

        result = list_sessions()

        assert len(result) == 2
        assert result[0]["id"] == "s1"
        assert result[1]["user_id"] == "u2"

    def test_list_sessions_empty(self, mock_langfuse_client):
        mock_langfuse_client.Langfuse.return_value.api.session.list.return_value = []

        from expflow.langfuse import list_sessions

        assert list_sessions() == []

    def test_get_session(self, mock_langfuse_client):
        mock_langfuse_client.Langfuse.return_value.api.session.get.return_value = (
            _make_mock_session("s1", "u1")
        )

        from expflow.langfuse import get_session

        result = get_session("s1")

        assert result["id"] == "s1"
        assert result["user_id"] == "u1"


# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════


class TestMetrics:
    """get_metrics() — aggregated usage/cost metrics."""

    def test_get_metrics_returns_dict(self, mock_langfuse_client):
        mock_langfuse_client.Langfuse.return_value.api.metrics.get.return_value = {
            "total_cost": 12.50,
            "total_traces": 100,
            "total_observations": 500,
        }

        from expflow.langfuse import get_metrics

        result = get_metrics()

        assert result["total_cost"] == 12.50
        assert result["total_traces"] == 100
