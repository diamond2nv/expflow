# expflow Phase 3: langfuse Integration — Implementation Plan

**Goal:** langfuse trace query, cost analysis, session search via expflow CLI

**Architecture:** `expflow/langfuse.py` wraps the langfuse Python SDK Public API client (`langfuse.Langfuse().api`)

**API surface:**
```python
def list_traces(limit=100, user_id=None, tags=None, session_id=None, ...) -> list[dict]
def get_trace(trace_id: str) -> dict
def get_trace_cost(trace_id: str) -> dict  # aggregated cost
def list_sessions(limit=100) -> list[dict]
def get_session(session_id: str) -> dict
def get_metrics(query: dict) -> dict  # aggregated usage/cost metrics
```

**CLI:**
```
expflow langfuse traces [--limit] [--user-id] [--tags] [--session-id]
expflow langfuse trace <id>
expflow langfuse trace-cost <id>
expflow langfuse sessions [--limit]
expflow langfuse session <id>
expflow langfuse metrics [--from] [--to]
```

**Mock strategy:** patch 'langfuse' in sys.modules, mock `Langfuse().api.trace.list/get`, etc.
