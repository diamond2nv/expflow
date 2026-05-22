#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semantic_client.py — HTTP client for semantic similarity sidecar.

Zero extra dependencies (stdlib only: urllib).
Connects to hfpapers-crawler's semantic_service sidecar at the configured URL.

Usage:
    client = SemanticClient()
    sim = client.similarity("text a", "text b")
    scores = client.classify("text", ["concept1", "concept2"])
    embedding = client.embed("text")
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("expflow_pde.semantic_client")

_DEFAULT_BASE_URL = os.environ.get(
    "EXPFLOW_SEMANTIC_URL", "http://127.0.0.1:8765"
)


class SemanticClient:
    """Thin HTTP client for the semantic similarity sidecar.

    Connects to the sentence-transformers service. All methods
    return None on connection error (graceful degradation).
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    # ── Public API ──

    def check_health(self) -> dict[str, Any] | None:
        """Check if the sidecar is running and healthy.

        Returns health dict or None if unreachable.
        """
        return self._get("/health")

    def embed(self, text: str) -> list[float] | None:
        """Compute embedding vector for text.

        Returns normalized embedding as list of floats, or None on error.
        """
        result = self._post("/embed", {"text": text})
        if result is None:
            return None
        return result.get("embedding")

    def similarity(self, text_a: str, text_b: str) -> float | None:
        """Compute cosine similarity between two texts.

        Returns 0.0-1.0 float, or None on error.
        """
        result = self._post(
            "/similarity", {"text_a": text_a, "text_b": text_b}
        )
        if result is None:
            return None
        return result.get("similarity")

    def classify(
        self, text: str, concepts: list[str]
    ) -> dict[str, Any] | None:
        """Classify text against reference concepts.

        Returns dict with 'scores', 'top_concept', 'top_score',
        or None on error.
        """
        return self._post("/classify", {"text": text, "concepts": concepts})

    # ── Internal HTTP helpers ──

    def _get(self, path: str) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError) as e:
            logger.debug("GET %s failed: %s", url, e)
            return None

    def _post(
        self, path: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError) as e:
            logger.debug("POST %s failed: %s", url, e)
            return None
