#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test for SemanticClient — all mocked HTTP, no real sidecar needed."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from expflow_pde.semantic_client import SemanticClient


class TestSemanticClient(unittest.TestCase):
    """Test SemanticClient with mocked HTTP calls."""

    def setUp(self):
        self.client = SemanticClient(base_url="http://test:9999")

    # ── check_health ──

    @patch("urllib.request.urlopen")
    def test_check_health_ok(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"status": "ok", "model_loaded": True, "device": "cpu"}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = self.client.check_health()
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["model_loaded"])

    @patch("urllib.request.urlopen")
    def test_check_health_unreachable(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = self.client.check_health()
        self.assertIsNone(result)

    # ── embed ──

    @patch("urllib.request.urlopen")
    def test_embed_ok(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"embedding": [0.1, 0.2, 0.3], "dimension": 3, "device": "cpu"}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = self.client.embed("test text")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.1)

    @patch("urllib.request.urlopen")
    def test_embed_unreachable(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = self.client.embed("test text")
        self.assertIsNone(result)

    # ── similarity ──

    @patch("urllib.request.urlopen")
    def test_similarity_ok(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"similarity": 0.85, "device": "cpu"}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = self.client.similarity("text a", "text b")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.85)

    @patch("urllib.request.urlopen")
    def test_similarity_unreachable(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = self.client.similarity("text a", "text b")
        self.assertIsNone(result)

    # ── classify ──

    @patch("urllib.request.urlopen")
    def test_classify_ok(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "scores": {"concept_a": 0.9, "concept_b": 0.3},
                "top_concept": "concept_a",
                "top_score": 0.9,
                "device": "cpu",
            }
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = self.client.classify(
            "some text", ["concept_a", "concept_b"]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["top_concept"], "concept_a")
        self.assertAlmostEqual(result["top_score"], 0.9)

    @patch("urllib.request.urlopen")
    def test_classify_unreachable(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = self.client.classify(
            "some text", ["concept_a", "concept_b"]
        )
        self.assertIsNone(result)

    # ── POST body verification ──

    @patch("urllib.request.urlopen")
    def test_similarity_sends_correct_body(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"similarity": 0.5, "device": "cpu"}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        self.client.similarity("foo", "bar")

        # Verify the request was constructed correctly
        args, _ = mock_urlopen.call_args
        req = args[0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["text_a"], "foo")
        self.assertEqual(body["text_b"], "bar")
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")

    @patch("urllib.request.urlopen")
    def test_timeout_on_slow_service(self, mock_urlopen):
        """SemanticClient should not hang forever on unreachable service."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("timed out")

        result = self.client.embed("slow test")
        self.assertIsNone(result)
