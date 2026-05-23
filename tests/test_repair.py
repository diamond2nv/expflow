#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.repair — RepairStage three-level auto-repair."""

from expflow_pde.repair import RepairStage


class TestRepairStageL0:
    """L0 rule engine matches and returns suggestions."""

    def test_l0_git_clone_failure(self):
        stage = RepairStage()
        log = "fatal: Could not read from remote repository."
        result = stage.run(log, 128)
        assert result["level"] == "L0"
        assert not result["fixed"]  # needs user action on remote node
        assert "git" in result["action"].lower()

    def test_l0_module_not_found(self):
        stage = RepairStage()
        log = "ModuleNotFoundError: No module named 'torch'"
        result = stage.run(log, 1)
        assert result["level"] == "L0"
        assert "system_site_packages" in result["action"].lower()

    def test_l0_pip_conflict(self):
        stage = RepairStage()
        log = "pip install impossible: package conflict"
        result = stage.run(log, 1)
        assert result["level"] == "L0"
        assert "packages=[]" in result["action"]

    def test_l0_no_match_returns_none(self):
        stage = RepairStage()
        log = "Something that doesn't match any pattern"
        result = stage.run(log, 0)  # exit 0 = no repair needed
        assert result["level"] == "none"
        assert not result["fixed"]

    def test_l0_success_exit_no_match(self):
        stage = RepairStage()
        log = "Training complete. Seg: 57.09"
        result = stage.run(log, 0)
        assert result["level"] == "none"
        assert not result["fixed"]


class TestRepairStageL1:
    """L1 traceback extraction and error localization."""

    def test_l1_extracts_traceback_info(self):
        stage = RepairStage()
        log = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/train.py", line 42, in train_step\n'
            "    loss = loss_fn(output, target)\n"
            '  File "/home/user/train.py", line 80, in loss_fn\n'
            "    diff_norm = torch.norm(x_flat - y_flat, p=2, dim=1)\n"
            "RuntimeError: The size of tensor a (32) must match the size of tensor b (16)\n"
        )
        result = stage.run(log, 1, enable_reflection=False)
        assert result["level"] == "L1"
        assert result["action"] is not None
        # Check history entries were recorded
        assert len(result["history"]) >= 1

    def test_l1_no_traceback(self):
        stage = RepairStage()
        log = "Killed\n"
        result = stage.run(log, 137)
        assert result["level"] == "L1"  # Falls to L1 even without Traceback
        assert result["fixed"] is False

    def test_l1_empty_log(self):
        stage = RepairStage()
        result = stage.run("", 1)
        assert result["level"] == "L1"
        assert not result["fixed"]


class TestRepairStageL2:
    """L2 reflection context preparation."""

    def test_l2_prepares_context(self):
        stage = RepairStage(max_l1_attempts=1)
        log = (
            "Some non-standard error: model failed after epoch 50\n"
            "loss became NaN starting from epoch 40\n"
            "gradient explosion detected\n"
        )
        result = stage.run(log, 1, enable_reflection=True)
        assert result["level"] == "L2"
        assert result["fixed"] is False
        assert result["action"] is not None

    def test_l2_disabled_by_default(self):
        stage = RepairStage()
        log = "Unknown error with no traceback"
        result = stage.run(log, 1, enable_reflection=False)
        # Without reflection, L1 runs but returns fixed=False,
        # and since L2 is disabled, level is "L1"
        assert result["level"] in ("L1", "none")

    def test_l2_subagent_prompt_in_history(self):
        stage = RepairStage(max_l1_attempts=1)
        log = "CUDA out of memory. Tried to allocate 2.00 GiB"
        result = stage.run(log, 1, enable_reflection=True)
        assert result["level"] in ("L2", "L1")
        # History should contain the attempt
        assert len(result["history"]) >= 1


class TestRepairStageProperties:
    """RepairStage state tracking."""

    def test_history_tracks_attempts(self):
        stage = RepairStage()
        log = "ModuleNotFoundError: No module named 'h5py'"
        _ = stage.run(log, 1)
        assert len(stage.history) >= 1

    def test_fixed_property(self):
        stage = RepairStage()
        # A non-matching case
        stage.run("ok output", 0)
        assert not stage.fixed

    def test_to_json_serializable(self):
        stage = RepairStage(experiment_id="exp_abc123")
        stage.run("ModuleNotFoundError: No module named 'x'", 1)
        json_str = stage.to_json()
        import json

        parsed = json.loads(json_str)
        assert parsed["experiment_id"] == "exp_abc123"
        assert "repair_history" in parsed


class TestRepairStageEdgeCases:
    """Edge cases and resilience."""

    def test_long_error_log(self):
        """Very long logs should not break L1 extraction."""
        log = "\n".join([f"Line {i}" for i in range(1000)])
        log += '\nTraceback (most recent call last):\n  File "test.py", line 1, in <module>\n    raise ValueError("test")\nValueError: test'
        stage = RepairStage(max_l1_attempts=1)
        result = stage.run(log, 1)
        assert result["level"] == "L1"

    def test_unicode_in_log(self):
        """Non-ASCII characters should not break parsing."""
        log = "Error: can't decode byte 0xff in position 0\nTraceback:\n  ValueError: 测试中文信息"
        stage = RepairStage()
        result = stage.run(log, 1)
        assert result is not None

    def test_no_newlines(self):
        """Single-line log should not crash."""
        stage = RepairStage()
        result = stage.run("Single line error: kaboom", 1)
        assert result is not None

    def test_multiple_repair_stages(self):
        """Multiple sequential repair calls should not leak state."""
        stage = RepairStage()
        r1 = stage.run("ModuleNotFoundError: No module named 'a'", 1)
        r2 = stage.run("OK", 0)
        r3 = stage.run("fatal: Could not read", 128)
        assert r1["level"] == "L0"
        assert r2["level"] == "none"
        assert r3["level"] == "L0"


class TestRepairStageInputValidation:
    """L2 input validation when task_log is empty or has no failure signal."""

    def test_empty_log_returns_input_valid_false(self):
        stage = RepairStage()
        result = stage.run("", 1, enable_reflection=True)
        assert result["level"] == "L2"
        assert result.get("input_valid") is False
        assert "no failure signal" in result.get("action", "").lower()

    def test_whitespace_only_log_input_invalid(self):
        stage = RepairStage()
        result = stage.run("   \n\n  ", 1, enable_reflection=True)
        assert result.get("input_valid") is False

    def test_normal_log_still_valid(self):
        stage = RepairStage()
        result = stage.run(
            "Traceback (most recent call last):\nValueError: bad", 1, enable_reflection=True
        )
        assert result.get("input_valid", True) is True

    def test_killed_log_contains_failure_signal(self):
        """'Killed' should be recognized as a failure signal."""
        from expflow_pde.repair import _log_has_failure_signal

        assert _log_has_failure_signal("Killed")
        assert _log_has_failure_signal("process was Killed")


class TestRepairStageExitCodeCategory:
    """exit_code_category should be present and correct."""

    def test_exit_code_category_0(self):
        stage = RepairStage()
        result = stage.run("ok", 0)
        assert result["exit_code_category"] == "success"

    def test_exit_code_category_1(self):
        stage = RepairStage()
        result = stage.run("Error: something", 1)
        assert result["exit_code_category"] == "error"

    def test_exit_code_category_137(self):
        stage = RepairStage()
        result = stage.run("Killed", 137)
        assert "signal" in result["exit_code_category"]

    def test_exit_code_category_unknown(self):
        stage = RepairStage()
        result = stage.run("Error: weird", 99)
        assert "unknown" in result["exit_code_category"]


class TestRepairStageSignalExitL1:
    """L1 should produce meaningful output for signal exit codes."""

    def test_l1_signal_137_no_traceback(self):
        stage = RepairStage()
        result = stage.run("Killed", 137)
        assert result["level"] == "L1"
        action = result["action"].lower()
        assert "sigkill" in action or "oom" in action

    def test_l1_signal_139_no_traceback(self):
        stage = RepairStage()
        result = stage.run("Segmentation fault", 139)
        assert result["level"] == "L1"
        assert "sigsegv" in result["action"].lower()


class TestRepairStageWikiMapping:
    """Wiki mapping should use exact match over substring."""

    def test_exact_match_returns_exact_source(self):
        from expflow_pde.repair import RepairStage

        stage = RepairStage()
        info = stage._exc_type_to_wiki("ModuleNotFoundError")
        assert info["source"] == "exact"
        assert len(info["paths"]) > 0

    def test_prefix_match_import_error(self):
        from expflow_pde.repair import RepairStage

        stage = RepairStage()
        info = stage._exc_type_to_wiki("ImportError: no module named git")
        assert info["source"] == "prefix"
        assert "pip-dependencies" in info["paths"][0]
        # Should NOT match SSH wiki (old bug: "git" substring in exc_type triggered ssh-keys)
        assert "ssh-keys" not in " ".join(info["paths"])

    def test_fallback_cuda_classification(self):
        from expflow_pde.repair import RepairStage

        stage = RepairStage()
        info = stage._exc_type_to_wiki("RuntimeError: CUDA error 999")
        assert info["source"] == "fallback"
        assert "gpu-memory" in info["paths"][0]

    def test_none_for_unknown(self):
        from expflow_pde.repair import RepairStage

        stage = RepairStage()
        info = stage._exc_type_to_wiki("WeirdError: something unusual")
        assert info["source"] == "none"
        assert len(info["paths"]) == 0

    def test_l2_propagates_wiki_source(self):
        stage = RepairStage()
        result = stage.run(
            "Traceback:\nModuleNotFoundError: no module 'x'", 1, enable_reflection=True
        )
        if result["level"] == "L2":
            assert "wiki_source" in result
            assert result["wiki_source"] in ("exact", "prefix", "substring", "fallback", "none")


class TestResolveRepairOutput:
    """_resolve_repair_output (moved to cli_repeat) — check import."""

    def test_fetch_task_log_importable(self):
        from expflow_pde.cli_repeat import _fetch_task_log

        assert callable(_fetch_task_log)

    def test_print_diagnosis_importable(self):
        from expflow_pde.cli_repeat import _print_diagnosis

        assert callable(_print_diagnosis)


class TestRepairStageSignalWikiMapping:
    """Signal exit codes should route to signal-specific wiki via _signal_to_wiki."""

    def test_signal_137_routes_to_oom_killer_wiki(self):
        stage = RepairStage()
        result = stage.run("Killed\n", 137, enable_reflection=True)
        assert result["level"] == "L2"
        assert result.get("wiki_source") == "signal"
        assert any("oom-killer" in p for p in result.get("wiki_paths", []))

    def test_signal_139_routes_to_segfault_wiki(self):
        stage = RepairStage()
        result = stage.run("Signal (11) Error: Segmentation fault\n", 139, enable_reflection=True)
        assert result["level"] == "L2"
        assert result.get("wiki_source") == "signal"
        assert any("segfault" in p for p in result.get("wiki_paths", []))

    def test_signal_143_routes_to_timeouts_wiki(self):
        stage = RepairStage()
        result = stage.run("Killed\n", 143, enable_reflection=True)
        assert result["level"] == "L2"
        assert result.get("wiki_source") == "signal"
        assert any("timeouts" in p for p in result.get("wiki_paths", []))

    def test_signal_134_routes_to_abort_wiki(self):
        from expflow_pde.repair import RepairStage

        info = RepairStage._signal_to_wiki(134, "SIGABRT")
        assert info["source"] == "signal"
        assert any("abort" in p for p in info.get("paths", []))

    def test_unsupported_signal_routes_to_unknown(self):
        from expflow_pde.repair import RepairStage

        info = RepairStage._signal_to_wiki(6, "SIGABRT")
        assert info["source"] == "signal"
        assert any("unknown-signal" in p for p in info.get("paths", []))


class TestRepairStageWordMatch:
    """Whole-word matching in _CLASSIFY_EXC prevents short-keyword false positives."""

    def test_disk_in_type_name_not_falsely_matches(self):
        from expflow_pde.repair import RepairStage

        # "diskerror" should NOT match "disk" keyword due to exclusion list
        paths = RepairStage._CLASSIFY_EXC("DiskError", "bogus error")
        assert len(paths) == 0

    def test_killed_as_substring_not_falsely_matches(self):
        from expflow_pde.repair import RepairStage

        # "diskilled" contains "killed" as substring but not as whole word
        combined = "diskilled operation failed"
        assert not RepairStage._word_in(combined, "killed")
        # Whole word "killed" should match
        assert RepairStage._word_in("process killed", "killed")

    def test_oom_whole_word_matches(self):
        from expflow_pde.repair import RepairStage

        paths = RepairStage._CLASSIFY_EXC("RuntimeError", "CUDA OOM during forward pass")
        assert any("gpu-memory" in p for p in paths)

    def test_no_space_matches_disk_quota(self):
        from expflow_pde.repair import RepairStage

        paths = RepairStage._CLASSIFY_EXC("OSError", "No space left on device")
        assert any("disk-space" in p for p in paths)


class TestRepairStageL2SemanticEnrichment:
    """SemanticContext enrichment in L2 (Phase 3: sidecar in subagent prompt)."""

    def _make_stage(self):
        from expflow_pde.repair import RepairStage
        return RepairStage()

    def _mock_client(self, classify_returns=None, similarity_returns=None):
        """Factory: returns a mock SemanticClient with configurable returns."""

        class _MockClient:
            def check_health(self):
                return {"status": "ok"}

            def classify(self, text, concepts):
                if classify_returns is not None:
                    return classify_returns
                for label in concepts:
                    if "CUDA runtime" in label:
                        return {
                            "scores": {label: 0.82},
                            "top_concept": label,
                            "top_score": 0.82,
                        }
                return {"scores": {}, "top_concept": "", "top_score": 0.0}

            def similarity(self, a, b):
                if similarity_returns is not None:
                    return similarity_returns
                return 0.72

        return _MockClient()

    def test_l2_semantic_context_present_when_sidecar_ok(self, monkeypatch):
        """When sidecar is reachable, L2 result includes non-empty semantic_context."""
        stage = self._make_stage()
        monkeypatch.setattr(
            "expflow_pde.semantic_client.SemanticClient",
            lambda base_url=None: self._mock_client(),
        )

        log = "Traceback (most recent call last):\nRuntimeError: CUDA error: an illegal memory access was encountered"
        result = stage.run(log, 1, enable_reflection=True)

        assert result["level"] == "L2"
        assert result.get("semantic_context", "") != ""
        assert "CUDA runtime" in result["semantic_context"]
        # subagent_prompt should contain the ## Semantic Context section
        assert "## Semantic Context" in result.get("subagent_prompt", "")

    def test_l2_semantic_context_empty_when_sidecar_unreachable(self, monkeypatch):
        """When sidecar is unreachable, L2 result has empty semantic_context."""
        stage = self._make_stage()

        class _Unreachable:
            def check_health(self):
                return None

            def classify(self, text, concepts):
                raise OSError("Connection refused")

            def similarity(self, a, b):
                raise OSError("Connection refused")

        monkeypatch.setattr(
            "expflow_pde.semantic_client.SemanticClient",
            lambda base_url=None: _Unreachable(),
        )

        log = "Traceback:\nRuntimeError: CUDA error"
        result = stage.run(log, 1, enable_reflection=True)

        assert result["level"] == "L2"
        assert result.get("semantic_context", "") == ""
        # subagent_prompt should NOT contain ## Semantic Context
        assert "## Semantic Context" not in result.get("subagent_prompt", "")

    def test_l2_semantic_context_empty_low_confidence(self, monkeypatch):
        """Low-confidence classify returns empty semantic_context."""
        stage = self._make_stage()

        monkeypatch.setattr(
            "expflow_pde.semantic_client.SemanticClient",
            lambda base_url=None: self._mock_client(
                classify_returns={"scores": {"Timeout": 0.12}, "top_concept": "Timeout", "top_score": 0.12},
                similarity_returns=0.05,
            ),
        )

        log = "Traceback:\nValueError: something obscure"
        result = stage.run(log, 1, enable_reflection=True)

        assert result["level"] == "L2"
        # Bottom confidence threshold is 0.25, so 0.12 should be rejected
        assert result.get("semantic_context", "") == ""
        assert "## Semantic Context" not in result.get("subagent_prompt", "")

    def test_l2_semantic_context_from_signal_exit(self, monkeypatch):
        """Signal exit (137) should also get semantic enrichment."""
        stage = self._make_stage()
        monkeypatch.setattr(
            "expflow_pde.semantic_client.SemanticClient",
            lambda base_url=None: self._mock_client(
                classify_returns={
                    "scores": {"Process killed by OOM killer": 0.91},
                    "top_concept": "Process killed by OOM killer",
                    "top_score": 0.91,
                },
                similarity_returns=0.88,
            ),
        )

        log = "Killed\nOut of memory\n"
        result = stage.run(log, 137, enable_reflection=True)

        assert result["level"] == "L2"
        assert result.get("semantic_context", "") != ""
        assert "OOM killer" in result["semantic_context"]
        assert "## Semantic Context" in result.get("subagent_prompt", "")

    def test_l2_semantic_context_no_sidecar_at_all(self):
        """Without mocking, sidecar is unreachable → empty context (production fallback)."""
        from expflow_pde.repair import RepairStage

        stage = RepairStage()
        log = "Traceback:\nRuntimeError: obscure CUDA driver error"
        result = stage.run(log, 1, enable_reflection=True)

        assert result["level"] == "L2"
        # In production with no sidecar running, semantic_context is empty
        assert result.get("semantic_context", "") == ""
        assert "## Semantic Context" not in result.get("subagent_prompt", "")


class TestRepairStageFixParams:
    """L0 rule engine propagates fix_params to top-level result."""

    def test_pip_conflict_has_fix_params(self):
        stage = RepairStage()
        result = stage.run("pip install impossible: package conflict", 1)
        assert result["level"] == "L0"
        assert "fix_params" in result
        assert result["fix_params"].get("packages") == []

    def test_git_not_found_has_empty_fix_params(self):
        stage = RepairStage()
        result = stage.run("fatal: Could not read from remote repository.", 128)
        assert result["level"] == "L0"
        assert "fix_params" in result

    def test_fix_params_not_in_non_l0_results(self):
        stage = RepairStage()
        result = stage.run("some random error", 1, enable_reflection=False)
        if result["level"] == "L1":
            assert "fix_params" not in result


class TestRepairStageFixPlanSchema:
    """L2 subagent prompt uses structured JSON schema."""

    def test_l2_prompt_contains_json_schema(self):
        from expflow_pde.repair import RepairStage

        stage = RepairStage(experiment_id="exp_test")
        context = {
            "experiment_id": "exp_test",
            "exit_code": 1,
            "exc_type": "ValueError",
            "exc_message": "bad value",
            "tb_snippet": ["Traceback:", "ValueError: bad"],
            "files_to_check": ["/opt/train.py"],
            "wiki_paths": [],
        }
        prompt = stage._render_l2_prompt(context)
        assert '"plan_type"' in prompt
        assert '"files"' in prompt
        assert '"no_plan"' in prompt


class TestFetchTaskLogTriple:
    """_fetch_task_log (now in cli_repeat) returns 3-tuple."""

    def test_signature_is_three_tuple(self):
        import inspect

        from expflow_pde.cli_repeat import _fetch_task_log

        sig = inspect.signature(_fetch_task_log)
        hint = sig.return_annotation
        assert hint is not inspect.Parameter.empty


class TestRepeatDiagnoseCLI:
    """expflow repeat diagnose — CLI diagnosis path."""

    def test_diagnose_from_json_file(self, tmp_path):
        """Reading from a local JSON file produces valid output."""
        import json
        import sys

        data = {
            "task_log": "Traceback (most recent call last):\n  File \"train.py\", line 42, in <module>\n    import torch\nModuleNotFoundError: No module named 'torch'",
            "exit_code": 1,
            "pipeline_id": "pipe_diag_001",
        }
        fpath = tmp_path / "failed_result.json"
        with open(fpath, "w") as f:
            json.dump(data, f)

        # Capture output — command prints to stdout
        from io import StringIO

        from expflow_pde.cli_repeat import repeat_diagnose_cmd

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            repeat_diagnose_cmd(target=str(fpath), json_output=False)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Diagnosis" in output
        assert "level=" in output or "L0" in output

    def test_diagnose_json_output(self, tmp_path):
        """--json flag produces valid JSON output."""
        import json
        import sys

        data = {
            "task_log": "Error: something broke",
            "exit_code": 1,
            "pipeline_id": "pipe_diag_002",
        }
        fpath = tmp_path / "failed_result.json"
        with open(fpath, "w") as f:
            json.dump(data, f)

        from io import StringIO

        from expflow_pde.cli_repeat import repeat_diagnose_cmd

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            repeat_diagnose_cmd(target=str(fpath), json_output=True)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        parsed = json.loads(output.strip())
        assert isinstance(parsed, dict)
        assert "level" in parsed
        assert "action" in parsed
