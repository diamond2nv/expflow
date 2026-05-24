"""Scan and mask wiki/skills directories for competition-specific content.

Two modes:
  - scan_directory(): audit mode — find violations without changing files
  - apply_mask(): create cleansed working copy with masked content

Usage:
    from expflow_pde.competition.mask.scanner import scan_directory, apply_mask
    from expflow_pde.competition.mask.rules import ALL_RULES

    results = scan_directory(Path("~/wiki"), ALL_RULES)
    manifest = apply_mask(Path("~/wiki"), Path("~/.competition/wiki"), ALL_RULES)
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .rules import MaskRule


def scan_file(path: Path | str, rules: list[MaskRule]) -> dict[str, Any]:
    """Scan a single file against all rules.

    Args:
        path: File path (Path or string).
        rules: List of MaskRule instances.

    Returns:
        Dict with path, violations (list), violation_count, and masked_content
        (only if violations found, else None).
    """
    path = Path(path) if isinstance(path, str) else path
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"path": str(path), "error": str(e), "violations": []}

    all_violations: list[str] = []
    masked = content
    for rule in rules:
        masked, violations = rule.apply(masked)
        all_violations.extend(violations)

    return {
        "path": str(path),
        "violations": all_violations,
        "violation_count": len(all_violations),
        "masked_content": masked if all_violations else None,
    }


def scan_directory(
    dir_path: Path,
    rules: list[MaskRule],
    file_globs: list[str] | None = None,
    max_files: int = 5000,
) -> list[dict[str, Any]]:
    """Recursively scan a directory for competition-specific content.

    Args:
        dir_path: Directory to scan.
        rules: List of MaskRule instances.
        file_globs: File patterns to scan (default: .md, .py, .yaml, .toml, .sh, .txt).
        max_files: Maximum number of files to scan (safety limit).

    Returns:
        List of scan result dicts (only files with violations or errors).
    """
    if file_globs is None:
        file_globs = ["*.md", "*.py", "*.yaml", "*.toml", "*.sh", "*.txt"]

    results: list[dict[str, Any]] = []
    count = 0
    for glob in file_globs:
        for fpath in sorted(dir_path.rglob(glob)):
            # Skip dot-directories and __pycache__
            parts = fpath.relative_to(dir_path).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            if count >= max_files:
                break
            result = scan_file(fpath, rules)
            if result.get("violations") or result.get("error"):
                results.append(result)
            count += 1
        if count >= max_files:
            break

    return results


def apply_mask(
    source_dir: Path,
    output_dir: Path,
    rules: list[MaskRule],
    file_globs: list[str] | None = None,
) -> dict[str, Any]:
    """Create a cleansed copy of source_dir at output_dir.

    Files with no violations are copied as-is.
    Files with violations are written with content masked.

    Args:
        source_dir: Source directory (e.g. ~/wiki).
        output_dir: Output directory (e.g. ~/.competition/wiki).
        rules: List of MaskRule instances.
        file_globs: File patterns to scan and mask.

    Returns:
        Manifest dict with stats.
    """
    if file_globs is None:
        file_globs = ["*.md", "*.py", "*.yaml", "*.toml", "*.sh", "*.txt"]

    # Remove existing output if present
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Copy all files first (preserving structure)
    _copy_tree(source_dir, output_dir)

    # Scan and mask
    all_results = scan_directory(output_dir, rules, file_globs)
    masked_count = 0
    for r in all_results:
        if r.get("masked_content") and r.get("path"):
            fpath = Path(r["path"])
            try:
                fpath.write_text(r["masked_content"], encoding="utf-8")
                masked_count += 1
            except OSError:
                pass

    manifest: dict[str, Any] = {
        "source": str(source_dir),
        "output": str(output_dir),
        "files_with_violations": len(all_results),
        "total_violations": sum(r["violation_count"] for r in all_results),
        "files_masked": masked_count,
        "violation_summary": _summarize_violations(all_results),
    }

    # Write manifest
    manifest_path = output_dir / "mask_manifest.json"
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
    except OSError:
        pass

    return manifest


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree, preserving structure, excluding hidden dirs."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item.name
        if s.startswith(".") or s == "__pycache__":
            continue
        sp = src / s
        dp = dst / s
        if sp.is_dir():
            _copy_tree(sp, dp)
        else:
            try:
                shutil.copy2(sp, dp)
            except (OSError, shutil.Error):
                pass


def _summarize_violations(results: list[dict]) -> dict[str, int]:
    """Count violations by rule name."""
    counts: Counter = Counter()
    for r in results:
        for v in r.get("violations", []):
            rule_name = v.split(":")[0] if ":" in v else "unknown"
            counts[rule_name] += 1
    return dict(counts)
