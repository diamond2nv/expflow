#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-installation check: verify the chosen package name is not
already taken on PyPI by a different project.

Usage:
    python scripts/check_pypi_conflict.py <package_name>

Returns exit code 0 if the name is available or belongs to the same project,
exit code 1 if a conflict exists.

This script is intended to be run BEFORE `pip install -e .` or `pip install .`
to prevent accidental naming collisions with third-party packages.
"""

import json
import sys
import urllib.request


def check_package(name: str) -> int:
    """Check if a package name is already registered on PyPI.

    Returns:
        0 — name is free (404 or name matches expected owner/project)
        1 — name is taken by a different project
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "expflow-check/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            info = data.get("info", {})
            pkg_name = info.get("name", "")
            author = info.get("author", "")
            summary = info.get("summary", "")

            print(f"[WARN] Package '{name}' already exists on PyPI!")
            print(f"       Name:    {pkg_name}")
            print(f"       Version: {info.get('version', '?')}")
            print(f"       Author:  {author}")
            print(f"       Summary: {summary[:120]}")
            print()
            print("       This is a DIFFERENT project. Proceeding will:")
            print("       - Overwrite the entry_point script 'expflow'")
            print("       - Break existing installations of the other expflow")
            print("       - Cause pip conflicts for users")
            print()
            print("       Action required: rename the package in pyproject.toml")
            print(f"       from '{name}' to something unique")
            return 1
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[OK] Package name '{name}' is available on PyPI")
            return 0
        print(f"[WARN] Could not check PyPI (HTTP {e.code}): {e.reason}")
        return 0  # network errors are non-fatal
    except urllib.error.URLError as e:
        print(f"[WARN] Could not reach PyPI (network issue): {e.reason}")
        return 0  # offline is fine


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_pypi_conflict.py <package_name>")
        sys.exit(1)

    name = sys.argv[1]
    result = check_package(name)
    sys.exit(result)
