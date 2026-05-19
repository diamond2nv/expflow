#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config loading: YAML + .env merge, dot-separated access."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _find_config() -> str | None:
    """Search for config.yaml in CWD and parent dirs."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return str(candidate)
    return None


def _load_env() -> dict[str, str]:
    """Load .env file, return env vars that differ from current."""
    load_dotenv(override=False)
    return dict(os.environ)


_config_cache: dict[str, Any] = {}
_env_cache: dict[str, str] | None = None


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML config, merge .env overrides. Returns dict."""
    global _env_cache

    if path is None:
        path = _find_config() or "config.yaml"

    config_path = Path(path)
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    _config_cache.clear()
    _config_cache.update(cfg)
    _env_cache = _load_env()
    return cfg


def get(key: str, default: Any = None) -> Any:
    """Dot-separated config access, e.g. get('search.queries')."""
    if not _config_cache:
        load_config()

    parts = key.split(".")
    val: Any = _config_cache
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
    return val if val is not None else default
