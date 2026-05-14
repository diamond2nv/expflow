#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest conftest — global fixtures for expflow tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def reset_config_cache() -> Generator[None, None, None]:
    """Reset config cache before each test to avoid cross-test pollution."""
    from expflow import config

    config._config_cache.clear()
    config._env_cache = None
    yield
    config._config_cache.clear()
    config._env_cache = None


@pytest.fixture
def temp_workdir() -> Generator[Path, None, None]:
    """Temporary working directory with config.yaml."""
    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        os.chdir(tmp)
        yield tmp
        os.chdir(str(old_cwd))
