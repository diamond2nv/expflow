#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow: experiment workflow orchestration toolkit for PDEBench/Agentic4Sci."""

__version__ = "0.1.0"

# Config is imported lazily — don't import it here to avoid dependency chain
# at import time. Users call `from expflow.config import load_config` explicitly.
__all__ = ["__version__"]
