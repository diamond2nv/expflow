#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal test: verify clearml agent can run a task on 5090."""

import socket
import os
import sys

# 检查 clearml
try:
    from clearml import Task
    task = Task.init(project_name="PDEBench Task1", task_name="5090_hello_test")
    task.connect({"hostname": socket.gethostname(), "user": os.environ.get("USER", "?")})
    print(f"[OK] Hostname: {socket.gethostname()}")
    print(f"[OK] Python: {sys.executable}")
    print(f"[OK] CWD: {os.getcwd()}")
    print(f"[OK] USER: {os.environ.get('USER', '?')}")
    print(f"[OK] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    cache_dir = os.environ.get("CLEARML_CACHE_DIR", "not set")
    print(f"[OK] CLEARML_CACHE_DIR: {cache_dir}")
    if cache_dir != "not set" and os.path.isdir(cache_dir):
        print(f"[OK] Cache dir exists: {os.listdir(cache_dir)[:5]}")
    print(f"[OK] Task {task.id} completed successfully!")
except Exception as e:
    print(f"[FAIL] {e}")
    raise
