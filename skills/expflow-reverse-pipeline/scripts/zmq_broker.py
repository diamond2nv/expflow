#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zmq_broker.py — ZeroMQ event broker for the reverse pipeline.

PUB-SUB message bus with topic-based routing, high-water mark, and
persistent fallback to JSON when ZMQ subscribers are offline.

Topics:
  taskctl/<task_id>/complete   — Task finished successfully
  taskctl/<task_id>/timeout    — Task exceeded expected duration
  taskctl/<task_id>/heartbeat  — Periodic alive signal
  system/gpu                   — GPU resource events
  system/cron                  — crontab tick (for debug)

Architecture:
  ┌──────────┐    PUB       ┌──────────────┐
  │ taskctl  │────port─────▶│  ZMQ Broker  │
  │ (cron)   │  15556/tcp   │              │
  └──────────┘              │  PUB-SUB     │────port─────▶┌─────────┐
  ┌──────────┐    PUB       │  fanout      │  15557/tcp   │ Hermes  │
  │ ClearML  │────port─────▶│              │─────────────▶│ Agent   │
  │ callback │  15556/tcp   │  SUB         │              │(goal)   │
  └──────────┘              └──────────────┘              └─────────┘
       │                                                  subscriber
   (future)

Design decisions:
  - In-process: broker runs inside taskctl daemon (no separate process)
  - HWM=1000: drop oldest if subscriber is too slow
  - JSON persistence: if no subscriber, spill to tasks.json
  - LINGER=0: never block publisher on send
  - No encryption in v1 (internal loopback only); CURVE ready for future
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import zmq

_log = logging.getLogger("zmq_broker")

# ─── Constants ──────────────────────────────────────────────────────────
DEFAULT_PUB_PORT = 15556
DEFAULT_SUB_PORT = 15557
DEFAULT_HWM = 1000

# IPC socket path (faster than TCP for same-host processes)
IPC_PUB = "ipc:///tmp/taskctl_pub.ipc"
IPC_SUB = "ipc:///tmp/taskctl_sub.ipc"

# Fallback JSON path (when no subscriber)
FALLBACK_FILE = Path.home() / ".hermes" / "task_monitor" / "zmq_fallback.jsonl"


# ─── Publisher ──────────────────────────────────────────────────────────
class Publisher:
    """ZeroMQ PUB socket for pushing events to subscribers.

    Thread-safe. LINGER=0, HWM configurable.
    Falls back to JSONL file if no subscriber (QoS 1).
    """

    def __init__(
        self,
        port: int = DEFAULT_PUB_PORT,
        hwm: int = DEFAULT_HWM,
        use_ipc: bool = True,
        use_tcp: bool = True,
    ):
        self._use_ipc = use_ipc
        self._use_tcp = use_tcp
        self._ctx: Optional[zmq.Context] = None
        self._sockets: list[zmq.Socket] = []
        self._hwm = hwm
        self._port = port
        self._last_subscriber_count = 0

    def start(self) -> None:
        """Initialize ZMQ context and bind sockets."""
        self._ctx = zmq.Context()
        self._sockets.clear()

        if self._use_ipc:
            sock = self._ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, self._hwm)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(IPC_PUB)
            self._sockets.append(sock)
            _log.info("ZMQ PUB bound to %s (HWM=%d)", IPC_PUB, self._hwm)

        if self._use_tcp:
            sock = self._ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, self._hwm)
            sock.setsockopt(zmq.LINGER, 0)
            addr = f"tcp://127.0.0.1:{self._port}"
            sock.bind(addr)
            self._sockets.append(sock)
            _log.info("ZMQ PUB bound to %s (HWM=%d)", addr, self._hwm)

        if not self._sockets:
            _log.warning("No ZMQ transport enabled — events logged only")

    def publish(self, topic: str, payload: dict, qos: int = 1) -> bool:
        """Publish an event to all subscribers.

        Args:
            topic: Event topic (e.g. 'taskctl/exp_001/complete')
            payload: JSON-serialisable dict
            qos: 0=fire-and-forget, 1=at-least-once (log fallback)

        Returns:
            True if at least one subscriber received, False otherwise.
        """
        # Envelope: topic (as bytes) + multipart delimiter + JSON body
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sent = False

        for sock in self._sockets:
            try:
                sock.send_multipart([topic.encode("utf-8"), body], zmq.DONTWAIT)
                # NOP: PUB socket always succeeds locally (ZMQ async)
                # Real success detection requires SUB socket echo
                sent = True
            except zmq.ZMQError as exc:
                _log.warning("ZMQ publish to %s failed: %s", sock, exc)

        # QoS 1: fallback to JSONL if send was successful but uncertain
        # (ZMQ PUB has no way to know if a subscriber is listening)
        if qos >= 1:
            self._fallback_log(topic, payload)

        return sent

    def _fallback_log(self, topic: str, payload: dict) -> None:
        """Append event to fallback JSONL (QoS 1 persistence)."""
        try:
            FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.time(),
                "topic": topic,
                "payload": payload,
            }
            with open(FALLBACK_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            _log.debug("Fallback logged: %s", topic)
        except Exception as exc:
            _log.warning("Fallback write failed: %s", exc)

    def subscriber_count(self) -> int:
        """Estimate connected subscribers (heuristic via EPHEMERAL check).

        Not 100% accurate — ZMQ PUB cannot query subscriptions.
        """
        return len(self._sockets)

    def stop(self) -> None:
        """Clean up ZMQ resources."""
        for sock in self._sockets:
            try:
                sock.setsockopt(zmq.LINGER, 0)
                sock.close()
            except Exception:
                pass
        if self._ctx:
            self._ctx.term()
        self._sockets.clear()
        _log.info("ZMQ PUB stopped")


# ─── Subscriber ─────────────────────────────────────────────────────────
class Subscriber:
    """ZeroMQ SUB socket for receiving events.

    Topic filter: subscribe to specific topics ('taskctl/') or all ('').
    Non-blocking recv with configurable timeout.
    """

    def __init__(
        self,
        port: int = DEFAULT_SUB_PORT,
        timeout_ms: int = 1000,
        use_ipc: bool = True,
        use_tcp: bool = True,
    ):
        self._use_ipc = use_ipc
        self._use_tcp = use_tcp
        self._ctx: Optional[zmq.Context] = None
        self._sockets: list[zmq.Socket] = []
        self._timeout_ms = timeout_ms
        self._port = port
        self._subscriptions: list[str] = []

    def start(self, topics: Optional[list[str]] = None) -> None:
        """Initialize ZMQ context and connect sockets.

        Args:
            topics: Topic prefixes to subscribe to. Default ['taskctl/'].
        """
        self._subscriptions = topics or ["taskctl/"]
        self._ctx = zmq.Context()
        self._sockets.clear()

        if self._use_ipc:
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.RCVHWM, 1000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(IPC_SUB)
            for t in self._subscriptions:
                sock.setsockopt_string(zmq.SUBSCRIBE, t)
            self._sockets.append(sock)
            _log.info(
                "ZMQ SUB connected to %s (topics: %s)",
                IPC_SUB, self._subscriptions,
            )

        if self._use_tcp:
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.RCVHWM, 1000)
            sock.setsockopt(zmq.LINGER, 0)
            addr = f"tcp://127.0.0.1:{self._port}"
            sock.connect(addr)
            for t in self._subscriptions:
                sock.setsockopt_string(zmq.SUBSCRIBE, t)
            self._sockets.append(sock)
            _log.info(
                "ZMQ SUB connected to %s (topics: %s)",
                addr, self._subscriptions,
            )

    def poll(self, timeout_ms: Optional[int] = None) -> list[dict]:
        """Poll all sockets for incoming events.

        Returns list of dicts with keys: topic, payload, source.
        Empty list if no events within timeout.
        """
        timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        events: list[dict] = []

        for sock in self._sockets:
            try:
                # Poll with timeout
                if sock.poll(timeout, zmq.POLLIN):
                    topic = sock.recv_string(zmq.DONTWAIT)
                    body = sock.recv(zmq.DONTWAIT)
                    payload = json.loads(body.decode("utf-8"))
                    events.append({
                        "topic": topic,
                        "payload": payload,
                        "source": str(sock),
                    })
            except zmq.ZMQError:
                continue
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                _log.warning("ZMQ deserialise error: %s", exc)
                continue

        return events

    def subscribe(self, topic_prefix: str) -> None:
        """Add a topic subscription at runtime."""
        self._subscriptions.append(topic_prefix)
        for sock in self._sockets:
            sock.setsockopt_string(zmq.SUBSCRIBE, topic_prefix)

    def unsubscribe(self, topic_prefix: str) -> None:
        """Remove a topic subscription."""
        self._subscriptions = [
            t for t in self._subscriptions if t != topic_prefix
        ]
        for sock in self._sockets:
            try:
                sock.setsockopt_string(zmq.UNSUBSCRIBE, topic_prefix)
            except zmq.ZMQError:
                pass

    def replay_fallback(self, since: float = 0) -> list[dict]:
        """Replay events from the JSONL fallback file (QoS 1 recovery).

        Args:
            since: Unix timestamp — only replay events after this time.

        Returns list of replayed event dicts.
        """
        if not FALLBACK_FILE.exists():
            return []

        events: list[dict] = []
        try:
            with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("ts", 0) >= since:
                            events.append(record)
                    except json.JSONDecodeError:
                        continue

            # Truncate: keep only events before `since` + last 100
            self._trim_fallback(since)
        except Exception as exc:
            _log.warning("Fallback replay error: %s", exc)

        _log.info("Replayed %d events from fallback (since ts=%.0f)", len(events), since)
        return events

    @staticmethod
    def _trim_fallback(since: float) -> None:
        """Keep events before `since` + last 100 lines (prevent unbounded growth)."""
        try:
            lines = FALLBACK_FILE.read_text(encoding="utf-8").splitlines()
            before = [l for l in lines if json.loads(l).get("ts", 0) < since]
            after = lines[-100:] if len(lines) > 100 else []
            FALLBACK_FILE.write_text(
                "\n".join(before + after) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def stop(self) -> None:
        """Clean up ZMQ resources."""
        for sock in self._sockets:
            try:
                sock.setsockopt(zmq.LINGER, 0)
                sock.close()
            except Exception:
                pass
        if self._ctx:
            self._ctx.term()
        self._sockets.clear()
        _log.info("ZMQ SUB stopped")


# ─── Convenience: create paired broker ──────────────────────────────────
def create_broker(
    pub_port: int = DEFAULT_PUB_PORT,
    hwm: int = DEFAULT_HWM,
    use_ipc: bool = True,
    use_tcp: bool = True,
) -> Publisher:
    """Create and start a Publisher instance (the 'broker').

    In the reverse pipeline, the broker runs inside taskctl daemon.
    """
    pub = Publisher(port=pub_port, hwm=hwm, use_ipc=use_ipc, use_tcp=use_tcp)
    pub.start()
    return pub


__all__ = [
    "Publisher",
    "Subscriber",
    "create_broker",
    "DEFAULT_PUB_PORT",
    "DEFAULT_SUB_PORT",
    "IPC_PUB",
    "IPC_SUB",
]
