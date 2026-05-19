#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIN protection module — hash-based 4-digit PIN with configurable storage.

Provides:
- Hashed PIN storage (SHA-256, not plaintext)
- PIN init / verify / clear / status
- Integration with expflow's config system (config.yaml + .env)
- A guard() helper for CLI commands that need confirmation

Design:
- PIN is stored as SHA-256 hex digest in config.yaml under `pin.hash`
  and/or as environment variable `EXPFLOW_PIN_HASH` in .env.
- PIN is never stored or logged in plaintext.
- Storage uses expflow's existing config system (config.py).

Usage:
    from expflow_pde.pin import pin_is_set, verify_pin, guard

    if guard("cancel experiment exp-001"):
        # proceed with cancellation
        ...
"""

import getpass
import hashlib
import os
from typing import Optional

from expflow_pde.config import get

# ── Config keys ──

_CONFIG_KEY = "pin.hash"
_ENV_KEY = "EXPFLOW_PIN_HASH"

# ── Module-level state for testability ──
# Set this to a tmp_path in tests to avoid touching ~/.expflow/
# Use _set_test_dir() (not direct assignment) for cross-file test safety.
_PIN_DIR: str | None = None


def _set_test_dir(path: str) -> None:
    """Set a temporary directory for PIN storage during testing.

    This is a module-level function so that all internal references to
    _PIN_DIR see the update, even when the test file's own top-level
    imports of _get_pin_dir / _pin_hash_path / _read_pin_hash were
    bound at parse time under a different module identity.

    Args:
        path: Absolute path to the test directory (e.g. str(tmp_path)).

    To restore, call ``_set_test_dir(None)`` or ``_set_test_dir("")``.
    """
    global _PIN_DIR  # noqa: PLW0603
    _PIN_DIR = path if path else None


def _get_pin_dir() -> str:
    """Return the PIN storage directory."""
    if _PIN_DIR is not None:
        return _PIN_DIR
    return os.path.expanduser("~/.expflow")


# ── Hashing ──


def _hash_pin(pin: str) -> str:
    """Return SHA-256 hex digest of a 4-digit PIN string."""
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


# ── Validation ──


def _validate_pin(pin: str) -> None:
    """Validate that pin is a 4-digit numeric string. Raises ValueError if not."""
    if not pin.isdigit() or len(pin) != 4:
        raise ValueError("PIN must be exactly 4 digits (0-9)")


# ── PIN status ──


def pin_is_set() -> bool:
    """Check if a PIN hash is configured (in config.yaml or .env)."""
    if _read_pin_hash() is not None:
        return True
    if get(_CONFIG_KEY):
        return True
    # Also check env directly (get() may not read raw env vars)
    from expflow_pde.config import _env_cache

    if _env_cache and _ENV_KEY in _env_cache:
        return True
    return False


def get_pin_hash() -> Optional[str]:
    """Return the stored PIN hash from config or .env.

    Precedence: .env (EXPFLOW_PIN_HASH) > config.yaml (pin.hash).
    """
    # .env override takes priority
    from expflow_pde.config import _env_cache

    if _env_cache and _ENV_KEY in _env_cache:
        return _env_cache[_ENV_KEY]
    return get(_CONFIG_KEY)


# ── Init / Clear ──


def init_pin(pin: str) -> str:
    """Initialize or update the PIN.

    Writes the hash to ~/.expflow/pin.hash (separate file, not config.yaml,
    to avoid accidentally committing the PIN hash to git).

    Args:
        pin: 4-digit numeric string.

    Returns:
        The hash of the PIN (for display/verification).

    Raises:
        ValueError: If pin is not a valid 4-digit number.
    """
    _validate_pin(pin)
    pin_hash = _hash_pin(pin)
    _write_pin_hash(pin_hash)
    return pin_hash


def clear_pin() -> None:
    """Remove the stored PIN hash file."""
    _write_pin_hash(None)


def _pin_hash_path() -> str:
    """Get path to the PIN hash file."""
    return os.path.join(_get_pin_dir(), "pin.hash")


def _write_pin_hash(pin_hash: Optional[str]) -> None:
    """Write or remove the PIN hash file."""
    path = _pin_hash_path()
    if pin_hash is None:
        if os.path.isfile(path):
            os.remove(path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(pin_hash + "\n")


def _read_pin_hash() -> Optional[str]:
    """Read PIN hash from file, or None."""
    path = _pin_hash_path()
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        val = f.readline().strip()
    return val if val else None


# ── Verify ──


def verify_pin(pin: str) -> bool:
    """Check if a PIN matches the stored hash.

    Checks sources in order:
    1. ~/.expflow/pin.hash file (from init_pin)
    2. config.yaml pin.hash
    3. .env EXPFLOW_PIN_HASH

    Args:
        pin: 4-digit numeric string to verify.

    Returns:
        True if the PIN matches, False otherwise.
    """
    pin_hash = _read_pin_hash()
    if pin_hash is None:
        pin_hash = get_pin_hash()
    if pin_hash is None:
        return False
    return _hash_pin(pin) == pin_hash


# ── Interactive guard ──


def guard(action_description: str) -> bool:
    """Interactive PIN guard for destructive operations.

    If no PIN is configured, returns True immediately (no guard).
    Otherwise prompts the user for their PIN and returns True only
    if it matches.

    Args:
        action_description: Human-readable description of the action
                            being guarded (e.g. "cancel experiment exp-001").

    Returns:
        True if the action should proceed, False if cancelled.
    """
    if not _read_pin_hash() and not get_pin_hash():
        return True  # No PIN configured — no guard

    print(f"\n  [!] Protected action: {action_description}")
    print("  [!] PIN protection is active. Enter your 4-digit PIN to proceed.")
    print("  [!] Press Ctrl+C or type 'quit' to abort.")

    try:
        raw = getpass.getpass("  PIN: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print("  Aborted.")
        return False

    if raw.lower() in ("quit", "exit", "abort", "q"):
        print("  Aborted.")
        return False

    if verify_pin(raw):
        print("  PIN verified. Proceeding...")
        return True
    else:
        print("  Incorrect PIN. Action cancelled.")
        return False
