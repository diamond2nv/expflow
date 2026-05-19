#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.pin — PIN protection module.

Covers:
- Hash consistency (same input → same hash)
- PIN validation (length, digits)
- init/clear/status lifecycle
- verify() matches
- guard() interactive flow
- Config integration
"""

import hashlib
import os

import pytest

from expflow_pde.pin import (
    _hash_pin,
    _pin_hash_path,
    _read_pin_hash,
    _validate_pin,
    clear_pin,
    guard,
    init_pin,
    pin_is_set,
    verify_pin,
)


class TestHash:
    """PIN hashing consistency."""

    def test_hash_consistency(self):
        h1 = _hash_pin("1234")
        h2 = _hash_pin("1234")
        assert h1 == h2

    def test_hash_different_pins_differ(self):
        h1 = _hash_pin("1234")
        h2 = _hash_pin("5678")
        assert h1 != h2

    def test_hash_is_sha256(self):
        h = _hash_pin("0000")
        expected = hashlib.sha256(b"0000").hexdigest()
        assert h == expected

    def test_hash_length(self):
        h = _hash_pin("9999")
        assert len(h) == 64  # SHA-256 hex


class TestValidation:
    """PIN format validation."""

    def test_valid_pin(self):
        _validate_pin("1234")  # no error

    def test_valid_pin_0000(self):
        _validate_pin("0000")  # edge case

    def test_valid_pin_9999(self):
        _validate_pin("9999")

    def test_too_short(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            _validate_pin("123")

    def test_too_long(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            _validate_pin("12345")

    def test_non_numeric(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            _validate_pin("abcd")

    def test_mixed(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            _validate_pin("12ab")

    def test_empty(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            _validate_pin("")

    def test_special_chars(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            _validate_pin("!@#$")


class TestInitClear:
    """PIN init/clear lifecycle — uses real filesystem via tmp_path."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        """Redirect PIN storage to tmp_path."""
        monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))
        yield

    def test_init_creates_hash_file(self):
        h = init_pin("4321")
        assert os.path.isfile(_pin_hash_path())
        with open(_pin_hash_path()) as f:
            assert f.read().strip() == h

    def test_init_returns_hash(self):
        h = init_pin("1234")
        expected = hashlib.sha256(b"1234").hexdigest()
        assert h == expected

    def test_init_overwrites_previous(self):
        h1 = init_pin("1111")
        h2 = init_pin("2222")
        assert h1 != h2
        with open(_pin_hash_path()) as f:
            assert f.read().strip() == h2

    def test_clear_removes_hash_file(self):
        init_pin("1234")
        assert os.path.isfile(_pin_hash_path())
        clear_pin()
        assert not os.path.isfile(_pin_hash_path())

    def test_clear_no_file_is_safe(self):
        clear_pin()  # no error if file doesn't exist

    def test_init_invalid_pin_raises(self):
        with pytest.raises(ValueError, match="exactly 4 digits"):
            init_pin("abc")
        assert not os.path.isfile(_pin_hash_path())


class TestVerify:
    """PIN verification."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))
        yield

    def test_verify_correct_pin(self):
        init_pin("1234")
        assert verify_pin("1234") is True

    def test_verify_incorrect_pin(self):
        init_pin("1234")
        assert verify_pin("5678") is False

    def test_verify_no_pin_configured(self):
        assert verify_pin("1234") is False

    def test_verify_after_clear(self):
        init_pin("1234")
        clear_pin()
        assert verify_pin("1234") is False

    def test_verify_wrong_length(self):
        init_pin("1234")
        assert verify_pin("12") is False
        assert verify_pin("12345") is False

    def test_verify_non_numeric(self):
        init_pin("1234")
        assert verify_pin("abcd") is False


class TestGuard:
    """Interactive guard — mocks getpass."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))
        yield

    def test_guard_no_pin_returns_true(self):
        """If no PIN is configured, guard always allows."""
        assert guard("cancel everything") is True

    def test_guard_correct_pin_returns_true(self, monkeypatch):
        init_pin("1234")
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "1234")
        assert guard("cancel experiment") is True

    def test_guard_incorrect_pin_returns_false(self, monkeypatch):
        init_pin("1234")
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "0000")
        assert guard("cancel experiment") is False

    def test_guard_quit_aborts(self, monkeypatch):
        init_pin("1234")
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "quit")
        assert guard("cancel experiment") is False

    def test_guard_keyboard_interrupt_aborts(self, monkeypatch):
        init_pin("1234")

        def _raise_interrupt(prompt=""):
            raise KeyboardInterrupt()

        monkeypatch.setattr("getpass.getpass", _raise_interrupt)
        assert guard("cancel experiment") is False


class TestPinIsSet:
    """pin_is_set() detection."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))
        yield

    def test_not_set_by_default(self):
        assert pin_is_set() is False

    def test_set_after_init(self):
        init_pin("1234")
        assert pin_is_set() is True

    def test_not_set_after_clear(self):
        init_pin("1234")
        clear_pin()
        assert pin_is_set() is False


class TestReadPinHash:
    """Read PIN hash from file."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))
        yield

    def test_read_after_init(self):
        h = init_pin("1234")
        assert _read_pin_hash() == h

    def test_read_no_file_returns_none(self):
        assert _read_pin_hash() is None

    def test_read_after_clear_returns_none(self):
        init_pin("1234")
        clear_pin()
        assert _read_pin_hash() is None
