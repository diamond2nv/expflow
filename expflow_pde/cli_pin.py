#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow pin CLI — PIN protection sub-commands."""

import getpass

import typer

pin_app = typer.Typer(
    name="pin",
    help="Manage 4-digit PIN protection for destructive operations",
    no_args_is_help=True,
)


@pin_app.command("init")
def pin_init(
    pin: str = typer.Argument(
        ...,
        help="4-digit numeric PIN. Omit for interactive prompt.",
    ),
) -> None:
    """Set or update the PIN.

    Stores a SHA-256 hash of the PIN in ~/.expflow/pin.hash.
    The PIN is never stored in plaintext.

    After setting, operations like 'expflow run cancel <id>' will
    require this PIN for confirmation.
    """
    from expflow_pde.pin import init_pin

    try:
        pin_hash = init_pin(pin)
        print("PIN set successfully.")
        print(f"  Hash: {pin_hash[:16]}...")
        print("")
        print("  PIN protection is now active.")
        print("  'expflow run cancel <id>' will require this PIN.")
    except ValueError as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=1)


@pin_app.command("check")
def pin_check() -> None:
    """Check if a PIN is configured and verify it.

    Prompts for the current PIN and reports whether it matches.
    """
    from expflow_pde.pin import pin_is_set, verify_pin

    if not pin_is_set():
        print("No PIN configured.")
        print("  Use 'expflow pin init <4-digit-pin>' to set one.")
        return

    raw = getpass.getpass("Enter current PIN: ").strip()

    if verify_pin(raw):
        print("PIN is correct ✓")
    else:
        print("PIN is incorrect ✗")
        raise typer.Exit(code=1)


@pin_app.command("clear")
def pin_clear(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Remove the PIN protection.

    Once cleared, destructive operations will no longer require a PIN.
    """
    from expflow_pde.pin import clear_pin, pin_is_set

    if not pin_is_set():
        print("No PIN configured.")
        return

    if not force:
        confirm = typer.confirm(
            "Are you sure you want to remove PIN protection?",
            default=False,
        )
        if not confirm:
            print("Cancelled.")
            raise typer.Exit(code=1)

    clear_pin()
    print("PIN protection removed.")


@pin_app.command("status")
def pin_status() -> None:
    """Show whether PIN protection is active."""
    from expflow_pde.pin import pin_is_set

    if pin_is_set():
        print("PIN protection: active ✓")
        print("")
        print("  Destructive operations will require a 4-digit PIN.")
        print("  Use 'expflow pin check' to verify your PIN.")
        print("  Use 'expflow pin clear' to remove protection.")
    else:
        print("PIN protection: inactive")
        print("  Use 'expflow pin init <4-digit-pin>' to set one.")
