#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qq_send.py — Send messages via QQ Bot REST API.

Pure REST API client that sends messages to QQ private chats. No LLM
calls, no Hermes gateway dependency. Credentials read from environment
variables (QQ_APP_ID, QQ_CLIENT_SECRET, QQBOT_HOME_CHANNEL).

Designed for zero-token notification from cron tasks, taskctl callbacks,
and server monitoring scripts.

Usage:
  python3 qq_send.py "Message text"
  python3 qq_send.py --file /path/to/message.txt
  echo "Message" | python3 qq_send.py
  python3 qq_send.py --test-token

Configuration:
  Set these environment variables in ~/.hermes/.env:
    QQ_APP_ID=your_app_id
    QQ_CLIENT_SECRET=your_client_secret
    QQBOT_HOME_CHANNEL=target_user_openid
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    import urllib.request
    import urllib.error

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("qq_send")

_LOG_DIR = Path.home() / ".hermes" / "task_monitor"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "qq_send.log"

_rotating_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_rotating_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] [qq_send] %(message)s")
)
_rotating_handler.setLevel(logging.INFO)
_log.addHandler(_rotating_handler)


# ─── Constants ───────────────────────────────────────────────────────────
API_BASE = os.getenv("QQ_API_BASE", "https://api.sgroup.qq.com")
TOKEN_URL = os.getenv(
    "QQ_TOKEN_URL",
    "https://bots.qq.com/app/getAppAccessToken",
)


# ─── Config Loading ──────────────────────────────────────────────────────
def _load_env(env_path: Optional[str] = None) -> None:
    """Load .env file without overriding existing environment variables."""
    paths_to_try = []
    if env_path:
        paths_to_try.append(Path(env_path))
    paths_to_try.extend([
        Path.home() / ".hermes" / ".env",
        Path.home() / ".env",
        Path.cwd() / ".env",
    ])

    for p in paths_to_try:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key not in os.environ:
                        os.environ[key] = value
                _log.debug("Loaded env: %s", p)
            except Exception as exc:
                _log.warning("Failed to load env %s: %s", p, exc)
            break


_load_env()

APP_ID = os.getenv("QQ_APP_ID", "")
CLIENT_SECRET = os.getenv("QQ_CLIENT_SECRET", "")
HOME_CHANNEL = (
    os.getenv("QQBOT_HOME_CHANNEL") or os.getenv("QQ_HOME_CHANNEL", "")
)

TEXT_TYPE = 0
MARKDOWN_TYPE = 2


# ─── Token Management ────────────────────────────────────────────────────
class TokenManager:
    """QQ Bot access token manager with caching."""

    _token: Optional[str] = None
    _expires_at: float = 0

    @classmethod
    def get_token(cls, force_refresh: bool = False) -> str:
        """Get a valid access token (cached)."""
        now = time.time()
        if not force_refresh and cls._token and now < cls._expires_at - 120:
            return cls._token
        token, expires_in = cls._fetch_token()
        cls._token = token
        cls._expires_at = now + expires_in
        _log.info("Acquired new token (expires in %ds)", expires_in)
        return token

    @classmethod
    def _fetch_token(cls) -> tuple[str, int]:
        """Fetch a new access token from the QQ Bot API."""
        payload = {"appId": APP_ID, "clientSecret": CLIENT_SECRET}
        if REQUESTS_AVAILABLE:
            resp = requests.post(TOKEN_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        else:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                TOKEN_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        token = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 7200))
        if not token:
            raise RuntimeError(f"Token fetch failed: {data}")
        return token, expires_in

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached token."""
        cls._token = None
        cls._expires_at = 0


# ─── API Call ────────────────────────────────────────────────────────────
def _api_call(
    method: str,
    path: str,
    payload: Optional[dict] = None,
    retries: int = 2,
) -> dict:
    """Make an authenticated QQ Bot API call."""
    token = TokenManager.get_token()
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"QQBot {token}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if REQUESTS_AVAILABLE:
                fn = requests.get if method == "GET" else requests.post
                resp = fn(
                    url, headers=headers,
                    json=payload if method == "POST" else None,
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()
            else:
                body = json.dumps(payload).encode("utf-8") if payload else None
                req = urllib.request.Request(
                    url, data=body, headers=headers, method=method
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "401" in err_str or "token" in err_str.lower():
                TokenManager.clear_cache()
                token = TokenManager.get_token(force_refresh=True)
                headers["Authorization"] = f"QQBot {token}"
                _log.info("Token expired, refreshed and retrying")
                continue
            if attempt < retries:
                wait = (attempt + 1) * 2
                _log.warning("API call failed, retry in %ds: %s", wait, exc)
                time.sleep(wait)

    err_msg = str(last_error) if last_error else "Unknown error"
    return {"error": True, "message": err_msg}


# ─── Send ────────────────────────────────────────────────────────────────
def send_to_qq(
    content: str,
    openid: Optional[str] = None,
    msg_type: int = TEXT_TYPE,
    retries: int = 2,
) -> dict:
    """Send a message via QQ Bot API.

    Args:
        content: Message text (plain or Markdown).
        openid: Target user OpenID. Defaults to QQBOT_HOME_CHANNEL.
        msg_type: 0=text, 2=markdown.
        retries: Max retry attempts on failure.

    Returns:
        {"success": True, "message_id": "xxx"} or
        {"success": False, "error": "description"}
    """
    target = openid or HOME_CHANNEL

    if not APP_ID or not CLIENT_SECRET:
        return {
            "success": False,
            "error": (
                "QQ_APP_ID and QQ_CLIENT_SECRET not configured. "
                "Set them in ~/.hermes/.env."
            ),
        }
    if not target:
        return {
            "success": False,
            "error": "Target OpenID not configured (QQBOT_HOME_CHANNEL)",
        }

    if msg_type == MARKDOWN_TYPE:
        msg_body = {
            "content": content,
            "msg_type": MARKDOWN_TYPE,
            "markdown": {"content": content},
        }
    else:
        msg_body = {
            "content": content,
            "msg_type": TEXT_TYPE,
        }

    path = f"/v2/users/{target}/messages"
    result = _api_call("POST", path, msg_body, retries=retries)

    if result.get("error"):
        return {"success": False, "error": result.get("message", str(result))}

    message_id = result.get("id", "")
    return {
        "success": bool(message_id),
        "message_id": message_id,
        "raw_response": result,
    }


# ─── CLI Entry ───────────────────────────────────────────────────────────
def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QQ Bot message sender (REST API, no LLM)"
    )
    parser.add_argument("message", nargs="?", help="Message text")
    parser.add_argument("--file", "-f", help="Read message from file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify configuration without sending",
    )
    parser.add_argument(
        "--test-token", action="store_true",
        help="Test token acquisition only",
    )
    parser.add_argument(
        "--md", action="store_true",
        help="Send as Markdown",
    )
    parser.add_argument("--openid", help="Target user OpenID")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        _log.setLevel(logging.DEBUG)

    # Test token
    if args.test_token:
        try:
            token = TokenManager.get_token(force_refresh=True)
            print(f"[OK] Token acquired: {token[:20]}...{token[-8:]}")
            return 0
        except Exception as exc:
            print(f"[ERR] Token failed: {exc}")
            return 1

    # Read message content
    content: Optional[str] = None
    if args.file:
        try:
            content = Path(args.file).expanduser().read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[ERR] Read file failed: {exc}")
            return 1
    elif args.message:
        content = args.message
    elif not sys.stdin.isatty():
        content = sys.stdin.read().strip()

    if not content:
        parser.print_help()
        return 1

    msg_type = MARKDOWN_TYPE if args.md else TEXT_TYPE

    # Dry run
    if args.dry_run:
        target = args.openid or HOME_CHANNEL
        print("[OK] Dry run:")
        print(f"  APP_ID:    {APP_ID[:4]}...{APP_ID[-4:] if APP_ID else 'N/A'}")
        print(f"  Target:    {target[:20] if target else 'N/A'}...")
        print(f"  Length:    {len(content)} chars")
        print(f"  Type:      {'Markdown' if args.md else 'Text'}")
        return 0

    # Send
    result = send_to_qq(content, openid=args.openid, msg_type=msg_type)
    if result.get("success"):
        print(f"[OK] Sent. message_id: {result['message_id']}")
        return 0
    else:
        print(f"[ERR] Send failed: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

__all__ = ["send_to_qq", "TokenManager"]
