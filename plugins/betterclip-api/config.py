"""
BetterClip API — Configuration
"""

import os
import secrets
from pathlib import Path

API_VERSION = "0.1.0"
HOST = "127.0.0.1"
PORT = 8765

# Token file lives at Wan2GP root, outside the plugin dir
_ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = _ROOT / "betterclip_token.txt"


def get_or_create_token() -> str:
    """Read the API token from disk, or generate one on first run."""
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    return token
