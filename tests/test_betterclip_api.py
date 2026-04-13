#!/usr/bin/env python3
"""
BetterClip API — Integration tests

Run with:
    python tests/test_betterclip_api.py

Requires the BetterClip API to be running on localhost:8765
(start Wan2GP with the betterclip-api plugin enabled).

Dependencies: requests (already in Wan2GP's environment)
"""

import sys
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8765"
TOKEN_FILE = Path(__file__).resolve().parents[1] / "betterclip_token.txt"


def read_token() -> str:
    """Read the API token from disk."""
    if not TOKEN_FILE.exists():
        print(f"FAIL: Token file not found at {TOKEN_FILE}")
        print("      Start Wan2GP with betterclip-api plugin first.")
        sys.exit(1)
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def headers(token: str) -> dict:
    return {"X-BetterClip-Token": token}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_ok(token: str) -> bool:
    """GET /health with valid token returns 200 + status ok."""
    r = requests.get(f"{BASE_URL}/health", headers=headers(token), timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "ok", f"Expected status 'ok', got {data['status']}"
    assert "version" in data, "Missing 'version' field"
    assert data["engine"] == "Wan2GP", f"Expected engine 'Wan2GP', got {data['engine']}"
    return True


def test_health_no_token() -> bool:
    """GET /health without token returns 401."""
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    return True


def test_health_bad_token() -> bool:
    """GET /health with invalid token returns 401."""
    r = requests.get(
        f"{BASE_URL}/health",
        headers={"X-BetterClip-Token": "this-is-not-the-right-token"},
        timeout=5,
    )
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    return True


def test_models_ok(token: str) -> bool:
    """GET /models with valid token returns non-empty families dict."""
    r = requests.get(f"{BASE_URL}/models", headers=headers(token), timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "families" in data, "Missing 'families' field"
    families = data["families"]
    assert isinstance(families, dict), "families should be a dict"
    assert len(families) > 0, "families should not be empty"

    # Spot-check: each family has expected shape
    for key, fam in families.items():
        assert "order" in fam, f"Family '{key}' missing 'order'"
        assert "label" in fam, f"Family '{key}' missing 'label'"
        assert "types" in fam, f"Family '{key}' missing 'types'"
        assert isinstance(fam["types"], list), f"Family '{key}' types should be a list"

    return True


def test_models_no_token() -> bool:
    """GET /models without token returns 401."""
    r = requests.get(f"{BASE_URL}/models", timeout=5)
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("health_ok", test_health_ok, True),
    ("health_no_token", test_health_no_token, False),
    ("health_bad_token", test_health_bad_token, False),
    ("models_ok", test_models_ok, True),
    ("models_no_token", test_models_no_token, False),
]


def main():
    token = read_token()
    passed = 0
    failed = 0

    print(f"\nBetterClip API Tests — {BASE_URL}")
    print(f"Token: {token[:8]}...\n")

    for name, func, needs_token in ALL_TESTS:
        try:
            if needs_token:
                func(token)
            else:
                func()
            print(f"  PASS  {name}")
            passed += 1
        except (AssertionError, requests.RequestException) as exc:  # noqa: E501
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 40}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
