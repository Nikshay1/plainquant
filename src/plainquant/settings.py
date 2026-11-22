"""Model settings saved from the web app, so nobody has to set environment variables.

The file lives on the user's own computer, readable only by them. Nothing is sent anywhere
except the model endpoint the user picked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIELDS = ("provider", "model", "base_url", "api_key")


def path() -> Path:
    if custom := os.environ.get("PLAINQUANT_CONFIG"):
        return Path(custom)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "plainquant" / "settings.json"


def load() -> dict | None:
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("provider"):
        return None
    return {k: data.get(k) or None for k in FIELDS}


def save(provider: str, model: str, base_url: str | None = None, api_key: str | None = None) -> Path:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"provider": provider, "model": model, "base_url": base_url or None, "api_key": api_key or None}
    tmp = p.with_suffix(".tmp")
    # Create the file owner-only before writing the key into it.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, p)
    return p


def clear() -> None:
    path().unlink(missing_ok=True)
