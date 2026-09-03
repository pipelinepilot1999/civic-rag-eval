"""Load .env without adding a dependency.

python-dotenv would do this, but it is one more pin in requirements.txt for
fifteen lines of parsing. Called by the eval harness and the API on startup.
"""

from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_env(path: str | pathlib.Path | None = None) -> None:
    """Set any KEY=VALUE from .env that is not already in the environment.

    Existing environment variables win, so an explicitly exported key or a CI
    secret is never silently overridden by a stale local file.
    """
    env_path = pathlib.Path(path) if path else ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
