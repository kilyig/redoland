"""Tiny stdlib .env loader (no python-dotenv dependency).

Loads KEY=VALUE lines from .env.local then .env (first wins; never overrides a
variable already set in the real environment). Called at CLI startup so settings
in .env.local are picked up automatically. The only variable the code reads is
`REDOLAND_CLAUDE_BIN` (path to the `claude` CLI, see model.py); there is no API
key — inference is `claude -p` on its own login.
"""

from __future__ import annotations

import os


def load_dotenv(paths=(".env.local", ".env")) -> None:
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass
