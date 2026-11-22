"""Entry point for the packaged desktop app (PyInstaller)."""

import os
import sys

# A windowed app has no console: send output to a log file next to the settings.
if sys.stdout is None or sys.stderr is None:
    from plainquant import settings

    log_dir = settings.path().parent
    os.makedirs(log_dir, exist_ok=True)
    log = open(log_dir / "app.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log

from plainquant.desktop import main

sys.exit(main())
