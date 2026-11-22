"""plainquant as a desktop app: the same interface in its own native window.

Starts the local server on a free port on this computer only, then opens a native window on
it (WebView2 on Windows, WebKit on macOS, Qt WebEngine on Linux). Closing the window stops
everything.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    try:
        import uvicorn
        import webview
    except ImportError:
        print("Install the desktop extras: pip install 'plainquant[desktop]'", file=sys.stderr)
        return 1
    from .server import create_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):  # up to 10 s for the server to come up
        if server.started:
            break
        time.sleep(0.05)
    else:
        print("plainquant: the local server didn't start", file=sys.stderr)
        return 1

    if os.environ.get("PLAINQUANT_SMOKE"):
        code = _smoke(port)
        server.should_exit = True
        thread.join(timeout=5)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # don't wait on the window toolkit's own shutdown (it can hang without a window)

    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True  # "Get a key" links open in the real browser
    webview.create_window(
        "plainquant",
        f"http://127.0.0.1:{port}/",
        width=1360,
        height=880,
        min_size=(960, 640),
        background_color="#060607",
    )
    webview.start(gui="qt" if sys.platform.startswith("linux") else None)
    server.should_exit = True
    thread.join(timeout=5)
    return 0


def _smoke(port: int) -> int:
    """Check a packaged build without a screen: the server, bundled files and window backend load."""
    import importlib
    import urllib.request

    base = f"http://127.0.0.1:{port}"
    try:
        for path in ("/", "/api/examples", "/assets/hero.webp", "/assets/Doto.woff2"):
            with urllib.request.urlopen(base + path, timeout=10) as r:
                if r.status != 200 or not r.read():
                    raise RuntimeError(f"{path} returned {r.status}")
            print(f"smoke: {path} ok", flush=True)
        backend = {"win32": "edgechromium", "darwin": "cocoa"}.get(sys.platform, "qt")
        importlib.import_module(f"webview.platforms.{backend}")
        print(f"smoke ok ({backend})", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 - report any failure as a failed check
        print(f"smoke failed: {type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
