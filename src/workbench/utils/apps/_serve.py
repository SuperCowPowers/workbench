"""Serving for REPL-launched Dash apps: background thread, free port, browser tab."""

import logging
import socket
import threading
import webbrowser

log = logging.getLogger("workbench")

# Where to start looking for a port. Each app takes the next free one.
BASE_PORT = 8050
PORT_SEARCH = 50


def _free_port(start: int = BASE_PORT) -> int:
    """First port at or above `start` that nothing is listening on."""
    for port in range(start, start + PORT_SEARCH):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port in {start}-{start + PORT_SEARCH}")


def serve(app, port: int = None, open_browser: bool = True) -> str:
    """Run a Dash app on a daemon thread and return its URL.

    The thread is a daemon so the REPL stays interactive and the process can still
    exit. Callbacks are registered at build time and a live Dash app rejects
    redefined ones, so each call takes a fresh port rather than reusing one —
    editing a callback means building a new app, and the old one is left behind
    on its thread until the process ends.

    Args:
        app (dash.Dash): A built app, callbacks already registered.
        port (int, optional): Port to serve on. Defaults to the first free one.
        open_browser (bool): Open a browser tab at the URL. Defaults to True.

    Returns:
        str: The URL the app is serving on.
    """
    port = port if port is not None else _free_port()
    url = f"http://127.0.0.1:{port}"

    # Werkzeug logs every request at INFO, and a hover callback is one request per
    # mouse move -- that buries the REPL. Dash's banner duplicates the line below.
    # `dash.dash` is named outright because Dash sets a level on it directly, which
    # a level on the parent `dash` logger would not override.
    for noisy in ("werkzeug", "dash", "dash.dash"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    threading.Thread(
        target=lambda: app.run(port=port, debug=False, use_reloader=False),
        daemon=True,
    ).start()

    log.important(f"Serving {url}")
    if open_browser:
        webbrowser.open(url)
    return url
