"""
BetterClip API — Wan2GP Plugin

Headless plugin (no UI tab).  Starts a FastAPI sidecar server in a
daemon thread so it dies automatically when Wan2GP exits.
"""

import socket
import threading
import time

from shared.utils.plugins import WAN2GPPlugin

from .config import HOST, PORT, API_VERSION, get_or_create_token

PlugIn_Name = "BetterClip API"
PlugIn_Id = "BetterClipAPI"


def _port_is_free(host: str, port: int) -> bool:
    """Return True if *port* on *host* is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


class BetterClipAPIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self._server_thread: threading.Thread | None = None
        self._actual_port: int = PORT

    # -- lifecycle -----------------------------------------------------------

    def setup_ui(self):
        """Called once during Wan2GP initialization.

        We request the globals we need and fire up the API server.
        No Gradio tab is added — this is a headless sidecar.
        """
        self.request_global("model_types_handlers")
        self.request_global("families_infos")

    def post_ui_setup(self, components):
        """Called after all UI components are wired.

        This is the right moment to start the server because globals
        have been injected by PluginManager at this point.
        """
        engine_globals = {
            "model_types_handlers": getattr(self, "model_types_handlers", {}),
            "families_infos": getattr(self, "families_infos", {}),
        }

        # Find a free port starting from the configured one
        port = PORT
        for attempt in range(10):
            if _port_is_free(HOST, port):
                break
            print(f"[BetterClip API] Port {port} busy, trying {port + 1}...")
            port += 1
        else:
            print(f"[BetterClip API] ERROR: no free port found in range {PORT}-{port}. API not started.")
            return {}

        self._actual_port = port
        token = get_or_create_token()

        self._server_thread = threading.Thread(
            target=self._run_server,
            args=(engine_globals, port),
            daemon=True,
            name="betterclip-api",
        )
        self._server_thread.start()

        # Give uvicorn a moment to bind before printing success
        time.sleep(0.3)

        print()
        print("=" * 56)
        print(f"  BetterClip API v{API_VERSION}")
        print(f"  Listening on http://{HOST}:{port}")
        print(f"  Token: {token}")
        print("=" * 56)
        print()

        return {}

    # -- server --------------------------------------------------------------

    @staticmethod
    def _run_server(engine_globals: dict, port: int):
        """Run uvicorn in the background thread."""
        import uvicorn
        from .server import create_app

        app = create_app(engine_globals)
        uvicorn.run(
            app,
            host=HOST,
            port=port,
            log_level="warning",
            access_log=False,
        )
