"""
BetterClip API — Wan2GP Plugin

Headless plugin (no UI tab).  Starts a FastAPI sidecar server in a
daemon thread so it dies automatically when Wan2GP exits.
"""

import logging
import threading

from shared.utils.plugins import WAN2GPPlugin

from .config import HOST, PORT, API_VERSION, get_or_create_token

logger = logging.getLogger("betterclip-api")

PlugIn_Name = "BetterClip API"
PlugIn_Id = "BetterClipAPI"


class BetterClipAPIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self._server_thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def setup_ui(self):
        """Called once during Wan2GP initialization.

        We request the globals we need and fire up the API server.
        No Gradio tab is added — this is a headless sidecar.
        """
        # Request access to Wan2GP internals we need for /models
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

        self._server_thread = threading.Thread(
            target=self._run_server,
            args=(engine_globals,),
            daemon=True,
            name="betterclip-api",
        )
        self._server_thread.start()

        token = get_or_create_token()
        logger.info("=" * 56)
        logger.info("  BetterClip API v%s", API_VERSION)
        logger.info("  Listening on http://%s:%s", HOST, PORT)
        logger.info("  Token: %s", token)
        logger.info("=" * 56)

        return {}

    # -- server --------------------------------------------------------------

    @staticmethod
    def _run_server(engine_globals: dict):
        """Run uvicorn in the background thread."""
        import uvicorn
        from .server import create_app

        app = create_app(engine_globals)
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="warning",
            access_log=False,
        )
