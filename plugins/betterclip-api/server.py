"""
BetterClip API — FastAPI server

Runs as a sidecar alongside Wan2GP's Gradio UI.
Endpoints:
  GET /health   — Liveness check
  GET /models   — List locally available model families and types
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_VERSION, get_or_create_token
from .auth import TokenAuthMiddleware


def create_app(engine_globals: dict | None = None) -> FastAPI:
    """Build the FastAPI app with auth and CORS configured.

    Parameters
    ----------
    engine_globals : dict, optional
        References to Wan2GP runtime objects injected by the plugin at
        startup.  Expected keys:
        - "model_types_handlers" : dict[str, handler]
        - "families_infos" : dict[str, tuple[int, str]]
    """
    app = FastAPI(
        title="BetterClip Engine API",
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
    )

    # --- Auth ---------------------------------------------------------------
    token = get_or_create_token()
    app.add_middleware(TokenAuthMiddleware, token=token)

    # --- CORS (localhost only) ----------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-BetterClip-Token", "Content-Type"],
    )

    # --- Shared state -------------------------------------------------------
    _engine = engine_globals or {}

    # --- Routes -------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": API_VERSION,
            "engine": "Wan2GP",
        }

    @app.get("/models")
    async def list_models():
        """Return available model families and their supported types.

        Response shape:
        {
          "families": {
            "flux": {
              "order": 100,
              "label": "Flux 1",
              "types": ["flux", "flux_schnell", ...]
            },
            ...
          }
        }
        """
        handlers: dict = _engine.get("model_types_handlers", {})
        families_infos: dict = _engine.get("families_infos", {})

        # Group model types by family
        # Each handler has query_family_infos() → {"family_key": (order, label)}
        # We reverse-map: for each model_type, find which family it belongs to
        family_types: dict[str, list[str]] = {}
        for model_type, handler in handlers.items():
            try:
                infos = handler.query_family_infos()
            except Exception:
                continue
            for fam_key in infos:
                family_types.setdefault(fam_key, [])
                # Only add if this type actually belongs to this family
                if model_type in (handler.query_supported_types() or []):
                    if model_type not in family_types[fam_key]:
                        family_types[fam_key].append(model_type)

        # Build response
        families = {}
        for fam_key, (order, label) in families_infos.items():
            if fam_key == "unknown":
                continue
            families[fam_key] = {
                "order": order,
                "label": label,
                "types": sorted(family_types.get(fam_key, [])),
            }

        return {"families": families}

    return app
