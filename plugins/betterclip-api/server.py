"""
BetterClip API — FastAPI server

Runs as a sidecar alongside Wan2GP's Gradio UI.
Endpoints:
  GET  /health           — Liveness check
  GET  /models           — List locally available model families and types
  GET  /generate/models  — List image models with speed/quality info
  POST /generate         — Submit image generation job
  GET  /jobs/{job_id}    — Check job status and get result
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from .config import API_VERSION, get_or_create_token
from .auth import TokenAuthMiddleware
from .generator import get_available_models, submit_generation, get_job, IMAGE_MODELS


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model_type: str = "flux_schnell"
    resolution: str = "1920x1080"
    num_inference_steps: int = 20
    seed: int = -1
    guidance_scale: float = 7.5
    output_dir: str = ""
    output_filename: str = ""
    image_guide: Optional[str] = None
    reference_strength: float = 0.35


def create_app(engine_globals: dict | None = None) -> FastAPI:
    app = FastAPI(
        title="BetterClip Engine API",
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
    )

    token = get_or_create_token()
    app.add_middleware(TokenAuthMiddleware, token=token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-BetterClip-Token", "Content-Type"],
    )

    _engine = engine_globals or {}

    # --- Health ---------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": API_VERSION, "engine": "Wan2GP"}

    # --- Models (families) ----------------------------------------------------

    @app.get("/models")
    async def list_models():
        handlers: dict = _engine.get("model_types_handlers", {})
        families_infos: dict = _engine.get("families_infos", {})

        family_types: dict[str, list[str]] = {}
        for model_type, handler in handlers.items():
            try:
                infos = handler.query_family_infos()
            except Exception:
                continue
            for fam_key in infos:
                family_types.setdefault(fam_key, [])
                if model_type in (handler.query_supported_types() or []):
                    if model_type not in family_types[fam_key]:
                        family_types[fam_key].append(model_type)

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

    # --- Image generation models info -----------------------------------------

    @app.get("/generate/models")
    async def list_image_models():
        handlers: dict = _engine.get("model_types_handlers", {})
        available = get_available_models(handlers)
        return {"models": available}

    # --- Submit generation job ------------------------------------------------

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        job_id = submit_generation(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            model_type=req.model_type,
            resolution=req.resolution,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            output_dir=req.output_dir,
            output_filename=req.output_filename,
            image_guide=req.image_guide,
            reference_strength=req.reference_strength,
        )
        return {"job_id": job_id, "status": "queued"}

    # --- Job status -----------------------------------------------------------

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        job = get_job(job_id)
        if not job:
            return {"error": "job not found"}, 404

        result = {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "model_type": job.model_type,
        }

        if job.status == "completed":
            result["output_path"] = job.output_path
        elif job.status == "failed":
            result["error"] = job.error

        return result

    return app
