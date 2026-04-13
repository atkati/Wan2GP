"""
BetterClip API — Image generation via WanGP Session API

Uses the shared.api.WanGPSession to submit image generation tasks
to the Wan2GP engine. Manages job queue and result retrieval.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Lazy import to avoid circular deps at module load time
_session = None
_session_lock = threading.Lock()


@dataclass
class GenerationJob:
    job_id: str
    status: str = "queued"  # queued, running, completed, failed
    prompt: str = ""
    model_type: str = ""
    output_path: str = ""
    error: str = ""
    progress: int = 0
    created_at: float = field(default_factory=time.time)


# In-memory job store
_jobs: dict[str, GenerationJob] = {}
_jobs_lock = threading.Lock()


# Model info database
IMAGE_MODELS = {
    "flux_schnell": {
        "type": "flux_schnell",
        "label": "Flux Schnell",
        "speed": "fast",
        "quality": "good",
        "vram": "~6 GB",
        "estimated_time": "~5s/image",
        "description": "Rapide, ideal pour le prototypage et les tests",
    },
    "flux": {
        "type": "flux",
        "label": "Flux Dev",
        "speed": "medium",
        "quality": "excellent",
        "vram": "~10 GB",
        "estimated_time": "~15s/image",
        "description": "Haute qualite, recommande pour le rendu final",
    },
    "flux2_dev": {
        "type": "flux2_dev",
        "label": "Flux 2 Dev",
        "speed": "medium",
        "quality": "excellent",
        "vram": "~12 GB",
        "estimated_time": "~20s/image",
        "description": "Derniere generation Flux, meilleure qualite",
    },
    "qwen": {
        "type": "qwen",
        "label": "Qwen Image",
        "speed": "medium",
        "quality": "good",
        "vram": "~8 GB",
        "estimated_time": "~10s/image",
        "description": "Bon pour les styles artistiques et illustrations",
    },
}


def get_available_models(model_types_handlers: dict) -> list[dict]:
    """Return info for models that are actually available in this Wan2GP install."""
    available = []
    for model_key, info in IMAGE_MODELS.items():
        if model_key in model_types_handlers:
            available.append(info)
    return available


def _get_session():
    """Lazy-init a WanGPSession that reuses the running Wan2GP instance."""
    global _session
    with _session_lock:
        if _session is None:
            from shared.api import WanGPSession
            _session = WanGPSession(console_output=False)
        return _session


def submit_generation(
    prompt: str,
    negative_prompt: str = "",
    model_type: str = "flux_schnell",
    resolution: str = "1920x1080",
    num_inference_steps: int = 20,
    seed: int = -1,
    guidance_scale: float = 7.5,
    output_dir: str = "",
    image_guide: Optional[str] = None,
    reference_strength: float = 0.35,
) -> str:
    """Submit an image generation job. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]
    job = GenerationJob(
        job_id=job_id,
        status="queued",
        prompt=prompt,
        model_type=model_type,
    )

    with _jobs_lock:
        _jobs[job_id] = job

    # Run in background thread
    thread = threading.Thread(
        target=_run_generation,
        args=(job, prompt, negative_prompt, model_type, resolution,
              num_inference_steps, seed, guidance_scale, output_dir,
              image_guide, reference_strength),
        daemon=True,
    )
    thread.start()

    return job_id


def _run_generation(
    job: GenerationJob,
    prompt: str,
    negative_prompt: str,
    model_type: str,
    resolution: str,
    num_inference_steps: int,
    seed: int,
    guidance_scale: float,
    output_dir: str,
    image_guide: Optional[str],
    reference_strength: float,
):
    """Background thread: run the generation via WanGPSession."""
    try:
        job.status = "running"
        job.progress = 10

        session = _get_session()

        # Build task settings
        settings: dict[str, Any] = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "text, watermark, letters, words, blurry, low quality",
            "resolution": resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "batch_size": 1,
        }

        if seed >= 0:
            settings["seed"] = seed

        # Image guide for img2img
        if image_guide and os.path.isfile(image_guide):
            settings["image_guide"] = image_guide
            settings["denoising_strength"] = reference_strength

        # Set output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            settings["save_path"] = output_dir

        job.progress = 20

        # Submit and wait
        result = session.run_task(settings)

        job.progress = 90

        if result.success and result.generated_files:
            job.output_path = result.generated_files[0]
            job.status = "completed"
            job.progress = 100
        else:
            errors = "; ".join(str(e) for e in result.errors) if result.errors else "Unknown error"
            job.status = "failed"
            job.error = errors

    except Exception as e:
        job.status = "failed"
        job.error = str(e)


def get_job(job_id: str) -> Optional[GenerationJob]:
    """Get job status."""
    with _jobs_lock:
        return _jobs.get(job_id)


def get_all_jobs() -> list[GenerationJob]:
    """Get all jobs."""
    with _jobs_lock:
        return list(_jobs.values())
