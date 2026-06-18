"""
BetterClip API — Image generation via WanGP Session API

Uses the shared.api.WanGPSession to submit image generation tasks.
After generation, copies/converts the result to the target output_dir as PNG.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_session = None
_session_lock = threading.Lock()
_queue_lock = threading.Lock()  # Ensures sequential generation — one at a time
_worker_running = False


@dataclass
class GenerationJob:
    job_id: str
    status: str = "queued"
    prompt: str = ""
    model_type: str = ""
    media_type: str = "image"  # "image" | "video"
    output_path: str = ""
    error: str = ""
    progress: int = 0
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, GenerationJob] = {}
_jobs_lock = threading.Lock()
_pending_queue: list[tuple] = []  # Queue of (job, args) tuples


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


VIDEO_MODELS = {
    "ltxv_13B": {
        "type": "ltxv_13B", "label": "LTX Video 13B", "family": "ltxv",
        "caps": ["t2v", "i2v"], "speed": "fast", "vram": "~12 GB",
        "estimated_time": "rapide",
        "description": "Tres rapide, videos longues — ideal cuts promo / previews",
    },
    "t2v": {
        "type": "t2v", "label": "Wan 2.1 T2V 14B", "family": "wan",
        "caps": ["t2v"], "speed": "medium", "vram": "~12 GB",
        "estimated_time": "moyen",
        "description": "Texte -> video, bonne qualite generale",
    },
    "t2v_2_2": {
        "type": "t2v_2_2", "label": "Wan 2.2 T2V", "family": "wan",
        "caps": ["t2v"], "speed": "medium", "vram": "~12 GB",
        "estimated_time": "moyen",
        "description": "Texte -> video, derniere generation Wan",
    },
    "i2v": {
        "type": "i2v", "label": "Wan 2.1 I2V 14B", "family": "wan",
        "caps": ["i2v"], "speed": "medium", "vram": "~12 GB",
        "estimated_time": "moyen",
        "description": "Image -> video : anime un visuel fixe (still Flux)",
    },
    "i2v_2_2": {
        "type": "i2v_2_2", "label": "Wan 2.2 I2V", "family": "wan",
        "caps": ["i2v"], "speed": "medium", "vram": "~12 GB",
        "estimated_time": "moyen",
        "description": "Image -> video, derniere generation Wan",
    },
    "vace_14B": {
        "type": "vace_14B", "label": "Vace 14B (reference)", "family": "vace",
        "caps": ["t2v", "i2v", "reference"], "speed": "medium", "vram": "~14 GB",
        "estimated_time": "moyen",
        "description": "Conditionnement par reference perso/decor — verrou de coherence",
    },
    "vace_14B_2_2": {
        "type": "vace_14B_2_2", "label": "Vace 14B 2.2 (reference)", "family": "vace",
        "caps": ["t2v", "i2v", "reference"], "speed": "medium", "vram": "~14 GB",
        "estimated_time": "moyen",
        "description": "Vace 2.2 : reference perso/decor, meilleure coherence",
    },
    "hunyuan": {
        "type": "hunyuan", "label": "Hunyuan Video T2V", "family": "hunyuan",
        "caps": ["t2v"], "speed": "slow", "vram": "~12 GB",
        "estimated_time": "lent",
        "description": "Texte -> video, rendu cinematographique",
    },
    "hunyuan_i2v": {
        "type": "hunyuan_i2v", "label": "Hunyuan Video I2V", "family": "hunyuan",
        "caps": ["i2v"], "speed": "slow", "vram": "~12 GB",
        "estimated_time": "lent",
        "description": "Image -> video Hunyuan",
    },
    "flf2v_720p": {
        "type": "flf2v_720p", "label": "Wan FLF2V 720p", "family": "wan",
        "caps": ["flf2v"], "speed": "medium", "vram": "~14 GB",
        "estimated_time": "moyen",
        "description": "Premiere + derniere image -> video (transitions)",
    },
}


def get_available_models(model_types_handlers: dict) -> list[dict]:
    available = []
    for model_key, info in IMAGE_MODELS.items():
        if model_key in model_types_handlers:
            available.append(info)
    return available


def get_available_video_models(model_types_handlers: dict) -> list[dict]:
    """Video models actually installed in this Wan2GP (same availability filter)."""
    return [info for key, info in VIDEO_MODELS.items() if key in model_types_handlers]


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            from shared.api import WanGPSession
            _session = WanGPSession(console_output=False)
        return _session


def _extract_first_frame(video_path: str, output_png: str) -> bool:
    """Extract first frame from a video file as PNG using FFmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vframes", "1", "-f", "image2", output_png
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.isfile(output_png)
    except Exception:
        return False


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
    output_filename: str = "",
    media_type: str = "image",
    num_frames: int = 1,
    fps: int = 16,
    image_start: Optional[str] = None,
) -> str:
    global _worker_running

    job_id = str(uuid.uuid4())[:8]
    job = GenerationJob(
        job_id=job_id,
        status="queued",
        prompt=prompt,
        model_type=model_type,
        media_type=media_type,
    )
    with _jobs_lock:
        _jobs[job_id] = job

    args = (job, prompt, negative_prompt, model_type, resolution,
            num_inference_steps, seed, guidance_scale, output_dir,
            image_guide, reference_strength, output_filename,
            media_type, num_frames, image_start)

    _pending_queue.append(args)

    # Start worker thread if not already running
    if not _worker_running:
        _worker_running = True
        thread = threading.Thread(target=_worker_loop, daemon=True)
        thread.start()

    return job_id


def _worker_loop():
    """Process generation jobs one at a time, sequentially."""
    global _worker_running
    try:
        while True:
            if not _pending_queue:
                break
            args = _pending_queue.pop(0)
            _run_generation(*args)
    finally:
        _worker_running = False


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
    output_filename: str,
    media_type: str,
    num_frames: int,
    image_start: Optional[str],
):
    try:
        job.status = "running"
        job.progress = 10

        session = _get_session()

        # Debridage video : video_length > 1 => vraie video ; sinon 1 frame = image
        is_video = (media_type == "video") and (num_frames or 0) > 1

        settings: dict[str, Any] = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "text, watermark, letters, words, blurry, low quality",
            "resolution": resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "batch_size": 1,
            "video_length": num_frames if is_video else 1,
        }

        if is_video:
            # Skip Layer Guidance (STG) desactive : sous offload son masque reste
            # sur CPU et casse la generation ("cuda:0 and cpu", ex. LTX). Off = stable.
            settings["perturbation_switch"] = 0

        if seed >= 0:
            settings["seed"] = seed

        # Image de reference (conditionnement style/perso)
        if image_guide and os.path.isfile(image_guide):
            settings["image_guide"] = image_guide
            settings["denoising_strength"] = reference_strength

        # Image-to-video : still de depart anime par le modele I2V
        if is_video and image_start and os.path.isfile(image_start):
            settings["image_start"] = image_start

        job.progress = 20

        result = session.run_task(settings)

        job.progress = 80

        if result.success and result.generated_files:
            source_file = result.generated_files[0]
            print(f"[betterclip-gen] Generated: {source_file}")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

                if is_video:
                    # Conserver la video produite telle quelle (MP4)
                    src_ext = os.path.splitext(source_file)[1].lower() or ".mp4"
                    fname = output_filename or f"shot_{job.job_id}{src_ext}"
                    target_path = os.path.join(output_dir, fname)
                    shutil.copy2(source_file, target_path)
                    job.output_path = target_path
                else:
                    # Chemin image (MVP Lyrics) : copie ou extraction 1re frame
                    fname = output_filename or f"frame_{job.job_id}.png"
                    target_path = os.path.join(output_dir, fname)
                    ext = os.path.splitext(source_file)[1].lower()
                    if ext in ('.png', '.jpg', '.jpeg', '.webp'):
                        shutil.copy2(source_file, target_path)
                        job.output_path = target_path
                    elif ext in ('.mp4', '.webm', '.mkv'):
                        if _extract_first_frame(source_file, target_path):
                            job.output_path = target_path
                        else:
                            target_path = os.path.join(output_dir, fname.replace('.png', ext))
                            shutil.copy2(source_file, target_path)
                            job.output_path = target_path
                    else:
                        shutil.copy2(source_file, target_path)
                        job.output_path = target_path

                print(f"[betterclip-gen] Saved to: {job.output_path}")
            else:
                job.output_path = source_file

            job.status = "completed"
            job.progress = 100
        else:
            errors = "; ".join(str(e) for e in result.errors) if result.errors else "Unknown error"
            job.status = "failed"
            job.error = errors
            print(f"[betterclip-gen] FAILED: {errors}")

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        print(f"[betterclip-gen] EXCEPTION: {e}")


def get_job(job_id: str) -> Optional[GenerationJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def get_all_jobs() -> list[GenerationJob]:
    with _jobs_lock:
        return list(_jobs.values())
