import argparse
import asyncio
import logging
import os
import random
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool


LOGGER = logging.getLogger("wan-move-api")
STOP_COMMAND = "__stop__"
GENERATE_COMMAND = "__generate__"


@dataclass
class ApiState:
    args: argparse.Namespace
    cfg: Any
    model: wan.WanMove
    rank: int
    world_size: int
    local_rank: int
    device: int
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)


STATE: Optional[ApiState] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed FastAPI service for Wan-Move generation."
    )
    parser.add_argument("--task", default="wan-move-i2v", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", default="480*832", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", default=os.getenv("WAN_MOVE_CKPT_DIR", "/models/Wan-Move-14B-480P"))
    parser.add_argument("--output_dir", default=os.getenv("WAN_MOVE_OUTPUT_DIR", "/outputs"))
    parser.add_argument("--host", default=os.getenv("WAN_MOVE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WAN_MOVE_PORT", "8000")))
    parser.add_argument("--offload_model", type=str2bool, default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=True)
    parser.add_argument("--no_t5_fsdp", dest="t5_fsdp", action="store_false")
    parser.add_argument("--dit_fsdp", action="store_true", default=True)
    parser.add_argument("--no_dit_fsdp", dest="dit_fsdp", action="store_false")
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max_jobs", type=int, default=128)
    parser.add_argument("--log_level", default=os.getenv("WAN_MOVE_LOG_LEVEL", "info"))
    args = parser.parse_args()

    validate_runtime_args(args)
    return args


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.ckpt_dir is None:
        raise ValueError("Please specify --ckpt_dir or WAN_MOVE_CKPT_DIR.")
    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unsupported task: {args.task}")
    if args.size not in SUPPORTED_SIZES[args.task]:
        supported = ", ".join(SUPPORTED_SIZES[args.task])
        raise ValueError(f"Unsupported size {args.size}; supported sizes are: {supported}")
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50
    if args.sample_shift is None:
        args.sample_shift = 3.0 if args.size in ["832*480", "480*832"] else 5.0
    if args.frame_num is None:
        args.frame_num = 81
    if args.frame_num <= 0 or (args.frame_num - 1) % 4 != 0:
        raise ValueError("--frame_num must be positive and equal to 4n+1.")


def setup_logging(rank: int, log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level if rank == 0 else logging.WARNING,
        format=f"[rank {rank}] [%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = local_rank

    setup_logging(rank, args.log_level)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    elif args.t5_fsdp or args.dit_fsdp or args.ulysses_size > 1 or args.ring_size > 1:
        raise ValueError("FSDP and context parallelism require torchrun with WORLD_SIZE > 1.")

    if args.ulysses_size > 1 or args.ring_size > 1:
        if args.ulysses_size * args.ring_size != world_size:
            raise ValueError("--ulysses_size * --ring_size must equal WORLD_SIZE.")
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    return rank, world_size, local_rank, device


def build_model(args: argparse.Namespace, rank: int, device: int) -> tuple[Any, wan.WanMove]:
    cfg = WAN_CONFIGS[args.task]
    if args.dtype == "fp32":
        cfg.param_dtype = torch.float32
    elif args.dtype == "fp16":
        cfg.param_dtype = torch.float16
    elif args.dtype == "bf16":
        cfg.param_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if cfg.num_heads % max(args.ulysses_size, 1) != 0:
        raise ValueError(f"Model num_heads={cfg.num_heads} is not divisible by ulysses_size={args.ulysses_size}.")

    LOGGER.info("Loading WanMove model from %s", args.ckpt_dir)
    model = wan.WanMove(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    LOGGER.info("Model loaded.")
    return cfg, model


def validate_arrays(track: np.ndarray, visibility: np.ndarray) -> None:
    if track.ndim == 3:
        frames, points, coords = track.shape
    elif track.ndim == 4:
        if track.shape[0] != 1:
            raise ValueError("Track batch dimension must be 1.")
        _, frames, points, coords = track.shape
    else:
        raise ValueError(f"Track array must have shape [F,N,2] or [1,F,N,2], got {track.shape}.")

    if coords != 2:
        raise ValueError(f"Track coordinate dimension must be 2, got {coords}.")

    if visibility.ndim == 2:
        vis_frames, vis_points = visibility.shape
    elif visibility.ndim == 3:
        if visibility.shape[0] != 1:
            raise ValueError("Visibility batch dimension must be 1.")
        _, vis_frames, vis_points = visibility.shape
    else:
        raise ValueError(f"Visibility array must have shape [F,N] or [1,F,N], got {visibility.shape}.")

    if frames != vis_frames or points != vis_points:
        raise ValueError(
            "Track and visibility shapes do not match: "
            f"track frames/points=({frames}, {points}), "
            f"visibility frames/points=({vis_frames}, {vis_points})."
        )


async def save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def load_job_inputs(job_dir: Path) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.open(job_dir / "input_image").convert("RGB")
    track = np.load(job_dir / "tracks.npy")
    visibility = np.load(job_dir / "visibility.npy")
    validate_arrays(track, visibility)
    return image, track, visibility


def run_generation(payload: dict[str, Any]) -> Optional[str]:
    assert STATE is not None

    job_dir = Path(payload["job_dir"])
    image, track, visibility = load_job_inputs(job_dir)
    seed = int(payload["seed"])
    size = payload["size"]
    sample_shift = float(payload["sample_shift"])
    frame_num = int(payload["frame_num"])
    sample_steps = int(payload["sample_steps"])

    video = STATE.model.generate(
        payload["prompt"],
        image,
        track,
        visibility,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver=payload["sample_solver"],
        sampling_steps=sample_steps,
        guide_scale=float(payload["sample_guide_scale"]),
        seed=seed,
        offload_model=bool(payload["offload_model"]),
        eval_bench=False,
    )

    if STATE.rank != 0:
        return None

    output_path = job_dir / "output.mp4"
    saved_path = cache_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=STATE.cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    if saved_path is None:
        raise RuntimeError("Video encoding failed.")
    return saved_path


def broadcast_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not dist.is_initialized():
        return payload
    objects = [payload if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def worker_loop() -> None:
    assert STATE is not None
    LOGGER.info("Worker rank waiting for generation jobs.")
    while True:
        payload = broadcast_payload({})
        if payload.get("command") == STOP_COMMAND:
            LOGGER.info("Worker rank stopping.")
            break
        if payload.get("command") != GENERATE_COMMAND:
            LOGGER.warning("Worker rank ignored unknown command: %s", payload.get("command"))
            continue
        try:
            run_generation(payload)
        except Exception:
            LOGGER.exception("Worker rank failed generation for job %s", payload.get("job_id"))
        finally:
            if dist.is_initialized():
                dist.barrier()


def create_app() -> FastAPI:
    app = FastAPI(title="Wan-Move API", version="1.0.0")

    @app.on_event("startup")
    async def startup() -> None:
        assert STATE is not None
        asyncio.create_task(process_jobs())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        assert STATE is not None
        return {
            "status": "ok",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "model_loaded": STATE.model is not None,
            "rank": STATE.rank,
            "world_size": STATE.world_size,
            "local_rank": STATE.local_rank,
            "task": STATE.args.task,
            "size": STATE.args.size,
            "dtype": STATE.args.dtype,
            "queue_depth": STATE.queue.qsize(),
            "jobs": len(STATE.jobs),
        }

    @app.post("/v1/generations")
    async def create_generation(
        prompt: str = Form(...),
        image: UploadFile = File(...),
        tracks: UploadFile = File(...),
        visibility: UploadFile = File(...),
        size: Optional[str] = Form(None),
        frame_num: Optional[int] = Form(None),
        sample_steps: Optional[int] = Form(None),
        sample_shift: Optional[float] = Form(None),
        sample_solver: Optional[str] = Form(None),
        sample_guide_scale: Optional[float] = Form(None),
        seed: Optional[int] = Form(None),
        offload_model: Optional[bool] = Form(None),
    ) -> dict[str, str]:
        assert STATE is not None
        if len(STATE.jobs) >= STATE.args.max_jobs:
            raise HTTPException(status_code=429, detail="Job history limit reached.")

        selected_size = size or STATE.args.size
        if selected_size not in SUPPORTED_SIZES[STATE.args.task]:
            supported = ", ".join(SUPPORTED_SIZES[STATE.args.task])
            raise HTTPException(status_code=400, detail=f"Unsupported size {selected_size}; supported sizes: {supported}")

        selected_solver = sample_solver or STATE.args.sample_solver
        if selected_solver not in {"unipc", "dpm++"}:
            raise HTTPException(status_code=400, detail="sample_solver must be 'unipc' or 'dpm++'.")

        selected_frame_num = frame_num or STATE.args.frame_num
        if selected_frame_num <= 0 or (selected_frame_num - 1) % 4 != 0:
            raise HTTPException(status_code=400, detail="frame_num must be positive and equal to 4n+1.")

        selected_steps = sample_steps or STATE.args.sample_steps
        if selected_steps <= 0:
            raise HTTPException(status_code=400, detail="sample_steps must be positive.")

        selected_shift = sample_shift if sample_shift is not None else (
            3.0 if selected_size in ["832*480", "480*832"] else STATE.args.sample_shift
        )
        selected_seed = seed if seed is not None and seed >= 0 else random.randint(0, sys.maxsize)

        job_id = uuid.uuid4().hex
        output_root = Path(STATE.args.output_dir)
        job_dir = output_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)

        image_path = job_dir / "input_image"
        track_path = job_dir / "tracks.npy"
        visibility_path = job_dir / "visibility.npy"

        try:
            await save_upload(image, image_path)
            await save_upload(tracks, track_path)
            await save_upload(visibility, visibility_path)
            load_job_inputs(job_dir)
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = {
            "command": GENERATE_COMMAND,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "prompt": prompt,
            "size": selected_size,
            "frame_num": selected_frame_num,
            "sample_steps": selected_steps,
            "sample_shift": selected_shift,
            "sample_solver": selected_solver,
            "sample_guide_scale": sample_guide_scale if sample_guide_scale is not None else STATE.args.sample_guide_scale,
            "seed": selected_seed,
            "offload_model": offload_model if offload_model is not None else STATE.args.offload_model,
        }
        STATE.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "seed": selected_seed,
            "error": None,
            "video_path": None,
            "request": {
                key: payload[key]
                for key in [
                    "prompt",
                    "size",
                    "frame_num",
                    "sample_steps",
                    "sample_shift",
                    "sample_solver",
                    "sample_guide_scale",
                    "offload_model",
                ]
            },
        }
        await STATE.queue.put(job_id)
        return {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        assert STATE is not None
        job = STATE.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        return job

    @app.get("/v1/jobs/{job_id}/video")
    async def get_job_video(job_id: str) -> FileResponse:
        assert STATE is not None
        job = STATE.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        if job["status"] != "succeeded" or job["video_path"] is None:
            raise HTTPException(status_code=409, detail="Video is not ready.")
        return FileResponse(
            job["video_path"],
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    return app


def execute_rank0_job(payload: dict[str, Any]) -> str:
    broadcast_payload(payload)
    try:
        video_path = run_generation(payload)
    finally:
        if dist.is_initialized():
            dist.barrier()
    if video_path is None:
        raise RuntimeError("Rank 0 did not produce a video path.")
    return video_path


async def process_jobs() -> None:
    assert STATE is not None
    while True:
        job_id = await STATE.queue.get()
        job = STATE.jobs[job_id]
        payload = {
            "command": GENERATE_COMMAND,
            "job_id": job_id,
            "job_dir": str(Path(STATE.args.output_dir) / job_id),
            **job["request"],
            "seed": job["seed"],
        }

        started = time.monotonic()
        job["status"] = "running"
        job["started_at"] = utc_now()
        try:
            video_path = await asyncio.to_thread(execute_rank0_job, payload)
            job["status"] = "succeeded"
            job["video_path"] = video_path
        except Exception as exc:
            LOGGER.exception("Generation failed for job %s", job_id)
            job["status"] = "failed"
            job["error"] = str(exc)
        finally:
            job["finished_at"] = utc_now()
            job["duration_seconds"] = round(time.monotonic() - started, 3)
            STATE.queue.task_done()


def main() -> None:
    global STATE
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed(args)
    cfg, model = build_model(args, rank, device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    STATE = ApiState(
        args=args,
        cfg=cfg,
        model=model,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )

    if rank == 0:
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        if dist.is_initialized():
            broadcast_payload({"command": STOP_COMMAND})
    else:
        worker_loop()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
