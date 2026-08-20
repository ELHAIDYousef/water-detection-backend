"""Image / video water detection endpoints."""

import os
import io
import time
import uuid
import threading
import numpy as np
from PIL import Image
import cv2

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import JSONResponse

from config import (
    MIN_COVERAGE_PERCENT,
    SAMPLE_EVERY_N_FRAMES,
    SEGMENT_GAP_TOLERANCE,
    MAX_SAMPLE_FRAMES,
    PROMPT,
    THRESHOLD,
)
from model import run_water_mask, coverage_percent, make_overlay, to_data_uri
from auth import get_current_user

def build_segments(timeline, sample_interval):
    segments, cur, missed = [], None, 0
    for pt in timeline:
        if pt["water"]:
            if cur is None:
                cur = {"start_sec": pt["time_sec"], "end_sec": pt["time_sec"],
                       "peak_coverage_percent": pt["coverage_percent"]}
            else:
                cur["end_sec"] = pt["time_sec"]
                cur["peak_coverage_percent"] = max(cur["peak_coverage_percent"], pt["coverage_percent"])
            missed = 0
        elif cur is not None:
            missed += 1
            if missed > SEGMENT_GAP_TOLERANCE:
                cur["end_sec"] = round(cur["end_sec"] + sample_interval, 2)
                segments.append(cur); cur, missed = None, 0
    if cur is not None:
        cur["end_sec"] = round(cur["end_sec"] + sample_interval, 2)
        segments.append(cur)
    return segments

def analyse_video(video_path, progress_cb=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sample_interval = SAMPLE_EVERY_N_FRAMES / fps

    timeline, internal = [], []
    to_process = max(1, total_frames // SAMPLE_EVERY_N_FRAMES)
    done, idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % SAMPLE_EVERY_N_FRAMES == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cov = coverage_percent(run_water_mask(pil))
            t = round(idx / fps, 2)
            water = cov >= MIN_COVERAGE_PERCENT
            timeline.append({"time_sec": t, "coverage_percent": cov, "water": water})
            internal.append({"frame_index": idx, "time_sec": t, "coverage_percent": cov, "water": water})
            done += 1
            if progress_cb:
                progress_cb(done, to_process)
        idx += 1
    cap.release()

    segments = build_segments(timeline, sample_interval)
    total_water = round(sum(s["end_sec"] - s["start_sec"] for s in segments), 2)
    max_cov = max((p["coverage_percent"] for p in timeline), default=0.0)

    water_frames = sorted([p for p in internal if p["water"]],
                          key=lambda p: p["coverage_percent"], reverse=True)[:MAX_SAMPLE_FRAMES]
    water_frames = sorted(water_frames, key=lambda p: p["time_sec"])
    sample_frames = []
    if water_frames:
        cap = cv2.VideoCapture(video_path)
        for wf in water_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, wf["frame_index"])
            ret, frame = cap.read()
            if not ret:
                continue
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sample_frames.append({"time_sec": wf["time_sec"],
                                  "overlay_url": to_data_uri(make_overlay(pil, run_water_mask(pil)))})
        cap.release()

    return {
        "water_present": len(segments) > 0,
        "video_duration_sec": round(total_frames / fps, 2) if total_frames else 0.0,
        "fps": round(fps, 2),
        "frames_analyzed": len(timeline),
        "sample_interval_sec": round(sample_interval, 3),
        "total_water_duration_sec": total_water,
        "water_coverage_max_percent": max_cov,
        "segments": segments,
        "timeline": timeline,
        "sample_frames": sample_frames,
    }

router = APIRouter(dependencies=[Depends(get_current_user)])

JOBS = {}

@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read()))
    except Exception:
        return JSONResponse(status_code=400, content={
            "status": "error", "error": {"code": "invalid_file", "message": "Not a supported image."}})
    t0 = time.time()
    W, H = pil.size
    mask = run_water_mask(pil)
    cov = coverage_percent(mask)
    return {
        "status": "ok",
        "water_present": cov >= MIN_COVERAGE_PERCENT,
        "coverage_percent": cov,
        "image_width": W, "image_height": H,
        "overlay_url": to_data_uri(make_overlay(pil, mask)),
        "mask_url": to_data_uri(Image.fromarray((mask * 255).astype(np.uint8))),
        "params": {"prompt": PROMPT, "threshold": THRESHOLD},
        "processing_ms": int((time.time() - t0) * 1000),
    }

def _process_video_job(job_id, path):
    def progress(done, total):
        JOBS[job_id].update(status="processing",
                            progress_percent=int(100 * done / max(1, total)),
                            frames_processed=done, frames_total=total)
    try:
        JOBS[job_id] = {"status": "done", "job_id": job_id, "result": analyse_video(path, progress)}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "job_id": job_id,
                        "error": {"code": "model_error", "message": str(e)}}
    finally:
        try: os.remove(path)
        except Exception: pass

@router.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    job_id = "vid_" + uuid.uuid4().hex[:8]
    path = f"/tmp/{job_id}.mp4"
    try:
        with open(path, "wb") as f:
            f.write(await file.read())
    except Exception:
        return JSONResponse(status_code=400, content={
            "status": "error", "error": {"code": "invalid_file", "message": "Could not read the video."}})
    JOBS[job_id] = {"status": "processing", "job_id": job_id,
                    "progress_percent": 0, "frames_processed": 0, "frames_total": 0}
    threading.Thread(target=_process_video_job, args=(job_id, path), daemon=True).start()
    return JSONResponse(status_code=202, content={"status": "processing", "job_id": job_id})

@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={
            "status": "error", "error": {"code": "job_not_found", "message": "No such job."}})
    return job
