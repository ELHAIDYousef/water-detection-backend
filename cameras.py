"""Simulated camera fleet — round-robin monitoring worker."""

import os
import glob
import time
import threading
import numpy as np
from PIL import Image
import cv2

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse, FileResponse

from config import (
    NUM_CAMERAS,
    CAMERA_OVERFLOW_THRESHOLD,
    FRAME_STEP,
    SLEEP_BETWEEN_CAMERAS,
    VIDEO_DIR,
    PLACES,
)
from model import run_water_mask, coverage_percent, make_overlay
from events import log_event
from notifications import notify_overflow
from auth import get_current_user

def _build_cameras():
    vids = sorted(
        glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
        + glob.glob(os.path.join(VIDEO_DIR, "*.avi"))
        + glob.glob(os.path.join(VIDEO_DIR, "*.mov"))
    )
    cams = []
    for i in range(NUM_CAMERAS):
        place = PLACES[i % len(PLACES)]
        cams.append({
            "id": f"cam_{i+1:02d}",
            "name": f"Camera {i+1:02d}",
            "place": place,
            "reference": f"{place.split(' ')[0][:3].upper()}-{i+1:03d}",
            "video": vids[i % len(vids)] if vids else None,
            "status": "unknown",
            "coverage_percent": 0.0,
            "updated_at": 0.0,
            "frame_pos": (i * 40),
        })
    return cams

CAMERAS = _build_cameras()
CAMERA_FRAMES = {}
_cam_lock = threading.Lock()

def _camera_worker():
    if not any(c["video"] for c in CAMERAS):
        print("[cameras] no videos found in", VIDEO_DIR, "- worker idle")
        return
    print(f"[cameras] worker started for {len(CAMERAS)} cameras")
    while True:
        for cam in CAMERAS:
            if not cam["video"]:
                continue
            try:
                cap = cv2.VideoCapture(cam["video"])
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                pos = cam["frame_pos"] % total
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    cam["frame_pos"] = 0
                    continue
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mask = run_water_mask(pil)
                cov = coverage_percent(mask)
                overlay = make_overlay(pil, mask)
                bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                new_status = "overflow" if cov >= CAMERA_OVERFLOW_THRESHOLD else "normal"
                prev_status = cam["status"]
                with _cam_lock:
                    if ok:
                        CAMERA_FRAMES[cam["id"]] = buf.tobytes()
                    cam["coverage_percent"] = cov
                    cam["status"] = new_status
                    cam["updated_at"] = time.time()
                if prev_status != "overflow" and new_status == "overflow":
                    log_event(cam, cov)
                    threading.Thread(target=notify_overflow, args=(cam, cov), daemon=True).start()

                cam["frame_pos"] = pos + FRAME_STEP
            except Exception as e:
                print("[cameras] error on", cam["id"], e)
            time.sleep(SLEEP_BETWEEN_CAMERAS)

def start_camera_worker():
    threading.Thread(target=_camera_worker, daemon=True).start()

router = APIRouter(dependencies=[Depends(get_current_user)])

@router.get("/cameras")
def list_cameras():
    with _cam_lock:
        out = [{
            "id": c["id"], "name": c["name"], "place": c["place"],
            "reference": c["reference"], "status": c["status"],
            "coverage_percent": round(c["coverage_percent"], 2),
            "updated_at": c["updated_at"],
            "frame_url": f"/cameras/{c['id']}/frame",
            "has_frame": c["id"] in CAMERA_FRAMES,
        } for c in CAMERAS]
        overflow = sum(1 for c in CAMERAS if c["status"] == "overflow")
    return {"cameras": out, "overflow_count": overflow, "total": len(CAMERAS)}

@router.get("/cameras/{cam_id}/frame")
def camera_frame(cam_id: str):
    with _cam_lock:
        data = CAMERA_FRAMES.get(cam_id)
    if data is None:
        return JSONResponse(status_code=404, content={
            "status": "error", "error": {"code": "no_frame", "message": "No frame yet."}})
    return Response(content=data, media_type="image/jpeg")

@router.get("/cameras/{cam_id}/video")
def camera_video(cam_id: str):
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if cam is None or not cam.get("video") or not os.path.exists(cam["video"]):
        return JSONResponse(status_code=404, content={
            "status": "error", "error": {"code": "no_video", "message": "No video for this camera."}})
    return FileResponse(cam["video"], media_type="video/mp4")
