"""Shared configuration and tunables for the water detection backend."""

import os
import torch
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ---- Detection tunables ----
MIN_COVERAGE_PERCENT   = 1.0
SAMPLE_EVERY_N_FRAMES  = 15
SEGMENT_GAP_TOLERANCE  = 1
MAX_SAMPLE_FRAMES      = 4

# ---- Camera simulation tunables ----
NUM_CAMERAS = 45
CAMERA_OVERFLOW_THRESHOLD = 5.0
FRAME_STEP = 25
SLEEP_BETWEEN_CAMERAS = 1.0

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
EVENTS_DB = os.path.join(os.path.dirname(__file__), "events.db")

PROMPT    = "muddy water"
THRESHOLD = 0.4

PLACES = [
    "Laverie - Ligne 1", "Laverie - Ligne 2", "Station de pompage",
    "Bassin decantation", "Zone filtration", "Conduite principale",
    "Silo phosphate", "Zone sechage", "Quai de chargement",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
