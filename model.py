"""CLIPSeg model loading and image inference helpers."""

import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from config import device, PROMPT, THRESHOLD

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.eval()

print("model loaded")

def run_water_mask(pil_image):
    image = pil_image.convert("RGB")
    W, H = image.size
    inputs = processor(text=[PROMPT], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()
    if probs.ndim == 3:
        probs = probs[0]
    probs = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)
    return (probs > THRESHOLD).astype(np.uint8)

def coverage_percent(mask):
    total = mask.size
    return round(100.0 * float(mask.sum()) / total, 2) if total else 0.0

def make_overlay(pil_image, mask):
    img = np.array(pil_image.convert("RGB"))
    color = np.array([95, 162, 60], dtype=np.uint8)   # OCP green
    alpha = 0.5
    m = mask.astype(bool)
    img[m] = (img[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(img)

def to_data_uri(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
