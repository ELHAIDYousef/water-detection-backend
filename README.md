# Water Detection Backend (standalone)

FastAPI app wrapping the water-detection model. Runs on CPU or GPU.
Same API as API_CONTRACT.md. No ngrok — on EC2 the instance has a public address.

## Run locally (test before AWS)

    pip install -r requirements.txt
    # CPU-only torch (smaller/faster to install) — optional:
    # pip install torch --index-url https://download.pytorch.org/whl/cpu
    uvicorn main:app --host 0.0.0.0 --port 8000

Then in the dashboard's config.js:
    BASE_URL = "http://localhost:8000"
    USE_MOCK = false

Open http://localhost:8000  -> should show {"status":"ok",...}
Open http://localhost:8000/docs -> test uploads without the dashboard.

## Notes

- First run downloads the CLIPSeg model (~1.5 GB) — one-time.
- On CPU: images take a few seconds; short videos take up to a minute or two.
  If video is too slow, raise SAMPLE_EVERY_N_FRAMES in main.py (fewer frames).
- To use a different model, edit the two ✏️ EDIT sections in main.py only.
