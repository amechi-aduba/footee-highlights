# Footee Vision

Footee Vision is an MVP soccer computer vision web app. Upload a highlight reel,
split it into scene-based segments, extract thumbnails, and return a structured
tactical analysis placeholder that is ready for future model integrations.

## Project structure

```text
footee-highlights/
  frontend/   React, Vite, TypeScript, and Tailwind
  backend/    FastAPI, OpenCV, and temporary local processing storage
```

## Run locally

Start the backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Start the frontend in a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. The backend API runs at `http://localhost:8000`,
with interactive docs at `http://localhost:8000/docs`.

## MVP flow

1. `POST /api/videos/upload` temporarily stores a video under a private random ID.
2. `POST /api/videos/{video_id}/process` reads metadata, detects scene cuts,
   saves thumbnails, and writes `backend/storage/results/{video_id}.json`.
3. `GET /api/videos/{video_id}/result` returns the saved JSON report.
4. `GET /api/videos/{video_id}/video` serves the original uploaded video.
5. `GET /api/videos/{video_id}/thumbnail/{segment_id}` serves segment poster images.
6. `DELETE /api/videos/{video_id}` removes the upload, thumbnails, cached
   detections, generated segments, and result JSON together.

## Upload privacy and retention

Uploads are not deduplicated or shared across sessions. The browser requests
deletion when the user leaves the page or starts a new upload. The backend also
removes abandoned sessions after one hour by default and scans every 15 minutes,
so a failed page-exit request cannot leave files indefinitely.

Runtime controls:

```powershell
$env:UPLOAD_RETENTION_SECONDS = "3600"
$env:UPLOAD_CLEANUP_INTERVAL_SECONDS = "900"
$env:MAX_UPLOAD_SIZE_MB = "500"
$env:FOOTEE_STORAGE_DIR = "C:\temporary\footee-vision"
$env:CORS_ORIGINS = "https://your-frontend.vercel.app"
```

`CORS_ORIGINS` accepts a comma-separated list. Local Vite origins are allowed
by default when the variable is not set.

Scene detection settings live in `backend/app/core/config.py`.

TransNetV2 is the primary shot-boundary detector. If its package or weights are unavailable,
the backend automatically falls back to the explainable hybrid detector, which combines:

- spatial HSV color histograms for background/location color changes
- edge-grid differences for camera/background structure changes
- optional YOLO object-layout differences for same-camera cuts where player locations change

Object-layout segmentation is slower because it runs YOLO during processing. Tune or disable it with:

```powershell
$env:SCENE_OBJECT_LAYOUT_ENABLED = "true"
$env:SCENE_OBJECT_LAYOUT_SAMPLE_EVERY_N_FRAMES = "30"
$env:SCENE_OBJECT_LAYOUT_DIFF_THRESHOLD = "0.62"
$env:SCENE_OBJECT_LAYOUT_MIN_SEGMENT_SECONDS = "2.0"
```

TransNetV2 can be configured with:

```powershell
$env:SCENE_DETECTION_METHOD = "transnetv2" # use "hybrid" to force the fallback
$env:TRANSNETV2_THRESHOLD = "0.5"
$env:TRANSNETV2_DEVICE = "auto" # or "cpu" / "cuda"
```

The integration uses the MIT-licensed
[TransNetV2 architecture](https://github.com/soCzech/TransNetV2) and a pinned
PyTorch distribution containing converted pretrained weights. CPU processing
can take several minutes for a long highlight reel; CUDA is selected
automatically when available.

## Football detector

After processing a video, scrub to the frame where the desired player appears.
Use **Detect objects in this frame**, click the desired player box, and then use
**Track selected player** to run ByteTrack through the remainder of that segment.
The saved JSON result stores per-frame focused-player boxes plus sampled-frame
detection counts.

The runtime now uses YOLO11m instead of YOLOv8n. Until a reviewed football
checkpoint is promoted, YOLO11m still maps COCO `person` to `player` and
`sports ball` to `ball`. The production checkpoint has four native classes:
`player`, `goalkeeper`, `referee`, and `ball`.

Create deduplicated frame samples and YOLO11m-assisted annotations:

```powershell
backend\.venv\Scripts\python backend\scripts\prepare_football_dataset.py
```

The dataset is generated under `backend/storage/training/football_dataset`.
Review every image/label pair, correct missed player and ball boxes, add
goalkeeper/referee labels, and change each `annotation_status` in `manifest.csv`
to `reviewed`. Training refuses to run while frames are pending or any class has
no examples.

Train and promote the reviewed checkpoint:

```powershell
backend\.venv\Scripts\python backend\scripts\train_football_detector.py --device 0
backend\.venv\Scripts\python backend\scripts\promote_football_detector.py
```

The full 960-pixel, 100-epoch YOLO11m run should use a CUDA GPU. After promotion,
restart the backend. Set `YOLO_MODEL_PATH` to test another checkpoint without
promoting it.

Focused-player re-association combines a pretrained neural appearance embedding
with predicted motion, bounding-box overlap, and size consistency. During brief
occlusions the identity remains reserved and the UI receives motion-predicted
boxes instead of immediately assigning a nearby player.
The default weights are 50% appearance, 25% position, 15% overlap, and 10%
size. Useful tuning variables are:

```powershell
$env:TRACKING_REASSOCIATION_THRESHOLD = "0.55"
$env:TRACKING_MIN_APPEARANCE_SIMILARITY = "0.35"
$env:TRACKING_REFERENCE_UPDATE_RATE = "0.03"
$env:TRACKING_BOX_SMOOTHING_ALPHA = "0.45"
$env:TRACKING_USE_NEURAL_REID = "true"
$env:TRACKING_REID_VALIDATE_EVERY_N_FRAMES = "5"
$env:TRACKING_TEAM_COLOR_WEIGHT = "0.25"
$env:TRACKING_REASSOCIATION_CONFIRM_FRAMES = "2"
$env:TRACKING_REASSOCIATION_MARGIN = "0.04"
$env:TRACKING_FRAME_STRIDE = "2"
$env:TRACKING_IMAGE_SIZE = "640"
$env:TRACKING_REID_MAX_CANDIDATES = "3"
$env:TRACKING_MAX_PREDICTION_FRAMES = "12"
$env:TRACKING_MAX_REASSOCIATION_FRAME_FRACTION = "0.20"
$env:TRACKING_LONG_GAP_FRAMES = "15"
$env:TRACKING_LONG_GAP_APPEARANCE = "0.60"
```

Tracking defaults to running YOLO/ByteTrack on every second source frame and
motion-predicting the frame between detections. Set `TRACKING_FRAME_STRIDE=1`
for maximum temporal precision or `3` for a faster CPU-oriented mode.
The API uses `models/football-yolo11n.pt` for tracking when that fine-tuned
checkpoint exists, while retaining YOLO11m for frame and role detection.

Frame detection also groups selectable player boxes into two inferred teams using
upper-torso jersey color. Selecting a player stores that color descriptor and uses
it as an additional ByteTrack re-association signal after occlusions.

## Next model integration points

Start in `backend/app/services/object_detection.py` for YOLO configuration and
`backend/app/services/analysis_builder.py` for the ByteTrack focused-player
tracking placeholder, tactical feature extraction, role classification, and
profile comparison.

For deployment, host the Vite frontend on Vercel and run the compute-heavy
FastAPI/OpenCV/YOLO backend on a container service. The backend intentionally
uses temporary storage; use durable object storage only if the product later
requires users to retain data explicitly.
