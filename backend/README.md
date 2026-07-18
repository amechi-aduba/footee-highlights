# Footee Vision Backend

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. OpenAPI docs are available at
`http://localhost:8000/docs`.

## Train the football detector

From the repository root:

```powershell
backend\.venv\Scripts\python backend\scripts\prepare_football_dataset.py
backend\.venv\Scripts\python backend\scripts\train_football_detector.py --device 0
backend\.venv\Scripts\python backend\scripts\promote_football_detector.py
```

`prepare` hashes uploads to remove exact duplicate videos and reserves a
contiguous time block for validation. YOLO11m labels are review suggestions,
not ground truth. `train` blocks until all manifest rows are reviewed and all
four classes have labels. `promote` installs the validated `best.pt` for the API.
