from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_CHECKPOINT = BACKEND_DIR / "models" / "football-yolo11n.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a validated lightweight tracking model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    checkpoint = parser.parse_args().checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    PRODUCTION_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, PRODUCTION_CHECKPOINT)
    print(f"Promoted tracking checkpoint to {PRODUCTION_CHECKPOINT}")
    print("Restart the backend to activate it.")
