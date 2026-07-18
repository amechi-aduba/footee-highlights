from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    BACKEND_DIR / "storage" / "training" / "runs" / "football-yolo11m" / "weights" / "best.pt"
)
PRODUCTION_CHECKPOINT = BACKEND_DIR / "models" / "football-yolo11m.pt"


def main(checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    PRODUCTION_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, PRODUCTION_CHECKPOINT)
    print(f"Promoted {checkpoint} to {PRODUCTION_CHECKPOINT}")
    print("Restart the backend to load the football-specific classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a validated football detector checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    main(parser.parse_args().checkpoint)
