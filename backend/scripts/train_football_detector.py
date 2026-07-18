from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BACKEND_DIR / "storage" / "training" / "football_dataset"
DEFAULT_MODEL_PATH = BACKEND_DIR / "yolo11m.pt"
REQUIRED_CLASSES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


def _validate_dataset(dataset_dir: Path) -> Counter[int]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("Run prepare_football_dataset.py first")
    with manifest_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    pending = [row["image"] for row in rows if row["annotation_status"] != "reviewed"]
    if pending:
        raise SystemExit(
            f"Training blocked: {len(pending)} frames still need human review in {manifest_path}"
        )

    counts: Counter[int] = Counter()
    for label_path in dataset_dir.glob("labels/*/*.txt"):
        for line in label_path.read_text(encoding="ascii").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    missing = [name for class_id, name in REQUIRED_CLASSES.items() if counts[class_id] == 0]
    if missing:
        raise SystemExit(f"Training blocked: no reviewed labels for {', '.join(missing)}")
    return counts


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(BACKEND_DIR / "storage" / "ultralytics"))
    from ultralytics import YOLO

    counts = _validate_dataset(args.dataset)
    print("Reviewed labels:", {REQUIRED_CLASSES[key]: value for key, value in counts.items()})
    if args.validate_only:
        print("Dataset validation passed; training was not started.")
        return
    model = YOLO(str(args.model))
    model.train(
        data=str(args.dataset / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(BACKEND_DIR / "storage" / "training" / "runs"),
        name="football-yolo11m",
        patience=20,
        cache=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11m on reviewed football labels.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
