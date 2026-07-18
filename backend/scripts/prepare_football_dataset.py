from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_VIDEOS_DIR = BACKEND_DIR / "storage" / "raw_videos"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "storage" / "training" / "football_dataset"
DEFAULT_MODEL_PATH = BACKEND_DIR / "yolo11m.pt"
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
COCO_CLASS_MAP = {0: 0, 32: 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_videos(video_dir: Path) -> list[tuple[Path, str]]:
    unique: dict[str, Path] = {}
    for path in sorted(video_dir.iterdir()):
        if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            continue
        fingerprint = _sha256(path)
        unique.setdefault(fingerprint, path)
    return [(path, fingerprint) for fingerprint, path in unique.items()]


def _write_dataset_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n",
        encoding="ascii",
    )


def _label_line(class_id: int, xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = xyxy
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def prepare_dataset(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(BACKEND_DIR / "storage" / "ultralytics"),
    )
    from ultralytics import YOLO

    output_dir = args.output.resolve()
    if (output_dir / "manifest.csv").exists():
        if not args.overwrite:
            raise SystemExit(
                f"Dataset already exists at {output_dir}. Use --overwrite only before manual review."
            )
        for directory in (output_dir / "images", output_dir / "labels"):
            if directory.exists():
                shutil.rmtree(directory)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    _write_dataset_yaml(output_dir)

    videos = _unique_videos(args.video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {args.video_dir}")
    model = YOLO(str(args.model))
    manifest_rows: list[dict[str, Any]] = []

    for source_index, (video_path, source_hash) in enumerate(videos):
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        timestamps = []
        timestamp = args.interval / 2
        while timestamp < duration and len(timestamps) < args.max_frames_per_video:
            timestamps.append(timestamp)
            timestamp += args.interval

        for sample_index, timestamp in enumerate(timestamps):
            # Keep a contiguous tail for validation to avoid adjacent-frame leakage.
            split = "val" if sample_index >= max(1, int(len(timestamps) * 0.8)) else "train"
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = capture.read()
            if not success:
                continue
            frame_number = round(timestamp * fps)
            stem = f"src{source_index:03d}_{source_hash[:10]}_f{frame_number:08d}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])

            result = model.predict(
                frame,
                conf=min(args.person_confidence, args.ball_confidence),
                imgsz=args.image_size,
                verbose=False,
            )[0]
            labels: list[str] = []
            counts = {name: 0 for name in CLASS_NAMES.values()}
            for box in result.boxes:
                source_class = int(box.cls.item())
                target_class = COCO_CLASS_MAP.get(source_class)
                if target_class is None:
                    continue
                confidence = float(box.conf.item())
                minimum_confidence = (
                    args.person_confidence if source_class == 0 else args.ball_confidence
                )
                if confidence < minimum_confidence:
                    continue
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                labels.append(_label_line(target_class, xyxy, frame.shape[1], frame.shape[0]))
                counts[CLASS_NAMES[target_class]] += 1
            label_path.write_text("\n".join(labels), encoding="ascii")
            manifest_rows.append(
                {
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "label": label_path.relative_to(output_dir).as_posix(),
                    "split": split,
                    "source_video": video_path.name,
                    "source_sha256": source_hash,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "annotation_status": "needs_human_review",
                    **{f"auto_{name}_count": counts[name] for name in CLASS_NAMES.values()},
                }
            )
        capture.release()

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="ascii") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Prepared {len(manifest_rows)} frames from {len(videos)} unique videos")
    print(f"Review every label and set annotation_status=reviewed in {manifest_path}")
    print("COCO pre-annotations only cover player and ball; label goalkeepers and referees manually.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reviewable football YOLO dataset.")
    parser.add_argument("--video-dir", type=Path, default=RAW_VIDEOS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-frames-per-video", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_dataset(parse_args())
