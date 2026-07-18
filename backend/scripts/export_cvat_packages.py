from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BACKEND_DIR / "storage" / "training" / "football_dataset"
CLASS_NAMES = ("player", "goalkeeper", "referee", "ball")


def _write_text(archive: zipfile.ZipFile, path: str, content: str) -> None:
    archive.writestr(path, content.encode("ascii"))


def export_split(dataset_dir: Path, split: str) -> None:
    export_dir = dataset_dir / "cvat_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted((dataset_dir / "images" / split).glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No {split} images found under {dataset_dir}")

    images_zip = export_dir / f"football_{split}_images.zip"
    annotations_zip = export_dir / f"football_{split}_yolo_annotations.zip"
    with zipfile.ZipFile(images_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in image_paths:
            archive.write(image_path, arcname=image_path.name)

    image_list = "\n".join(f"data/obj_train_data/{path.name}" for path in image_paths)
    with zipfile.ZipFile(annotations_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_text(archive, "obj.names", "\n".join(CLASS_NAMES) + "\n")
        _write_text(
            archive,
            "obj.data",
            "classes = 4\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        )
        _write_text(archive, "train.txt", image_list + "\n")
        for image_path in image_paths:
            label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
            archive.write(label_path, arcname=f"obj_train_data/{label_path.name}")

    print(f"Created {images_zip}")
    print(f"Created {annotations_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package generated labels for CVAT YOLO 1.1 import.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    args = parser.parse_args()
    splits = ("train", "val") if args.split == "all" else (args.split,)
    for selected_split in splits:
        export_split(args.dataset.resolve(), selected_split)
