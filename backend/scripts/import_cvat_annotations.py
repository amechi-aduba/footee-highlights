from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BACKEND_DIR / "storage" / "training" / "football_dataset"
EXPECTED_CLASSES = ("player", "goalkeeper", "referee", "ball")


def _annotation_entries(archive: ZipFile) -> dict[str, str]:
    entries: dict[str, str] = {}
    for name in archive.namelist():
        path = Path(name)
        if not name.lower().endswith(".txt"):
            continue
        if path.name.lower() in {"train.txt", "valid.txt", "val.txt", "test.txt"}:
            continue
        entries[path.stem] = name
    return entries


def _validate_label_text(text: str, source: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{source}:{line_number} must contain five YOLO values")
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
        if class_id not in range(len(EXPECTED_CLASSES)):
            raise SystemExit(f"{source}:{line_number} has invalid class ID {class_id}")
        if not all(0 <= value <= 1 for value in coordinates):
            raise SystemExit(f"{source}:{line_number} has coordinates outside 0..1")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise SystemExit(f"{source}:{line_number} has an empty box")
        counts[class_id] += 1
    return counts


def _backup_labels(dataset_dir: Path) -> Path:
    backup_dir = dataset_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"pre_cvat_labels_{timestamp}.zip"
    with ZipFile(backup_path, "w", compression=ZIP_DEFLATED) as archive:
        for label_path in dataset_dir.glob("labels/*/*.txt"):
            archive.write(label_path, label_path.relative_to(dataset_dir).as_posix())
    return backup_path


def import_split(dataset_dir: Path, split: str, archive_path: Path) -> Counter[int]:
    image_stems = {path.stem for path in (dataset_dir / "images" / split).glob("*.jpg")}
    if not image_stems:
        raise SystemExit(f"No local {split} images found")

    with ZipFile(archive_path) as archive:
        names_entry = next(
            (name for name in archive.namelist() if Path(name).name.lower() == "obj.names"),
            None,
        )
        if names_entry is None:
            raise SystemExit(f"{archive_path} does not contain obj.names")
        class_names = tuple(
            line.strip() for line in archive.read(names_entry).decode("utf-8").splitlines() if line.strip()
        )
        if class_names != EXPECTED_CLASSES:
            raise SystemExit(
                f"Class order mismatch in {archive_path}: expected {EXPECTED_CLASSES}, got {class_names}"
            )

        entries = _annotation_entries(archive)
        exported_stems = set(entries)
        if exported_stems != image_stems:
            missing = sorted(image_stems - exported_stems)
            extra = sorted(exported_stems - image_stems)
            raise SystemExit(
                f"{split} coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )

        validated: dict[str, str] = {}
        counts: Counter[int] = Counter()
        for stem, entry in entries.items():
            text = archive.read(entry).decode("utf-8")
            counts.update(_validate_label_text(text, entry))
            validated[stem] = text

    labels_dir = (dataset_dir / "labels" / split).resolve()
    expected_parent = (dataset_dir.resolve() / "labels" / split)
    if labels_dir != expected_parent:
        raise SystemExit(f"Refusing to replace labels outside {expected_parent}")
    labels_dir.mkdir(parents=True, exist_ok=True)
    for old_label in labels_dir.glob("*.txt"):
        old_label.unlink()
    for stem, text in validated.items():
        (labels_dir / f"{stem}.txt").write_text(text, encoding="ascii")
    return counts


def mark_manifest_reviewed(dataset_dir: Path, imported_splits: set[str]) -> None:
    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("Dataset manifest is empty")
    columns = list(rows[0])
    for row in rows:
        if row["split"] in imported_splits:
            row["annotation_status"] = "reviewed"
    with manifest_path.open("w", newline="", encoding="ascii") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset.resolve()
    archives = {"train": args.train.resolve(), "val": args.val.resolve()}
    for archive_path in archives.values():
        if not archive_path.exists():
            raise SystemExit(f"CVAT export does not exist: {archive_path}")

    # Validate both archives before modifying either split.
    for split, archive_path in archives.items():
        with ZipFile(archive_path) as archive:
            entries = _annotation_entries(archive)
            image_stems = {path.stem for path in (dataset_dir / "images" / split).glob("*.jpg")}
            if set(entries) != image_stems:
                raise SystemExit(f"{split} archive does not match the local images")
            names_entry = next(
                (name for name in archive.namelist() if Path(name).name.lower() == "obj.names"),
                None,
            )
            if names_entry is None:
                raise SystemExit(f"{archive_path} does not contain obj.names")
            class_names = tuple(
                line.strip()
                for line in archive.read(names_entry).decode("utf-8").splitlines()
                if line.strip()
            )
            if class_names != EXPECTED_CLASSES:
                raise SystemExit(f"Class order mismatch in {archive_path}: {class_names}")
            for entry in entries.values():
                _validate_label_text(archive.read(entry).decode("utf-8"), entry)

    backup_path = _backup_labels(dataset_dir)
    total_counts: Counter[int] = Counter()
    for split, archive_path in archives.items():
        counts = import_split(dataset_dir, split, archive_path)
        total_counts.update(counts)
        reviewed_dir = dataset_dir / "reviewed_exports"
        reviewed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_path, reviewed_dir / f"{split}_reviewed.zip")
    mark_manifest_reviewed(dataset_dir, set(archives))

    print(f"Backed up previous labels to {backup_path}")
    print("Imported reviewed labels:")
    for class_id, class_name in enumerate(EXPECTED_CLASSES):
        print(f"  {class_name}: {total_counts[class_id]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import reviewed CVAT YOLO 1.1 annotations.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    main(parser.parse_args())
