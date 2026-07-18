from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BACKEND_DIR / "storage" / "training" / "football_dataset"


def main(dataset_dir: Path, validation_sources: set[str]) -> None:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("Dataset manifest is empty")

    source_hashes = {row["source_sha256"] for row in rows}
    resolved_validation = {
        source_hash
        for source_hash in source_hashes
        if any(source_hash.startswith(prefix) for prefix in validation_sources)
    }
    if len(resolved_validation) != len(validation_sources):
        raise SystemExit("One or more validation source prefixes did not match exactly one source")

    referenced = {
        "images": {Path(row["image"]).stem for row in rows},
        "labels": {Path(row["label"]).stem for row in rows},
    }
    backup_dir = dataset_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stale_backup = backup_dir / f"stale_samples_{timestamp}.zip"
    with ZipFile(stale_backup, "w", compression=ZIP_DEFLATED) as archive:
        for kind, extension in (("images", "*.jpg"), ("labels", "*.txt")):
            for path in dataset_dir.glob(f"{kind}/*/{extension}"):
                if path.stem not in referenced[kind]:
                    archive.write(path, path.relative_to(dataset_dir).as_posix())
                    path.unlink()

    split_counts = {"train": 0, "val": 0}
    for row in rows:
        target_split = "val" if row["source_sha256"] in resolved_validation else "train"
        for kind, column, extension in (
            ("images", "image", ".jpg"),
            ("labels", "label", ".txt"),
        ):
            current_path = dataset_dir / row[column]
            if not current_path.exists():
                raise SystemExit(f"Missing reviewed file: {current_path}")
            target_dir = dataset_dir / kind / target_split
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{current_path.stem}{extension}"
            if current_path.resolve() != target_path.resolve():
                current_path.replace(target_path)
            row[column] = target_path.relative_to(dataset_dir).as_posix()
        row["split"] = target_split
        split_counts[target_split] += 1

    with manifest_path.open("w", newline="", encoding="ascii") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Source-level split complete: {split_counts}")
    print(f"Validation sources: {sorted(hash[:10] for hash in resolved_validation)}")
    print(f"Archived stale samples to {stale_backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a reviewed dataset by complete source video.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--val-source", action="append", required=True)
    args = parser.parse_args()
    main(args.dataset, set(args.val_source))
