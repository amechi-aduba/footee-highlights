"""Build static, read-only sample reels without spending production compute.

Run from the repository root:

    backend/.venv/Scripts/python.exe backend/scripts/build_demo_samples.py --source-directory C:/Users/Amech/Downloads

The script runs scene splitting once on the local machine, copies the supplied
videos and generated thumbnails into ``frontend/public/samples``, and writes a
static result JSON for each reel. The deployed frontend can then review these
clips without calling the FastAPI backend.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIRECTORY = REPOSITORY_ROOT / "backend"
PUBLIC_SAMPLES_DIRECTORY = REPOSITORY_ROOT / "frontend" / "public" / "samples"

# Keep one-time generation bounded and inexpensive. The hybrid detector and
# grass-based filtering run locally, never in Azure, and are not part of the
# deployed application's runtime.
os.environ.setdefault("FOOTEE_STORAGE_DIR", "storage/demo-build")
os.environ.setdefault("LOW_MEMORY_MODE", "true")
os.environ.setdefault("SCENE_DETECTION_METHOD", "hybrid")
os.environ.setdefault("SCENE_OBJECT_LAYOUT_ENABLED", "false")
os.environ.setdefault("SCENE_SECOND_PASS_ENABLED", "false")
os.environ.setdefault("SEGMENT_FILTER_USE_PLAYER_MODEL", "false")

sys.path.insert(0, str(BACKEND_DIRECTORY))

from app.core.config import THUMBNAILS_DIR  # noqa: E402
from app.services.video_processing import process_video  # noqa: E402


SAMPLES = (
    {
        "id": "alistair-johnston",
        "filename": "Alistair Johnston  2019 Wake Forest Highlights.mp4",
        "title": "Alistair Johnston",
        "description": "A preprocessed version of the 2019 Wake Forest highlight reel.",
    },
    {
        "id": "jack-harrison",
        "filename": "Jack Harrison Highlights #11.mp4",
        "title": "Jack Harrison",
        "description": "A preprocessed version of the number 11 highlight reel.",
    },
    {
        "id": "ousseni-bouda",
        "filename": "Ousseni Bouda Senior Year High School Soccer Highlight video..mp4",
        "title": "Ousseni Bouda",
        "description": "A preprocessed version of the senior-year high-school reel.",
    },
)


def _build_sample(source_directory: Path, sample: dict[str, str], force: bool) -> None:
    source = source_directory / sample["filename"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing sample video: {source}")

    destination = PUBLIC_SAMPLES_DIRECTORY / sample["id"]
    if destination.exists():
        if not force:
            print(f"Skipping {sample['title']}: {destination} already exists", flush=True)
            return
        resolved_destination = destination.resolve()
        resolved_samples_root = PUBLIC_SAMPLES_DIRECTORY.resolve()
        if resolved_destination.parent != resolved_samples_root:
            raise RuntimeError(f"Refusing to replace unexpected path: {resolved_destination}")
        shutil.rmtree(resolved_destination)
    destination.mkdir(parents=True)

    video_id = f"demo-{sample['id']}"
    last_report = 0.0

    def report(
        _: str,
        stage: str,
        fraction: float,
        message: str,
        completed: int | None,
        total: int | None,
    ) -> dict:
        nonlocal last_report
        now = monotonic()
        if now - last_report >= 1.0 or fraction >= 1.0:
            count = f" ({completed}/{total})" if completed is not None and total else ""
            print(f"  {stage}: {fraction * 100:5.1f}% {message}{count}", flush=True)
            last_report = now
        return {}

    print(f"Processing {sample['title']} from {source.name}", flush=True)
    result = process_video(video_id, source, report)

    thumbnail_source = THUMBNAILS_DIR / video_id
    thumbnail_destination = destination / "thumbnails"
    if not thumbnail_source.is_dir():
        raise RuntimeError(f"Thumbnail generation did not create {thumbnail_source}")
    shutil.copytree(thumbnail_source, thumbnail_destination)
    shutil.copy2(source, destination / "video.mp4")

    result["video_id"] = video_id
    result["demo"] = {
        "id": sample["id"],
        "title": sample["title"],
        "description": sample["description"],
        "video_path": f"/samples/{sample['id']}/video.mp4",
        "read_only": True,
    }
    for segment in result["segments"]:
        segment["focused_player_status"] = "demo"
        segment["thumbnail_path"] = (
            f"/samples/{sample['id']}/thumbnails/{segment['segment_id']}.jpg"
        )

    (destination / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built {sample['title']}: {len(result['segments'])} clips, "
        f"{source.stat().st_size / 1024**2:.1f} MB video",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sample",
        action="append",
        choices=[sample["id"] for sample in SAMPLES],
        help="Build only the selected sample ID. May be supplied more than once.",
    )
    arguments = parser.parse_args()

    selected_ids = set(arguments.sample or [])
    for sample in SAMPLES:
        if selected_ids and sample["id"] not in selected_ids:
            continue
        _build_sample(arguments.source_directory, sample, arguments.force)


if __name__ == "__main__":
    main()
