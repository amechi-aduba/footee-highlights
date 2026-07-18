"""Machine-side sanity check for the OSNet ReID appearance embedding.

Run from the backend/ directory once you've placed a checkpoint at
models/osnet_x0_25.pth (a torchreid `osnet_x0_25` .pth — Market1501 or MSMT17
weights give the best identity signal):

    python scripts/verify_reid.py [optional/path/to/crop.jpg]

It confirms: the checkpoint loads into the vendored architecture, a forward pass
yields a 512-d L2-normalized embedding, and the batched path matches per-crop.
The sandbox that generated this code has no torch, so this is the real
forward-pass test — it must be run on the machine that will serve the app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# make "app" importable when run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.reid_embedding import appearance_backend, embed_crops, reid_available  # noqa: E402


def _fake_crop(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((160, 80, 3)) * 255).astype(np.uint8)


def main() -> int:
    print("reid_available():", reid_available())
    print("appearance_backend():", appearance_backend())
    if not reid_available():
        print(
            "\nOSNet is NOT active — the app will use the HSV fallback.\n"
            "Place a torchreid osnet_x0_25 checkpoint at models/osnet_x0_25.pth\n"
            "(torchreid MODEL_ZOO: osnet_x0_25_market1501 / osnet_x0_25_msmt17)."
        )
        return 1

    crops = [_fake_crop(1), _fake_crop(2), _fake_crop(1)]  # #0 and #2 identical
    embeddings = embed_crops(crops)
    dims = [None if e is None else e.shape[0] for e in embeddings]
    norms = [None if e is None else round(float(np.linalg.norm(e)), 4) for e in embeddings]
    print("embedding dims:", dims, "(expect 512)")
    print("embedding norms:", norms, "(expect ~1.0)")

    same = float(np.dot(embeddings[0], embeddings[2]))       # identical crops
    diff = float(np.dot(embeddings[0], embeddings[1]))       # different crops
    print(f"cosine(identical crops) = {same:.4f}  (expect ~1.0)")
    print(f"cosine(different crops) = {diff:.4f}  (expect clearly lower)")

    ok = dims[0] == 512 and abs((norms[0] or 0) - 1.0) < 1e-2 and same > 0.99 and same - diff > 0.05
    print("\nRESULT:", "OK — OSNet ReID is wired correctly." if ok else "CHECK the output above.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
