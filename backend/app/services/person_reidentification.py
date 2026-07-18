from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch


@lru_cache(maxsize=1)
def _load_embedding_model() -> tuple[torch.nn.Module, torch.device]:
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, device


def neural_appearance_embedding(crop: Any) -> np.ndarray | None:
    """Create a normalized neural appearance embedding for a player crop."""
    if crop is None or crop.size == 0:
        return None

    model, device = _load_embedding_model()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    standard_deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / standard_deviation).unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model(tensor)[0].cpu().numpy().astype(np.float32)
    embedding /= float(np.linalg.norm(embedding) + 1e-8)
    return embedding
