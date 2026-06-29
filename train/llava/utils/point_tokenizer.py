import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


HIST_TOKEN = "<HIST_POINTS>"
FUT_TOKEN = "<FUT_POINTS>"
POINT_SPECIAL_TOKENS = [HIST_TOKEN, FUT_TOKEN]


class PointEmbedding(nn.Module):
    def __init__(self, centers: np.ndarray, embed_dim: int, k: int = 16, tau: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32))
        self.embed = nn.Parameter(torch.empty(self.centers.shape[0], embed_dim))
        nn.init.orthogonal_(self.embed)
        self.k = k
        self.tau = tau
        self.dropout = nn.Dropout(dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(points.unsqueeze(1), self.centers.unsqueeze(0), p=2).squeeze(1)
        knn_dists, knn_idx = torch.topk(dists, self.k, largest=False, dim=1)
        weights = torch.softmax(-knn_dists / self.tau, dim=1).unsqueeze(2)
        embeds = self.embed[knn_idx]
        return self.dropout((embeds * weights).sum(dim=1))


def _build_point_embedding(state_dict: dict, device: torch.device) -> PointEmbedding:
    centers = state_dict["embedding.C"].cpu().numpy()
    embed_dim = int(state_dict["embedding.E"].shape[1])
    k = int(state_dict.get("embedding.K", 16))
    module = PointEmbedding(centers, embed_dim, k)
    module.load_state_dict({k: v for k, v in state_dict.items() if k.startswith("embedding.")}, strict=False)
    module.to(device)
    module.eval()
    return module


def load_point_tokenizer(ckpt_path: str, transform_path: Optional[str] = None, device: Optional[str] = None) -> Tuple[nn.Module, Optional[Tuple[float, float, float]]]:
    device = torch.device(device or "cpu")
    state = torch.load(ckpt_path, map_location=device)
    if "embedding.C" not in state:
        raise ValueError(f"Checkpoint {ckpt_path} missing embedding weights")
    model = _build_point_embedding(state, device)

    transform = None
    if transform_path and os.path.exists(transform_path):
        with open(transform_path, "r") as f:
            data = json.load(f)
        transform = (
            float(data.get("scale", 1.0)),
            float(data.get("translate_x", 0.0)),
            float(data.get("translate_y", 0.0)),
        )

    return model, transform


def apply_point_transform(points: np.ndarray, transform: Optional[Tuple[float, float, float]]):
    if transform is None:
        return points
    scale, tx, ty = transform
    pts = points.astype(np.float32).copy()
    pts[:, 0] = (pts[:, 0] + tx) * scale
    pts[:, 1] = (pts[:, 1] + ty) * scale
    return pts
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


HIST_TOKEN = "<HIST_POINTS>"
FUT_TOKEN = "<FUT_POINTS>"
POINT_SPECIAL_TOKENS = [HIST_TOKEN, FUT_TOKEN]


class PointEmbedding(nn.Module):
    def __init__(self, centers: np.ndarray, embed_dim: int, k: int = 16, tau: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32))
        self.embed = nn.Parameter(torch.empty(self.centers.shape[0], embed_dim))
        nn.init.orthogonal_(self.embed)
        self.k = k
        self.tau = tau
        self.dropout = nn.Dropout(dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(points.unsqueeze(1), self.centers.unsqueeze(0), p=2).squeeze(1)
        knn_dists, knn_idx = torch.topk(dists, self.k, largest=False, dim=1)
        weights = torch.softmax(-knn_dists / self.tau, dim=1).unsqueeze(2)
        embeds = self.embed[knn_idx]
        return self.dropout((embeds * weights).sum(dim=1))


def _build_point_embedding(state_dict: dict, device: torch.device) -> PointEmbedding:
    centers = state_dict["embedding.C"].cpu().numpy()
    embed_dim = int(state_dict["embedding.E"].shape[1])
    k = int(state_dict.get("embedding.K", 16))
    module = PointEmbedding(centers, embed_dim, k)
    module.load_state_dict({k: v for k, v in state_dict.items() if k.startswith("embedding.")}, strict=False)
    module.to(device)
    module.eval()
    return module


def load_point_tokenizer(ckpt_path: str, transform_path: Optional[str] = None, device: Optional[str] = None) -> Tuple[nn.Module, Optional[Tuple[float, float, float]]]:
    device = torch.device(device or "cpu")
    state = torch.load(ckpt_path, map_location=device)
    if "embedding.C" not in state:
        raise ValueError(f"Checkpoint {ckpt_path} missing embedding weights")
    model = _build_point_embedding(state, device)

    transform = None
    if transform_path and os.path.exists(transform_path):
        with open(transform_path, "r") as f:
            data = json.load(f)
        transform = (
            float(data.get("scale", 1.0)),
            float(data.get("translate_x", 0.0)),
            float(data.get("translate_y", 0.0)),
        )

    return model, transform


def apply_point_transform(points: np.ndarray, transform: Optional[Tuple[float, float, float]]):
    if transform is None:
        return points
    scale, tx, ty = transform
    pts = points.astype(np.float32).copy()
    pts[:, 0] = (pts[:, 0] + tx) * scale
    pts[:, 1] = (pts[:, 1] + ty) * scale
    return pts

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class PointEmbedding(nn.Module):
    def __init__(self, centers: np.ndarray, embed_dim: int, k: int, dropout: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32))
        self.embed = nn.Parameter(torch.empty(self.centers.shape[0], embed_dim))
        nn.init.orthogonal_(self.embed)
        self.k = k
        self.tau = tau
        self.dropout = nn.Dropout(dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(points.unsqueeze(1), self.centers.unsqueeze(0), p=2).squeeze(1)
        knn_dists, knn_idx = torch.topk(dists, self.k, largest=False, dim=1)
        weights = torch.softmax(-knn_dists / self.tau, dim=1).unsqueeze(2)
        embeds = self.embed[knn_idx]
        return self.dropout((embeds * weights).sum(dim=1))


def _build_point_embedding(state_dict: dict, device: torch.device) -> PointEmbedding:
    centers = state_dict["embedding.C"].cpu().numpy()
    embed_dim = int(state_dict["embedding.E"].shape[1])
    k = int(state_dict.get("embedding.K", 16))
    module = PointEmbedding(centers, embed_dim, k)
    module.load_state_dict({k: v for k, v in state_dict.items() if k.startswith("embedding.")}, strict=False)
    module.to(device)
    module.eval()
    return module


def load_point_tokenizer(ckpt_path: str, transform_path: Optional[str] = None, device: Optional[str] = None) -> Tuple[nn.Module, Optional[Tuple[float, float, float]]]:
    device = torch.device(device or "cpu")
    state = torch.load(ckpt_path, map_location=device)
    if "embedding.C" not in state:
        raise ValueError(f"Checkpoint {ckpt_path} missing embedding weights")
    model = _build_point_embedding(state, device)

    transform = None
    if transform_path and os.path.exists(transform_path):
        with open(transform_path, "r") as f:
            data = json.load(f)
        transform = (
            float(data.get("scale", 1.0)),
            float(data.get("translate_x", 0.0)),
            float(data.get("translate_y", 0.0)),
        )

    return model, transform


def apply_point_transform(points: np.ndarray, transform: Optional[Tuple[float, float, float]]):
    if transform is None:
        return points
    scale, tx, ty = transform
    pts = points.astype(np.float32).copy()
    pts[:, 0] = (pts[:, 0] + tx) * scale
    pts[:, 1] = (pts[:, 1] + ty) * scale
    return pts

