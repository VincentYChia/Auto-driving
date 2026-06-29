import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import torch
import math
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple, Union


import tqdm

from tokenizer.nuscenes_dataset import NuScenesDataset

torch.manual_seed(42)


def get_xy_points(dataset: NuScenesDataset):
    """
    Get all the things. Same as "fut_waypoints_ego" from gen_data.py

    Returns:
        (N, 2) array of all points in the ego vehicle frame
    """
    xy_points = []

    # Iterate through every sample in the dataset and collect future waypoints
    # in the ego vehicle frame. Equivalent conceptually to the fut_waypoints_ego
    # list in gen_data.py (future ego-frame waypoints), but aggregated across
    # the entire dataset for clustering / codebook creation.
    for idx in tqdm.tqdm(range(len(dataset))):
        item = dataset[idx]
        traj = item.get("trajectory", None)
        if traj is None:
            continue
        if traj.shape[0] == 0:
            continue
        # Take only x,y (ignore z) -> (T,2)
        xy = traj[:, :2]
        xy_points.append(xy)

    if not xy_points:
        return np.empty((0, 2), dtype=np.float32)

    all_xy = np.vstack(xy_points).astype(np.float32)
    return all_xy

def save_data():
    load_dotenv()

    VERSION = 'v1.0-trainval'
    DATAROOT = os.getenv("NUSCENES_ROOT")
    dataset = NuScenesDataset(nuscenes_path=DATAROOT, version=VERSION, split='train', future_seconds=3, future_hz=2, get_img_data=False)
    xy_points = get_xy_points(dataset)
    np.save("points_xy.npy", xy_points)

    print(f"Total number of points: {xy_points.shape[0]}")
    plt.figure(figsize=(8, 8))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.1, alpha=0.5)
    plt.title("Future Waypoints in Ego Vehicle Frame")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("future_waypoints_ego_frame.png", dpi=300)

def get_clusters(points: np.ndarray, num_clusters=2048):
    nbrs = NearestNeighbors(n_neighbors=16).fit(points)
    dists, _ = nbrs.kneighbors(points)
    # larger mean distance => lower density => higher weight
    w = (dists.mean(axis=1) + 1e-6)**2
    w /= w.mean()

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=16384, random_state=42)
    kmeans.fit(points, sample_weight=w)
    return kmeans

def viz(kmeans, points: np.ndarray):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.5, label='Data Points')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=20, marker='x', label='Cluster Centers')
    plt.title("K-Means Clustering of Future Waypoints")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("kmeans_clusters.png", dpi=450)

def preprocess(points: np.ndarray, kmeans: MiniBatchKMeans):
    """
    Given a set of points/their clusters, normalize this data
    s.t. all points are in the range [-1, 1].
    """
    minx, maxx = points[:, 0].min(), points[:, 0].max()
    miny, maxy = points[:, 1].min(), points[:, 1].max()

    print(f"x range: {minx} to {maxx}")
    print(f"y range: {miny} to {maxy}")

    # Scale and translate points to [-1, 1]
    scale = 2 / (max(maxx - minx, maxy - miny) + 1e-6)
    translate_x = -(maxx + minx) / 2.0
    translate_y = -(maxy + miny) / 2.0
    points_normalized = points.copy()
    points_normalized[:, 0] = (points[:, 0] + translate_x)
    points_normalized[:, 1] = (points[:, 1] + translate_y)
    points_normalized[:, 0] *= scale
    points_normalized[:, 1] *= scale

    # Apply same transformation to cluster centers
    centers = kmeans.cluster_centers_.copy()
    centers[:, 0] = (centers[:, 0] + translate_x) * scale
    centers[:, 1] = (centers[:, 1] + translate_y) * scale
    kmeans.cluster_centers_ = centers
    return points_normalized, kmeans, (scale, translate_x, translate_y)
    

def _as_centers(kmeans_or_centers: Union[MiniBatchKMeans, np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Utility to unwrap MiniBatchKMeans objects into a centers array while supporting
    direct numpy / torch inputs.
    """
    if isinstance(kmeans_or_centers, MiniBatchKMeans):
        centers = kmeans_or_centers.cluster_centers_
    elif isinstance(kmeans_or_centers, torch.Tensor):
        centers = kmeans_or_centers.detach().cpu().numpy()
    else:
        centers = np.asarray(kmeans_or_centers)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Expected centers to have shape (N, 2), got {centers.shape}")
    return centers.astype(np.float32)


class PointEmbedding(nn.Module):
    def __init__(self, kmeans_or_centers: Union[MiniBatchKMeans, np.ndarray, torch.Tensor],
                 D: int = 4096, dropout: float = 0.00, tau: float = 1.0):
        super().__init__()
        centers = _as_centers(kmeans_or_centers)
        self.register_buffer("C", torch.as_tensor(centers, dtype=torch.float32))  # (num_clusters, 2)
        self.dropout = nn.Dropout(dropout)

        self.D = D
        self.E = nn.Parameter(nn.init.orthogonal_(torch.empty(self.C.shape[0], self.D)))  # (num_clusters, D)
        self.tau = tau

    def forward(self, X: torch.Tensor, K: int = 1, return_indices: bool = False):
        """
        X: (B, 2) tensor of points
        Returns:
            (B, D) tensor of embeddings
        """
        if K > self.C.shape[0]:
            raise ValueError(f"K={K} exceeds number of centers ({self.C.shape[0]}).")
        distances = torch.cdist(X.unsqueeze(1), self.C.unsqueeze(0), p=2).squeeze(1)  # (B, num_clusters)
        knn_dists, knn_indices = torch.topk(distances, K, largest=False, dim=1)  # (B, K)

        weights = F.softmax(-knn_dists / self.tau, dim=1)  # (B, K)
        embeddings = self.E[knn_indices]  # (B, K, D)
        weights = weights.unsqueeze(2)  # (B, K, 1)
        weighted_embeddings = (embeddings * weights).sum(dim=1)  # (B, D)

        output = self.dropout(weighted_embeddings)
        if return_indices:
            return output, knn_indices[:, 0]
        return output

    def nearest_indices(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the nearest (hard) center index for each input point.
        """
        distances = torch.cdist(X, self.C, p=2)
        return distances.argmin(dim=1)
    
class EmbeddingDecoder(nn.Module):
    def __init__(self, kmeans_or_centers: Union[MiniBatchKMeans, np.ndarray, torch.Tensor],
                 D: int = 4096, hidden_dim: int = 256):
        super().__init__()
        centers = _as_centers(kmeans_or_centers)
        self.register_buffer("C", torch.as_tensor(centers, dtype=torch.float32))  # (num_clusters, 2)
        self.D = D
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(D, 2)

        self.mlp = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) tensor of embeddings
        Returns:
            (B, 2) tensor of reconstructed points
        """
        recon_points = self.linear(embeddings) + self.mlp(embeddings)  # (B, 2)
        return recon_points


class PointTokenizer(nn.Module):
    """
    Combines the embedding/decoder stack with geometry utilities so the tokenizer
    can be used without explicitly materialising the k-means centroids or
    normalization transforms outside the module.
    """

    def __init__(
        self,
        kmeans_or_centers: Union[MiniBatchKMeans, np.ndarray, torch.Tensor],
        D: int = 4096,
        dropout: float = 0.0,
        hidden_dim: int = 256,
        transform: Optional[Tuple[float, float, float]] = None,
        tau: float = 1.0,
    ):
        super().__init__()
        centers = _as_centers(kmeans_or_centers)
        if transform is None:
            transform = (1.0, 0.0, 0.0)
        scale, translate_x, translate_y = transform
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("translate", torch.as_tensor([translate_x, translate_y], dtype=torch.float32))

        self.embedding = PointEmbedding(centers, D=D, dropout=dropout, tau=tau)
        self.decoder = EmbeddingDecoder(centers, D=D, hidden_dim=hidden_dim)

    @property
    def centers(self) -> torch.Tensor:
        return self.embedding.C

    def _ensure_point_tensor(self, xy: Union[np.ndarray, torch.Tensor, Tuple[float, float]]):
        xy_tensor = torch.as_tensor(xy, dtype=torch.float32)
        if xy_tensor.numel() == 0:
            raise ValueError("Point tensor is empty.")
        if xy_tensor.dim() == 1:
            if xy_tensor.numel() != 2:
                raise ValueError(f"Expected a 2D point, got shape {tuple(xy_tensor.shape)}")
            xy_tensor = xy_tensor.unsqueeze(0)
            squeeze = True
        elif xy_tensor.dim() == 2 and xy_tensor.size(-1) == 2:
            squeeze = False
        else:
            raise ValueError(f"Expected shape (2,) or (N, 2); received {tuple(xy_tensor.shape)}")
        if xy_tensor.device != self.scale.device:
            xy_tensor = xy_tensor.to(self.scale.device)
        return xy_tensor, squeeze

    def _ensure_index_tensor(self, indices: Union[int, np.ndarray, torch.Tensor]):
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        if idx_tensor.numel() == 0:
            raise ValueError("Index tensor is empty.")
        if idx_tensor.dim() == 0:
            idx_tensor = idx_tensor.unsqueeze(0)
            squeeze = True
        elif idx_tensor.dim() == 1:
            squeeze = False
        else:
            raise ValueError(f"Expected scalar or 1D indices; received {tuple(idx_tensor.shape)}")
        if idx_tensor.device != self.centers.device:
            idx_tensor = idx_tensor.to(self.centers.device)
        return idx_tensor, squeeze

    def _normalize_tensor(self, xy):
        xy_tensor, squeeze = self._ensure_point_tensor(xy)
        normalized = (xy_tensor + self.translate) * self.scale
        return normalized, squeeze

    def _denormalize_tensor(self, xy):
        xy_tensor, squeeze = self._ensure_point_tensor(xy)
        denormalized = xy_tensor / self.scale - self.translate
        return denormalized, squeeze

    def normalize_points(self, xy):
        normalized, squeeze = self._normalize_tensor(xy)
        return normalized.squeeze(0) if squeeze else normalized

    def denormalize_points(self, xy):
        denormalized, squeeze = self._denormalize_tensor(xy)
        return denormalized.squeeze(0) if squeeze else denormalized

    def points_to_embeddings(self, xy, K: int = 1):
        xy_norm, squeeze = self._normalize_tensor(xy)
        embeddings = self.embedding(xy_norm, K=K)
        return embeddings.squeeze(0) if squeeze else embeddings

    def points_to_indices(self, xy) -> torch.Tensor:
        xy_norm, squeeze = self._normalize_tensor(xy)
        indices = self.embedding.nearest_indices(xy_norm)
        return indices.squeeze(0) if squeeze else indices

    def encode_points(self, xy, K: int = 1):
        """
        Convenience wrapper that returns both embeddings and discrete token indices.
        """
        xy_norm, squeeze = self._normalize_tensor(xy)
        embeddings, indices = self.embedding(xy_norm, K=K, return_indices=True)
        if squeeze:
            embeddings = embeddings.squeeze(0)
            indices = indices.squeeze(0)
        return embeddings, indices

    def indices_to_points(self, indices):
        idx_tensor, squeeze = self._ensure_index_tensor(indices)
        centers = self.centers[idx_tensor]
        points = self.denormalize_points(centers)
        return points.squeeze(0) if squeeze else points

    def indices_to_embeddings(self, indices):
        idx_tensor, squeeze = self._ensure_index_tensor(indices)
        embeddings = self.embedding.E[idx_tensor]
        return embeddings.squeeze(0) if squeeze else embeddings

    def decode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Wrapper around the decoder that accepts either a single embedding or a batch.
        """
        embeddings_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=self.centers.device)
        squeeze = False
        if embeddings_tensor.dim() == 1:
            embeddings_tensor = embeddings_tensor.unsqueeze(0)
            squeeze = True
        elif embeddings_tensor.dim() != 2:
            raise ValueError(f"Expected embedding tensor with shape (D,) or (N, D); got {tuple(embeddings_tensor.shape)}")
        decoded = self.decoder(embeddings_tensor)
        return decoded.squeeze(0) if squeeze else decoded

@torch.no_grad()
def geometry_loss(embeddings: torch.Tensor, points: torch.Tensor,
                  n_pairs=1024, detach_scales=True):
    """
    embeddings: (B, D)
    points: (B, 2)
    """
    B = embeddings.size(0)
    if B < 2:
        return embeddings.new_tensor(0.0)

    i = torch.randint(0, B, (n_pairs,), device=embeddings.device)
    j = torch.randint(0, B, (n_pairs,), device=embeddings.device)

    dE  = torch.norm(embeddings[i] - embeddings[j], dim=1)
    dUV = torch.norm(points[i] - points[j], dim=1)

    # robust scaling (median)
    scaleE  = dE.median() + 1e-9
    scaleUV = dUV.median() + 1e-9
    if detach_scales:
        scaleE = scaleE.detach()
        scaleUV = scaleUV.detach()

    return ((dE / scaleE - dUV / scaleUV) ** 2).mean()

def assign_batch_labels(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 2) points (already normalized)
    centers: (K, 2)
    Returns: (B,) cluster index per point
    """
    # (B, K)
    dists = torch.cdist(x, centers, p=2)
    return dists.argmin(dim=1)

def supervised_contrastive_loss(z, labels, temperature=0.07, eps=1e-12):
    z = F.normalize(z, dim=1)
    logits = torch.matmul(z, z.T) / temperature

    logits_mask = torch.ones_like(logits, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T) & logits_mask
    neg_mask = logits_mask & ~positive_mask

    # log-softmax over all non-self pairs
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    pos_counts = positive_mask.sum(dim=1)
    valid = pos_counts > 0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_counts.clamp_min(1)
    return -(mean_log_prob_pos[valid]).mean()
    
def train_loop(points, kmeans, transform=None, num_epochs=100, batch_size=512, lr=1e-3, device='cuda'):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(points).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    embedding_dimension = 4096
    model = PointTokenizer(
        kmeans_or_centers=kmeans,
        D=embedding_dimension,
        dropout=0.0,
        hidden_dim=256,
        transform=transform,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-7)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_geom = 0.0
        total_contrastive = 0.0
        grad_norms = []

        # freeze decoder for first 10 epochs
        if epoch < 10:
            for p in model.decoder.parameters(): p.requires_grad = False
        else:
            for p in model.decoder.parameters(): p.requires_grad = True

        for batch in dataloader:        
            x = batch[0].to(device)  # (B, 2)

            # Here, embeddings --> decoder can go through soft assignment, but geom loss is calculated on hard assignments
            optimizer.zero_grad()
            # make it easier to learn a useful embedding in the beginning
            # K = 16 if epoch < 20 else 1
            K = max(1, 2 ** max(0, 5 - epoch // 10)) # goes from 16 to 1 in first 50 epochs
            embeddings = model.embedding(x, K=K)  # (B, D)
            recon_points = model.decoder(embeddings)  # (B, 2)

            mse_loss = mse_loss_fn(recon_points, x)
            geom_loss = geometry_loss(embeddings, x, n_pairs=4096)
            
            # Compute contrastive loss
            labels = assign_batch_labels(x, model.embedding.C)
            contrastive_loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)
            
            # Combine losses with warmup for contrastive loss
            # constrasive_weight --> goes from 0 to 0.5 in first 10 epochs
            contrastive_weight = min(0.5, 0.05 * (epoch / 10))

            # loss = mse_loss # NOTE: just mse
            # de-weight geom and constrastive losses heavily
            loss = mse_loss + (0.01 * geom_loss) + (contrastive_weight * 0.01 * contrastive_loss)
            loss.backward()

            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_recon += mse_loss.item() * bs
            total_geom += geom_loss.item() * bs
            total_contrastive += contrastive_loss.item() * bs

        denom = len(dataset)
        avg_loss = total_loss / denom
        avg_recon = total_recon / denom
        avg_geom = total_geom / denom
        avg_contrastive = total_contrastive / denom
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        print(f"{epoch+1}/{num_epochs} | total {avg_loss:.6f} | recon {avg_recon:.6f} | geom {avg_geom:.6f} | contrastive {avg_contrastive:.6f} | grad {avg_grad_norm:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")
        wandb.log({
            "epoch": epoch+1,
            "loss": avg_loss,
            "loss_recon": avg_recon,
            "loss_geom": avg_geom,
            "loss_contrastive": avg_contrastive,
            "grad_norm": avg_grad_norm,
        })
        scheduler.step()

    return model

if __name__ == "__main__":
    # save_data() # Uncomment to re-save data from NuScenes
    points = np.load("points_xy.npy")
    kmeans = get_clusters(points, num_clusters=(nc:=256))
    # viz(kmeans, points)

    points_normalized, kmeans, transform = preprocess(points, kmeans)
    viz(kmeans, points_normalized)

    # train
    wandb.init(project="vladiffusion", name=f"tok_{nc}pts_3s")
    model = train_loop(points_normalized, kmeans, transform=transform, num_epochs=100, batch_size=4096, lr=1.5e-4, device='cuda')
    torch.save(model.state_dict(), "tokenizer_model.pth")
    wandb.finish()
