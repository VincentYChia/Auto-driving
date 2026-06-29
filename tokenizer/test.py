from tqdm import tqdm
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import os
from dotenv import load_dotenv
from nuscenes_dataset import NuScenesDataset
from pyquaternion import Quaternion

from tokenizer import viz

def get_xy_points_by_horizon(dataset, horizons=(1, 2, 3)):
    hz = dataset.future_hz
    idx_for = {t: max(0, int(round(t * hz)) - 1) for t in horizons}
    buckets = {t: [] for t in horizons}

    for sample in tqdm(dataset.samples):
        traj_world = dataset._get_future_trajectory(sample)
        if traj_world.size == 0:
            continue

        lidar_sd = dataset.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = dataset.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        t0 = np.asarray(ego_pose["translation"])
        R_world_to_ego = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
        traj_local = (R_world_to_ego @ (traj_world - t0).T).T.astype(np.float32)

        for t, idx in idx_for.items():
            idx = min(idx, traj_local.shape[0] - 1)  # clamp if the track is shorter
            buckets[t].append(traj_local[idx, :2])

    return {t: np.vstack(points).astype(np.float32) for t, points in buckets.items() if points}


def save_data():
    load_dotenv()

    VERSION = 'v1.0-trainval'
    DATAROOT = os.getenv("NUSCENES_ROOT")
    dataset = NuScenesDataset(nuscenes_path=DATAROOT, version=VERSION, split='train', future_seconds=3, future_hz=2, get_img_data=False)
    xy_by_t = get_xy_points_by_horizon(dataset)

    for t, arr in xy_by_t.items():
        np.save(f"points_xy_{t}s.npy", arr)
        plt.figure(figsize=(8, 8))
        plt.scatter(arr[:, 0], arr[:, 1], s=0.1, alpha=0.5)
        plt.title(f"Future Waypoints @ {t}s")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(f"future_waypoints_{t}s.png", dpi=300)
        plt.close()
        print(f"Saved {len(arr)} points for {t}s horizon")

def get_clusters(points: np.ndarray, num_clusters=2048):
    nbrs = NearestNeighbors(n_neighbors=16).fit(points)
    dists, _ = nbrs.kneighbors(points)
    # larger mean distance => lower density => higher weight
    w = (dists.mean(axis=1) + 1e-6)**2
    w /= w.mean()

    # actually, make w uniform
    w = np.ones_like(w)

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, 
                             batch_size=16384, 
                             random_state=42,
                             reassignment_ratio=0.05,
                             max_iter=1000,
                             n_init=2,
                             verbose=1)
    kmeans.fit(points, sample_weight=w)
    return kmeans

def farthest_point_clustering(points: np.ndarray, num_clusters=2048):
    N, _ = points.shape
    centers = np.zeros((num_clusters, 2), dtype=np.float32)
    chosen = np.zeros(N, dtype=bool)

    # 1) pick first center randomly
    idx = np.random.randint(N)
    centers[0] = points[idx]
    chosen[idx] = True

    # 2) iteratively pick the farthest point from existing centers
    dists = np.linalg.norm(points - centers[0:1], axis=1)  # (N,)
    for k in range(1, num_clusters):
        idx = np.argmax(dists * (~chosen))  # ignore already chosen points
        centers[k] = points[idx]
        chosen[idx] = True
        new_dists = np.linalg.norm(points - centers[k:k+1], axis=1)
        dists = np.minimum(dists, new_dists)
        if k % 100 == 0:
            print(f"Chose {k} centers")

    class DummyKMeans:
        def __init__(self, centers):
            self.cluster_centers_ = centers

    return DummyKMeans(centers)

def get_clusters_with_outlier_removal(points: np.ndarray, num_clusters=2048, outlier_percentile=95):
    """
    Remove outliers based on neighbor density, cluster on dense regions, then assign outliers back
    """
    print(f"Original points: {len(points)}")
    
    # 1) Calculate density (inverse of average distance to neighbors)
    nbrs = NearestNeighbors(n_neighbors=4).fit(points)
    dists, _ = nbrs.kneighbors(points)
    avg_neighbor_dist = dists.mean(axis=1)  # Average distance to 16 nearest neighbors
    
    # Higher distance = lower density = more likely to be outlier
    threshold = np.percentile(avg_neighbor_dist, outlier_percentile)
    
    # Split into core (dense) and outlier (sparse) points
    is_core = avg_neighbor_dist <= threshold
    core_points = points[is_core]
    outlier_points = points[~is_core]
    
    print(f"Core points: {len(core_points)}, Outliers: {len(outlier_points)}")
    print(f"Outlier threshold (avg neighbor dist): {threshold:.4f}")
    
    # 2) Cluster only on core points, using density weighting
    core_dists = avg_neighbor_dist[is_core]
    w = (core_dists + 1e-6)**2  # Higher distance = higher weight (same as original)
    w /= w.mean()
    
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, 
        batch_size=16384, 
        random_state=42,
        reassignment_ratio=0.05,
        max_iter=1000,
        n_init=2,
        verbose=1
    )
    kmeans.fit(core_points, sample_weight=w)
    
    print(f"Clustered {len(core_points)} core points into {num_clusters} clusters")
    
    # 3) Assign outliers to nearest centers (but don't move the centers)
    if len(outlier_points) > 0:
        outlier_labels = kmeans.predict(outlier_points)
        print(f"Assigned {len(outlier_points)} outliers to existing clusters")
    
    return kmeans

def hierarchical_density_clustering(points, num_clusters=4096, hierarchy_levels=3):
    """
    Multi-level clustering that allocates anchors proportionally to local density
    """
    print(f"Starting hierarchical clustering with {len(points)} points")
    
    # Level 1: Macro clustering to identify major regions
    macro_clusters = max(32, num_clusters // 128)  # ~32 macro regions
    print(f"\n=== Level 1: Creating {macro_clusters} macro regions ===")
    
    macro_kmeans = MiniBatchKMeans(
        n_clusters=macro_clusters,
        batch_size=16384,
        random_state=42,
        n_init=10,
        verbose=0
    )
    macro_labels = macro_kmeans.fit_predict(points)
    
    # Level 2: Analyze each macro region and allocate micro-clusters
    all_centers = []
    
    for macro_id in range(macro_clusters):
        macro_mask = macro_labels == macro_id
        macro_points = points[macro_mask]
        n_points = len(macro_points)
        
        if n_points == 0:
            continue
        
        # Allocate clusters proportionally to sqrt(point count)
        # sqrt gives better balance than linear proportion
        cluster_allocation = max(1, int(num_clusters * np.sqrt(n_points) / np.sqrt(len(points) * 2)))
        
        print(f"Macro region {macro_id}: {n_points} points -> {cluster_allocation} clusters")
        
        # Level 3: Cluster within this macro region
        if cluster_allocation == 1:
            # Just use centroid
            all_centers.append(macro_points.mean(axis=0))
        else:
            local_kmeans = MiniBatchKMeans(
                n_clusters=min(cluster_allocation, len(macro_points)),
                batch_size=min(4096, len(macro_points)),
                random_state=42,
                n_init=3,
                verbose=0
            )
            local_kmeans.fit(macro_points)
            all_centers.extend(local_kmeans.cluster_centers_)
    
    # Combine all centers
    final_centers = np.array(all_centers)
    print(f"\n=== Created {len(final_centers)} total anchors ===")
    
    # If we have too many or too few, adjust
    if len(final_centers) != num_clusters:
        print(f"Adjusting from {len(final_centers)} to {num_clusters} clusters...")
        adjustment_kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            init=final_centers[:num_clusters] if len(final_centers) > num_clusters else 'k-means++',
            batch_size=16384,
            n_init=1,
            max_iter=100,
            verbose=1
        )
        adjustment_kmeans.fit(points)
        final_centers = adjustment_kmeans.cluster_centers_
    
    # Create a mock kmeans object for compatibility
    class MockKMeans:
        def __init__(self, centers):
            self.cluster_centers_ = centers
            self._tree = KDTree(centers)
        
        def predict(self, X):
            _, labels = self._tree.query(X)
            return labels
    
    return MockKMeans(final_centers)

if __name__ == "__main__":
    save_data()

    points = np.load("points_xy_1s.npy")
    print(f"Loaded {len(points)} points")
    kmeans = get_clusters(points, num_clusters=4096)
    # kmeans = farthest_point_clustering(points, num_clusters=4096)
    # kmeans = get_clusters_with_outlier_removal(points, num_clusters=4096, outlier_percentile=98)
    # kmeans = hierarchical_density_clustering(points, num_clusters=4096, hierarchy_levels=3)
    viz(kmeans, points)
    centers = torch.from_numpy(kmeans.cluster_centers_).float()


    # Process in batches to avoid OOM
    batch_size = 10000
    all_dists = []

    for i in tqdm(range(0, len(points), batch_size)):
        batch_pts = torch.from_numpy(points[i:i+batch_size]).float()
        batch_dists = torch.cdist(batch_pts, centers, p=2).min(dim=1)[0].numpy()
        all_dists.append(batch_dists)

    dists = np.concatenate(all_dists)

    plt.figure(figsize=(8, 6))
    plt.hist(dists, bins=1500, alpha=0.7)
    plt.xlabel("Distance to nearest codebook center")
    plt.ylabel("Count")
    plt.title("Distribution of distances to codebook")
    plt.savefig("imgs/codebook_distance_histogram.png", dpi=500)

    print(f"Mean distance to nearest center: {dists.mean()}")
    print(f"Median distance to nearest center: {np.median(dists)}")