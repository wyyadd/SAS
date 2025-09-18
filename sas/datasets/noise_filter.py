import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def detect_noise_dbscan(vel: torch.Tensor, a_type: int, plot: bool = False) -> torch.Tensor:
    vel_norm = torch.norm(vel, dim=-1).numpy()
    nonzero_mask = vel_norm > 0
    if np.sum(nonzero_mask) < 4:
        if plot:
            return torch.tensor([], dtype=torch.int)
        else:
            return torch.arange(vel.shape[0])

    data = vel.numpy()

    q20, q80 = np.quantile(vel_norm[nonzero_mask], [0.2, 0.8])
    eps_min = {
        1: 2.0,  # VEHICLE
        2: 0.3,  # PEDESTRIAN
        3: 1.0,  # CYCLIST
    }[a_type]
    eps = max((q80 - q20) * 1.5, eps_min)
    min_samples = max(data.shape[0] // 10, 4)

    # DBSCAN train
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_

    # noise label is -1
    noise_indices = np.where(labels == -1)[0]

    if plot and len(noise_indices) > 0:
        plt.figure(figsize=(6, 6))
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, col in zip(unique_labels, colors):
            cluster_points = data[labels == label]
            if label == -1:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c="k", marker="x", s=100, label="Outliers")
            else:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[col], alpha=0.6, label=f"Cluster {label}")

        plt.legend()
        plt.title(f"type={int(a_type)} eps={eps:.2f}, min_samples={min_samples}")
        plt.xlabel("vx")
        plt.ylabel("vy")
        plt.grid(True)
        plt.show()

    return torch.from_numpy(noise_indices)


def filter_noise(
        mask: torch.Tensor,
        vel: torch.Tensor,
        pos: torch.Tensor,
        agent_type: torch.Tensor,
        plot: bool = False,
) -> torch.Tensor:
    new_mask = mask.clone()
    for n in range(pos.size(0)):  # traverse agent
        valid_idx = torch.where(mask[n])[0]
        if valid_idx.numel() < 2:
            new_mask[n] = False
            continue

        gaps = torch.where(valid_idx[1:] - valid_idx[:-1] > 1)[0] + 1
        splits = torch.tensor_split(valid_idx, gaps.tolist())

        for seg in splits:
            p = pos[n, seg]  # [L, 2]
            v = torch.cat([vel[n, seg][0:1], (p[1:] - p[:-1]) / 0.1], dim=0)
            noise_indices = detect_noise_dbscan(v, int(agent_type[n]), plot)

            if len(noise_indices) > 0:
                new_mask[n][seg][noise_indices] = False

            if plot and len(noise_indices) > 0:
                traj = p.numpy()
                traj = traj - traj[0]
                fig, ax = plt.subplots(figsize=(6, 6))

                ax.scatter(traj[:, 0], traj[:, 1], s=20)
                ax.scatter(traj[0, 0], traj[0, 1], c="green", s=50, marker="x")  # start
                ax.scatter(traj[-1, 0], traj[-1, 1], c="red", s=50, marker="o")  # end

                ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
                ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
                ax.set_aspect("equal", adjustable="box")
                # ax.set_title(title)
                plt.show()

    return new_mask
