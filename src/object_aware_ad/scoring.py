import numpy as np
import torch


def compute_patch_anomaly(query_patch, normal_bank, chunk_size=512, device=None):
    values = []
    if device is None:
        device = normal_bank.device
    normal_bank = normal_bank.to(device, non_blocking=True)
    query_patch = query_patch.to(device, non_blocking=True)

    for i in range(0, query_patch.shape[0], chunk_size):
        q = query_patch[i:i + chunk_size]
        sim = q @ normal_bank.T
        max_sim = sim.max(dim=1).values
        values.append((1.0 - max_sim).detach().cpu())

    return torch.cat(values, dim=0).numpy()


def compute_image_scores(anomaly_flat, top_patch_nums, top_ratios):
    sorted_scores = np.sort(anomaly_flat)
    scores = {
        "score_mean": float(anomaly_flat.mean()),
        "score_max": float(anomaly_flat.max()),
    }

    for k in top_patch_nums:
        kk = min(int(k), len(sorted_scores))
        scores[f"score_top{int(k)}patch_mean"] = float(sorted_scores[-kk:].mean())

    for ratio in top_ratios:
        k = max(1, int(len(sorted_scores) * float(ratio)))
        pct = int(round(float(ratio) * 100))
        scores[f"score_top{pct}pct_mean"] = float(sorted_scores[-k:].mean())

    return scores
