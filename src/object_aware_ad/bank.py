import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def normalize_inplace(features, eps=1e-12):
    norm = features.norm(dim=-1, keepdim=True).clamp_min(eps)
    features.div_(norm)
    return features


def keep_num_from_ratio(n, ratio):
    if ratio is None or ratio >= 1:
        return n
    return max(1, min(n, int(np.ceil(n * ratio))))


def random_sample(bank, keep_num):
    if keep_num >= bank.shape[0]:
        return bank
    idx = torch.randperm(bank.shape[0])[:keep_num]
    return bank[idx]


def random_projection(features, out_dim, seed=42):
    if out_dim is None or out_dim <= 0 or features.shape[1] <= out_dim:
        return features
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    proj = torch.randn(features.shape[1], out_dim, generator=generator, dtype=features.dtype)
    proj = F.normalize(proj, dim=0)
    return features @ proj


def greedy_coreset_indices(features, keep_num):
    if keep_num >= features.shape[0]:
        return torch.arange(features.shape[0])

    features = F.normalize(features, dim=-1)
    selected = [torch.randint(0, features.shape[0], (1,)).item()]
    min_dist = torch.full((features.shape[0],), float("inf"))

    for _ in tqdm(range(1, keep_num), desc="Coreset", leave=False):
        last = features[selected[-1]].unsqueeze(0)
        sim = (features @ last.T).squeeze(1)
        dist = 1.0 - sim
        min_dist = torch.minimum(min_dist, dist)
        selected.append(torch.argmax(min_dist).item())

    return torch.tensor(selected)


def greedy_coreset(bank, keep_num):
    idx = greedy_coreset_indices(bank, keep_num)
    return bank[idx]


def fast_greedy_coreset(bank, keep_num, max_candidates=120000, projection_dim=128):
    if keep_num >= bank.shape[0]:
        return bank

    bank = normalize_inplace(bank)
    candidate_count = min(int(max_candidates), int(bank.shape[0]))

    if candidate_count < bank.shape[0]:
        candidate_idx = torch.randperm(bank.shape[0])[:candidate_count]
        candidates = bank[candidate_idx]
    else:
        candidate_idx = None
        candidates = bank

    work = random_projection(candidates, projection_dim)
    idx_in_candidates = greedy_coreset_indices(work, min(keep_num, candidates.shape[0]))
    return candidates[idx_in_candidates[:keep_num]]


def adaptive_cluster_count(n, keep_num, target_size=20000, min_clusters=16, max_clusters=256):
    if n <= 0:
        return 0
    k = int(round(float(n) / float(max(1, target_size))))
    k = max(int(min_clusters), min(int(max_clusters), k))
    k = min(k, int(n), int(keep_num))
    return max(1, k)


def allocate_cluster_quota(counts, keep_num, mode="sqrt"):
    counts = np.asarray(counts, dtype=np.int64)
    nonzero = counts > 0
    quota = np.zeros_like(counts, dtype=np.int64)
    if keep_num <= 0 or not nonzero.any():
        return quota

    if mode == "uniform":
        weights = nonzero.astype(np.float64)
    elif mode == "proportional":
        weights = counts.astype(np.float64)
    elif mode == "sqrt":
        weights = np.sqrt(counts.astype(np.float64))
    else:
        raise ValueError(f"Unknown cluster allocation mode: {mode}")

    weights[~nonzero] = 0.0
    raw = weights / weights.sum() * int(keep_num)
    quota = np.floor(raw).astype(np.int64)

    if keep_num >= int(nonzero.sum()):
        quota[(quota == 0) & nonzero] = 1

    quota = np.minimum(quota, counts)
    remaining = int(keep_num - quota.sum())
    fractional_order = np.argsort(-(raw - np.floor(raw)))

    while remaining > 0:
        progressed = False
        for c in fractional_order:
            if remaining <= 0:
                break
            if counts[c] > quota[c]:
                quota[c] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    while remaining < 0:
        for c in np.argsort(quota):
            if remaining >= 0:
                break
            if quota[c] > 0:
                quota[c] -= 1
                remaining += 1

    return quota


def cluster_balanced_sample(
    bank,
    keep_num,
    allocation="sqrt",
    target_size=20000,
    min_clusters=16,
    max_clusters=256,
    batch_size=50000,
    projection_dim=0,
    seed=42,
):
    if keep_num >= bank.shape[0]:
        return bank, {
            "cluster_count": 0,
            "cluster_allocation": allocation,
            "cluster_projection_dim": int(projection_dim or 0),
        }

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError("cluster_balanced sampling requires scikit-learn.") from exc

    bank = normalize_inplace(bank.cpu())
    n = int(bank.shape[0])
    k = adaptive_cluster_count(
        n,
        keep_num,
        target_size=target_size,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    work_bank = random_projection(bank, int(projection_dim), seed=seed) if projection_dim else bank
    work_bank = normalize_inplace(work_bank.contiguous())

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        batch_size=int(batch_size),
        n_init=3,
        reassignment_ratio=0.0,
        verbose=0,
    )

    for start in tqdm(range(0, n, int(batch_size)), desc="Fit cluster bank", leave=False):
        chunk = work_bank[start:start + int(batch_size)].numpy()
        kmeans.partial_fit(chunk)

    counts = np.zeros(k, dtype=np.int64)
    for start in tqdm(range(0, n, int(batch_size)), desc="Count clusters", leave=False):
        chunk = work_bank[start:start + int(batch_size)].numpy()
        labels = kmeans.predict(chunk)
        counts += np.bincount(labels, minlength=k)

    quota = allocate_cluster_quota(counts, keep_num, mode=allocation)
    candidates = {c: (np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)) for c in range(k) if quota[c] > 0}
    centers = kmeans.cluster_centers_.astype(np.float32, copy=False)

    for start in tqdm(range(0, n, int(batch_size)), desc="Select cluster reps", leave=False):
        chunk = work_bank[start:start + int(batch_size)].numpy()
        labels = kmeans.predict(chunk)
        diff = chunk - centers[labels]
        dist = np.einsum("ij,ij->i", diff, diff)

        for c in np.unique(labels):
            q = int(quota[c])
            if q <= 0:
                continue
            local = np.where(labels == c)[0]
            if local.size > q:
                local = local[np.argpartition(dist[local], q - 1)[:q]]
            old_dist, old_idx = candidates[c]
            merged_dist = np.concatenate([old_dist, dist[local].astype(np.float32, copy=False)])
            merged_idx = np.concatenate([old_idx, (start + local).astype(np.int64, copy=False)])
            if merged_idx.size > q:
                best = np.argpartition(merged_dist, q - 1)[:q]
                merged_dist = merged_dist[best]
                merged_idx = merged_idx[best]
            candidates[c] = (merged_dist, merged_idx)

    selected = np.concatenate([idx for _, idx in candidates.values()])
    if selected.size > keep_num:
        selected = selected[:keep_num]
    selected = np.sort(selected)

    info = {
        "cluster_count": int(k),
        "cluster_allocation": allocation,
        "cluster_projection_dim": int(projection_dim or 0),
        "cluster_target_size": int(target_size),
        "cluster_min": int(min_clusters),
        "cluster_max": int(max_clusters),
        "cluster_batch_size": int(batch_size),
        "cluster_nonempty": int((counts > 0).sum()),
        "cluster_quota_min": int(quota[quota > 0].min()) if (quota > 0).any() else 0,
        "cluster_quota_max": int(quota.max()) if quota.size else 0,
    }
    return bank[torch.from_numpy(selected).long()], info


def trim_normal_outliers(bank, trim_ratio):
    if trim_ratio <= 0:
        return bank, 0
    trim_ratio = min(max(float(trim_ratio), 0.0), 0.95)

    bank = normalize_inplace(bank)
    center = F.normalize(bank.mean(dim=0, keepdim=True), dim=-1)
    dist = (1.0 - (bank @ center.T).squeeze(1)).cpu().numpy()

    keep_num = max(1, int(np.ceil(len(dist) * (1.0 - trim_ratio))))
    keep_idx = np.argsort(dist)[:keep_num]
    return bank[torch.tensor(keep_idx)], int(len(dist) - keep_num)


def sample_bank(
    bank,
    sampling,
    bank_ratio,
    trim_ratio=0.0,
    fast_coreset_max_candidates=120000,
    fast_coreset_projection_dim=128,
    cluster_allocation="sqrt",
    cluster_target_size=20000,
    cluster_min=16,
    cluster_max=256,
    cluster_batch_size=50000,
    cluster_projection_dim=0,
    seed=42,
):
    before = int(bank.shape[0])
    bank = normalize_inplace(bank)
    trimmed = 0
    actual_sampling = sampling
    sampling_info = {}

    if sampling in {"trimmed_random", "trimmed_coreset"}:
        bank, trimmed = trim_normal_outliers(bank, trim_ratio)

    keep_num = keep_num_from_ratio(int(bank.shape[0]), bank_ratio)

    if sampling in {"none", "all"}:
        sampled = bank
        actual_sampling = "none"
    elif sampling in {"random", "trimmed_random"}:
        sampled = random_sample(bank, keep_num)
    elif sampling in {"coreset", "trimmed_coreset"}:
        sampled = fast_greedy_coreset(
            bank,
            keep_num,
            max_candidates=fast_coreset_max_candidates,
            projection_dim=fast_coreset_projection_dim,
        )
    elif sampling == "cluster_balanced":
        sampled, sampling_info = cluster_balanced_sample(
            bank,
            keep_num,
            allocation=cluster_allocation,
            target_size=cluster_target_size,
            min_clusters=cluster_min,
            max_clusters=cluster_max,
            batch_size=cluster_batch_size,
            projection_dim=cluster_projection_dim,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown sampling: {sampling}")

    sampled = normalize_inplace(sampled)
    return {
        "bank": sampled,
        "patches_before_trim": before,
        "patches_trimmed": trimmed,
        "patches_before_sampling": int(bank.shape[0]),
        "patches_after_sampling": int(sampled.shape[0]),
        "actual_sampling": actual_sampling,
        **sampling_info,
    }


def build_normal_bank(
    records,
    get_patch_fn,
    sampling,
    bank_ratio,
    trim_ratio,
    fast_coreset_max_candidates=120000,
    fast_coreset_projection_dim=128,
    cluster_allocation="sqrt",
    cluster_target_size=20000,
    cluster_min=16,
    cluster_max=256,
    cluster_batch_size=50000,
    cluster_projection_dim=0,
    seed=42,
):
    rs = records.copy()
    random.shuffle(rs)

    patches = []
    used = []
    grid_hw = None

    for r in tqdm(rs, desc="Build normal bank", leave=False):
        patch, grid_hw = get_patch_fn(r)
        patches.append(patch)
        used.append(r["resolved_crop_path"])

    if not patches:
        return None

    bank = torch.cat(patches, dim=0)
    del patches
    sampled = sample_bank(
        bank,
        sampling=sampling,
        bank_ratio=bank_ratio,
        trim_ratio=trim_ratio,
        fast_coreset_max_candidates=fast_coreset_max_candidates,
        fast_coreset_projection_dim=fast_coreset_projection_dim,
        cluster_allocation=cluster_allocation,
        cluster_target_size=cluster_target_size,
        cluster_min=cluster_min,
        cluster_max=cluster_max,
        cluster_batch_size=cluster_batch_size,
        cluster_projection_dim=cluster_projection_dim,
        seed=seed,
    )
    sampled["grid_hw"] = grid_hw
    sampled["used_bank_paths"] = used
    sampled["normal_bank_images"] = len(used)
    return sampled
