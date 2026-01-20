from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from collapse_operator.stats_buffer import StatsBuffer
from utils.metrics import js_divergence


def _entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    probs = probs / (probs.sum() + eps)
    return float(-np.sum(probs * np.log(probs + eps)))


def merge_split_operator(
    stats: StatsBuffer,
    codebook: np.ndarray,
    num_classes: int,
    delta_merge: float,
    delta_merge_inv: float,
    delta_split: float,
    eps_split_gain: float,
    env_ids: List[int],
) -> Dict[str, List[Dict[str, float]]]:
    num_codes = stats.num_codes
    merges = []
    splits = []

    active_codes = [c for c in range(num_codes) if stats.counts[c] > 0]

    # Merge candidates
    for i in range(len(active_codes)):
        for j in range(i + 1, len(active_codes)):
            ci = active_codes[i]
            cj = active_codes[j]
            p_i = stats.distribution(ci, num_classes)
            p_j = stats.distribution(cj, num_classes)
            d_pred = js_divergence(p_i, p_j)
            d_inv = 0.0
            for e in env_ids:
                pe_i = stats.distribution_env(e, ci, num_classes)
                pe_j = stats.distribution_env(e, cj, num_classes)
                d_inv = max(d_inv, js_divergence(pe_i, pe_j))
            if d_pred <= delta_merge and d_inv <= delta_merge_inv:
                merges.append({"code_i": ci, "code_j": cj, "d_pred": d_pred, "d_inv": d_inv})

    # Apply merges (greedy)
    merged_codes = set()
    for merge in merges:
        ci = merge["code_i"]
        cj = merge["code_j"]
        if ci in merged_codes or cj in merged_codes:
            continue
        total = stats.counts[ci] + stats.counts[cj]
        if total == 0:
            continue
        weight_i = stats.counts[ci] / total
        weight_j = stats.counts[cj] / total
        codebook[ci] = weight_i * codebook[ci] + weight_j * codebook[cj]
        # free capacity by reinitializing cj
        codebook[cj] = np.random.normal(0.0, 1.0, size=codebook[cj].shape)
        merged_codes.update([ci, cj])

    # Split candidates
    for c in active_codes:
        p_c = stats.distribution(c, num_classes)
        entropy = _entropy(p_c)
        if entropy < delta_split:
            continue
        embeddings = stats.code_embeddings(c)
        if embeddings.shape[0] < 10:
            continue
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=0)
        cluster_ids = kmeans.fit_predict(embeddings)
        # compute NLL gain using actual future labels per cluster
        counts_before = np.zeros(num_classes, dtype=np.float64)
        for lbl in stats.code_labels(c):
            if lbl < num_classes:
                counts_before[lbl] += 1
        nll_before = -np.sum(counts_before * np.log(counts_before / (counts_before.sum() + 1e-8) + 1e-8))

        counts_a = np.zeros(num_classes, dtype=np.float64)
        counts_b = np.zeros(num_classes, dtype=np.float64)
        for lbl, cluster_id in zip(stats.code_labels(c), cluster_ids):
            if lbl >= num_classes:
                continue
            if cluster_id == 0:
                counts_a[lbl] += 1
            else:
                counts_b[lbl] += 1
        p_a = counts_a / (counts_a.sum() + 1e-8)
        p_b = counts_b / (counts_b.sum() + 1e-8)
        nll_after = -np.sum(counts_a * np.log(p_a + 1e-8)) - np.sum(counts_b * np.log(p_b + 1e-8))
        gain = nll_before - nll_after
        if gain >= eps_split_gain:
            # find least used code to reassign
            free_code = int(np.argmin(stats.counts))
            if free_code == c:
                continue
            codebook[c] = kmeans.cluster_centers_[0]
            codebook[free_code] = kmeans.cluster_centers_[1]
            splits.append({"code": c, "new_code": free_code, "gain": float(gain)})

    return {"merges": merges, "splits": splits}
