from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def code_perplexity(counts: np.ndarray, eps: float = 1e-8) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / (counts.sum() + eps)
    entropy = -np.sum(probs * np.log(probs + eps))
    return float(np.exp(entropy))


def active_code_count(counts: np.ndarray, min_freq: float = 1e-3) -> int:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() == 0:
        return 0
    probs = counts / counts.sum()
    return int(np.sum(probs >= min_freq))


def clustering_scores(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return {"ari": float(ari), "nmi": float(nmi)}


def purity_scores(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[float, float]:
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    clusters = np.unique(pred_labels)
    classes = np.unique(true_labels)

    # purity
    total = len(true_labels)
    purity = 0.0
    for c in clusters:
        idx = pred_labels == c
        if idx.sum() == 0:
            continue
        counts = [np.sum(true_labels[idx] == k) for k in classes]
        purity += np.max(counts)
    purity /= max(total, 1)

    # inverse purity
    inv_purity = 0.0
    for k in classes:
        idx = true_labels == k
        if idx.sum() == 0:
            continue
        counts = [np.sum(pred_labels[idx] == c) for c in clusters]
        inv_purity += np.max(counts)
    inv_purity /= max(total, 1)

    return float(purity), float(inv_purity)
