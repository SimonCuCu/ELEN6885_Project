"""Top-K ranking metrics for recommendation evaluation."""

import torch
from typing import List, Dict


def recall_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10
) -> float:
    """Compute Recall@K metric.

    Recall@K = (# relevant items in top-K) / (# total relevant items)
    For single-target case: Recall@K = 1 if target in top-K, else 0

    Args:
        predictions: (batch, num_actions) scores/logits for all actions
        targets: (batch,) ground truth action indices
        k: Number of top predictions to consider

    Returns:
        Recall@K score in [0, 1], averaged over batch

    Example:
        >>> predictions = torch.randn(32, 3900)
        >>> targets = torch.randint(0, 3900, (32,))
        >>> recall = recall_at_k(predictions, targets, k=10)
        >>> assert 0 <= recall <= 1
    """
    batch_size = predictions.shape[0]

    # Get top-K predictions
    _, top_k_indices = predictions.topk(k, dim=-1)  # (batch, k)

    # Check if target is in top-K for each sample
    # Expand targets to (batch, k) for comparison
    targets_expanded = targets.unsqueeze(1).expand(-1, k)

    # hits: (batch,) boolean, True if target in top-K
    hits = (top_k_indices == targets_expanded).any(dim=1)

    # Recall is fraction of hits
    recall = hits.float().mean().item()

    return recall


def ndcg_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10
) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain).

    NDCG@K measures ranking quality with position-based discount.
    DCG = sum(rel_i / log2(i + 2)) for i in top-K
    NDCG = DCG / IDCG (ideal DCG)

    For binary relevance: rel_i = 1 if item is relevant, 0 otherwise

    Args:
        predictions: (batch, num_actions) scores for all actions
        targets: (batch,) ground truth action indices
        k: Number of top predictions to consider

    Returns:
        NDCG@K score in [0, 1], averaged over batch

    Example:
        >>> predictions = torch.randn(32, 3900)
        >>> targets = torch.randint(0, 3900, (32,))
        >>> ndcg = ndcg_at_k(predictions, targets, k=10)
        >>> assert 0 <= ndcg <= 1
    """
    batch_size = predictions.shape[0]

    # Get top-K predictions
    _, top_k_indices = predictions.topk(k, dim=-1)  # (batch, k)

    # Compute DCG for each sample
    dcg = torch.zeros(batch_size, device=predictions.device)

    for i in range(batch_size):
        # Find position of target in top-K (if present)
        target = targets[i]
        top_k = top_k_indices[i]

        # Check if target is in top-K
        mask = (top_k == target)

        if mask.any():
            # Position in top-K (0-indexed)
            position = mask.nonzero(as_tuple=True)[0][0].item()

            # DCG with position discount: 1 / log2(position + 2)
            dcg[i] = 1.0 / torch.log2(torch.tensor(position + 2.0))

    # IDCG for single relevant item at position 0: 1 / log2(2) = 1.0
    idcg = 1.0

    # NDCG = DCG / IDCG
    ndcg = (dcg / idcg).mean().item()

    return ndcg


def hitrate_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10
) -> float:
    """Compute HitRate@K metric.

    HitRate@K = (# users with relevant item in top-K) / (# users)
    Equivalent to Recall@K for single-target case.

    Args:
        predictions: (batch, num_actions) scores for all actions
        targets: (batch,) ground truth action indices
        k: Number of top predictions to consider

    Returns:
        HitRate@K score in [0, 1]

    Example:
        >>> predictions = torch.randn(32, 3900)
        >>> targets = torch.randint(0, 3900, (32,))
        >>> hitrate = hitrate_at_k(predictions, targets, k=10)
        >>> assert 0 <= hitrate <= 1
    """
    # HitRate@K is same as Recall@K for single-target case
    return recall_at_k(predictions, targets, k)


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Compute all Top-K metrics for multiple K values.

    Args:
        predictions: (batch, num_actions) scores for all actions
        targets: (batch,) ground truth action indices
        k_values: List of K values to evaluate (default: [5, 10, 20])

    Returns:
        Dictionary with keys like:
            - 'recall@5', 'recall@10', 'recall@20'
            - 'ndcg@5', 'ndcg@10', 'ndcg@20'
            - 'hitrate@5', 'hitrate@10', 'hitrate@20'

    Example:
        >>> predictions = torch.randn(32, 3900)
        >>> targets = torch.randint(0, 3900, (32,))
        >>> metrics = compute_all_metrics(predictions, targets)
        >>> print(metrics.keys())
        dict_keys(['recall@5', 'recall@10', 'recall@20', ...])
    """
    metrics = {}

    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(predictions, targets, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
        metrics[f'hitrate@{k}'] = hitrate_at_k(predictions, targets, k)

    return metrics
