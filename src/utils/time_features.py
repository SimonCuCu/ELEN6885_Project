"""Time feature processing utilities.

This module provides functions to process temporal information:
- Bucketing time deltas for embedding
"""

import torch


def bucket_time_delta(
    timestamps: torch.Tensor,
    num_buckets: int = 20
) -> torch.Tensor:
    """Convert timestamps to bucketed time deltas.

    Computes time differences between consecutive timestamps,
    applies log-scale transformation, and buckets into discrete bins.

    Args:
        timestamps: (batch, seq_len) tensor of timestamps in seconds
        num_buckets: Number of discrete buckets (default: 20)

    Returns:
        Bucketed time deltas: (batch, seq_len) tensor with values in [0, num_buckets-1]
        First position (no previous timestamp) gets bucket 0.

    Example:
        >>> timestamps = torch.tensor([[1000, 2000, 3000, 10000]])
        >>> buckets = bucket_time_delta(timestamps, num_buckets=20)
        >>> buckets.shape
        torch.Size([1, 4])
        >>> buckets[0, 0].item()  # First position
        0
    """
    batch_size, seq_len = timestamps.shape
    device = timestamps.device

    # Initialize buckets (all zeros) - on same device as input
    buckets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    if seq_len <= 1:
        return buckets

    # Compute time deltas (current - previous)
    # Shape: (batch, seq_len - 1)
    time_deltas = timestamps[:, 1:] - timestamps[:, :-1]

    # Clamp negative deltas to 0 (shouldn't happen in well-ordered data)
    time_deltas = torch.clamp(time_deltas, min=1)

    # Apply log transform to compress large deltas
    # Add 1 to avoid log(0)
    log_deltas = torch.log(time_deltas.float() + 1.0)

    # Normalize to [0, 1] range for bucketing
    # Use a heuristic max value (e.g., log(1 year in seconds) ≈ 17.5)
    max_log_delta = 20.0  # Covers deltas up to ~485M seconds (~15 years)
    normalized = log_deltas / max_log_delta
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Convert to bucket indices [0, num_buckets-1]
    bucket_indices = (normalized * (num_buckets - 1)).long()

    # Assign buckets (skip first position, keep it as 0)
    buckets[:, 1:] = bucket_indices

    return buckets
