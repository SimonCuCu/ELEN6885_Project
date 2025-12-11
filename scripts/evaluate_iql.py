"""Evaluate IQL policy using NX_0 and Top-K metrics.

This script evaluates a trained IQL policy using:
1. NX_0 off-policy evaluation with BC behavior policy
2. Top-K ranking metrics (Recall@K, NDCG@K, HitRate@K)

Usage:
    python scripts/evaluate.py --iql_checkpoint checkpoints/iql/iql_best.pt --bc_checkpoint checkpoints/bc/bc_policy_best.pt
    python scripts/evaluate.py --data_dir data/processed --k_values 5 10 20

Example:
    python scripts/evaluate.py --iql_checkpoint checkpoints/iql/iql_final.pt --bc_checkpoint checkpoints/bc/bc_policy_final.pt --output_path results/eval.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data_module.dataset import MovieLensIQLDataset
from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder
from src.models.iql.state_encoder import StateEncoder
from src.models.iql.networks import PolicyNetwork
from src.models.bc_policy import BCPolicy
from src.evaluation.nx0_evaluator import NX0Evaluator
from src.evaluation.metrics import compute_all_metrics
from src.utils.seed import set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate IQL policy with NX_0 and Top-K metrics"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--iql_checkpoint",
        type=str,
        required=True,
        help="Path to IQL checkpoint"
    )
    parser.add_argument(
        "--bc_checkpoint",
        type=str,
        required=True,
        help="Path to BC policy checkpoint"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed trajectory data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/evaluation.json",
        help="Path to save evaluation results"
    )

    # Model arguments
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Maximum sequence length"
    )

    # Evaluation arguments
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for Top-K metrics"
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=20.0,
        help="Maximum importance weight for NX_0"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small constant for numerical stability"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Split to evaluate
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on"
    )

    return parser.parse_args()


def load_processed_data(data_dir: str, split: str = "test"):
    """Load preprocessed trajectory data.

    Args:
        data_dir: Directory containing processed data
        split: Which split to load ('val' or 'test')

    Returns:
        Tuple of (trajectories, zipcode_vocab, metadata)
    """
    data_path = Path(data_dir)

    # Load trajectories
    traj_file = f"{split}_trajectories.pkl"
    with open(data_path / traj_file, "rb") as f:
        trajectories = pickle.load(f)

    # Load vocabulary
    with open(data_path / "zipcode_vocab.json", "r") as f:
        zipcode_vocab = json.load(f)

    # Load metadata
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return trajectories, zipcode_vocab, metadata


def create_state_encoder(
    max_movie_id: int,
    num_zipcode_buckets: int,
    device: torch.device,
    max_seq_len: int = 50
) -> StateEncoder:
    """Create StateEncoder with default config.

    Args:
        max_movie_id: Maximum movie ID (for embedding size)
        num_zipcode_buckets: Number of zipcode buckets
        device: Device to use
        max_seq_len: Maximum sequence length

    Returns:
        StateEncoder instance
    """
    # Create static feature encoder
    static_encoder = StaticFeatureEncoder(
        num_zipcode_buckets=num_zipcode_buckets,
        zipcode_embed_dim=8
    )

    # Create SASRec encoder
    sasrec_encoder = SASRecEncoder(
        num_movies=max_movie_id,  # Use max_movie_id to ensure all IDs fit
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=max_seq_len
    )

    # Create state encoder
    state_encoder = StateEncoder(static_encoder, sasrec_encoder)
    return state_encoder.to(device)


def compute_popularity_from_trajectories(
    trajectories,
    max_movie_id: int
) -> np.ndarray:
    """Compute item popularity counts from training trajectories.

    Each movie's popularity is defined as the number of times it appears
    in the interaction sequences over all users.

    Args:
        trajectories: List of Trajectory objects from the *train* split
        max_movie_id: Maximum movie ID (inclusive)

    Returns:
        1D numpy array of length (max_movie_id + 1) with popularity counts.
        Index i corresponds to movie ID i.
    """
    counts = np.zeros(max_movie_id + 1, dtype=np.int64)

    for traj in trajectories:
        # traj.movie_ids is a 1D numpy array of movie IDs for this user
        movie_ids = traj.movie_ids

        # Make sure IDs are within valid range and ignore non-positive IDs
        mask = (movie_ids > 0) & (movie_ids <= max_movie_id)
        valid_ids = movie_ids[mask]

        # Increment counts for each movie ID
        np.add.at(counts, valid_ids, 1)

    return counts

def encode_batch_states(
    state_encoder: StateEncoder,
    batch: dict,
    device: torch.device
) -> torch.Tensor:
    """Encode batch of trajectories into states.

    Args:
        state_encoder: StateEncoder instance
        batch: Batch dictionary from DataLoader
        device: Device to use

    Returns:
        States tensor of shape (batch, state_dim)
    """
    # Move batch to device
    gender = batch["gender"].to(device)
    age = batch["age"].to(device)
    occupation = batch["occupation"].to(device)
    zipcode_bucket = batch["zipcode_bucket"].to(device)
    movie_sequence = batch["movie_sequence"].to(device)
    rating_sequence = batch["rating_sequence"].to(device)
    timestamp_sequence = batch["timestamp_sequence"].to(device)
    sequence_mask = batch["sequence_mask"].to(device)

    # Encode states
    states = state_encoder(
        gender=gender,
        age=age,
        occupation=occupation,
        zipcode_bucket=zipcode_bucket,
        movie_sequence=movie_sequence,
        rating_sequence=rating_sequence,
        timestamp_sequence=timestamp_sequence,
        sequence_mask=sequence_mask
    )

    return states


def evaluate_nx0(
    evaluator: NX0Evaluator,
    state_encoder: StateEncoder,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate policy using NX_0.

    Args:
        evaluator: NX0Evaluator instance
        state_encoder: StateEncoder for encoding states
        dataloader: Data loader
        device: Device to use

    Returns:
        Dictionary with NX_0 metrics
    """
    print("\nRunning NX_0 evaluation...")

    all_nx0_values = []
    all_ess_values = []
    all_mean_weights = []

    state_encoder.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Encode states
            states = encode_batch_states(state_encoder, batch, device)

            # Get actions and rewards
            actions = batch["actions"][:, 0].to(device)  # First timestep
            rewards = batch["rewards"][:, 0].to(device)

            # Create mask (all valid since we're using first timestep only)
            mask = torch.ones(states.shape[0], dtype=torch.bool, device=device)

            # Evaluate
            result = evaluator.evaluate_batch(
                states.unsqueeze(1),  # Add time dimension
                actions.unsqueeze(1),
                rewards.unsqueeze(1),
                mask.unsqueeze(1)
            )

            all_nx0_values.append(result['nx0_value'])
            all_ess_values.append(result['effective_sample_size'])
            all_mean_weights.append(result['mean_weight'])

    # Aggregate results
    nx0_metrics = {
        'nx0_value': float(np.mean(all_nx0_values)),
        'nx0_std': float(np.std(all_nx0_values)),
        'effective_sample_size': float(np.mean(all_ess_values)),
        'mean_weight': float(np.mean(all_mean_weights)),
        'max_weight': float(np.max(all_mean_weights)),  # Use max of mean weights
        'weight_std': float(np.std(all_mean_weights))
    }

    return nx0_metrics


def evaluate_topk(
    policy: PolicyNetwork,
    state_encoder: StateEncoder,
    dataloader: DataLoader,
    k_values: List[int],
    device: torch.device
) -> Dict[str, float]:
    """Evaluate policy using Top-K metrics.

    Args:
        policy: IQL policy network
        state_encoder: StateEncoder for encoding states
        dataloader: Data loader
        k_values: List of K values to evaluate
        device: Device to use

    Returns:
        Dictionary with Top-K metrics
    """
    print("\nRunning Top-K evaluation...")

    all_predictions = []
    all_targets = []

    policy.eval()
    state_encoder.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Encode states
            states = encode_batch_states(state_encoder, batch, device)

            # Get target actions (first timestep)
            actions = batch["actions"][:, 0].to(device)

            # Get policy predictions
            predictions = policy(states)

            all_predictions.append(predictions.cpu())
            all_targets.append(actions.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Top-K metrics
    topk_metrics = compute_all_metrics(
        all_predictions,
        all_targets,
        k_values=k_values
    )

    return topk_metrics

def evaluate_topk_popularity(
    popularity_scores: torch.Tensor,
    dataloader: DataLoader,
    k_values: List[int],
    device: torch.device
) -> Dict[str, float]:
    """Evaluate POPULARITY baseline using Top-K metrics.

    This baseline ignores the user state and always recommends movies
    ranked by their global popularity estimated from the *training* split.

    Args:
        popularity_scores: (num_actions,) tensor with popularity scores
        dataloader: Data loader over the evaluation split
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
        device: Device to use

    Returns:
        Dictionary with keys like:
            - 'pop_recall@5', 'pop_ndcg@5', 'pop_hitrate@5', etc.
    """
    print("\nRunning Top-K evaluation for POPULARITY baseline...")

    all_predictions = []
    all_targets = []

    # Normalize to device once
    popularity_scores = popularity_scores.to(device)

    with torch.no_grad():
        for batch in dataloader:
            # Target action: first timestep of the trajectory
            targets = batch["actions"][:, 0].to(device)
            batch_size = targets.shape[0]

            # Same popularity-based scores for every sample in the batch
            preds = popularity_scores.unsqueeze(0).expand(batch_size, -1)

            all_predictions.append(preds)
            all_targets.append(targets)

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute base Top-K metrics
    base_metrics = compute_all_metrics(
        all_predictions,
        all_targets,
        k_values=k_values
    )

    # Add 'pop_' prefix to distinguish from IQL metrics
    prefixed_metrics = {
        f"pop_{name}": value for name, value in base_metrics.items()
    }

    return prefixed_metrics


def evaluate_topk_random(
    num_actions: int,
    dataloader: DataLoader,
    k_values: List[int],
    device: torch.device,
    seed: int = 42
) -> Dict[str, float]:
    """Evaluate RANDOM baseline using Top-K metrics.

    This baseline ignores the user state and recommends items by
    assigning random scores independently for each user.

    Args:
        num_actions: Total number of actions (movies)
        dataloader: Data loader over the evaluation split
        k_values: List of K values to evaluate
        device: Device to use
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys like:
            - 'rand_recall@5', 'rand_ndcg@5', 'rand_hitrate@5', etc.
    """
    print("\nRunning Top-K evaluation for RANDOM baseline...")

    all_predictions = []
    all_targets = []

    # Set RNG seed for reproducibility
    torch.manual_seed(seed)

    with torch.no_grad():
        for batch in dataloader:
            targets = batch["actions"][:, 0].to(device)
            batch_size = targets.shape[0]

            # Random scores ~ Uniform(0, 1) for each user and each action
            preds = torch.rand(batch_size, num_actions, device=device)

            all_predictions.append(preds)
            all_targets.append(targets)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    base_metrics = compute_all_metrics(
        all_predictions,
        all_targets,
        k_values=k_values
    )

    # Add 'rand_' prefix to distinguish from IQL metrics
    prefixed_metrics = {
        f"rand_{name}": value for name, value in base_metrics.items()
    }

    return prefixed_metrics

def print_results(results: Dict[str, float]):
    """Print evaluation results in formatted table.

    Args:
        results: Dictionary with all evaluation metrics
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # NX_0 metrics
    if 'nx0_value' in results:
        print("\nNX_0 Off-Policy Evaluation:")
        print("-" * 70)
        print(f"  NX_0 Value:              {results['nx0_value']:.4f} ± {results.get('nx0_std', 0):.4f}")
        print(f"  Effective Sample Size:   {results['effective_sample_size']:.2f}")
        print(f"  Mean Weight:             {results['mean_weight']:.4f} ± {results.get('weight_std', 0):.4f}")
        print(f"  Max Weight:              {results['max_weight']:.4f}")

    # # Top-K metrics
    # print("\nTop-K Ranking Metrics:")
    # print("-" * 70)
    #
    # # Group by K value
    # k_values = sorted(set(
    #     int(key.split('@')[1])
    #     for key in results.keys()
    #     if '@' in key
    # ))
    #
    # for k in k_values:
    #     print(f"\n  K={k}:")
    #     if f'recall@{k}' in results:
    #         print(f"    Recall@{k}:   {results[f'recall@{k}']:.4f}")
    #     if f'ndcg@{k}' in results:
    #         print(f"    NDCG@{k}:     {results[f'ndcg@{k}']:.4f}")
    #     if f'hitrate@{k}' in results:
    #         print(f"    HitRate@{k}:  {results[f'hitrate@{k}']:.4f}")
    # Top-K metrics
    print("\nTop-K Ranking Metrics:")
    print("-" * 70)

    # Group by K value
    k_values = sorted(set(
        int(key.split('@')[1])
        for key in results.keys()
        if '@' in key
    ))

    for k in k_values:
        print(f"\n  K={k}:")
        if f'recall@{k}' in results:
            print(f"    Recall@{k}:   {results[f'recall@{k}']:.4f}")
        if f'ndcg@{k}' in results:
            print(f"    NDCG@{k}:     {results[f'ndcg@{k}']:.4f}")
        if f'hitrate@{k}' in results:
            print(f"    HitRate@{k}:  {results[f'hitrate@{k}']:.4f}")

    print("\n" + "=" * 70)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)

    print("=" * 70)
    print("IQL Policy Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"IQL checkpoint: {args.iql_checkpoint}")
    print(f"BC checkpoint: {args.bc_checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"K values: {args.k_values}")
    print(f"Clip weight: {args.clip_weight}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading evaluation data...")
    trajectories, zipcode_vocab, metadata = load_processed_data(
        args.data_dir,
        split=args.split
    )
    print(f"  Loaded {len(trajectories)} {args.split} trajectories")
    print(f"  Movies: {metadata['num_movies']}")
    print(f"  Zipcode buckets: {metadata['num_zipcode_buckets']}")

    # Create dataset and dataloader
    dataset = MovieLensIQLDataset(
        trajectories,
        zipcode_vocab,
        max_seq_len=args.max_seq_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Also load TRAIN split for computing popularity baseline
    # We only need the trajectories, not the vocab/metadata again.
    train_trajectories, _, _ = load_processed_data(
        args.data_dir,
        split="train"
    )

    # Compute popularity counts over training trajectories
    popularity_counts = compute_popularity_from_trajectories(
        train_trajectories,
        max_movie_id=metadata["max_movie_id"]
    )

    # Convert to float tensor for scoring (counts themselves are valid scores)
    popularity_scores = torch.from_numpy(
        popularity_counts.astype(np.float32)
    )

    # Create state encoder
    print("\n[2/4] Creating models...")
    state_encoder = create_state_encoder(
        max_movie_id=metadata["max_movie_id"],
        num_zipcode_buckets=metadata["num_zipcode_buckets"],
        device=device,
        max_seq_len=args.max_seq_len
    )

    # Load IQL checkpoint
    print("\n[3/4] Loading checkpoints...")
    iql_checkpoint = torch.load(args.iql_checkpoint, map_location=device, weights_only=False)

    # Load state encoder weights
    if "state_encoder" in iql_checkpoint:
        state_encoder.load_state_dict(iql_checkpoint["state_encoder"])
        print("  Loaded state encoder weights")

    # Create and load IQL policy
    iql_policy = PolicyNetwork(
        state_dim=state_encoder.output_dim,
        num_actions=metadata["max_movie_id"] + 1
    ).to(device)

    if "policy_network" in iql_checkpoint:
        iql_policy.load_state_dict(iql_checkpoint["policy_network"])
    else:
        iql_policy.load_state_dict(iql_checkpoint)
    iql_policy.eval()
    print("  Loaded IQL policy")

    # Load BC policy
    bc_policy = BCPolicy(
        state_dim=state_encoder.output_dim,
        num_actions=metadata["max_movie_id"] + 1
    ).to(device)

    bc_checkpoint = torch.load(args.bc_checkpoint, map_location=device, weights_only=True)
    if "bc_policy" in bc_checkpoint:
        bc_policy.load_state_dict(bc_checkpoint["bc_policy"])
    else:
        bc_policy.load_state_dict(bc_checkpoint)
    bc_policy.eval()
    print("  Loaded BC policy")

    # # Create NX_0 evaluator
    # print("\n[4/4] Running evaluation...")
    # evaluator = NX0Evaluator(
    #     bc_policy=bc_policy,
    #     target_policy=iql_policy,
    #     clip_weight=args.clip_weight,
    #     epsilon=args.epsilon
    # )
    #
    # # Run evaluations
    # nx0_metrics = evaluate_nx0(
    #     evaluator, state_encoder, dataloader, device
    # )
    #
    # topk_metrics = evaluate_topk(
    #     iql_policy, state_encoder, dataloader, args.k_values, device
    # )
    #
    # # Combine results
    # results = {
    #     "split": args.split,
    #     "iql_checkpoint": args.iql_checkpoint,
    #     "bc_checkpoint": args.bc_checkpoint,
    #     **nx0_metrics,
    #     **topk_metrics
    # }

    # Run NX_0 evaluation for IQL policy (with BC as behavior policy)
    nx0_metrics = evaluate_nx0(
        evaluator, state_encoder, dataloader, device
    )

    # Top-K evaluation for IQL policy (state-dependent)
    topk_metrics = evaluate_topk(
        iql_policy, state_encoder, dataloader, args.k_values, device
    )

    # Top-K evaluation for POPULARITY baseline (state-agnostic)
    pop_topk_metrics = evaluate_topk_popularity(
        popularity_scores=popularity_scores,
        dataloader=dataloader,
        k_values=args.k_values,
        device=device
    )

    # Top-K evaluation for RANDOM baseline (state-agnostic)
    rand_topk_metrics = evaluate_topk_random(
        num_actions=metadata["max_movie_id"] + 1,
        dataloader=dataloader,
        k_values=args.k_values,
        device=device,
        seed=args.seed
    )

    # Combine all results
    results = {
        "split": args.split,
        "iql_checkpoint": args.iql_checkpoint,
        "bc_checkpoint": args.bc_checkpoint,
        **nx0_metrics,
        **topk_metrics,       # IQL (un-prefixed) -> e.g., 'recall@5'
        **pop_topk_metrics,   # POP baseline -> 'pop_recall@5', ...
        **rand_topk_metrics,  # RANDOM baseline -> 'rand_recall@5', ...
    }

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
