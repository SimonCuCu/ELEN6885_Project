"""Train IQL policy on MovieLens-1M trajectory data.

This script implements the complete IQL training pipeline:
1. Load preprocessed trajectory data
2. Initialize StateEncoder (Static + SASRec) and IQL networks
3. Optionally load pretrained BC policy weights
4. Train IQL with V/Q/Policy network updates
5. Validate and log metrics to TensorBoard
6. Save checkpoints

Usage:
    python scripts/train.py
    python scripts/train.py --config-name default training.num_epochs=10
    python scripts/train.py training.device=cuda training.batch_size=256

Example with Hydra:
    python scripts/train.py training.num_epochs=5 model.iql.beta_awr=5.0
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Attempt to import hydra (optional dependency)
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

from src.data_module.dataset import MovieLensIQLDataset
from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder
from src.models.iql.state_encoder import StateEncoder
from src.models.iql.trainer import IQLTrainer
from src.models.bc_policy import BCPolicy
from src.evaluation.nx0_evaluator import NX0Evaluator
from src.evaluation.metrics import compute_all_metrics
from src.utils.seed import set_seed, get_device, worker_init_fn


def load_processed_data(
    data_dir: str
) -> Tuple[list, list, list, dict, dict]:
    """Load preprocessed trajectory data.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (train_trajs, val_trajs, test_trajs, zipcode_vocab, metadata)
    """
    data_path = Path(data_dir)

    with open(data_path / "train_trajectories.pkl", "rb") as f:
        train_trajs = pickle.load(f)

    with open(data_path / "val_trajectories.pkl", "rb") as f:
        val_trajs = pickle.load(f)

    with open(data_path / "test_trajectories.pkl", "rb") as f:
        test_trajs = pickle.load(f)

    with open(data_path / "zipcode_vocab.json", "r") as f:
        zipcode_vocab = json.load(f)

    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return train_trajs, val_trajs, test_trajs, zipcode_vocab, metadata


def create_dataloaders(
    train_trajs: list,
    val_trajs: list,
    zipcode_vocab: dict,
    max_seq_len: int = 50,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        train_trajs: Training trajectories
        val_trajs: Validation trajectories
        zipcode_vocab: Zipcode vocabulary mapping
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = MovieLensIQLDataset(
        train_trajs,
        zipcode_vocab,
        max_seq_len=max_seq_len
    )

    val_dataset = MovieLensIQLDataset(
        val_trajs,
        zipcode_vocab,
        max_seq_len=max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True  # Avoid small batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def create_models(
    max_movie_id: int,
    num_zipcode_buckets: int,
    cfg: dict,
    device: torch.device
) -> Tuple[StateEncoder, IQLTrainer]:
    """Create StateEncoder and IQL Trainer.

    Args:
        max_movie_id: Maximum movie ID (for embedding size)
        num_zipcode_buckets: Number of zipcode buckets
        cfg: Configuration dictionary
        device: Device to use

    Returns:
        Tuple of (state_encoder, iql_trainer)
    """
    # Create static feature encoder
    static_encoder = StaticFeatureEncoder(
        num_zipcode_buckets=num_zipcode_buckets,
        zipcode_embed_dim=cfg.get("static", {}).get("zipcode_embed_dim", 8)
    )

    # Create SASRec encoder
    sasrec_cfg = cfg.get("sasrec", {})
    sasrec_encoder = SASRecEncoder(
        num_movies=max_movie_id,  # Use max_movie_id to ensure all IDs fit
        d_model=sasrec_cfg.get("d_model", 64),
        nhead=sasrec_cfg.get("nhead", 2),
        num_layers=sasrec_cfg.get("num_layers", 2),
        dim_feedforward=sasrec_cfg.get("dim_feedforward", 256),
        dropout=sasrec_cfg.get("dropout", 0.1),
        max_seq_len=cfg.get("max_seq_len", 50)
    )

    # Create state encoder
    state_encoder = StateEncoder(static_encoder, sasrec_encoder)
    state_encoder = state_encoder.to(device)

    # Create IQL trainer
    iql_cfg = cfg.get("iql", {})
    iql_trainer = IQLTrainer(
        state_dim=state_encoder.output_dim,
        num_actions=max_movie_id + 1,  # +1 for padding
        hidden_dim=iql_cfg.get("hidden_dim", 256),
        num_hidden_layers=iql_cfg.get("num_hidden_layers", 2),
        lr=cfg.get("learning_rate", 1e-4),
        gamma=iql_cfg.get("gamma", 0.99),
        tau_expectile=iql_cfg.get("tau_expectile", 0.7),
        beta_awr=iql_cfg.get("beta_awr", 3.0),
        clip_weight=iql_cfg.get("clip_weight", 20.0),
        tau_target=iql_cfg.get("tau_target", 0.005)
    )

    # Move IQL networks to device
    iql_trainer.v_network = iql_trainer.v_network.to(device)
    iql_trainer.q_network = iql_trainer.q_network.to(device)
    iql_trainer.q_target = iql_trainer.q_target.to(device)
    iql_trainer.policy_network = iql_trainer.policy_network.to(device)

    return state_encoder, iql_trainer


def load_bc_weights(
    iql_trainer: IQLTrainer,
    bc_checkpoint_path: str,
    device: torch.device
) -> bool:
    """Load pretrained BC policy weights into IQL policy network.

    Args:
        iql_trainer: IQL trainer instance
        bc_checkpoint_path: Path to BC checkpoint
        device: Device to load to

    Returns:
        True if loaded successfully, False otherwise
    """
    bc_path = Path(bc_checkpoint_path)

    if not bc_path.exists():
        print(f"Warning: BC checkpoint not found at {bc_path}")
        return False

    try:
        checkpoint = torch.load(bc_path, map_location=device, weights_only=True)

        # Get BC policy state dict
        if "bc_policy" in checkpoint:
            bc_state_dict = checkpoint["bc_policy"]
        else:
            bc_state_dict = checkpoint

        # Load into IQL policy network
        iql_trainer.policy_network.load_state_dict(bc_state_dict)
        print(f"Loaded BC weights from {bc_path}")
        return True

    except Exception as e:
        print(f"Warning: Failed to load BC weights: {e}")
        return False


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
        States tensor of shape (batch * seq_len, state_dim)
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


def train_epoch(
    state_encoder: StateEncoder,
    iql_trainer: IQLTrainer,
    train_loader: DataLoader,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    log_interval: int = 100
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        state_encoder: StateEncoder instance
        iql_trainer: IQLTrainer instance
        train_loader: Training dataloader
        device: Device to use
        grad_clip_norm: Gradient clipping norm
        log_interval: Log every N steps

    Returns:
        Dictionary with average epoch metrics
    """
    state_encoder.train()
    iql_trainer.v_network.train()
    iql_trainer.q_network.train()
    iql_trainer.policy_network.train()

    epoch_metrics = {
        "v_loss": [],
        "q_loss": [],
        "policy_loss": [],
        "mean_q_value": [],
        "mean_v_value": [],
        "mean_advantage": []
    }

    for step, batch in enumerate(train_loader):
        # Encode states (current timestep only for simplicity)
        # Detach states to avoid multiple backward passes through encoder
        with torch.no_grad():
            states = encode_batch_states(state_encoder, batch, device)

        # Get actions, rewards, and create next states
        actions = batch["actions"][:, 0].to(device)  # First action
        rewards = batch["rewards"][:, 0].to(device)  # First reward
        dones = batch["dones"][:, 0].bool().to(device)

        # For next states, use same states (simplified)
        # In practice, you'd compute actual next states from trajectory
        next_states = states.clone()

        # IQL update
        metrics = iql_trainer.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )

        # Gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) +
                list(iql_trainer.v_network.parameters()) +
                list(iql_trainer.q_network.parameters()) +
                list(iql_trainer.policy_network.parameters()),
                grad_clip_norm
            )

        # Accumulate metrics
        for key, value in metrics.items():
            epoch_metrics[key].append(value)

        # Log progress
        if (step + 1) % log_interval == 0:
            avg_metrics = {k: np.mean(v[-log_interval:]) for k, v in epoch_metrics.items()}
            print(
                f"  Step {step+1}/{len(train_loader)}: "
                f"V={avg_metrics['v_loss']:.4f}, "
                f"Q={avg_metrics['q_loss']:.4f}, "
                f"π={avg_metrics['policy_loss']:.4f}"
            )

    # Average metrics over epoch
    return {k: np.mean(v) for k, v in epoch_metrics.items()}


def validate(
    state_encoder: StateEncoder,
    iql_trainer: IQLTrainer,
    val_loader: DataLoader,
    device: torch.device,
    bc_policy: Optional[BCPolicy] = None
) -> Dict[str, float]:
    """Validate model on validation set.

    Args:
        state_encoder: StateEncoder instance
        iql_trainer: IQLTrainer instance
        val_loader: Validation dataloader
        device: Device to use
        bc_policy: Optional BC policy for NX_0 evaluation

    Returns:
        Dictionary with validation metrics
    """
    state_encoder.eval()
    iql_trainer.v_network.eval()
    iql_trainer.q_network.eval()
    iql_trainer.policy_network.eval()

    val_metrics = {
        "v_loss": [],
        "q_loss": [],
        "policy_loss": [],
        "mean_q_value": [],
        "mean_v_value": []
    }

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # Encode states
            states = encode_batch_states(state_encoder, batch, device)

            # Get targets
            actions = batch["actions"][:, 0].to(device)
            rewards = batch["rewards"][:, 0].to(device)
            dones = batch["dones"][:, 0].bool().to(device)
            next_states = states

            # Compute losses (without updating)
            # V loss
            q_values_all = iql_trainer.q_network(states)
            q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            v_values = iql_trainer.v_network(states).squeeze(1)

            from src.models.iql.losses import expectile_loss, q_loss
            v_loss = expectile_loss(v_values, q_values, iql_trainer.tau_expectile)

            # Q loss
            next_v_values = iql_trainer.q_target(next_states).max(dim=1)[0]
            q_loss_val = q_loss(
                q_values_all, actions, rewards,
                next_v_values, dones, iql_trainer.gamma
            )

            val_metrics["v_loss"].append(v_loss.item())
            val_metrics["q_loss"].append(q_loss_val.item())
            val_metrics["mean_q_value"].append(q_values.mean().item())
            val_metrics["mean_v_value"].append(v_values.mean().item())

            # Collect predictions for Top-K
            predictions = iql_trainer.policy_network(states)
            all_predictions.append(predictions.cpu())
            all_targets.append(actions.cpu())

    # Compute Top-K metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    topk_metrics = compute_all_metrics(
        all_predictions, all_targets, k_values=[5, 10, 20]
    )

    # Combine metrics
    result = {k: np.mean(v) for k, v in val_metrics.items()}
    result.update(topk_metrics)

    return result


def save_checkpoint(
    state_encoder: StateEncoder,
    iql_trainer: IQLTrainer,
    epoch: int,
    metrics: dict,
    path: str
):
    """Save training checkpoint.

    Args:
        state_encoder: StateEncoder instance
        iql_trainer: IQLTrainer instance
        epoch: Current epoch
        metrics: Current metrics
        path: Save path
    """
    checkpoint = {
        "epoch": epoch,
        "metrics": metrics,
        "state_encoder": state_encoder.state_dict(),
        "v_network": iql_trainer.v_network.state_dict(),
        "q_network": iql_trainer.q_network.state_dict(),
        "q_target": iql_trainer.q_target.state_dict(),
        "policy_network": iql_trainer.policy_network.state_dict(),
        "v_optimizer": iql_trainer.v_optimizer.state_dict(),
        "q_optimizer": iql_trainer.q_optimizer.state_dict(),
        "policy_optimizer": iql_trainer.policy_optimizer.state_dict()
    }
    torch.save(checkpoint, path)


def parse_args():
    """Parse command line arguments for non-Hydra mode."""
    parser = argparse.ArgumentParser(description="Train IQL on MovieLens-1M")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--max_seq_len", type=int, default=50)

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)

    # IQL hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau_expectile", type=float, default=0.7)
    parser.add_argument("--beta_awr", type=float, default=3.0)
    parser.add_argument("--clip_weight", type=float, default=20.0)

    # BC pretraining
    parser.add_argument("--bc_checkpoint", type=str, default="checkpoints/bc/bc_policy_best.pt")
    parser.add_argument("--no_bc_init", action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/iql")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--tensorboard_dir", type=str, default="runs")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main_with_args(args):
    """Main training function with parsed arguments."""
    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(args.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter(tensorboard_dir / f"iql_{time.strftime('%Y%m%d_%H%M%S')}")

    # Load data
    print("\n[1/5] Loading preprocessed data...")
    train_trajs, val_trajs, test_trajs, zipcode_vocab, metadata = load_processed_data(
        args.data_dir
    )
    print(f"  Train: {len(train_trajs)} trajectories")
    print(f"  Val: {len(val_trajs)} trajectories")
    print(f"  Movies: {metadata['num_movies']}")
    print(f"  Zipcode buckets: {metadata['num_zipcode_buckets']}")

    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_trajs, val_trajs, zipcode_vocab,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create models
    print("\n[3/5] Creating models...")
    cfg = {
        "sasrec": {"d_model": 64, "nhead": 2, "num_layers": 2},
        "static": {"zipcode_embed_dim": 8},
        "iql": {
            "hidden_dim": 256,
            "gamma": args.gamma,
            "tau_expectile": args.tau_expectile,
            "beta_awr": args.beta_awr,
            "clip_weight": args.clip_weight
        },
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len
    }

    state_encoder, iql_trainer = create_models(
        max_movie_id=metadata["max_movie_id"],
        num_zipcode_buckets=metadata["num_zipcode_buckets"],
        cfg=cfg,
        device=device
    )
    print(f"  State dim: {state_encoder.output_dim}")

    # Load BC weights
    if not args.no_bc_init:
        print("\n[4/5] Loading BC policy weights...")
        load_bc_weights(iql_trainer, args.bc_checkpoint, device)
    else:
        print("\n[4/5] Skipping BC weight initialization")

    # Training loop
    print("\n[5/5] Starting training...")
    print("=" * 70)

    best_val_loss = float("inf")
    training_history = []

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            state_encoder, iql_trainer, train_loader,
            device, args.grad_clip_norm, args.log_interval
        )

        # Validate
        val_metrics = validate(
            state_encoder, iql_trainer, val_loader,
            device
        )

        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch)

        # Print summary
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train - V: {train_metrics['v_loss']:.4f}, Q: {train_metrics['q_loss']:.4f}, π: {train_metrics['policy_loss']:.4f}")
        print(f"  Val - V: {val_metrics['v_loss']:.4f}, Q: {val_metrics['q_loss']:.4f}")
        print(f"  Val - R@5: {val_metrics.get('recall@5', 0):.4f}, R@10: {val_metrics.get('recall@10', 0):.4f}")

        # Save history
        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": epoch_time
        })

        # Save checkpoints
        if epoch % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"iql_epoch_{epoch}.pt"
            save_checkpoint(state_encoder, iql_trainer, epoch, val_metrics, str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        val_loss = val_metrics["v_loss"] + val_metrics["q_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "iql_best.pt"
            save_checkpoint(state_encoder, iql_trainer, epoch, val_metrics, str(best_path))
            print(f"  New best model! Saved to {best_path}")

    # Save final model
    final_path = checkpoint_dir / "iql_final.pt"
    save_checkpoint(state_encoder, iql_trainer, args.num_epochs, val_metrics, str(final_path))
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    writer.close()

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluate.py --iql_checkpoint {final_path}")
    print(f"  2. View logs: tensorboard --logdir {tensorboard_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("IQL Training on MovieLens-1M")
    print("=" * 70)
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"IQL params: γ={args.gamma}, τ={args.tau_expectile}, β={args.beta_awr}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    main_with_args(args)


if __name__ == "__main__":
    main()

