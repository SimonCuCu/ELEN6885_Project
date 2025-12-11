"""Train Behavioral Cloning policy on MovieLens-1M trajectory data.

This script trains a BC policy to estimate the behavior policy μ(a|s) from
historical trajectories. The trained BC policy is used in NX_0 off-policy
evaluation.

Usage:
    python scripts/train_bc.py --data_dir data/processed --output_dir checkpoints/bc
    python scripts/train_bc.py --epochs 10 --batch_size 256 --lr 1e-4

Example:
    python scripts/train_bc.py --epochs 5 --lr 1e-3 --save_every 1
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from src.data_module.dataset import MovieLensIQLDataset
from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder
from src.models.iql.state_encoder import StateEncoder
from src.models.bc_policy import BCPolicy
from src.training.bc_trainer import BCPolicyTrainer
from src.utils.seed import set_seed, get_device, worker_init_fn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Behavioral Cloning policy"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed trajectory data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/bc",
        help="Directory to save BC policy checkpoints"
    )

    # Model arguments
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for BC policy network"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training (auto/cuda/cpu)"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    # Logging arguments
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="TensorBoard log directory"
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Freeze encoder
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze StateEncoder during BC training"
    )

    return parser.parse_args()


def load_processed_data(data_dir: str):
    """Load preprocessed trajectory data.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (train_trajs, val_trajs, zipcode_vocab, metadata)
    """
    data_path = Path(data_dir)

    with open(data_path / "train_trajectories.pkl", "rb") as f:
        train_trajs = pickle.load(f)

    with open(data_path / "val_trajectories.pkl", "rb") as f:
        val_trajs = pickle.load(f)

    with open(data_path / "zipcode_vocab.json", "r") as f:
        zipcode_vocab = json.load(f)

    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return train_trajs, val_trajs, zipcode_vocab, metadata


def create_dataloaders(
    train_trajs: list,
    val_trajs: list,
    zipcode_vocab: dict,
    max_seq_len: int,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        train_trajs: Training trajectories
        val_trajs: Validation trajectories
        zipcode_vocab: Zipcode vocabulary
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers

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
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_state_encoder(
    max_movie_id: int,
    num_zipcode_buckets: int,
    max_seq_len: int,
    device: torch.device
) -> StateEncoder:
    """Create StateEncoder.

    Args:
        max_movie_id: Maximum movie ID (for embedding size)
        num_zipcode_buckets: Number of zipcode buckets
        max_seq_len: Maximum sequence length
        device: Device to use

    Returns:
        StateEncoder instance
    """
    static_encoder = StaticFeatureEncoder(
        num_zipcode_buckets=num_zipcode_buckets,
        zipcode_embed_dim=8
    )

    sasrec_encoder = SASRecEncoder(
        num_movies=max_movie_id,  # Use max_movie_id to ensure all IDs fit
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=max_seq_len
    )

    state_encoder = StateEncoder(static_encoder, sasrec_encoder)
    return state_encoder.to(device)


def encode_batch_states(
    state_encoder: StateEncoder,
    batch: dict,
    device: torch.device
) -> torch.Tensor:
    """Encode batch of trajectories into states.

    Args:
        state_encoder: StateEncoder instance
        batch: Batch dictionary
        device: Device to use

    Returns:
        States tensor of shape (batch, state_dim)
    """
    gender = batch["gender"].to(device)
    age = batch["age"].to(device)
    occupation = batch["occupation"].to(device)
    zipcode_bucket = batch["zipcode_bucket"].to(device)
    movie_sequence = batch["movie_sequence"].to(device)
    rating_sequence = batch["rating_sequence"].to(device)
    timestamp_sequence = batch["timestamp_sequence"].to(device)
    sequence_mask = batch["sequence_mask"].to(device)

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
    trainer: BCPolicyTrainer,
    train_loader: DataLoader,
    epoch: int,
    log_interval: int,
    device: torch.device,
    freeze_encoder: bool = False
) -> Dict[str, float]:
    """Train BC policy for one epoch.

    Args:
        state_encoder: StateEncoder instance
        trainer: BC policy trainer
        train_loader: Training data loader
        epoch: Current epoch number
        log_interval: Log metrics every N steps
        device: Device to use
        freeze_encoder: Whether encoder is frozen

    Returns:
        Dictionary with epoch metrics
    """
    if freeze_encoder:
        state_encoder.eval()
    else:
        state_encoder.train()

    epoch_losses = []
    epoch_accuracies = []

    for step, batch in enumerate(train_loader):
        # Encode states
        with torch.set_grad_enabled(not freeze_encoder):
            states = encode_batch_states(state_encoder, batch, device)

        # Get actions (first timestep)
        actions = batch["actions"][:, 0].to(device)

        # Train step
        metrics = trainer.train_step(states, actions)

        epoch_losses.append(metrics['bc_loss'])
        epoch_accuracies.append(metrics['accuracy'])

        if (step + 1) % log_interval == 0:
            avg_loss = np.mean(epoch_losses[-log_interval:])
            avg_acc = np.mean(epoch_accuracies[-log_interval:])
            print(
                f"  Epoch {epoch} Step {step + 1}/{len(train_loader)}: "
                f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}"
            )

    return {
        'train_loss': np.mean(epoch_losses),
        'train_accuracy': np.mean(epoch_accuracies)
    }


def evaluate(
    state_encoder: StateEncoder,
    trainer: BCPolicyTrainer,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate BC policy on validation set.

    Args:
        state_encoder: StateEncoder instance
        trainer: BC policy trainer
        val_loader: Validation data loader
        device: Device to use

    Returns:
        Dictionary with validation metrics
    """
    state_encoder.eval()

    all_losses = []
    all_accuracies = []
    all_perplexities = []

    with torch.no_grad():
        for batch in val_loader:
            states = encode_batch_states(state_encoder, batch, device)
            actions = batch["actions"][:, 0].to(device)

            metrics = trainer.evaluate(states, actions)

            all_losses.append(metrics['loss'])
            all_accuracies.append(metrics['accuracy'])
            all_perplexities.append(metrics['perplexity'])

    return {
        'val_loss': np.mean(all_losses),
        'val_accuracy': np.mean(all_accuracies),
        'val_perplexity': np.mean(all_perplexities)
    }


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard directory
    tensorboard_dir = Path(args.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter(tensorboard_dir / f"bc_{time.strftime('%Y%m%d_%H%M%S')}")

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Behavioral Cloning Policy Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading processed data...")
    train_trajs, val_trajs, zipcode_vocab, metadata = load_processed_data(args.data_dir)
    print(f"  Train: {len(train_trajs)} trajectories")
    print(f"  Val: {len(val_trajs)} trajectories")
    print(f"  Movies: {metadata['num_movies']}")

    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_trajs, val_trajs, zipcode_vocab,
        args.max_seq_len, args.batch_size, args.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create models
    print("\n[3/4] Creating models...")
    state_encoder = create_state_encoder(
        max_movie_id=metadata["max_movie_id"],
        num_zipcode_buckets=metadata["num_zipcode_buckets"],
        max_seq_len=args.max_seq_len,
        device=device
    )
    print(f"  State dim: {state_encoder.output_dim}")

    # Freeze encoder if requested
    if args.freeze_encoder:
        for param in state_encoder.parameters():
            param.requires_grad = False
        print("  StateEncoder frozen")

    # Create BC policy
    bc_policy = BCPolicy(
        state_dim=state_encoder.output_dim,
        num_actions=metadata["max_movie_id"] + 1,  # +1 for padding
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers
    ).to(device)

    # Create trainer
    trainer = BCPolicyTrainer(bc_policy, lr=args.lr)

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        print(f"  Resuming from checkpoint: {args.resume_from}")
        trainer.load(args.resume_from)

    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(
            state_encoder, trainer, train_loader,
            epoch, args.log_interval, device,
            freeze_encoder=args.freeze_encoder
        )

        # Evaluate
        val_metrics = evaluate(state_encoder, trainer, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        writer.add_scalar("train/loss", train_metrics['train_loss'], epoch)
        writer.add_scalar("train/accuracy", train_metrics['train_accuracy'], epoch)
        writer.add_scalar("val/loss", val_metrics['val_loss'], epoch)
        writer.add_scalar("val/accuracy", val_metrics['val_accuracy'], epoch)
        writer.add_scalar("val/perplexity", val_metrics['val_perplexity'], epoch)

        # Combine metrics
        epoch_metrics = {
            'epoch': epoch,
            **train_metrics,
            **val_metrics,
            'time': epoch_time
        }
        training_history.append(epoch_metrics)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Acc:  {train_metrics['train_accuracy']:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['val_accuracy']:.4f}")
        print(f"  Val Perplexity: {val_metrics['val_perplexity']:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"bc_policy_epoch_{epoch}.pt"
            trainer.save(str(checkpoint_path))
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_checkpoint_path = output_dir / "bc_policy_best.pt"
            trainer.save(str(best_checkpoint_path))
            # Also save state encoder
            torch.save(state_encoder.state_dict(), output_dir / "state_encoder_best.pt")
            print(f"  New best model! Saved to {best_checkpoint_path}")

    # Save final model
    final_checkpoint_path = output_dir / "bc_policy_final.pt"
    trainer.save(str(final_checkpoint_path))
    torch.save(state_encoder.state_dict(), output_dir / "state_encoder_final.pt")
    print(f"\nTraining complete! Final model saved to {final_checkpoint_path}")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    writer.close()

    print("\n" + "=" * 60)
    print("BC Training Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Train IQL: python scripts/train.py --bc_checkpoint {best_checkpoint_path}")
    print(f"  2. View logs: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()
