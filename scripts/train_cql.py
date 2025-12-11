import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.data_module.dataset import MovieLensIQLDataset
from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder
from src.models.iql.state_encoder import StateEncoder
from src.models.cql.trainer import CQLTrainer 
from src.models.bc_policy import BCPolicy 
from src.evaluation.metrics import compute_all_metrics
from src.utils.seed import set_seed, get_device, worker_init_fn


def load_processed_data(
    data_dir: str
) -> Tuple[list, list, list, dict, dict]:
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
        drop_last=True
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
) -> Tuple[StateEncoder, CQLTrainer]:
    """
    Create StateEncoder and CQL Trainer.
    """
    # Static feature encoder
    static_encoder = StaticFeatureEncoder(
        num_zipcode_buckets=num_zipcode_buckets,
        zipcode_embed_dim=cfg.get("static", {}).get("zipcode_embed_dim", 8)
    )

    # SASRec encoder
    sasrec_cfg = cfg.get("sasrec", {})
    sasrec_encoder = SASRecEncoder(
        num_movies=max_movie_id,
        d_model=sasrec_cfg.get("d_model", 64),
        nhead=sasrec_cfg.get("nhead", 2),
        num_layers=sasrec_cfg.get("num_layers", 2),
        dim_feedforward=sasrec_cfg.get("dim_feedforward", 256),
        dropout=sasrec_cfg.get("dropout", 0.1),
        max_seq_len=cfg.get("max_seq_len", 50)
    )

    state_encoder = StateEncoder(static_encoder, sasrec_encoder)
    state_encoder = state_encoder.to(device)

    # CQL trainer
    cql_cfg = cfg.get("cql", {})
    cql_trainer = CQLTrainer(
        state_dim=state_encoder.output_dim,
        num_actions=max_movie_id + 1,  # +1 for padding
        hidden_dim=cql_cfg.get("hidden_dim", 256),
        num_hidden_layers=cql_cfg.get("num_hidden_layers", 2),
        lr=cfg.get("learning_rate", 1e-4),
        gamma=cql_cfg.get("gamma", 0.99),
        cql_alpha=cql_cfg.get("alpha", 1.0),
        tau_target=cql_cfg.get("tau_target", 0.005)
    ).to(device)

    return state_encoder, cql_trainer


def encode_batch_states(
    state_encoder: StateEncoder,
    batch: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Encode batch of trajectories into states.
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
    cql_trainer: CQLTrainer,
    train_loader: DataLoader,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    log_interval: int = 100
) -> Dict[str, float]:
    """
    Train one epoch of CQL.
    """
    state_encoder.eval()
    cql_trainer.q_network.train()

    epoch_metrics = {
        "q_loss": [],
        "cql_loss": [],
        "total_loss": [],
        "mean_q_value": [],
        "mean_target_q_value": []
    }

    for step, batch in enumerate(train_loader):
        with torch.no_grad():
            states = encode_batch_states(state_encoder, batch, device)

        actions = batch["actions"][:, 0].to(device)
        rewards = batch["rewards"][:, 0].to(device)
        dones = batch["dones"][:, 0].bool().to(device)

        next_states = states.clone()

        metrics = cql_trainer.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(cql_trainer.q_network.parameters()),
                grad_clip_norm
            )

        for key, value in metrics.items():
            epoch_metrics[key].append(value)

        if (step + 1) % log_interval == 0:
            avg_metrics = {k: np.mean(v[-log_interval:]) for k, v in epoch_metrics.items()}
            print(
                f"  Step {step+1}/{len(train_loader)}: "
                f"Q={avg_metrics['q_loss']:.4f}, "
                f"CQL={avg_metrics['cql_loss']:.4f}, "
                f"Total={avg_metrics['total_loss']:.4f}"
            )

    return {k: np.mean(v) for k, v in epoch_metrics.items()}


def validate(
    state_encoder: StateEncoder,
    cql_trainer: CQLTrainer,
    val_loader: DataLoader,
    device: torch.device,
    bc_policy: Optional[BCPolicy] = None 
) -> Dict[str, float]:
    """
    Validate CQL model on validation set.
    """
    state_encoder.eval()
    cql_trainer.q_network.eval()

    val_metrics = {
        "q_loss": [],
        "cql_loss": [],
        "total_loss": [],
        "mean_q_value": [],
        "mean_target_q_value": []
    }

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            states = encode_batch_states(state_encoder, batch, device)
            actions = batch["actions"][:, 0].to(device)
            rewards = batch["rewards"][:, 0].to(device)
            dones = batch["dones"][:, 0].bool().to(device)
            next_states = states

            q_values_all = cql_trainer.q_network(states)
            q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            next_q_target_all = cql_trainer.q_target(next_states)
            next_q_target = next_q_target_all.max(dim=1)[0]
            target = rewards + cql_trainer.gamma * (1.0 - dones.float()) * next_q_target

            q_loss = torch.mean((q_values - target) ** 2)

            logsumexp_q = torch.logsumexp(q_values_all, dim=1)
            cql_reg = (logsumexp_q - q_values).mean()
            cql_loss = cql_trainer.cql_alpha * cql_reg
            total_loss = q_loss + cql_loss

            val_metrics["q_loss"].append(q_loss.item())
            val_metrics["cql_loss"].append(cql_loss.item())
            val_metrics["total_loss"].append(total_loss.item())
            val_metrics["mean_q_value"].append(q_values.mean().item())
            val_metrics["mean_target_q_value"].append(next_q_target.mean().item())

            all_predictions.append(q_values_all.cpu())
            all_targets.append(actions.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    topk_metrics = compute_all_metrics(
        all_predictions, all_targets, k_values=[5, 10, 20]
    )

    result = {k: np.mean(v) for k, v in val_metrics.items()}
    result.update(topk_metrics)

    return result


def save_checkpoint(
    state_encoder: StateEncoder,
    cql_trainer: CQLTrainer,
    epoch: int,
    metrics: dict,
    path: str
):
    checkpoint = {
        "algorithm": "cql",
        "epoch": epoch,
        "metrics": metrics,
        "state_encoder": state_encoder.state_dict(),
        "q_network": cql_trainer.q_network.state_dict(),
        "q_target": cql_trainer.q_target.state_dict(),
        "q_optimizer": cql_trainer.q_optimizer.state_dict()
    }
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CQL on MovieLens-1M")

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

    # CQL hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--tau_target", type=float, default=0.005)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cql")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--tensorboard_dir", type=str, default="runs")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main_with_args(args):
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(args.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(tensorboard_dir / f"cql_{time.strftime('%Y%m%d_%H%M%S')}")

    print("\n[1/5] Loading preprocessed data...")
    train_trajs, val_trajs, test_trajs, zipcode_vocab, metadata = load_processed_data(
        args.data_dir
    )
    print(f"  Train: {len(train_trajs)} trajectories")
    print(f"  Val: {len(val_trajs)} trajectories")
    print(f"  Movies: {metadata['num_movies']}")
    print(f"  Zipcode buckets: {metadata['num_zipcode_buckets']}")

    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_trajs, val_trajs, zipcode_vocab,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    print("\n[3/5] Creating models (StateEncoder + CQL Q-network)...")
    cfg = {
        "sasrec": {"d_model": 64, "nhead": 2, "num_layers": 2},
        "static": {"zipcode_embed_dim": 8},
        "cql": {
            "hidden_dim": 256,
            "gamma": args.gamma,
            "alpha": args.cql_alpha,
            "tau_target": args.tau_target
        },
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len
    }

    state_encoder, cql_trainer = create_models(
        max_movie_id=metadata["max_movie_id"],
        num_zipcode_buckets=metadata["num_zipcode_buckets"],
        cfg=cfg,
        device=device
    )
    print(f"  State dim: {state_encoder.output_dim}")

    print("\n[5/5] Starting CQL training...")
    print("=" * 70)

    best_val_loss = float("inf")
    training_history = []

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)

        train_metrics = train_epoch(
            state_encoder, cql_trainer, train_loader,
            device, args.grad_clip_norm, args.log_interval
        )

        val_metrics = validate(
            state_encoder, cql_trainer, val_loader,
            device
        )

        epoch_time = time.time() - epoch_start

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch)

        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(
            f"  Train - Q: {train_metrics['q_loss']:.4f}, "
            f"CQL: {train_metrics['cql_loss']:.4f}, "
            f"Total: {train_metrics['total_loss']:.4f}"
        )
        print(
            f"  Val   - Q: {val_metrics['q_loss']:.4f}, "
            f"CQL: {val_metrics['cql_loss']:.4f}, "
            f"R@5: {val_metrics.get('recall@5', 0):.4f}, "
            f"R@10: {val_metrics.get('recall@10', 0):.4f}"
        )

        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": epoch_time
        })

        if epoch % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"cql_epoch_{epoch}.pt"
            save_checkpoint(state_encoder, cql_trainer, epoch, val_metrics, str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

        val_loss = val_metrics["total_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "cql_best.pt"
            save_checkpoint(state_encoder, cql_trainer, epoch, val_metrics, str(best_path))
            print(f"  New best model! Saved to {best_path}")

    final_path = checkpoint_dir / "cql_final.pt"
    save_checkpoint(state_encoder, cql_trainer, args.num_epochs, val_metrics, str(final_path))
    print(f"\nTraining complete! Final model saved to {final_path}")

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    writer.close()

    print("\n" + "=" * 70)
    print("CQL training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Evaluate (you may write scripts/evaluate_cql.py using q_network).")
    print(f"  2. View logs: tensorboard --logdir {tensorboard_dir}")


def main():
    args = parse_args()

    print("=" * 70)
    print("CQL Training on MovieLens-1M")
    print("=" * 70)
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"CQL params: γ={args.gamma}, α={args.cql_alpha}, τ_target={args.tau_target}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    main_with_args(args)


if __name__ == "__main__":
    main()
