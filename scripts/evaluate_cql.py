import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data_module.dataset import MovieLensIQLDataset
from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder
from src.models.iql.state_encoder import StateEncoder
from src.models.iql.networks import PolicyNetwork
from src.models.cql.trainer import MLPQNetwork
from src.evaluation.metrics import compute_all_metrics
from src.utils.seed import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Top-K evaluation for IQL and CQL policies")
    parser.add_argument("--iql_checkpoint", type=str, default=None)
    parser.add_argument("--cql_checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_path", type=str, default="results/evaluation_topk.json")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_processed_data(data_dir: str, split: str = "test"):
    data_path = Path(data_dir)
    traj_file = f"{split}_trajectories.pkl"
    with open(data_path / traj_file, "rb") as f:
        trajectories = pickle.load(f)
    with open(data_path / "zipcode_vocab.json", "r") as f:
        zipcode_vocab = json.load(f)
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return trajectories, zipcode_vocab, metadata


def create_state_encoder(max_movie_id: int, num_zipcode_buckets: int, device: torch.device, max_seq_len: int = 50) -> StateEncoder:
    static_encoder = StaticFeatureEncoder(num_zipcode_buckets=num_zipcode_buckets, zipcode_embed_dim=8)
    sasrec_encoder = SASRecEncoder(
        num_movies=max_movie_id,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=max_seq_len
    )
    state_encoder = StateEncoder(static_encoder, sasrec_encoder)
    return state_encoder.to(device)


def compute_popularity_from_trajectories(trajectories, max_movie_id: int) -> np.ndarray:
    counts = np.zeros(max_movie_id + 1, dtype=np.int64)
    for traj in trajectories:
        movie_ids = traj.movie_ids
        mask = (movie_ids > 0) & (movie_ids <= max_movie_id)
        valid_ids = movie_ids[mask]
        np.add.at(counts, valid_ids, 1)
    return counts


def encode_batch_states(state_encoder: StateEncoder, batch: dict, device: torch.device) -> torch.Tensor:
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


def evaluate_topk(policy: torch.nn.Module, state_encoder: StateEncoder, dataloader: DataLoader, k_values: List[int], device: torch.device) -> Dict[str, float]:
    all_predictions = []
    all_targets = []
    policy.eval()
    state_encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            states = encode_batch_states(state_encoder, batch, device)
            actions = batch["actions"][:, 0].to(device)
            predictions = policy(states)
            all_predictions.append(predictions.cpu())
            all_targets.append(actions.cpu())
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    topk_metrics = compute_all_metrics(all_predictions, all_targets, k_values=k_values)
    return topk_metrics


def evaluate_topk_popularity(popularity_scores: torch.Tensor, dataloader: DataLoader, k_values: List[int], device: torch.device) -> Dict[str, float]:
    all_predictions = []
    all_targets = []
    popularity_scores = popularity_scores.to(device)
    with torch.no_grad():
        for batch in dataloader:
            targets = batch["actions"][:, 0].to(device)
            batch_size = targets.shape[0]
            preds = popularity_scores.unsqueeze(0).expand(batch_size, -1)
            all_predictions.append(preds)
            all_targets.append(targets)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    base_metrics = compute_all_metrics(all_predictions, all_targets, k_values=k_values)
    return {f"pop_{k}": v for k, v in base_metrics.items()}


def evaluate_topk_random(num_actions: int, dataloader: DataLoader, k_values: List[int], device: torch.device, seed: int = 42) -> Dict[str, float]:
    all_predictions = []
    all_targets = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch in dataloader:
            targets = batch["actions"][:, 0].to(device)
            batch_size = targets.shape[0]
            preds = torch.rand(batch_size, num_actions, device=device)
            all_predictions.append(preds)
            all_targets.append(targets)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    base_metrics = compute_all_metrics(all_predictions, all_targets, k_values=k_values)
    return {f"rand_{k}": v for k, v in base_metrics.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    trajectories, zipcode_vocab, metadata = load_processed_data(args.data_dir, split="test")
    dataset = MovieLensIQLDataset(trajectories, zipcode_vocab, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_trajectories, _, _ = load_processed_data(args.data_dir, split="train")
    popularity_counts = compute_popularity_from_trajectories(train_trajectories, max_movie_id=metadata["max_movie_id"])
    popularity_scores = torch.from_numpy(popularity_counts.astype(np.float32))

    state_encoder = create_state_encoder(
        max_movie_id=metadata["max_movie_id"],
        num_zipcode_buckets=metadata["num_zipcode_buckets"],
        device=device,
        max_seq_len=args.max_seq_len
    )

    results: Dict[str, Any] = {
        "split": "test",
        "iql_checkpoint": args.iql_checkpoint,
        "cql_checkpoint": args.cql_checkpoint
    }

    if args.iql_checkpoint is not None:
        iql_ckpt = torch.load(args.iql_checkpoint, map_location=device, weights_only=False)
        if "state_encoder" in iql_ckpt:
            state_encoder.load_state_dict(iql_ckpt["state_encoder"])
        iql_policy = PolicyNetwork(
            state_dim=state_encoder.output_dim,
            num_actions=metadata["max_movie_id"] + 1
        ).to(device)
        if "policy_network" in iql_ckpt:
            iql_policy.load_state_dict(iql_ckpt["policy_network"])
        else:
            iql_policy.load_state_dict(iql_ckpt)
        iql_metrics = evaluate_topk(iql_policy, state_encoder, dataloader, args.k_values, device)
        iql_metrics = {f"iql_{k}": v for k, v in iql_metrics.items()}
        results.update(iql_metrics)

    if args.cql_checkpoint is not None:
        cql_ckpt = torch.load(args.cql_checkpoint, map_location=device, weights_only=False)
        if "state_encoder" in cql_ckpt:
            state_encoder.load_state_dict(cql_ckpt["state_encoder"])
        cql_policy = MLPQNetwork(
            state_dim=state_encoder.output_dim,
            num_actions=metadata["max_movie_id"] + 1,
            hidden_dim=256,
            num_hidden_layers=2
        ).to(device)
        if "q_network" in cql_ckpt:
            cql_policy.load_state_dict(cql_ckpt["q_network"])
        else:
            cql_policy.load_state_dict(cql_ckpt)
        cql_metrics = evaluate_topk(cql_policy, state_encoder, dataloader, args.k_values, device)
        cql_metrics = {f"cql_{k}": v for k, v in cql_metrics.items()}
        results.update(cql_metrics)

    pop_metrics = evaluate_topk_popularity(popularity_scores, dataloader, args.k_values, device)
    rand_metrics = evaluate_topk_random(metadata["max_movie_id"] + 1, dataloader, args.k_values, device, seed=args.seed)
    results.update(pop_metrics)
    results.update(rand_metrics)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
