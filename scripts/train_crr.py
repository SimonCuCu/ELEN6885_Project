import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data_module.dataset import MovieLensIQLDataset
from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder
from src.models.iql.state_encoder import StateEncoder
from src.models.iql.networks import PolicyNetwork
from src.models.crr.networks import CRRCritic
from src.utils.seed import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRR policy on MovieLens-1M")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints/crr")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max_weight", type=float, default=20.0)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def load_processed_data(data_dir: str, split: str = "train"):
    import json
    import pickle

    data_path = Path(data_dir)
    with open(data_path / f"{split}_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    with open(data_path / "zipcode_vocab.json", "r") as f:
        zipcode_vocab = json.load(f)
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return trajectories, zipcode_vocab, metadata


def create_state_encoder(max_movie_id: int, num_zipcode_buckets: int, device: torch.device, max_seq_len: int):
    static_encoder = StaticFeatureEncoder(num_zipcode_buckets=num_zipcode_buckets, zipcode_embed_dim=8)
    sasrec_encoder = SASRecEncoder(
        num_movies=max_movie_id,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=max_seq_len,
    )
    state_encoder = StateEncoder(static_encoder, sasrec_encoder)
    return state_encoder.to(device)


def encode_batch_states(state_encoder: StateEncoder, batch: dict, device: torch.device, prefix: str = ""):
    gender = batch[f"{prefix}gender"].to(device)
    age = batch[f"{prefix}age"].to(device)
    occupation = batch[f"{prefix}occupation"].to(device)
    zipcode_bucket = batch[f"{prefix}zipcode_bucket"].to(device)
    movie_sequence = batch[f"{prefix}movie_sequence"].to(device)
    rating_sequence = batch[f"{prefix}rating_sequence"].to(device)
    timestamp_sequence = batch[f"{prefix}timestamp_sequence"].to(device)
    sequence_mask = batch[f"{prefix}sequence_mask"].to(device)

    states = state_encoder(
        gender=gender,
        age=age,
        occupation=occupation,
        zipcode_bucket=zipcode_bucket,
        movie_sequence=movie_sequence,
        rating_sequence=rating_sequence,
        timestamp_sequence=timestamp_sequence,
        sequence_mask=sequence_mask,
    )
    return states


def soft_update(target, source, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def train():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    trajectories, zipcode_vocab, metadata = load_processed_data(args.data_dir, split=args.split)

    dataset = MovieLensIQLDataset(trajectories, zipcode_vocab, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    state_encoder = create_state_encoder(
        max_movie_id=metadata["max_movie_id"],
        num_zipcode_buckets=metadata["num_zipcode_buckets"],
        device=device,
        max_seq_len=args.max_seq_len,
    )

    state_dim = state_encoder.output_dim
    num_actions = metadata["max_movie_id"] + 1

    policy = PolicyNetwork(state_dim=state_dim, num_actions=num_actions).to(device)
    critic = CRRCritic(state_dim=state_dim, num_actions=num_actions).to(device)

    target_policy = PolicyNetwork(state_dim=state_dim, num_actions=num_actions).to(device)
    target_critic = CRRCritic(state_dim=state_dim, num_actions=num_actions).to(device)
    target_policy.load_state_dict(policy.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optim = torch.optim.Adam(policy.parameters(), lr=args.lr_actor)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        policy.train()
        critic.train()
        state_encoder.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            actions = batch["actions"].to(device).long()
            rewards = batch["rewards"].to(device).float()
            dones = batch["dones"].to(device).float()

            states = encode_batch_states(state_encoder, batch, device, prefix="")
            next_states = encode_batch_states(state_encoder, batch, device, prefix="next_")

            q_values = critic(states)
            q_sa = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_q = target_critic(next_states)
                next_logits = target_policy(next_states)
                next_probs = F.softmax(next_logits, dim=-1)
                v_next = (next_probs * next_q).sum(dim=-1)
                target_q = rewards + args.gamma * (1.0 - dones) * v_next

            critic_loss = F.mse_loss(q_sa, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            with torch.no_grad():
                q_values_detached = critic(states)
                logits_detached = policy(states)
                probs_detached = F.softmax(logits_detached, dim=-1)
                v = (probs_detached * q_values_detached).sum(dim=-1)
                q_sa_detached = q_values_detached.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                advantages = q_sa_detached - v
                weights = torch.exp(advantages / args.beta).clamp(max=args.max_weight)

            logits = policy(states)
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_a = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            actor_loss = -(weights * log_pi_a).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            soft_update(target_critic, critic, args.target_tau)
            soft_update(target_policy, policy, args.target_tau)

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            num_batches += 1

        avg_actor_loss = total_actor_loss / max(1, num_batches)
        avg_critic_loss = total_critic_loss / max(1, num_batches)
        epoch_loss = avg_actor_loss + avg_critic_loss

        print(
            f"Epoch {epoch}: actor_loss={avg_actor_loss:.4f}, "
            f"critic_loss={avg_critic_loss:.4f}, total={epoch_loss:.4f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt = {
                "state_encoder": state_encoder.state_dict(),
                "policy_network": policy.state_dict(),
                "critic": critic.state_dict(),
                "metadata": metadata,
                "args": vars(args),
            }
            ckpt_path = output_dir / "crr_best.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved best CRR checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()