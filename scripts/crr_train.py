import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Config
# ============================================================

DATA_DIR = "data/processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    lr: float = 3e-3
    batch_size: int = 4096
    epochs: int = 10

    emb_dim: int = 64
    beta: float = 1.0          # CRR temperature
    max_w: float = 20.0        # weight clip
    lambda_crr: float = 1.0
    lambda_bc: float = 0.1


CFG = Config()

# ============================================================
# Dataset wrapper
# ============================================================


class OfflineDataset(Dataset):
    def __init__(self, d):
        self.s = d["states"]
        self.a = d["actions"]
        self.r = d["rewards"]
        self.sn = d["next_states"]
        self.dn = d["dones"]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.sn[idx],
            self.dn[idx],
        )


# ============================================================
# Shared embedding matrix (VERY IMPORTANT)
# ============================================================

class SharedEmb(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)


# ============================================================
# Q-network (critic)
# ============================================================

class MFQNet(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, users):
        u = self.shared.user_emb(users)          # [B, D]
        v = self.shared.item_emb.weight          # [I, D]
        q = u @ v.t()                             # [B, I]
        return q


# ============================================================
# Policy network (actor)
# ============================================================

class MFPolicyNet(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, users):
        u = self.shared.user_emb(users)          # [B, D]
        v = self.shared.item_emb.weight          # [I, D]
        logits = u @ v.t()                       # [B, I]
        probs = F.softmax(logits, dim=-1)
        return probs, logits


# ============================================================
# Top-K Evaluation
# ============================================================

def evaluate_topk(policy_net, loader, K=20):
    policy_net.eval()
    total = 0
    hits = 0

    with torch.no_grad():
        for s, a, r, _, _ in loader:
            s = s.to(DEVICE)
            a = a.to(DEVICE)

            probs, _ = policy_net(s)                 # [B, I]
            topk_items = probs.topk(K, dim=-1).indices  # [B, K]

            a_exp = a.unsqueeze(1)                   # [B, 1]
            hit = (topk_items == a_exp).any(dim=1)   # [B]

            hits += hit.sum().item()
            total += s.size(0)

    policy_net.train()
    return hits / max(total, 1)


# ============================================================
# TRAINING LOOP — CRR + BC
# ============================================================

def train_crr():

    # 1) Load data
    train_obj = torch.load(os.path.join(DATA_DIR, "train.pt"))
    val_obj = torch.load(os.path.join(DATA_DIR, "val.pt"))

    num_users = train_obj["num_users"]
    num_items = train_obj["num_items"]

    train_ds = OfflineDataset(train_obj["data"])
    val_ds = OfflineDataset(val_obj["data"])

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    # 2) Shared embeddings for both actor & critic
    shared = SharedEmb(num_users, num_items, CFG.emb_dim).to(DEVICE)

    q_net = MFQNet(shared).to(DEVICE)
    policy_net = MFPolicyNet(shared).to(DEVICE)

    optim_q = torch.optim.Adam(q_net.parameters(), lr=CFG.lr)
    optim_pi = torch.optim.Adam(policy_net.parameters(), lr=CFG.lr)

    print(f"Using device: {DEVICE}")
    print(f"Users={num_users}, Items={num_items}")

    # 3) Training
    for ep in range(1, CFG.epochs + 1):
        for s, a, r, _, _ in train_loader:

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)

            # --------------------------------------------------
            # Critic: Q(s,a) ≈ r (bandit, no bootstrap)
            # --------------------------------------------------
            q_values = q_net(s)                        # [B, I]
            q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            critic_loss = F.mse_loss(q_sa, r)

            optim_q.zero_grad()
            critic_loss.backward()
            optim_q.step()

            # --------------------------------------------------
            # Actor (CRR): weights = exp(beta*A(s,a))
            # --------------------------------------------------
            with torch.no_grad():
                q_det = q_net(s)                       # [B, I]

                # More stable bandit value estimate:
                v_s = q_det.mean(dim=1)                # [B]
                adv = q_sa - v_s                       # [B]

                # Normalize advantage
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)

                # CRR weighting
                weights = torch.exp(CFG.beta * adv)
                weights = torch.clamp(weights, max=CFG.max_w)

            probs, logits = policy_net(s)

            log_pi = F.log_softmax(logits, dim=-1)
            log_pi_a = log_pi.gather(1, a.unsqueeze(1)).squeeze(1)

            crr_loss = -(weights * log_pi_a).mean()
            bc_loss = F.cross_entropy(logits, a)

            actor_loss = CFG.lambda_crr * crr_loss + CFG.lambda_bc * bc_loss

            optim_pi.zero_grad()
            actor_loss.backward()
            optim_pi.step()

        # ------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------
        top10 = evaluate_topk(policy_net, val_loader, K=10)
        top20 = evaluate_topk(policy_net, val_loader, K=20)
        top50 = evaluate_topk(policy_net, val_loader, K=50)

        print(
            f"Epoch {ep:02d} | "
            f"critic_loss={critic_loss:.4f} | actor_loss={actor_loss:.4f} | "
            f"Top10={top10:.4f} | Top20={top20:.4f} | Top50={top50:.4f}"
        )


if __name__ == "__main__":
    train_crr()
