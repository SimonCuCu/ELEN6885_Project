# src/models/cql/trainer.py

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden_dim: int = 256, num_hidden_layers: int = 2):
        super().__init__()
        layers = []
        input_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CQLTrainer:
    """
    Conservative Q-Learning trainer for discrete-action offline RL.
    Q-learning style, no separate policy network: policy is derived from Q.

    Loss:
        L = MSE(Q(s,a), r + γ max_{a'} Q_target(s',a')) +
            α [ logsumexp_a Q(s,a) - Q(s, a_behavior) ]
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        lr: float = 1e-4,
        gamma: float = 0.99,
        cql_alpha: float = 1.0,
        tau_target: float = 0.005,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.tau_target = tau_target

        self.q_network = MLPQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        self.q_target = copy.deepcopy(self.q_network)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # For compatibility with evaluation code that expects .policy_network
        self.policy_network = self.q_network

    def to(self, device: torch.device):
        self.q_network.to(device)
        self.q_target.to(device)
        return self

    @torch.no_grad()
    def soft_update_target(self):
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau_target)
            target_param.data.add_(self.tau_target * param.data)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One gradient step of CQL + Bellman update.

        Args:
            states: [B, state_dim]
            actions: [B]
            rewards: [B]
            next_states: [B, state_dim]
            dones: [B] bool

        Returns:
            dict with q_loss, cql_loss, total_loss, mean_q_value, mean_target_q_value
        """
        # Q(s, ·) for all actions
        q_values_all = self.q_network(states)  # [B, A]
        # Q(s, a_behavior)
        q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        with torch.no_grad():
            next_q_target_all = self.q_target(next_states)  # [B, A]
            next_q_target = next_q_target_all.max(dim=1)[0]  # [B]
            target = rewards + self.gamma * (1.0 - dones.float()) * next_q_target

        # 1) Bellman error
        q_loss = F.mse_loss(q_values, target)

        # 2) CQL regularizer (CQL(H) style)
        logsumexp_q = torch.logsumexp(q_values_all, dim=1)  # [B]
        # E_{a~πβ} Q(s,a) ≈ Q(s, a_behavior) in offline dataset
        cql_reg = (logsumexp_q - q_values).mean()
        cql_loss = self.cql_alpha * cql_reg

        total_loss = q_loss + cql_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        # Soft update target network
        self.soft_update_target()

        with torch.no_grad():
            mean_q = q_values.mean().item()
            mean_target_q = next_q_target.mean().item()

        return {
            "q_loss": q_loss.item(),
            "cql_loss": cql_loss.item(),
            "total_loss": total_loss.item(),
            "mean_q_value": mean_q,
            "mean_target_q_value": mean_target_q,
        }