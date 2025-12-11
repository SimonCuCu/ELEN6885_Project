import torch
import torch.nn as nn


class CRRCritic(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)