"""Behavioral Cloning policy for offline evaluation.

This module implements a policy network trained via supervised learning
to clone the behavior policy from historical trajectories.
"""

import torch
import torch.nn as nn
from typing import List


class BCPolicy(nn.Module):
    """Behavioral Cloning policy for estimating behavior policy μ(a|s).

    Uses supervised learning to clone the behavior policy from historical
    (state, action) pairs. Architecture identical to PolicyNetwork but
    with independent parameters.

    Args:
        state_dim: Dimension of state representation (default: 101)
        num_actions: Number of actions (movies) (default: 3900)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_hidden_layers: Number of hidden layers (default: 2)

    Example:
        >>> bc_policy = BCPolicy(state_dim=101, num_actions=3900)
        >>> states = torch.randn(32, 101)
        >>> logits = bc_policy(states)
        >>> probs = bc_policy.get_all_probs(states)
        >>> probs.shape
        torch.Size([32, 3900])
    """

    def __init__(
        self,
        state_dim: int = 101,
        num_actions: int = 3900,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        # Build MLP layers (same as PolicyNetwork)
        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (logits for each action)
        layers.append(nn.Linear(hidden_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute behavior policy logits.

        Args:
            state: (batch, state_dim) state representation

        Returns:
            Logits: (batch, num_actions)
            Apply softmax to get μ(a|s)
        """
        return self.network(state)

    def get_action_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Get probabilities for specific actions.

        Args:
            states: (batch, state_dim)
            actions: (batch,) action indices

        Returns:
            (batch,) probabilities μ(a|s) for given actions
        """
        logits = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        return probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    def get_all_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get full probability distribution over all actions.

        Args:
            states: (batch, state_dim)

        Returns:
            (batch, num_actions) probability distribution
            Each row sums to 1.0
        """
        logits = self.forward(states)
        return torch.softmax(logits, dim=-1)
