"""IQL network components.

This module implements the core neural networks for IQL:
- VNetwork: Value function V(s)
- QNetwork: Q-function Q(s, .) with full action space output
- PolicyNetwork: Policy pi(a|s) with full action space output
"""

import torch
import torch.nn as nn
from typing import List


class VNetwork(nn.Module):
    """Value network for IQL.

    Estimates state value V(s) using a multi-layer perceptron.

    Args:
        state_dim: Dimension of state representation (default: 101)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_hidden_layers: Number of hidden layers (default: 2)

    Example:
        >>> vnet = VNetwork(state_dim=101, hidden_dim=256)
        >>> state = torch.randn(32, 101)
        >>> value = vnet(state)
        >>> value.shape
        torch.Size([32, 1])
    """

    def __init__(
        self,
        state_dim: int = 101,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        # Build MLP layers
        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (single value)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            state: (batch, state_dim) state representation

        Returns:
            Value estimates: (batch, 1)
        """
        return self.network(state)


class QNetwork(nn.Module):
    """Q-network for IQL with full action space.

    Estimates Q-values Q(s, a) for all actions simultaneously.
    Outputs Q-values for the full action space (approximately 3900 movies).

    Args:
        state_dim: Dimension of state representation (default: 101)
        num_actions: Number of actions (movies) (default: 3900)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_hidden_layers: Number of hidden layers (default: 2)

    Example:
        >>> qnet = QNetwork(state_dim=101, num_actions=3900)
        >>> state = torch.randn(32, 101)
        >>> q_values = qnet(state)
        >>> q_values.shape
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

        # Build MLP layers
        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (Q-value for each action)
        layers.append(nn.Linear(hidden_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate Q-values for all actions.

        Args:
            state: (batch, state_dim) state representation

        Returns:
            Q-values: (batch, num_actions)
            Q-value estimates for each action
        """
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Policy network for IQL with full action space.

    Outputs policy logits pi(a|s) for all actions simultaneously.
    Uses Advantage-Weighted Regression (AWR) for training.

    Args:
        state_dim: Dimension of state representation (default: 101)
        num_actions: Number of actions (movies) (default: 3900)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_hidden_layers: Number of hidden layers (default: 2)

    Example:
        >>> policy = PolicyNetwork(state_dim=101, num_actions=3900)
        >>> state = torch.randn(32, 101)
        >>> logits = policy(state)
        >>> probs = torch.softmax(logits, dim=-1)
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

        # Build MLP layers
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
        """Compute policy logits for all actions.

        Args:
            state: (batch, state_dim) state representation

        Returns:
            Logits: (batch, num_actions)
            Unnormalized log-probabilities for each action
            Apply softmax to get probabilities: pi(a|s) = softmax(logits)
        """
        return self.network(state)
