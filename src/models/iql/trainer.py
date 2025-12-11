"""IQL trainer implementation.

This module implements the IQL training loop that coordinates:
- V-network updates (expectile regression)
- Q-network updates (TD regression)
- Policy network updates (advantage-weighted regression)
- Target network updates (Polyak averaging)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import copy

from src.models.iql.networks import VNetwork, QNetwork, PolicyNetwork
from src.models.iql.losses import expectile_loss, q_loss, awr_policy_loss


class IQLTrainer:
    """IQL trainer for offline reinforcement learning.

    Manages the training of V-network, Q-network, and Policy network using
    the Implicit Q-Learning algorithm.

    Args:
        state_dim: Dimension of state representation (default: 101)
        num_actions: Number of actions (movies) (default: 3900)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_hidden_layers: Number of hidden layers (default: 2)
        lr: Learning rate for all networks (default: 1e-4)
        gamma: Discount factor (default: 0.99)
        tau_expectile: Expectile level for V-network (default: 0.7)
        beta_awr: Temperature for AWR policy loss (default: 3.0)
        clip_weight: Maximum weight for AWR (default: 20.0)
        tau_target: Target network update rate (default: 0.005)

    Example:
        >>> trainer = IQLTrainer(state_dim=101, num_actions=3900)
        >>> states = torch.randn(32, 101)
        >>> actions = torch.randint(0, 3900, (32,))
        >>> rewards = torch.randn(32)
        >>> next_states = torch.randn(32, 101)
        >>> dones = torch.zeros(32, dtype=torch.bool)
        >>> metrics = trainer.update(states, actions, rewards, next_states, dones)
    """

    def __init__(
        self,
        state_dim: int = 101,
        num_actions: int = 3900,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau_expectile: float = 0.7,
        beta_awr: float = 3.0,
        clip_weight: float = 20.0,
        tau_target: float = 0.005
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau_expectile = tau_expectile
        self.beta_awr = beta_awr
        self.clip_weight = clip_weight
        self.tau_target = tau_target

        # Initialize networks
        self.v_network = VNetwork(state_dim, hidden_dim, num_hidden_layers)
        self.q_network = QNetwork(state_dim, num_actions, hidden_dim, num_hidden_layers)
        self.q_target = copy.deepcopy(self.q_network)
        self.policy_network = PolicyNetwork(state_dim, num_actions, hidden_dim, num_hidden_layers)

        # Freeze target network (no gradients)
        for param in self.q_target.parameters():
            param.requires_grad = False

        # Initialize optimizers
        self.v_optimizer = optim.Adam(self.v_network.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step for all networks.

        Args:
            states: (batch, state_dim) current states
            actions: (batch,) actions taken
            rewards: (batch,) rewards received
            next_states: (batch, state_dim) next states
            dones: (batch,) terminal flags

        Returns:
            Dictionary of training metrics:
                - v_loss: V-network loss
                - q_loss: Q-network loss
                - policy_loss: Policy network loss
                - mean_q_value: Average Q-value
                - mean_v_value: Average V-value
                - mean_advantage: Average advantage (Q - V)
        """
        batch_size = states.shape[0]

        # ===== Update V-network =====
        with torch.no_grad():
            # Compute Q-values for taken actions
            q_values_all = self.q_network(states)
            q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # V-network forward
        v_values = self.v_network(states).squeeze(1)

        # Expectile regression loss
        v_loss = expectile_loss(v_values, q_values, self.tau_expectile)

        # Update V-network
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # ===== Update Q-network =====
        # Q-network forward
        q_values_all = self.q_network(states)

        # Target V-values for next states
        with torch.no_grad():
            next_v_values = self.q_target(next_states)
            # Use max Q-value from target network as V(s')
            next_v_values = next_v_values.max(dim=1)[0]

        # TD loss
        q_loss_value = q_loss(
            q_values_all,
            actions,
            rewards,
            next_v_values,
            dones,
            self.gamma
        )

        # Update Q-network
        self.q_optimizer.zero_grad()
        q_loss_value.backward()
        self.q_optimizer.step()

        # ===== Update Policy network =====
        with torch.no_grad():
            # Compute advantages: A(s,a) = Q(s,a) - V(s)
            q_values_all = self.q_network(states)
            q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            v_values = self.v_network(states).squeeze(1)
            advantages = q_values - v_values

        # Policy forward
        logits = self.policy_network(states)

        # AWR loss
        policy_loss_value = awr_policy_loss(
            logits,
            actions,
            advantages,
            self.beta_awr,
            self.clip_weight
        )

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss_value.backward()
        self.policy_optimizer.step()

        # ===== Update target network (Polyak averaging) =====
        self._update_target_network()

        # ===== Compute metrics =====
        with torch.no_grad():
            q_values_all = self.q_network(states)
            q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            v_values = self.v_network(states).squeeze(1)

            metrics = {
                "v_loss": v_loss.item(),
                "q_loss": q_loss_value.item(),
                "policy_loss": policy_loss_value.item(),
                "mean_q_value": q_values.mean().item(),
                "mean_v_value": v_values.mean().item(),
                "mean_advantage": (q_values - v_values).mean().item()
            }

        return metrics

    def _update_target_network(self):
        """Update target network using Polyak averaging.

        theta_target = tau * theta + (1 - tau) * theta_target
        """
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_param.data.copy_(
                self.tau_target * param.data + (1 - self.tau_target) * target_param.data
            )

    def get_action(self, states: torch.Tensor, greedy: bool = True) -> torch.Tensor:
        """Select actions using the policy network.

        Args:
            states: (batch, state_dim) states
            greedy: If True, select argmax action. If False, sample from policy.

        Returns:
            (batch,) action indices
        """
        with torch.no_grad():
            logits = self.policy_network(states)

            if greedy:
                # Greedy: select action with highest logit
                actions = logits.argmax(dim=-1)
            else:
                # Stochastic: sample from policy distribution
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(1)

        return actions

    def save(self, path: str):
        """Save trainer state to file.

        Args:
            path: File path to save to
        """
        state = {
            "v_network": self.v_network.state_dict(),
            "q_network": self.q_network.state_dict(),
            "q_target": self.q_target.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load trainer state from file.

        Args:
            path: File path to load from
        """
        state = torch.load(path)
        self.v_network.load_state_dict(state["v_network"])
        self.q_network.load_state_dict(state["q_network"])
        self.q_target.load_state_dict(state["q_target"])
        self.policy_network.load_state_dict(state["policy_network"])
        self.v_optimizer.load_state_dict(state["v_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
