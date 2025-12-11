"""NX_0 Off-Policy Evaluator using Self-Normalized Importance Sampling.

This module implements the NX_0 estimator for offline policy evaluation,
which uses importance sampling with weight clipping and self-normalization
to estimate the value of a target policy from trajectories collected under
a behavior policy.

Reference:
    Thomas & Brunskill (2016). "Data-Efficient Off-Policy Policy Evaluation
    for Reinforcement Learning"
"""

import torch
import torch.nn as nn
from typing import Dict


class NX0Evaluator:
    """NX_0 estimator for off-policy evaluation.

    Uses self-normalized importance sampling (SNIS) with weight clipping
    to estimate the value of a target policy π from trajectories collected
    under a behavior policy μ.

    Formula:
        w_i = min(C, π(a_i|s_i) / (μ(a_i|s_i) + ε))
        normalized_w_i = w_i / Σ_j w_j
        NX_0 = Σ_i (normalized_w_i * r_i)

    Args:
        bc_policy: Behavioral Cloning policy estimating μ(a|s)
        target_policy: Target policy π(a|s) to evaluate
        clip_weight: Maximum importance weight (default: 20.0)
        epsilon: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> from src.models.bc_policy import BCPolicy
        >>> from src.models.iql.networks import PolicyNetwork
        >>> bc_policy = BCPolicy(state_dim=101, num_actions=3900)
        >>> target_policy = PolicyNetwork(state_dim=101, num_actions=3900)
        >>> evaluator = NX0Evaluator(bc_policy, target_policy)
        >>> result = evaluator.evaluate_trajectory(states, actions, rewards, mask)
        >>> print(result['nx0_value'])
    """

    def __init__(
        self,
        bc_policy: nn.Module,
        target_policy: nn.Module,
        clip_weight: float = 20.0,
        epsilon: float = 1e-8
    ):
        self.bc_policy = bc_policy
        self.target_policy = target_policy
        self.clip_weight = clip_weight
        self.epsilon = epsilon

        # Freeze BC policy parameters (it should not be trained)
        for param in self.bc_policy.parameters():
            param.requires_grad = False

        self.bc_policy.eval()

    def compute_importance_weights(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute clipped importance weights for state-action pairs.

        Args:
            states: (batch, state_dim) or (seq_len, state_dim)
            actions: (batch,) or (seq_len,) action indices

        Returns:
            (batch,) or (seq_len,) clipped importance weights
        """
        with torch.no_grad():
            self.bc_policy.eval()
            self.target_policy.eval()

            # Get behavior policy probabilities μ(a|s)
            bc_logits = self.bc_policy(states)
            bc_probs = torch.softmax(bc_logits, dim=-1)
            mu = bc_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Get target policy probabilities π(a|s)
            target_logits = self.target_policy(states)
            target_probs = torch.softmax(target_logits, dim=-1)
            pi = target_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute importance weights with clipping
            # w = min(C, π / (μ + ε))
            weights = torch.clamp(
                pi / (mu + self.epsilon),
                max=self.clip_weight
            )

        return weights

    def evaluate_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate a single trajectory using NX_0.

        Args:
            states: (seq_len, state_dim)
            actions: (seq_len,) action indices
            rewards: (seq_len,) rewards
            mask: (seq_len,) boolean mask (True for valid timesteps)

        Returns:
            Dictionary with:
                - nx0_value: NX_0 estimate
                - effective_sample_size: ESS metric
                - mean_weight: Average importance weight
                - max_weight: Maximum importance weight
        """
        # Compute importance weights
        weights = self.compute_importance_weights(states, actions)

        # Apply mask (only consider valid timesteps)
        weights = weights * mask.float()
        rewards = rewards * mask.float()

        # Self-normalize weights
        total_weight = weights.sum()
        if total_weight > 0:
            normalized_weights = weights / total_weight
        else:
            # Handle edge case: all weights are zero
            normalized_weights = torch.zeros_like(weights)

        # Compute NX_0 value
        nx0_value = (normalized_weights * rewards).sum().item()

        # Compute effective sample size
        # ESS = (Σ w_i)^2 / Σ w_i^2
        sum_weights = weights.sum()
        sum_weights_sq = (weights ** 2).sum()
        if sum_weights_sq > 0:
            ess = ((sum_weights ** 2) / sum_weights_sq).item()
        else:
            ess = 0.0

        # Compute diagnostics
        valid_weights = weights[mask]
        mean_weight = valid_weights.mean().item() if mask.any() else 0.0
        max_weight = valid_weights.max().item() if mask.any() else 0.0

        return {
            'nx0_value': nx0_value,
            'effective_sample_size': ess,
            'mean_weight': mean_weight,
            'max_weight': max_weight
        }

    def evaluate_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate multiple trajectories using NX_0.

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len) action indices
            rewards: (batch, seq_len) rewards
            mask: (batch, seq_len) boolean mask

        Returns:
            Dictionary with aggregated metrics:
                - nx0_value: Average NX_0 across trajectories
                - effective_sample_size: Total ESS
                - mean_weight: Average importance weight
        """
        batch_size, seq_len, state_dim = states.shape

        # Flatten batch and sequence dimensions
        states_flat = states.view(-1, state_dim)
        actions_flat = actions.view(-1)
        rewards_flat = rewards.view(-1)
        mask_flat = mask.view(-1)

        # Compute importance weights for all timesteps
        weights_flat = self.compute_importance_weights(
            states_flat, actions_flat
        )

        # Apply mask
        weights_flat = weights_flat * mask_flat.float()
        rewards_flat = rewards_flat * mask_flat.float()

        # Self-normalize weights
        total_weight = weights_flat.sum()
        if total_weight > 0:
            normalized_weights = weights_flat / total_weight
        else:
            normalized_weights = torch.zeros_like(weights_flat)

        # Compute NX_0 value
        nx0_value = (normalized_weights * rewards_flat).sum().item()

        # Compute effective sample size
        sum_weights = weights_flat.sum()
        sum_weights_sq = (weights_flat ** 2).sum()
        if sum_weights_sq > 0:
            ess = ((sum_weights ** 2) / sum_weights_sq).item()
        else:
            ess = 0.0

        # Compute diagnostics
        valid_weights = weights_flat[mask_flat]
        mean_weight = valid_weights.mean().item() if mask_flat.any() else 0.0

        return {
            'nx0_value': nx0_value,
            'effective_sample_size': ess,
            'mean_weight': mean_weight
        }
