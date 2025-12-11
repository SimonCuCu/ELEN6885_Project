"""IQL loss functions.

This module implements the loss functions for IQL:
- expectile_loss: Asymmetric regression loss for V-network
- q_loss: Temporal difference loss for Q-network
- awr_policy_loss: Advantage-weighted regression loss for policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.7
) -> torch.Tensor:
    """Compute expectile regression loss.

    Expectile loss is an asymmetric squared loss that penalizes positive and
    negative errors differently. Used for training the V-network in IQL.

    Loss = |tau - I(error < 0)| * error^2
    where error = target - prediction

    Args:
        prediction: (batch, *) predicted values
        target: (batch, *) target values
        tau: Expectile level in [0, 1]. tau=0.5 is equivalent to MSE.
            Higher tau puts more weight on positive errors (target > prediction).
            Typical values: 0.7-0.9 for IQL.

    Returns:
        Scalar loss value

    Example:
        >>> prediction = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([2.0, 2.0, 1.0])
        >>> loss = expectile_loss(prediction, target, tau=0.7)
    """
    error = target - prediction
    weight = torch.abs(tau - (error < 0).float())
    return (weight * error ** 2).mean()


def q_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """Compute Q-network loss using temporal difference.

    Computes MSE loss between Q(s,a) and the TD target:
    target = r + gamma * V(s') * (1 - done)

    Args:
        q_values: (batch, num_actions) Q-values for all actions
        actions: (batch,) action indices taken
        rewards: (batch,) immediate rewards
        next_values: (batch,) V(s') values from target V-network
        dones: (batch,) boolean tensor, True if episode ended
        gamma: Discount factor

    Returns:
        Scalar loss value

    Example:
        >>> q_values = torch.randn(32, 3900)
        >>> actions = torch.randint(0, 3900, (32,))
        >>> rewards = torch.randn(32)
        >>> next_values = torch.randn(32)
        >>> dones = torch.zeros(32, dtype=torch.bool)
        >>> loss = q_loss(q_values, actions, rewards, next_values, dones)
    """
    # Select Q-values for taken actions
    q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute TD target: r + gamma * V(s') * (1 - done)
    with torch.no_grad():
        q_target = rewards + gamma * next_values * (~dones).float()

    # MSE loss
    return F.mse_loss(q_pred, q_target)


def awr_policy_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 3.0,
    clip_weight: float = 20.0
) -> torch.Tensor:
    """Compute Advantage-Weighted Regression (AWR) policy loss.

    Computes weighted cross-entropy loss where weights are exponential
    advantages: weight = min(clip_weight, exp(advantage / beta))

    Loss = -mean(weight * log pi(a|s))

    Args:
        logits: (batch, num_actions) policy logits for all actions
        actions: (batch,) action indices taken in behavior policy
        advantages: (batch,) advantage values A(s,a) = Q(s,a) - V(s)
        beta: Temperature parameter controlling advantage sensitivity.
            Higher beta makes policy less sensitive to advantages.
            Typical values: 1.0-5.0 for IQL.
        clip_weight: Maximum weight to prevent explosion from large advantages.
            Typical values: 20.0-100.0

    Returns:
        Scalar loss value

    Example:
        >>> logits = torch.randn(32, 3900)
        >>> actions = torch.randint(0, 3900, (32,))
        >>> advantages = torch.randn(32)
        >>> loss = awr_policy_loss(logits, actions, advantages, beta=3.0)
    """
    # Compute AWR weights: exp(A / beta)
    weights = torch.exp(advantages / beta)

    # Clip weights to prevent explosion
    weights = torch.clamp(weights, max=clip_weight)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Select log probabilities for taken actions
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Weighted negative log likelihood
    loss = -(weights * selected_log_probs).mean()

    return loss
