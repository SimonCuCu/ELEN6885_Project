"""Trainer for Behavioral Cloning policy."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict


class BCPolicyTrainer:
    """Trainer for Behavioral Cloning policy.

    Trains BC policy using cross-entropy loss on historical (state, action) pairs.

    Args:
        bc_policy: BCPolicy model to train
        lr: Learning rate (default: 1e-4)

    Example:
        >>> from src.models.bc_policy import BCPolicy
        >>> bc_policy = BCPolicy(state_dim=101, num_actions=3900)
        >>> trainer = BCPolicyTrainer(bc_policy, lr=1e-4)
        >>> metrics = trainer.train_step(states, actions)
        >>> print(metrics['bc_loss'], metrics['accuracy'])
    """

    def __init__(
        self,
        bc_policy: nn.Module,
        lr: float = 1e-4
    ):
        self.bc_policy = bc_policy
        self.optimizer = optim.Adam(bc_policy.parameters(), lr=lr)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step.

        Args:
            states: (batch, state_dim)
            actions: (batch,) ground truth actions

        Returns:
            Dictionary with:
                - bc_loss: Cross-entropy loss
                - accuracy: Top-1 accuracy
        """
        self.bc_policy.train()

        # Forward pass
        logits = self.bc_policy(states)

        # Compute loss
        loss = F.cross_entropy(logits, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == actions).float().mean()

        return {
            'bc_loss': loss.item(),
            'accuracy': accuracy.item()
        }

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate BC policy on validation data.

        Args:
            states: (batch, state_dim)
            actions: (batch,) ground truth actions

        Returns:
            Dictionary with:
                - loss: Cross-entropy loss
                - accuracy: Top-1 accuracy
                - perplexity: exp(loss)
        """
        self.bc_policy.eval()

        with torch.no_grad():
            logits = self.bc_policy(states)
            loss = F.cross_entropy(logits, actions)

            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == actions).float().mean()

            perplexity = torch.exp(loss)

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item()
        }

    def save(self, path: str):
        """Save BC policy checkpoint.

        Args:
            path: File path to save checkpoint
        """
        state = {
            'bc_policy': self.bc_policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load BC policy checkpoint.

        Args:
            path: File path to load checkpoint
        """
        state = torch.load(path)
        self.bc_policy.load_state_dict(state['bc_policy'])
        self.optimizer.load_state_dict(state['optimizer'])
