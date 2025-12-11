"""State encoder for IQL.

This module combines static user features with sequential interaction history
to produce a unified state representation.
"""

import torch
import torch.nn as nn

from src.data_module.static_encoder import StaticFeatureEncoder
from src.models.sasrec_encoder import SASRecEncoder


class StateEncoder(nn.Module):
    """State encoder combining static features and sequence encoding.

    Combines user static features (gender, age, occupation, zipcode) with
    their interaction history encoded by SASRec to produce a unified state
    representation for IQL.

    Args:
        static_encoder: StaticFeatureEncoder instance
        sasrec_encoder: SASRecEncoder instance

    Attributes:
        output_dim: Total dimension of state representation
            = static_encoder.output_dim + sasrec_encoder.d_model
            = 37 + 64 = 101 (with default configurations)

    Example:
        >>> static_encoder = StaticFeatureEncoder(num_zipcode_buckets=50)
        >>> sasrec_encoder = SASRecEncoder(num_movies=4000, d_model=64)
        >>> state_encoder = StateEncoder(static_encoder, sasrec_encoder)
        >>> state_encoder.output_dim
        101
    """

    def __init__(
        self,
        static_encoder: StaticFeatureEncoder,
        sasrec_encoder: SASRecEncoder
    ):
        super().__init__()
        self.static_encoder = static_encoder
        self.sasrec_encoder = sasrec_encoder

        # Total state dimension
        self.output_dim = static_encoder.output_dim + sasrec_encoder.d_model

    def forward(
        self,
        gender: torch.Tensor,
        age: torch.Tensor,
        occupation: torch.Tensor,
        zipcode_bucket: torch.Tensor,
        movie_sequence: torch.Tensor,
        rating_sequence: torch.Tensor,
        timestamp_sequence: torch.Tensor,
        sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode state from static features and interaction sequence.

        Args:
            gender: (batch,) gender indices (0=M, 1=F)
            age: (batch,) age category indices (0-6)
            occupation: (batch,) occupation indices (0-20)
            zipcode_bucket: (batch,) zipcode bucket IDs
            movie_sequence: (batch, seq_len) movie IDs
            rating_sequence: (batch, seq_len) ratings (1-5, 0 for padding)
            timestamp_sequence: (batch, seq_len) timestamps in seconds
            sequence_mask: (batch, seq_len) boolean mask (True=valid, False=padding)

        Returns:
            State representation: (batch, output_dim)
            Concatenation of [static_features ; last_step_sequence_encoding]
        """
        batch_size = gender.shape[0]

        # Pack static features into dictionary for StaticFeatureEncoder
        static_features_dict = {
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "zipcode_bucket": zipcode_bucket
        }

        # Encode static features (batch, 37)
        static_features = self.static_encoder(static_features_dict)

        # Encode sequence (batch, seq_len, d_model)
        sequence_encoding = self.sasrec_encoder(
            movie_ids=movie_sequence,
            ratings=rating_sequence,
            timestamps=timestamp_sequence,
            padding_mask=sequence_mask
        )

        # Extract last valid step from sequence encoding
        # Find last valid position for each sample in batch
        # sequence_mask: (batch, seq_len) with True for valid positions
        sequence_lengths = sequence_mask.sum(dim=1)  # (batch,) number of valid positions

        # Extract encoding at last valid position for each sample
        # batch_indices: [0, 1, 2, ..., batch_size-1]
        batch_indices = torch.arange(batch_size, device=sequence_encoding.device)
        # last_valid_indices: [seq_len-1, seq_len-1, ...] for each sample
        last_valid_indices = (sequence_lengths - 1).clamp(min=0)

        # last_step_encoding: (batch, d_model)
        last_step_encoding = sequence_encoding[batch_indices, last_valid_indices]

        # Concatenate static features and sequence encoding
        state = torch.cat([static_features, last_step_encoding], dim=-1)

        return state
