"""Static feature encoder for user demographics.

This module provides:
- StaticFeatureEncoder: PyTorch module to encode user static features
- build_zipcode_vocabulary: Create zipcode prefix to bucket mapping
"""

import torch
import torch.nn as nn
import pandas as pd
from typing import Dict

from src.data_module.constants import UNK_TOKEN


def build_zipcode_vocabulary(users_df: pd.DataFrame) -> Dict[str, int]:
    """Build zipcode prefix to bucket_id mapping.

    Creates a vocabulary mapping zipcode prefixes to integer bucket IDs.
    UNK_TOKEN is always mapped to bucket 0.

    Args:
        users_df: DataFrame with 'zipcode_prefix' column

    Returns:
        Dict mapping zipcode_prefix string to bucket_id (int)
        UNK_TOKEN maps to 0, other zipcodes to 1, 2, 3, ...

    Example:
        >>> users_df = pd.DataFrame({
        ...     "UserID": [1, 2, 3],
        ...     "zipcode_prefix": ["480", "123", "480"]
        ... })
        >>> vocab = build_zipcode_vocabulary(users_df)
        >>> vocab["<UNK>"]
        0
        >>> vocab["480"]
        1
    """
    # Get unique zipcode prefixes
    unique_zipcodes = users_df["zipcode_prefix"].unique()

    # Initialize vocabulary with UNK_TOKEN at index 0
    vocab = {UNK_TOKEN: 0}

    # Add other zipcodes (skip UNK_TOKEN if it appears in data)
    bucket_id = 1
    for zipcode in sorted(unique_zipcodes):
        if zipcode not in vocab:
            vocab[zipcode] = bucket_id
            bucket_id += 1

    return vocab


class StaticFeatureEncoder(nn.Module):
    """Encode user static features into a fixed-size vector.

    Encodes:
    - Gender: 1-d (0 for M, 1 for F)
    - Age: 7-d one-hot
    - Occupation: 21-d one-hot
    - Zipcode: embedded from bucket (8-d by default)

    Output dimension: 1 + 7 + 21 + zipcode_embed_dim = 37 (default)

    Args:
        num_zipcode_buckets: Number of zipcode buckets (vocabulary size)
        zipcode_embed_dim: Embedding dimension for zipcode (default: 8)

    Example:
        >>> encoder = StaticFeatureEncoder(num_zipcode_buckets=50)
        >>> static_features = {
        ...     "gender": torch.tensor([0, 1]),
        ...     "age": torch.tensor([[1, 0, 0, 0, 0, 0, 0],
        ...                          [0, 1, 0, 0, 0, 0, 0]]),
        ...     "occupation": torch.zeros(2, 21),
        ...     "zipcode_bucket": torch.tensor([0, 5])
        ... }
        >>> output = encoder(static_features)
        >>> output.shape
        torch.Size([2, 37])
    """

    def __init__(
        self,
        num_zipcode_buckets: int,
        zipcode_embed_dim: int = 8
    ):
        super().__init__()

        self.num_zipcode_buckets = num_zipcode_buckets
        self.zipcode_embed_dim = zipcode_embed_dim

        # Zipcode embedding layer
        self.zipcode_embedding = nn.Embedding(
            num_embeddings=num_zipcode_buckets,
            embedding_dim=zipcode_embed_dim
        )

        # Calculate total output dimension
        # gender(1) + age(7) + occupation(21) + zipcode_embed(8) = 37
        self.output_dim = 1 + 7 + 21 + zipcode_embed_dim

    def forward(self, static_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode static features into fixed-size vector.

        Args:
            static_features: Dictionary containing:
                - gender: (batch_size,) tensor with values 0 or 1
                - age: (batch_size, 7) one-hot tensor
                - occupation: (batch_size, 21) one-hot tensor
                - zipcode_bucket: (batch_size,) tensor with bucket IDs

        Returns:
            Encoded features: (batch_size, output_dim) tensor

        Raises:
            KeyError: If required keys are missing from static_features
        """
        # Extract features
        gender = static_features["gender"].float().unsqueeze(1)  # (batch, 1)
        age = static_features["age"].float()  # (batch, 7)
        occupation = static_features["occupation"].float()  # (batch, 21)
        zipcode_bucket = static_features["zipcode_bucket"]  # (batch,)

        # Embed zipcode
        zipcode_embed = self.zipcode_embedding(zipcode_bucket)  # (batch, embed_dim)

        # Concatenate all features
        encoded = torch.cat([
            gender,           # 1-d
            age,             # 7-d
            occupation,      # 21-d
            zipcode_embed    # 8-d (default)
        ], dim=1)

        return encoded
