"""SASRec (Self-Attentive Sequential Recommendation) encoder.

This module implements a Transformer-based sequence encoder for
user interaction histories.
"""

import torch
import torch.nn as nn
import math

from src.utils.time_features import bucket_time_delta


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds position information to sequence embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length

    Example:
        >>> pe = PositionalEncoding(d_model=64, max_len=100)
        >>> x = torch.randn(4, 10, 64)  # (batch, seq_len, d_model)
        >>> output = pe(x)
        >>> output.shape
        torch.Size([4, 10, 64])
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            x + positional encoding, same shape as input
        """
        seq_len = x.size(1)
        # Add positional encoding (broadcast across batch)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class SASRecEncoder(nn.Module):
    """Self-Attentive Sequential Recommendation encoder.

    Encodes user interaction sequences using:
    - Movie, rating, and time delta embeddings
    - Positional encoding
    - Multi-layer Transformer with causal masking
    - LayerNorm and residual connections

    Args:
        num_movies: Number of unique movies (vocabulary size)
        d_model: Model dimension (default: 64)
        nhead: Number of attention heads (default: 2)
        num_layers: Number of Transformer layers (default: 2)
        dim_feedforward: FFN hidden dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
        max_seq_len: Maximum sequence length (default: 50)

    Example:
        >>> encoder = SASRecEncoder(num_movies=4000, d_model=64)
        >>> movie_ids = torch.randint(1, 4000, (32, 20))
        >>> ratings = torch.randint(1, 6, (32, 20))
        >>> timestamps = torch.arange(1000, 1020).unsqueeze(0).repeat(32, 1)
        >>> padding_mask = torch.ones(32, 20, dtype=torch.bool)
        >>> output = encoder(movie_ids, ratings, timestamps, padding_mask)
        >>> output.shape
        torch.Size([32, 20, 64])
    """

    def __init__(
        self,
        num_movies: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        super().__init__()

        self.d_model = d_model
        self.num_movies = num_movies
        self.max_seq_len = max_seq_len

        # Embeddings
        # Movie embedding (padding_idx=0 for padding tokens)
        self.movie_embedding = nn.Embedding(
            num_embeddings=num_movies + 1,  # +1 for padding
            embedding_dim=d_model,
            padding_idx=0
        )

        # Rating embedding (ratings 1-5, plus 0 for padding)
        # Dimension is d_model // 4 to save parameters
        rating_embed_dim = max(d_model // 4, 8)
        self.rating_embedding = nn.Embedding(
            num_embeddings=6,  # 0-5
            embedding_dim=rating_embed_dim,
            padding_idx=0
        )

        # Time delta embedding (20 buckets)
        time_embed_dim = max(d_model // 4, 8)
        self.time_embedding = nn.Embedding(
            num_embeddings=20,
            embedding_dim=time_embed_dim,
            padding_idx=0
        )

        # Projection layer to combine embeddings to d_model
        combined_embed_dim = d_model + rating_embed_dim + time_embed_dim
        self.input_projection = nn.Linear(combined_embed_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        movie_ids: torch.Tensor,
        ratings: torch.Tensor,
        timestamps: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode user interaction sequence.

        Args:
            movie_ids: (batch, seq_len) movie IDs
            ratings: (batch, seq_len) ratings (1-5, 0 for padding)
            timestamps: (batch, seq_len) timestamps in seconds
            padding_mask: (batch, seq_len) boolean mask (True=valid, False=padding)

        Returns:
            Sequence encoding: (batch, seq_len, d_model)
            Each position contains the encoded representation up to that point.
        """
        batch_size, seq_len = movie_ids.shape

        # Embed movie IDs
        movie_embed = self.movie_embedding(movie_ids)  # (batch, seq, d_model)

        # Embed ratings
        rating_embed = self.rating_embedding(ratings)  # (batch, seq, rating_dim)

        # Bucket time deltas and embed
        time_buckets = bucket_time_delta(timestamps, num_buckets=20)
        time_embed = self.time_embedding(time_buckets)  # (batch, seq, time_dim)

        # Combine embeddings
        combined = torch.cat([movie_embed, rating_embed, time_embed], dim=-1)
        combined = self.input_projection(combined)  # (batch, seq, d_model)

        # Add positional encoding
        combined = self.positional_encoding(combined)

        # Apply dropout
        combined = self.dropout(combined)

        # Create causal mask (prevent attending to future positions)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(movie_ids.device)

        # Create key padding mask (inverted: True=ignore, False=attend)
        # PyTorch expects True for positions to IGNORE
        key_padding_mask = ~padding_mask  # Invert: True->False, False->True

        # Apply Transformer
        output = self.transformer(
            combined,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )

        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask to prevent attending to future positions.

        Args:
            sz: Sequence length

        Returns:
            Mask tensor of shape (sz, sz)
            Values: 0.0 for allowed positions, -inf for masked positions
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.masked_fill(mask, float('-inf'))
        mask = mask.masked_fill(~mask, 0.0)
        return mask
