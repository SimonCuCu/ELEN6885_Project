"""PyTorch Dataset for IQL training on MovieLens trajectories.

This module provides MovieLensIQLDataset which converts user trajectories
into format suitable for IQL training.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
import numpy as np

from src.data_module.trajectory import Trajectory
from src.data_module.constants import (
    UNK_TOKEN,
    AGE_CATEGORIES,
    OCCUPATION_RANGE
)


class MovieLensIQLDataset(Dataset):
    """Dataset for IQL training on full trajectories.

    Each sample is a user trajectory. Returns:
    - user_id: User identifier
    - Static features (for StaticFeatureEncoder):
      - gender: scalar (0=M, 1=F)
      - age: 7-d one-hot vector
      - occupation: 21-d one-hot vector
      - zipcode_bucket: scalar (bucket ID from vocabulary)
    - Sequence features:
      - movie_sequence: (max_seq_len,) padded movie IDs
      - rating_sequence: (max_seq_len,) padded ratings
      - timestamp_sequence: (max_seq_len,) padded timestamps
      - sequence_mask: (max_seq_len,) boolean mask for valid positions
      - sequence_length: scalar (actual length before padding)
    - IQL targets:
      - actions: (max_seq_len,) target actions (next movie)
      - rewards: (max_seq_len,) rewards (ratings)
      - dones: (max_seq_len,) terminal flags

    Args:
        trajectories: List of Trajectory objects
        zipcode_vocab: Dict mapping zipcode_prefix to bucket_id
        max_seq_len: Maximum sequence length (default: 50)
        pad_movie_id: Movie ID used for padding (default: 0)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = MovieLensIQLDataset(train_trajs, zipcode_vocab)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     print(batch["user_id"].shape)
        ...     break
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        zipcode_vocab: Dict[str, int],
        max_seq_len: int = 50,
        pad_movie_id: int = 0
    ):
        self.trajectories = trajectories
        self.zipcode_vocab = zipcode_vocab
        self.max_seq_len = max_seq_len
        self.pad_movie_id = pad_movie_id

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single trajectory as dict of tensors.

        Args:
            idx: Trajectory index

        Returns:
            Dictionary with all features and targets as tensors
        """
        traj = self.trajectories[idx]

        # Encode static features
        gender = self._encode_gender(traj.static_features["Gender"])
        age = self._encode_age(traj.static_features["Age"])
        occupation = self._encode_occupation(traj.static_features["Occupation"])
        zipcode_bucket = self._encode_zipcode(traj.static_features["zipcode_prefix"])

        # Process sequence (truncate if too long, pad if too short)
        seq_len = min(len(traj.movie_ids), self.max_seq_len)
        movie_sequence = self._pad_sequence(traj.movie_ids, seq_len)
        rating_sequence = self._pad_sequence(traj.ratings, seq_len)
        timestamp_sequence = self._pad_sequence(traj.timestamps, seq_len)

        # Create sequence mask (True for valid positions, False for padding)
        sequence_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        sequence_mask[:seq_len] = True

        # Create IQL targets
        # Actions: next movie in sequence (for each state)
        # For trajectory [m1, m2, m3], actions are [m2, m3, 0]
        actions = torch.zeros(self.max_seq_len, dtype=torch.long)
        if seq_len > 0:
            # Shift movies by 1
            actions[:seq_len-1] = torch.from_numpy(
                traj.movie_ids[1:seq_len].copy()
            ).long()
            # Last action is padded (or could be special end token)
            actions[seq_len-1] = self.pad_movie_id

        # Rewards: ratings at each step
        rewards = torch.zeros(self.max_seq_len, dtype=torch.float32)
        if seq_len > 0:
            rewards[:seq_len] = torch.from_numpy(
                traj.ratings[:seq_len].copy()
            ).float()

        # Dones: mark terminal states
        dones = torch.zeros(self.max_seq_len, dtype=torch.float32)
        if seq_len > 0:
            dones[seq_len-1] = 1.0  # Last valid position is terminal

        return {
            "user_id": torch.tensor(traj.user_id, dtype=torch.long),
            # Static features
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "zipcode_bucket": zipcode_bucket,
            # Sequence features
            "movie_sequence": movie_sequence,
            "rating_sequence": rating_sequence,
            "timestamp_sequence": timestamp_sequence,
            "sequence_mask": sequence_mask,
            "sequence_length": torch.tensor(seq_len, dtype=torch.long),
            # IQL targets
            "actions": actions,
            "rewards": rewards,
            "dones": dones
        }

    def _encode_gender(self, gender: str) -> torch.Tensor:
        """Encode gender as 0 (M) or 1 (F).

        Args:
            gender: 'M' or 'F'

        Returns:
            Scalar tensor (0 or 1)
        """
        return torch.tensor(0 if gender == "M" else 1, dtype=torch.long)

    def _encode_age(self, age: int) -> torch.Tensor:
        """Encode age as 7-d one-hot vector.

        Args:
            age: Age category (1, 18, 25, 35, 45, 50, 56)

        Returns:
            7-d one-hot tensor
        """
        age_to_idx = {age_cat: idx for idx, age_cat in enumerate(AGE_CATEGORIES)}
        age_idx = age_to_idx.get(age, 0)  # Default to first category if not found

        one_hot = torch.zeros(len(AGE_CATEGORIES), dtype=torch.float32)
        one_hot[age_idx] = 1.0
        return one_hot

    def _encode_occupation(self, occupation: int) -> torch.Tensor:
        """Encode occupation as 21-d one-hot vector.

        Args:
            occupation: Occupation code (0-20)

        Returns:
            21-d one-hot tensor
        """
        num_occupations = OCCUPATION_RANGE[1] - OCCUPATION_RANGE[0] + 1  # 21

        # Clamp to valid range
        occupation = max(OCCUPATION_RANGE[0], min(OCCUPATION_RANGE[1], occupation))

        one_hot = torch.zeros(num_occupations, dtype=torch.float32)
        one_hot[occupation] = 1.0
        return one_hot

    def _encode_zipcode(self, zipcode_prefix: str) -> torch.Tensor:
        """Encode zipcode prefix to bucket ID.

        Args:
            zipcode_prefix: Zipcode prefix string

        Returns:
            Scalar tensor with bucket ID
        """
        # Use UNK_TOKEN bucket if not in vocabulary
        bucket_id = self.zipcode_vocab.get(zipcode_prefix, self.zipcode_vocab[UNK_TOKEN])
        return torch.tensor(bucket_id, dtype=torch.long)

    def _pad_sequence(
        self,
        sequence: np.ndarray,
        seq_len: int
    ) -> torch.Tensor:
        """Pad or truncate sequence to max_seq_len.

        Args:
            sequence: NumPy array to pad/truncate
            seq_len: Actual valid length

        Returns:
            Padded tensor of shape (max_seq_len,)
        """
        # Truncate if needed
        sequence = sequence[:self.max_seq_len]

        # Convert to tensor
        tensor = torch.from_numpy(sequence.copy())

        # Pad if needed
        if len(tensor) < self.max_seq_len:
            padding = torch.zeros(
                self.max_seq_len - len(tensor),
                dtype=tensor.dtype
            )
            tensor = torch.cat([tensor, padding])

        return tensor
