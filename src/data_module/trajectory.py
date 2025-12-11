"""Trajectory construction and temporal splitting.

This module provides:
- Trajectory dataclass for user interaction sequences
- Functions to build trajectories from ratings and user data
- Temporal splitting into train/val/test sets
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


@dataclass
class Trajectory:
    """Single user trajectory with full interaction history.

    Attributes:
        user_id: User identifier
        movie_ids: Array of movie IDs in temporal order
        ratings: Array of ratings (aligned with movie_ids)
        timestamps: Array of timestamps (aligned with movie_ids)
        static_features: Dict with Gender, Age, Occupation, zipcode_prefix
        split: 'train' | 'val' | 'test' | 'none'
    """
    user_id: int
    movie_ids: np.ndarray  # (T,)
    ratings: np.ndarray    # (T,)
    timestamps: np.ndarray # (T,)
    static_features: Dict[str, any]
    split: str


def build_trajectories(
    ratings_df: pd.DataFrame,
    users_df: pd.DataFrame,
    min_interactions: int = 20
) -> List[Trajectory]:
    """Build per-user trajectories from ratings and user data.

    For each user with at least min_interactions:
    1. Extract all ratings, sorted by timestamp
    2. Merge with static user features
    3. Create Trajectory object

    Args:
        ratings_df: Cleaned ratings DataFrame
        users_df: Cleaned users DataFrame (with zipcode_prefix)
        min_interactions: Minimum number of interactions required per user

    Returns:
        List of Trajectory objects, one per qualifying user

    Example:
        >>> ratings = parse_ratings("datasets/ratings.dat")
        >>> users = clean_users(parse_users("datasets/users.dat"))
        >>> trajectories = build_trajectories(ratings, users, min_interactions=20)
        >>> print(f"Built {len(trajectories)} trajectories")
    """
    trajectories = []

    # Group ratings by user
    user_groups = ratings_df.groupby("UserID")

    for user_id, user_ratings in user_groups:
        # Filter users with insufficient interactions
        if len(user_ratings) < min_interactions:
            continue

        # Sort by timestamp (ascending)
        user_ratings = user_ratings.sort_values("Timestamp")

        # Extract sequences
        movie_ids = user_ratings["MovieID"].values
        ratings = user_ratings["Rating"].values
        timestamps = user_ratings["Timestamp"].values

        # Get static features from users_df
        user_info = users_df[users_df["UserID"] == user_id]

        if len(user_info) == 0:
            # Skip users not in users_df
            continue

        user_info = user_info.iloc[0]

        static_features = {
            "Gender": user_info["Gender"],
            "Age": user_info["Age"],
            "Occupation": user_info["Occupation"],
            "zipcode_prefix": user_info["zipcode_prefix"]
        }

        # Create trajectory
        trajectory = Trajectory(
            user_id=int(user_id),
            movie_ids=movie_ids,
            ratings=ratings,
            timestamps=timestamps,
            static_features=static_features,
            split="none"
        )

        trajectories.append(trajectory)

    return trajectories


def temporal_split_trajectories(
    trajectories: List[Trajectory],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """Split each trajectory temporally into train/val/test.

    For each user, split their interaction sequence by time:
    - Train: first train_ratio of interactions
    - Val: next val_ratio of interactions
    - Test: remaining interactions

    This ensures no future data leakage into training.

    Args:
        trajectories: List of user trajectories
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)

    Returns:
        Tuple of (train_trajectories, val_trajectories, test_trajectories)

    Example:
        >>> train, val, test = temporal_split_trajectories(trajectories)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    train_trajectories = []
    val_trajectories = []
    test_trajectories = []

    test_ratio = 1.0 - train_ratio - val_ratio

    for traj in trajectories:
        n_interactions = len(traj.movie_ids)

        # Calculate split indices
        train_end = int(n_interactions * train_ratio)
        val_end = int(n_interactions * (train_ratio + val_ratio))

        # Ensure at least 1 interaction in each split
        # For very short trajectories, use minimum allocation
        if n_interactions <= 3:
            train_end = 1
            val_end = 2
        else:
            train_end = max(1, train_end)
            val_end = max(train_end + 1, min(val_end, n_interactions - 1))

        # Split train
        train_traj = Trajectory(
            user_id=traj.user_id,
            movie_ids=traj.movie_ids[:train_end].copy(),
            ratings=traj.ratings[:train_end].copy(),
            timestamps=traj.timestamps[:train_end].copy(),
            static_features=traj.static_features.copy(),
            split="train"
        )
        train_trajectories.append(train_traj)

        # Split val
        val_traj = Trajectory(
            user_id=traj.user_id,
            movie_ids=traj.movie_ids[train_end:val_end].copy(),
            ratings=traj.ratings[train_end:val_end].copy(),
            timestamps=traj.timestamps[train_end:val_end].copy(),
            static_features=traj.static_features.copy(),
            split="val"
        )
        val_trajectories.append(val_traj)

        # Split test
        test_traj = Trajectory(
            user_id=traj.user_id,
            movie_ids=traj.movie_ids[val_end:].copy(),
            ratings=traj.ratings[val_end:].copy(),
            timestamps=traj.timestamps[val_end:].copy(),
            static_features=traj.static_features.copy(),
            split="test"
        )
        test_trajectories.append(test_traj)

    return train_trajectories, val_trajectories, test_trajectories
