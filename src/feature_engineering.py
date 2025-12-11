import numpy as np
import pandas as pd
from typing import Dict


def encode_user_features(users_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """One-hot encode user attributes."""
    required_cols = ['UserID', 'Gender', 'Age', 'Occupation']
    for col in required_cols:
        if col not in users_df.columns:
            raise KeyError(f"Missing required column: {col}")

    user_features = {}

    gender_map = {'M': 0, 'F': 1}

    age_categories = [1, 18, 25, 35, 45, 50, 56]

    for _, row in users_df.iterrows():
        user_id = row['UserID']

        features = np.zeros(30, dtype=np.float32)

        gender = row['Gender']
        if gender not in gender_map:
            raise ValueError(f"Invalid gender value: {gender}. Expected 'M' or 'F'")
        gender_idx = gender_map[gender]
        features[gender_idx] = 1

        age = row['Age']
        if age not in age_categories:
            raise ValueError(f"Invalid age value: {age}. Expected one of {age_categories}")
        age_idx = age_categories.index(age)
        features[2 + age_idx] = 1

        occ = row['Occupation']
        if not (0 <= occ <= 20):
            raise ValueError(f"Invalid occupation value: {occ}. Expected 0-20")
        features[9 + occ] = 1

        user_features[user_id] = features

    return user_features


def encode_movie_features(movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Multi-hot encode movie genres."""
    required_cols = ['MovieID', 'Title', 'Genres']
    for col in required_cols:
        if col not in movies_df.columns:
            raise KeyError(f"Missing required column: {col}")

    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    genre_to_idx = {genre: i for i, genre in enumerate(genres_list)}

    movie_features = {}

    for _, row in movies_df.iterrows():
        movie_id = row['MovieID']
        genres_str = row['Genres']

        features = np.zeros(18, dtype=np.float32)

        if not isinstance(genres_str, str):
            raise ValueError(f"Invalid Genres format for MovieID {movie_id}: {genres_str}")

        genres = genres_str.split('|')

        for genre in genres:
            if genre in genre_to_idx:
                features[genre_to_idx[genre]] = 1

        movie_features[movie_id] = features

    return movie_features


def create_context(
    user_features: np.ndarray,
    movie_features: np.ndarray
) -> np.ndarray:
    """Concatenate user and movie features into a 48-dim context."""
    if user_features.shape != (30,):
        raise ValueError(f"User features must be 30-dimensional, got {user_features.shape}")
    if movie_features.shape != (18,):
        raise ValueError(f"Movie features must be 18-dimensional, got {movie_features.shape}")

    return np.concatenate([user_features, movie_features])


def rating_to_reward(rating: float, threshold: float = 4.0) -> float:
    """Convert rating to binary reward using the given threshold."""
    if not (1 <= rating <= 5):
        raise ValueError(f"Rating must be in range [1, 5], got {rating}")

    return 1.0 if rating >= threshold else 0.0


def batch_create_contexts(
    user_features_dict: Dict[int, np.ndarray],
    movie_features_dict: Dict[int, np.ndarray],
    user_ids: np.ndarray,
    movie_ids: np.ndarray
) -> np.ndarray:
    """Create context vectors for multiple user-movie pairs."""
    if len(user_ids) != len(movie_ids):
        raise ValueError(
            f"user_ids and movie_ids must have same length. "
            f"Got {len(user_ids)} and {len(movie_ids)}"
        )

    n_pairs = len(user_ids)
    contexts = np.zeros((n_pairs, 48), dtype=np.float32)

    for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
        if user_id not in user_features_dict:
            raise KeyError(f"UserID {user_id} not found in user_features_dict")
        if movie_id not in movie_features_dict:
            raise KeyError(f"MovieID {movie_id} not found in movie_features_dict")

        contexts[i] = create_context(
            user_features_dict[user_id],
            movie_features_dict[movie_id]
        )

    return contexts
