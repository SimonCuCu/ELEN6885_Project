import pandas as pd
from typing import Dict, Tuple


def load_users(filepath: str) -> pd.DataFrame:
    """Load users.dat into a DataFrame."""
    users = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        header=None,
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode'],
        encoding='latin-1'
    )
    return users


def load_movies(filepath: str) -> pd.DataFrame:
    """Load movies.dat into a DataFrame."""
    movies = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        header=None,
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    return movies


def load_ratings(filepath: str, sort_by_time: bool = True) -> pd.DataFrame:
    """Load ratings.dat and optionally sort by timestamp."""
    ratings = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='latin-1'
    )

    if sort_by_time:
        ratings = ratings.sort_values('Timestamp').reset_index(drop=True)

    return ratings


def create_movie_id_mapping(movies_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build bidirectional MovieID/index mappings."""
    movie_ids = sorted(movies_df['MovieID'].unique())
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    movie_idx_to_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}

    return movie_id_to_idx, movie_idx_to_id


def create_user_id_mapping(users_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build bidirectional UserID/index mappings."""
    user_ids = sorted(users_df['UserID'].unique())
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    user_idx_to_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}

    return user_id_to_idx, user_idx_to_id
