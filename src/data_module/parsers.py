"""Data parsers for MovieLens-1M dataset.

This module provides functions to parse the three main data files:
- ratings.dat: User ratings for movies
- users.dat: User demographic information
- movies.dat: Movie metadata
"""

import pandas as pd
from pathlib import Path
from typing import Union

from src.data_module.constants import (
    DATA_DELIMITER,
    RATINGS_COLUMNS,
    USERS_COLUMNS,
    MOVIES_COLUMNS,
    GENRE_SEPARATOR,
    MIN_RATING,
    MAX_RATING
)


def parse_ratings(file_path: Union[str, Path]) -> pd.DataFrame:
    """Parse ratings.dat file.

    Args:
        file_path: Path to ratings.dat file

    Returns:
        DataFrame with columns [UserID, MovieID, Rating, Timestamp]
        Rows with missing fields are filtered out.

    Example:
        >>> df = parse_ratings("datasets/ratings.dat")
        >>> print(df.head())
    """
    file_path = Path(file_path)

    # Read with explicit column names
    df = pd.read_csv(
        file_path,
        sep=DATA_DELIMITER,
        names=RATINGS_COLUMNS,
        engine='python',
        encoding='latin-1'
    )

    # Remove rows with any missing values
    df = df.dropna()

    # Convert to appropriate data types
    df["UserID"] = df["UserID"].astype(int)
    df["MovieID"] = df["MovieID"].astype(int)
    df["Rating"] = df["Rating"].astype(int)
    df["Timestamp"] = df["Timestamp"].astype(int)

    # Filter invalid ratings
    df = df[(df["Rating"] >= MIN_RATING) & (df["Rating"] <= MAX_RATING)]

    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=["UserID", "MovieID", "Timestamp"], keep='first')

    return df


def parse_users(file_path: Union[str, Path]) -> pd.DataFrame:
    """Parse users.dat file.

    Args:
        file_path: Path to users.dat file

    Returns:
        DataFrame with columns [UserID, Gender, Age, Occupation, Zipcode]

    Example:
        >>> df = parse_users("datasets/users.dat")
        >>> print(df.head())
    """
    file_path = Path(file_path)

    # Read with explicit column names
    df = pd.read_csv(
        file_path,
        sep=DATA_DELIMITER,
        names=USERS_COLUMNS,
        engine='python',
        encoding='latin-1'
    )

    # Remove rows with missing critical fields
    df = df.dropna(subset=["UserID", "Gender", "Age", "Occupation"])

    # Convert to appropriate data types
    df["UserID"] = df["UserID"].astype(int)
    df["Gender"] = df["Gender"].astype(str)
    df["Age"] = df["Age"].astype(int)
    df["Occupation"] = df["Occupation"].astype(int)
    df["Zipcode"] = df["Zipcode"].astype(str)

    return df


def parse_movies(file_path: Union[str, Path]) -> pd.DataFrame:
    """Parse movies.dat file, handle duplicates.

    Args:
        file_path: Path to movies.dat file

    Returns:
        DataFrame with columns [MovieID, Title, Genres]
        Genres is a list of strings.
        Duplicate MovieIDs are removed (keeping first occurrence).

    Example:
        >>> df = parse_movies("datasets/movies.dat")
        >>> print(df.head())
        >>> print(df.iloc[0]["Genres"])  # List of genre strings
    """
    file_path = Path(file_path)

    # Read with explicit column names
    df = pd.read_csv(
        file_path,
        sep=DATA_DELIMITER,
        names=MOVIES_COLUMNS,
        engine='python',
        encoding='latin-1'
    )

    # Remove rows with missing MovieID
    df = df.dropna(subset=["MovieID"])

    # Convert MovieID to int
    df["MovieID"] = df["MovieID"].astype(int)

    # Split genres into list
    df["Genres"] = df["Genres"].apply(
        lambda x: x.split(GENRE_SEPARATOR) if pd.notna(x) else []
    )

    # Remove duplicate MovieIDs (keep first)
    df = df.drop_duplicates(subset=["MovieID"], keep='first')

    return df
