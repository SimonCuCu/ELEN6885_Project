"""Data cleaning utilities for MovieLens-1M dataset.

This module provides functions to clean and preprocess data:
- Remove invalid ratings
- Extract zipcode prefixes
- Handle non-standard zipcodes
"""

import pandas as pd
import re
from typing import Set

from src.data_module.constants import UNK_TOKEN, ZIPCODE_PREFIX_LEN


def clean_ratings(
    ratings_df: pd.DataFrame,
    valid_movie_ids: Set[int]
) -> pd.DataFrame:
    """Remove invalid ratings from dataset.

    Filters out:
    - Ratings for movies not in valid_movie_ids
    - Duplicate ratings (same UserID, MovieID, Timestamp)

    Args:
        ratings_df: DataFrame with ratings data
        valid_movie_ids: Set of valid MovieIDs from movies.dat

    Returns:
        Cleaned DataFrame with only valid ratings

    Example:
        >>> movies_df = parse_movies("datasets/movies.dat")
        >>> valid_ids = set(movies_df["MovieID"])
        >>> cleaned = clean_ratings(ratings_df, valid_ids)
    """
    # Filter out ratings for invalid movie IDs
    df = ratings_df[ratings_df["MovieID"].isin(valid_movie_ids)].copy()

    # Remove duplicate ratings (keep first occurrence)
    df = df.drop_duplicates(subset=["UserID", "MovieID", "Timestamp"], keep='first')

    return df


def extract_zipcode_prefix(
    zipcode: str,
    prefix_len: int = ZIPCODE_PREFIX_LEN
) -> str:
    """Extract numeric prefix from zipcode.

    Handles various zipcode formats:
    - US zipcodes: "48067" -> "480"
    - Canadian zipcodes: "A1B2C3" -> UNK_TOKEN
    - Zipcodes with dashes: "48067-1234" -> "480"
    - Non-numeric or mixed alphanumeric: -> UNK_TOKEN

    Args:
        zipcode: Zipcode string
        prefix_len: Number of digits to extract (default: 3)

    Returns:
        Numeric prefix or UNK_TOKEN for invalid zipcodes

    Example:
        >>> extract_zipcode_prefix("48067")
        '480'
        >>> extract_zipcode_prefix("A1B2C3")
        '<UNK>'
    """
    if pd.isna(zipcode) or zipcode == "":
        return UNK_TOKEN

    # Convert to string and strip whitespace
    zipcode = str(zipcode).strip()

    # Check if starts with non-digit (e.g., Canadian zipcode)
    if not zipcode or not zipcode[0].isdigit():
        return UNK_TOKEN

    # Extract leading digits (before any non-digit character)
    match = re.match(r'^(\d+)', zipcode)
    if not match:
        return UNK_TOKEN

    leading_digits = match.group(1)

    # Check if there are letters mixed within the first prefix_len characters
    # If the original zipcode has letters within the prefix range, return UNK
    prefix_substring = zipcode[:min(len(zipcode), prefix_len)]
    if re.search(r'[a-zA-Z]', prefix_substring):
        return UNK_TOKEN

    # Return first N digits (or less if zipcode is shorter)
    return leading_digits[:prefix_len]


def clean_users(users_df: pd.DataFrame) -> pd.DataFrame:
    """Clean user data and add zipcode prefix.

    Adds a new column 'zipcode_prefix' containing the extracted
    zipcode prefix (first 3 digits) or UNK_TOKEN for invalid zipcodes.

    Args:
        users_df: DataFrame with user data

    Returns:
        DataFrame with added 'zipcode_prefix' column

    Example:
        >>> users_df = parse_users("datasets/users.dat")
        >>> cleaned = clean_users(users_df)
        >>> print(cleaned[["Zipcode", "zipcode_prefix"]].head())
    """
    df = users_df.copy()

    # Extract zipcode prefix
    df["zipcode_prefix"] = df["Zipcode"].apply(extract_zipcode_prefix)

    return df
