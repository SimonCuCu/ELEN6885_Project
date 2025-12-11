"""Constants for data processing.

This module defines all constants used in data parsing and preprocessing.
"""

from typing import List, Tuple

# Special tokens
UNK_TOKEN: str = "<UNK>"
PAD_TOKEN: str = "<PAD>"

# User demographic constants
GENDER_VALUES: List[str] = ["M", "F"]
AGE_CATEGORIES: List[int] = [1, 18, 25, 35, 45, 50, 56]
OCCUPATION_RANGE: Tuple[int, int] = (0, 20)

# Zipcode processing
ZIPCODE_PREFIX_LEN: int = 3
MAX_ZIPCODE_BUCKETS: int = 50

# Rating constants
MIN_RATING: int = 1
MAX_RATING: int = 5
RATING_SCALE: int = 5

# Data file delimiters
DATA_DELIMITER: str = "::"

# Column names
RATINGS_COLUMNS: List[str] = ["UserID", "MovieID", "Rating", "Timestamp"]
USERS_COLUMNS: List[str] = ["UserID", "Gender", "Age", "Occupation", "Zipcode"]
MOVIES_COLUMNS: List[str] = ["MovieID", "Title", "Genres"]

# Genre separator
GENRE_SEPARATOR: str = "|"

# Minimum interactions per user
MIN_INTERACTIONS_PER_USER: int = 20
