"""Preprocess MovieLens-1M data into trajectories.

This script:
1. Parses raw .dat files (ratings, users, movies)
2. Cleans and preprocesses data
3. Builds user trajectories
4. Performs temporal train/val/test split
5. Builds zipcode vocabulary
6. Saves processed data for training

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --data_dir datasets --output_dir data/processed
    python scripts/preprocess_data.py --min_interactions 20 --train_ratio 0.7

Example:
    python scripts/preprocess_data.py --output_dir data/ml-1m-processed
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from src.data_module.parsers import parse_ratings, parse_users, parse_movies
from src.data_module.cleaners import clean_ratings, clean_users
from src.data_module.trajectory import (
    Trajectory,
    build_trajectories,
    temporal_split_trajectories
)
from src.data_module.static_encoder import build_zipcode_vocabulary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens-1M data into trajectories"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="Directory containing raw .dat files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=20,
        help="Minimum interactions per user"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation"
    )

    return parser.parse_args()


def print_stats(
    name: str,
    trajectories: List[Trajectory]
) -> Dict[str, float]:
    """Print and return statistics for a trajectory set.

    Args:
        name: Name of the dataset split
        trajectories: List of trajectories

    Returns:
        Dictionary with statistics
    """
    if not trajectories:
        print(f"{name}: 0 trajectories")
        return {"count": 0, "avg_length": 0, "total_interactions": 0}

    lengths = [len(t.movie_ids) for t in trajectories]

    stats = {
        "count": int(len(trajectories)),
        "avg_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "median_length": float(np.median(lengths)),
        "total_interactions": int(sum(lengths))
    }

    print(f"\n{name}:")
    print(f"  Trajectories: {stats['count']}")
    print(f"  Avg length: {stats['avg_length']:.2f}")
    print(f"  Min/Max length: {stats['min_length']} / {stats['max_length']}")
    print(f"  Median length: {stats['median_length']:.2f}")
    print(f"  Total interactions: {stats['total_interactions']}")

    return stats


def main():
    """Main preprocessing function."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MovieLens-1M Data Preprocessing")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Min interactions: {args.min_interactions}")
    print(f"Train/Val/Test ratio: {args.train_ratio:.0%}/{args.val_ratio:.0%}/{1-args.train_ratio-args.val_ratio:.0%}")
    print("=" * 70)

    # Step 1: Parse raw data files
    print("\n[1/6] Parsing raw data files...")

    ratings_path = data_dir / "ratings.dat"
    users_path = data_dir / "users.dat"
    movies_path = data_dir / "movies.dat"

    ratings_df = parse_ratings(ratings_path)
    users_df = parse_users(users_path)
    movies_df = parse_movies(movies_path)

    print(f"  Ratings: {len(ratings_df):,} records")
    print(f"  Users: {len(users_df):,} records")
    print(f"  Movies: {len(movies_df):,} records")

    # Step 2: Clean data
    print("\n[2/6] Cleaning data...")

    # Get valid movie IDs
    valid_movie_ids = set(movies_df["MovieID"])
    print(f"  Valid movie IDs: {len(valid_movie_ids):,}")

    # Clean ratings (remove invalid movie IDs)
    ratings_cleaned = clean_ratings(ratings_df, valid_movie_ids)
    print(f"  Cleaned ratings: {len(ratings_cleaned):,} ({len(ratings_cleaned)/len(ratings_df)*100:.1f}% retained)")

    # Clean users (add zipcode prefix)
    users_cleaned = clean_users(users_df)

    # Analyze zipcode distribution
    zipcode_counts = users_cleaned["zipcode_prefix"].value_counts()
    print(f"  Unique zipcode prefixes: {len(zipcode_counts)}")
    print(f"  Top 5 zipcodes: {list(zipcode_counts.head().index)}")

    # Step 3: Build trajectories
    print("\n[3/6] Building user trajectories...")

    trajectories = build_trajectories(
        ratings_cleaned,
        users_cleaned,
        min_interactions=args.min_interactions
    )

    print(f"  Total trajectories: {len(trajectories)}")
    print_stats("All trajectories", trajectories)

    # Step 4: Temporal split
    print("\n[4/6] Performing temporal split...")

    train_trajs, val_trajs, test_trajs = temporal_split_trajectories(
        trajectories,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    train_stats = print_stats("Train", train_trajs)
    val_stats = print_stats("Validation", val_trajs)
    test_stats = print_stats("Test", test_trajs)

    # Step 5: Build zipcode vocabulary
    print("\n[5/6] Building zipcode vocabulary...")

    zipcode_vocab = build_zipcode_vocabulary(users_cleaned)
    print(f"  Vocabulary size: {len(zipcode_vocab)}")

    # Step 6: Save processed data
    print("\n[6/6] Saving processed data...")

    # Save trajectories
    with open(output_dir / "train_trajectories.pkl", "wb") as f:
        pickle.dump(train_trajs, f)
    print(f"  Saved: train_trajectories.pkl")

    with open(output_dir / "val_trajectories.pkl", "wb") as f:
        pickle.dump(val_trajs, f)
    print(f"  Saved: val_trajectories.pkl")

    with open(output_dir / "test_trajectories.pkl", "wb") as f:
        pickle.dump(test_trajs, f)
    print(f"  Saved: test_trajectories.pkl")

    # Save zipcode vocabulary
    with open(output_dir / "zipcode_vocab.json", "w") as f:
        json.dump(zipcode_vocab, f, indent=2)
    print(f"  Saved: zipcode_vocab.json")

    # Save movie ID mapping (for reference)
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(sorted(valid_movie_ids), start=1)}
    with open(output_dir / "movie_id_mapping.json", "w") as f:
        json.dump(movie_id_to_idx, f)
    print(f"  Saved: movie_id_mapping.json")

    # Save metadata
    metadata = {
        "num_users": int(len(trajectories)),
        "num_movies": int(len(valid_movie_ids)),
        "num_zipcode_buckets": int(len(zipcode_vocab)),
        "max_movie_id": int(max(valid_movie_ids)),
        "min_interactions": int(args.min_interactions),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "train": train_stats,
        "val": val_stats,
        "test": test_stats
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")

    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - train_trajectories.pkl ({train_stats['count']} users)")
    print(f"  - val_trajectories.pkl ({val_stats['count']} users)")
    print(f"  - test_trajectories.pkl ({test_stats['count']} users)")
    print(f"  - zipcode_vocab.json ({len(zipcode_vocab)} buckets)")
    print(f"  - movie_id_mapping.json ({len(valid_movie_ids)} movies)")
    print(f"  - metadata.json")

    print("\nNext steps:")
    print("  1. Train BC policy: python scripts/train_bc.py --data_dir data/processed")
    print("  2. Train IQL: python scripts/train.py data.processed_dir=data/processed")


if __name__ == "__main__":
    main()

