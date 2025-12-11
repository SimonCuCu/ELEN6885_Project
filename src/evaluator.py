import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict


class OfflineEvaluator:
    """Replay-based offline evaluator."""

    def __init__(
        self,
        bandit_algorithm,
        user_features: Dict[int, np.ndarray],
        movie_features: Dict[int, np.ndarray],
        movie_id_to_idx: Dict[int, int],
        movie_idx_to_id: Dict[int, int],
        reward_threshold: float = 4.0
    ):
        if not (1 <= reward_threshold <= 5):
            raise ValueError(f"reward_threshold must be in [1, 5], got {reward_threshold}")

        self.bandit = bandit_algorithm
        self.user_features = user_features
        self.movie_features = movie_features
        self.movie_id_to_idx = movie_id_to_idx
        self.movie_idx_to_id = movie_idx_to_id
        self.reward_threshold = reward_threshold

        self.total_rounds = 0
        self.matched_rounds = 0
        self.cumulative_reward = 0.0
        self.reward_history = []
        self.ctr_history = []
        self.time_points = []

    def replay_evaluation(
        self,
        ratings_df: pd.DataFrame,
        max_rounds: Optional[int] = None,
        candidate_strategy: str = 'random_k',
        k_candidates: int = 20,
        verbose: bool = True,
        record_interval: int = 1000
    ) -> Dict[str, Any]:
        required_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        if not all(col in ratings_df.columns for col in required_columns):
            raise ValueError(f"ratings_df must contain columns: {required_columns}")

        if len(ratings_df) == 0:
            raise ValueError("ratings_df cannot be empty")

        n_rounds = len(ratings_df) if max_rounds is None else min(max_rounds, len(ratings_df))

        for i, (_, row) in enumerate(ratings_df.iloc[:n_rounds].iterrows()):
            if i >= n_rounds:
                break

            user_id = int(row['UserID'])
            actual_movie_id = int(row['MovieID'])
            actual_rating = float(row['Rating'])

            if user_id not in self.user_features:
                continue
            if actual_movie_id not in self.movie_features:
                continue
            if actual_movie_id not in self.movie_id_to_idx:
                continue

            user_vec = self.user_features[user_id]
            actual_movie_vec = self.movie_features[actual_movie_id]

            if candidate_strategy == 'all':
                candidate_movie_ids = list(self.movie_features.keys())
            elif candidate_strategy == 'random_k':
                candidate_movie_ids = self._generate_random_candidates(
                    actual_movie_id,
                    k_candidates
                )
            else:
                raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

            candidate_indices = [
                self.movie_id_to_idx[mid]
                for mid in candidate_movie_ids
                if mid in self.movie_id_to_idx
            ]

            if len(candidate_indices) == 0:
                continue

            is_hybrid = hasattr(self.bandit, 'shared_dim')

            if is_hybrid:
                shared_context = user_vec
                arm_contexts = {
                    idx: self.movie_features[self.movie_idx_to_id[idx]]
                    for idx in candidate_indices
                }
                selected_idx, _ = self.bandit.select_arm(
                    shared_context,
                    arm_contexts,
                    candidate_arms=candidate_indices
                )
            else:
                if isinstance(self.bandit, RandomBaseline):
                    context = self._create_context(user_vec, actual_movie_vec)
                    selected_idx, _ = self.bandit.select_arm(
                        context,
                        candidate_arms=candidate_indices
                    )
                else:
                    best_idx = None
                    best_ucb = -float("inf")

                    for idx in candidate_indices:
                        movie_id = self.movie_idx_to_id[idx]
                        movie_vec = self.movie_features.get(movie_id)
                        if movie_vec is None:
                            continue

                        context = self._create_context(user_vec, movie_vec)
                        _, ucb = self.bandit.select_arm(
                            context,
                            candidate_arms=[idx]
                        )

                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_idx = idx

                    if best_idx is None:
                        continue

                    selected_idx = best_idx

            recommended_movie_id = self.movie_idx_to_id[selected_idx]

            self.total_rounds += 1

            if recommended_movie_id == actual_movie_id:
                self.matched_rounds += 1
                reward = self._rating_to_reward(actual_rating)
                self.cumulative_reward += reward

                if is_hybrid:
                    shared_context = user_vec
                    arm_context = actual_movie_vec
                    self.bandit.update(selected_idx, shared_context, arm_context, reward)
                else:
                    context = self._create_context(user_vec, actual_movie_vec)
                    self.bandit.update(selected_idx, context, reward)

            if i % record_interval == 0 and i > 0:
                self._record_metrics(i)
                if verbose:
                    ctr = self.matched_rounds / self.total_rounds if self.total_rounds > 0 else 0
                    print(f"Round {i}/{n_rounds}: CTR={ctr:.4f}, "
                          f"Cumulative Reward={self.cumulative_reward:.2f}")

        self._record_metrics(n_rounds)

        return self.compute_metrics()

    def _create_context(
        self,
        user_features: np.ndarray,
        movie_features: np.ndarray
    ) -> np.ndarray:
        return np.concatenate([user_features, movie_features])

    def _generate_random_candidates(
        self,
        actual_movie: int,
        k: int
    ) -> List[int]:
        """Return random candidates that include the actual movie."""
        candidates = [actual_movie]

        all_movies = list(self.movie_features.keys())

        other_movies = [m for m in all_movies if m != actual_movie]

        if len(other_movies) > 0:
            n_to_sample = min(k - 1, len(other_movies))
            random_movies = np.random.choice(
                other_movies,
                size=n_to_sample,
                replace=False
            )
            candidates.extend(random_movies.tolist())

        return candidates

    def _rating_to_reward(self, rating: float) -> float:
        """Convert rating to binary reward."""
        return 1.0 if rating >= self.reward_threshold else 0.0

    def _record_metrics(self, time_point: int) -> None:
        """Record current metrics to history."""
        self.time_points.append(time_point)
        self.reward_history.append(self.cumulative_reward)
        ctr = self.matched_rounds / self.total_rounds if self.total_rounds > 0 else 0
        self.ctr_history.append(ctr)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final evaluation metrics."""
        ctr = self.matched_rounds / self.total_rounds if self.total_rounds > 0 else 0
        avg_reward = (
            self.cumulative_reward / self.matched_rounds
            if self.matched_rounds > 0 else 0
        )

        return {
            'CTR': ctr,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': avg_reward,
            'total_rounds': self.total_rounds,
            'matched_rounds': self.matched_rounds
        }

    def plot_learning_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Plot cumulative reward learning curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.reward_history, linewidth=2, color='#2E86AB')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)
        plt.title('Learning Curve: Cumulative Reward Over Time',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_ctr_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Plot CTR (Click-Through Rate) over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ctr_history, linewidth=2, color='#A23B72')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('CTR (Click-Through Rate)', fontsize=12)
        plt.title('CTR Evolution Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


class RandomBaseline:
    """Random recommendation baseline."""

    def __init__(self, n_arms: int, n_features: int):
        self.n_arms = n_arms
        self.n_features = n_features

    def select_arm(
        self,
        context: np.ndarray,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        if candidate_arms is None:
            candidate_arms = list(range(self.n_arms))

        selected_arm = np.random.choice(candidate_arms)
        return int(selected_arm), 0.0

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        pass


class PopularityBaseline:
    """Popularity-based recommendation baseline."""

    def __init__(self, n_arms: int, n_features: int):
        self.n_arms = n_arms
        self.n_features = n_features
        self.arm_counts = defaultdict(int)
        self.total_count = 0

    def select_arm(
        self,
        context: np.ndarray,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        if candidate_arms is None:
            candidate_arms = list(range(self.n_arms))

        best_arm = max(
            candidate_arms,
            key=lambda arm: self.arm_counts[arm]
        )

        popularity_score = self.arm_counts[best_arm] / max(self.total_count, 1)

        return int(best_arm), float(popularity_score)

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.arm_counts[arm] += 1
        self.total_count += 1
