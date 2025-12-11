"""Generate replay-based CTR/Reward and alpha-sensitivity plots for UCB."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
DATA_DIR = BASE_DIR / "datasets"

from src.data_loader import (  
    create_movie_id_mapping,
    load_movies,
    load_ratings,
    load_users,
)
from src.feature_engineering import encode_movie_features, encode_user_features  
from src.evaluator import OfflineEvaluator, RandomBaseline  
from src.linucb import HybridLinUCB, LinUCB  


SEED = 42
MAX_ROUNDS = 20000
K_CANDIDATES = 20
RECORD_INTERVAL = 1000
ALPHAS = [0.5, 1.0, 2.0, 3.0]
TIME_STEPS = [5000, 10000, 15000]


def build_data():
    users = load_users(str(DATA_DIR / "users.dat"))
    movies = load_movies(str(DATA_DIR / "movies.dat"))
    ratings = load_ratings(str(DATA_DIR / "ratings.dat"), sort_by_time=True)
    user_features = encode_user_features(users)
    movie_features = encode_movie_features(movies)
    movie_id_to_idx, movie_idx_to_id = create_movie_id_mapping(movies)
    return ratings, user_features, movie_features, movie_id_to_idx, movie_idx_to_id


def run_replay(bandit, ratings, user_features, movie_features, movie_id_to_idx, movie_idx_to_id):
    evaluator = OfflineEvaluator(
        bandit,
        user_features=user_features,
        movie_features=movie_features,
        movie_id_to_idx=movie_id_to_idx,
        movie_idx_to_id=movie_idx_to_id,
        reward_threshold=4.0,
    )
    metrics = evaluator.replay_evaluation(
        ratings_df=ratings,
        max_rounds=MAX_ROUNDS,
        candidate_strategy="random_k",
        k_candidates=K_CANDIDATES,
        verbose=False,
        record_interval=RECORD_INTERVAL,
    )
    return metrics, evaluator


def alpha_sweep(bandit_cls, shared_kwargs, ratings, user_features, movie_features, movie_id_to_idx, movie_idx_to_id):
    ctrs = []
    rewards = []
    for alpha in ALPHAS:
        bandit = bandit_cls(alpha=alpha, **shared_kwargs)
        metrics, _ = run_replay(bandit, ratings, user_features, movie_features, movie_id_to_idx, movie_idx_to_id)
        ctrs.append(metrics["CTR"])
        rewards.append(metrics["cumulative_reward"])
        print(f"{bandit_cls.__name__} alpha={alpha}: CTR={metrics['CTR']*100:.2f}%, Reward={metrics['cumulative_reward']:.1f}")
    return ctrs, rewards


def plot_results(curves, alpha_ctrs, alpha_rewards):
    colors = {
        "Random": "#F18F01",
        "Disjoint LinUCB": "#2E86AB",
        "Hybrid LinUCB": "#A23B72",
    }
    linestyles = {
        "Random": "--",
        "Disjoint LinUCB": "-",
        "Hybrid LinUCB": "-.",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Cumulative reward curve
    for name, data in curves.items():
        axes[0, 0].plot(
            data["time_points"],
            data["reward_history"],
            label=name,
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2.3 if name != "Random" else 2.0,
        )
    axes[0, 0].set_xlabel("Time Steps")
    axes[0, 0].set_ylabel("Cumulative Reward")
    axes[0, 0].set_title("Cumulative Reward (Replay, 20K rounds)")
    axes[0, 0].grid(True, alpha=0.3, linestyle="--")
    axes[0, 0].legend(loc="best")

    # CTR curve
    for name, data in curves.items():
        ctr_percent = [ctr * 100 for ctr in data["ctr_history"]]
        axes[0, 1].plot(
            data["time_points"],
            ctr_percent,
            label=name,
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2.3 if name != "Random" else 2.0,
        )
    axes[0, 1].set_xlabel("Time Steps")
    axes[0, 1].set_ylabel("CTR (%)")
    axes[0, 1].set_title("CTR Over Time (Replay, 20K rounds)")
    axes[0, 1].grid(True, alpha=0.3, linestyle="--")
    axes[0, 1].legend(loc="best")

    # Alpha vs CTR
    axes[1, 0].plot(ALPHAS, alpha_ctrs["Disjoint"], marker="o", color=colors["Disjoint LinUCB"], label="Disjoint LinUCB")
    axes[1, 0].plot(ALPHAS, alpha_ctrs["Hybrid"], marker="s", color=colors["Hybrid LinUCB"], label="Hybrid LinUCB")
    axes[1, 0].set_xlabel("Alpha")
    axes[1, 0].set_ylabel("Final CTR")
    axes[1, 0].set_title("Alpha Sensitivity: CTR")
    axes[1, 0].grid(True, alpha=0.3, linestyle="--")
    axes[1, 0].legend(loc="best")

    # Alpha vs Reward
    axes[1, 1].plot(ALPHAS, alpha_rewards["Disjoint"], marker="o", color=colors["Disjoint LinUCB"], label="Disjoint LinUCB")
    axes[1, 1].plot(ALPHAS, alpha_rewards["Hybrid"], marker="s", color=colors["Hybrid LinUCB"], label="Hybrid LinUCB")
    axes[1, 1].set_xlabel("Alpha")
    axes[1, 1].set_ylabel("Final Cumulative Reward")
    axes[1, 1].set_title("Alpha Sensitivity: Cumulative Reward")
    axes[1, 1].grid(True, alpha=0.3, linestyle="--")
    axes[1, 1].legend(loc="best")

    plt.suptitle("Replay Evaluation: UCB Variants", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = BASE_DIR / "UCB_replay.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to: {output_path}")


def _nearest(values: List[int], target: int) -> int:
    return min(values, key=lambda x: abs(x - target))


def plot_replay_bars(curves):
    colors = {
        "Random": "#F18F01",
        "Disjoint LinUCB": "#2E86AB",
        "Hybrid LinUCB": "#A23B72",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    time_labels = []
    reward_data = {"Random": [], "Disjoint LinUCB": [], "Hybrid LinUCB": []}
    ctr_data = {"Random": [], "Disjoint LinUCB": [], "Hybrid LinUCB": []}

    for t in TIME_STEPS:
        time_labels.append(str(t))
        for name, data in curves.items():
            tp = _nearest(data["time_points"], t)
            idx = data["time_points"].index(tp)
            reward_data[name].append(data["reward_history"][idx])
            ctr_data[name].append(data["ctr_history"][idx] * 100.0)

    x = np.arange(len(TIME_STEPS))
    width = 0.22

    for i, name in enumerate(["Random", "Disjoint LinUCB", "Hybrid LinUCB"]):
        axes[0].bar(
            x + (i - 1) * width,
            reward_data[name],
            width,
            label=name,
            color=colors[name],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(time_labels)
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Cumulative Reward (selected steps)")
    axes[0].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].legend(loc="upper left")

    for i, name in enumerate(["Random", "Disjoint LinUCB", "Hybrid LinUCB"]):
        axes[1].bar(
            x + (i - 1) * width,
            ctr_data[name],
            width,
            label=name,
            color=colors[name],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(time_labels)
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("CTR (%)")
    axes[1].set_title("CTR (selected steps)")
    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[1].legend(loc="upper left")

    plt.suptitle("Replay Evaluation (Bar): UCB Variants", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = BASE_DIR / "UCB_replay1.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to: {output_path}")


def plot_alpha_lines(alpha_ctrs, alpha_rewards):
    colors = {
        "Disjoint LinUCB": "#2E86AB",
        "Hybrid LinUCB": "#A23B72",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ALPHAS, alpha_ctrs["Disjoint"], marker="o", color=colors["Disjoint LinUCB"], label="Disjoint LinUCB")
    axes[0].plot(ALPHAS, alpha_ctrs["Hybrid"], marker="s", color=colors["Hybrid LinUCB"], label="Hybrid LinUCB")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Final CTR")
    axes[0].set_title("Alpha Sensitivity: CTR")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(loc="best")

    axes[1].plot(ALPHAS, alpha_rewards["Disjoint"], marker="o", color=colors["Disjoint LinUCB"], label="Disjoint LinUCB")
    axes[1].plot(ALPHAS, alpha_rewards["Hybrid"], marker="s", color=colors["Hybrid LinUCB"], label="Hybrid LinUCB")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Final Cumulative Reward")
    axes[1].set_title("Alpha Sensitivity: Cumulative Reward")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(loc="best")

    plt.suptitle("Alpha Sensitivity (UCB Variants)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = BASE_DIR / "UCB_replay2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    np.random.seed(SEED)
    print("=" * 80)
    print("UCB Replay Curves & Alpha Sensitivity")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    ratings, user_features, movie_features, movie_id_to_idx, movie_idx_to_id = build_data()

    # Curves with alpha=1.0
    bandits_for_curves = {
        "Random": RandomBaseline(n_arms=len(movie_features), n_features=48),
        "Disjoint LinUCB": LinUCB(n_arms=len(movie_features), n_features=48, alpha=1.0),
        "Hybrid LinUCB": HybridLinUCB(n_arms=len(movie_features), shared_dim=30, arm_dim=18, alpha=1.0),
    }

    curves = {}
    for name, bandit in bandits_for_curves.items():
        metrics, evaluator = run_replay(bandit, ratings, user_features, movie_features, movie_id_to_idx, movie_idx_to_id)
        curves[name] = {
            "reward_history": evaluator.reward_history,
            "ctr_history": evaluator.ctr_history,
            "time_points": evaluator.time_points,
            "metrics": metrics,
        }
        print(f"{name}: CTR={metrics['CTR']*100:.2f}%, Reward={metrics['cumulative_reward']:.1f}")

    # Alpha sweeps
    disjoint_ctrs, disjoint_rewards = alpha_sweep(
        LinUCB,
        {"n_arms": len(movie_features), "n_features": 48},
        ratings,
        user_features,
        movie_features,
        movie_id_to_idx,
        movie_idx_to_id,
    )
    hybrid_ctrs, hybrid_rewards = alpha_sweep(
        HybridLinUCB,
        {"n_arms": len(movie_features), "shared_dim": 30, "arm_dim": 18},
        ratings,
        user_features,
        movie_features,
        movie_id_to_idx,
        movie_idx_to_id,
    )

    alpha_ctrs = {"Disjoint": disjoint_ctrs, "Hybrid": hybrid_ctrs}
    alpha_rewards = {"Disjoint": disjoint_rewards, "Hybrid": hybrid_rewards}

    plot_results(curves, alpha_ctrs, alpha_rewards)
    plot_replay_bars(curves)
    plot_alpha_lines(alpha_ctrs, alpha_rewards)
