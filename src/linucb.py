import numpy as np
from typing import List, Tuple, Optional


class LinUCB:
    """Disjoint LinUCB implementation."""

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0
    ):

        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        self.A = np.array([np.identity(n_features, dtype=np.float64) for _ in range(n_arms)])

        self.b = np.zeros((n_arms, n_features), dtype=np.float64)

    def select_arm(
        self,
        context: np.ndarray,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
       
        if context.shape != (self.n_features,):
            raise ValueError(
                f"Context must have shape ({self.n_features},), got {context.shape}"
            )

        if candidate_arms is None:
            candidate_arms = list(range(self.n_arms))

        if len(candidate_arms) == 0:
            raise ValueError("candidate_arms cannot be empty")

        for arm in candidate_arms:
            if not (0 <= arm < self.n_arms):
                raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        best_arm = None
        best_ucb = -float('inf')

        for arm in candidate_arms:
            theta = np.linalg.solve(self.A[arm], self.b[arm])

            p = np.dot(theta, context)

            A_inv_x = np.linalg.solve(self.A[arm], context)
            confidence = self.alpha * np.sqrt(np.dot(context, A_inv_x))

            ucb = p + confidence

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm, float(best_ucb)

    def update(
        self,
        arm: int,
        context: np.ndarray,
        reward: float
    ) -> None:
       
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        if context.shape != (self.n_features,):
            raise ValueError(
                f"Context must have shape ({self.n_features},), got {context.shape}"
            )

        if not np.isfinite(reward):
            raise ValueError(f"Reward must be a finite number, got {reward}")

        self.A[arm] += np.outer(context, context)

        self.b[arm] += reward * context

    def get_theta(self, arm: int) -> np.ndarray:
        
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        theta = np.linalg.solve(self.A[arm], self.b[arm])
        return theta

    def get_arm_count(self) -> np.ndarray:
        
        counts = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            counts[arm] = np.trace(self.A[arm]) - self.n_features
        return counts

    def reset(self) -> None:
        
        self.A = np.array(
            [np.identity(self.n_features, dtype=np.float64) for _ in range(self.n_arms)]
        )
        self.b = np.zeros((self.n_arms, self.n_features), dtype=np.float64)


class HybridLinUCB:
    """Hybrid LinUCB with shared and arm-specific features."""

    def __init__(
        self,
        n_arms: int,
        shared_dim: int,
        arm_dim: int,
        alpha: float = 1.0
    ):

        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}")
        if arm_dim <= 0:
            raise ValueError(f"arm_dim must be positive, got {arm_dim}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.n_arms = n_arms
        self.shared_dim = shared_dim
        self.arm_dim = arm_dim
        self.alpha = alpha

        self.A0 = np.identity(shared_dim, dtype=np.float64)
        self.b0 = np.zeros(shared_dim, dtype=np.float64)

        self.A = np.array(
            [np.identity(arm_dim, dtype=np.float64) for _ in range(n_arms)]
        )
        self.B = np.zeros((n_arms, arm_dim, shared_dim), dtype=np.float64)
        self.b = np.zeros((n_arms, arm_dim), dtype=np.float64)

    def select_arm(
        self,
        shared_context: np.ndarray,
        arm_contexts: dict,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
       
        if shared_context.shape != (self.shared_dim,):
            raise ValueError(
                f"shared_context must have shape ({self.shared_dim},), "
                f"got {shared_context.shape}"
            )

        if candidate_arms is None:
            candidate_arms = list(arm_contexts.keys())

        if len(candidate_arms) == 0:
            raise ValueError("candidate_arms cannot be empty")

        beta_hat = np.linalg.solve(self.A0, self.b0)

        A0_inv_z = np.linalg.solve(self.A0, shared_context)

        best_arm = None
        best_ucb = -float('inf')

        for arm in candidate_arms:
            arm_context = arm_contexts.get(arm)
            if arm_context is None:
                continue

            if not (0 <= arm < self.n_arms):
                raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

            if arm_context.shape != (self.arm_dim,):
                raise ValueError(
                    f"arm_context for arm {arm} must have shape ({self.arm_dim},), "
                    f"got {arm_context.shape}"
                )

            theta_hat = np.linalg.solve(
                self.A[arm],
                self.b[arm] - self.B[arm] @ beta_hat
            )

            pred_reward = np.dot(shared_context, beta_hat) + np.dot(arm_context, theta_hat)

            Aa_inv_xa = np.linalg.solve(self.A[arm], arm_context)

            term1 = np.dot(shared_context, A0_inv_z)

            term2 = -2.0 * np.dot(A0_inv_z, self.B[arm].T @ Aa_inv_xa)

            term3 = np.dot(arm_context, Aa_inv_xa)

            Bt_Aa_inv_xa = self.B[arm].T @ Aa_inv_xa
            A0_inv_Bt_Aa_inv_xa = np.linalg.solve(self.A0, Bt_Aa_inv_xa)
            term4 = np.dot(Bt_Aa_inv_xa, A0_inv_Bt_Aa_inv_xa)

            s_t_a = term1 + term2 + term3 + term4

            s_t_a = max(s_t_a, 0.0)

            confidence = self.alpha * np.sqrt(s_t_a)

            ucb = pred_reward + confidence

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        if best_arm is None:
            raise ValueError("No valid arm found in candidates")

        return best_arm, float(best_ucb)

    def update(
        self,
        arm: int,
        shared_context: np.ndarray,
        arm_context: np.ndarray,
        reward: float
    ) -> None:
       
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        if shared_context.shape != (self.shared_dim,):
            raise ValueError(
                f"shared_context must have shape ({self.shared_dim},), "
                f"got {shared_context.shape}"
            )

        if arm_context.shape != (self.arm_dim,):
            raise ValueError(
                f"arm_context must have shape ({self.arm_dim},), "
                f"got {arm_context.shape}"
            )

        if not np.isfinite(reward):
            raise ValueError(f"Reward must be a finite number, got {reward}")

        self.A[arm] += np.outer(arm_context, arm_context)

        self.B[arm] += np.outer(arm_context, shared_context)

        self.b[arm] += reward * arm_context

        self.A0 += np.outer(shared_context, shared_context)

        self.b0 += reward * shared_context

    def get_beta(self) -> np.ndarray:
        beta = np.linalg.solve(self.A0, self.b0)
        return beta

    def get_theta(self, arm: int) -> np.ndarray:
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        beta = self.get_beta()
        theta = np.linalg.solve(self.A[arm], self.b[arm] - self.B[arm] @ beta)
        return theta

    def reset(self) -> None:
        """Reset matrices and vectors to initial state."""
        self.A0 = np.identity(self.shared_dim, dtype=np.float64)
        self.b0 = np.zeros(self.shared_dim, dtype=np.float64)
        self.A = np.array(
            [np.identity(self.arm_dim, dtype=np.float64) for _ in range(self.n_arms)]
        )
        self.B = np.zeros((self.n_arms, self.arm_dim, self.shared_dim), dtype=np.float64)
        self.b = np.zeros((self.n_arms, self.arm_dim), dtype=np.float64)
