"""Generalized advantage estimation with explicit episode boundaries."""

from collections.abc import Sequence

import numpy as np


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return advantages and lambda returns; terminal transitions never bootstrap."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    raw_dones = np.asarray(dones)
    if rewards.ndim != 1 or values.shape != rewards.shape or raw_dones.shape != rewards.shape:
        raise ValueError("rewards, values and dones must be equal-length vectors")
    if raw_dones.dtype != np.bool_ and raw_dones.size:
        raise ValueError("dones must contain booleans")
    dones = raw_dones.astype(bool)
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("rewards and values must be finite")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    if not len(rewards):
        return rewards.copy(), values.copy()
    if not dones[-1] and bootstrap_value is None:
        raise ValueError("An unfinished rollout requires bootstrap_value")
    bootstrap = 0.0 if dones[-1] else float(bootstrap_value)
    if not np.isfinite(bootstrap):
        raise ValueError("bootstrap_value must be finite")
    advantages = np.zeros_like(rewards)
    following_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuing = float(not dones[index])
        following_value = values[index + 1] if index + 1 < len(values) else bootstrap
        residual = rewards[index] + gamma * following_value * continuing - values[index]
        following_advantage = residual + gamma * gae_lambda * continuing * following_advantage
        advantages[index] = following_advantage
    return advantages, advantages + values
