import numpy as np
import pytest

from rl_mac_mkg.training.advantages import compute_gae


def test_gae_has_zero_terminal_bootstrap():
    advantages, returns = compute_gae(
        [1.0, 2.0], [0.5, 0.25], [False, True], gamma=1.0, gae_lambda=1.0
    )
    np.testing.assert_allclose(advantages, [2.5, 1.75])
    np.testing.assert_allclose(returns, [3.0, 2.0])


def test_gae_does_not_cross_episode_boundary():
    advantages, returns = compute_gae([1, 100], [0, 0], [True, True], gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(advantages, [1, 100])
    np.testing.assert_allclose(returns, [1, 100])


def test_truncated_rollout_requires_explicit_bootstrap():
    with pytest.raises(ValueError, match="bootstrap"):
        compute_gae([1], [0], [False])
    _, returns = compute_gae([1], [0], [False], bootstrap_value=2, gamma=0.5)
    np.testing.assert_allclose(returns, [2])


@pytest.mark.parametrize("rewards,values,dones", [([1], [], [True]), ([np.nan], [0], [True])])
def test_gae_rejects_invalid_arrays(rewards, values, dones):
    with pytest.raises(ValueError):
        compute_gae(rewards, values, dones)
