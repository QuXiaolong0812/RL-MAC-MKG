"""On-policy PPO and supervised operation/object initialization."""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..controller.context import DecisionContext
from .advantages import compute_gae


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    batch_size: int = 128
    minibatch_size: int = 16
    epochs: int = 4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5

    def __post_init__(self):
        if not all(
            math.isfinite(value)
            for value in (
                self.learning_rate,
                self.gamma,
                self.gae_lambda,
                self.clip_epsilon,
                self.value_coefficient,
                self.entropy_coefficient,
                self.max_gradient_norm,
            )
        ):
            raise ValueError("PPO parameters must be finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Discounts must lie in [0, 1]")
        if not 0 < self.clip_epsilon < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid PPO clipping or learning rate")
        if min(self.batch_size, self.minibatch_size, self.epochs) < 1:
            raise ValueError("Batch and epoch counts must be positive")
        if min(self.value_coefficient, self.entropy_coefficient) < 0 or self.max_gradient_norm <= 0:
            raise ValueError("Loss coefficients must be nonnegative and gradient norm positive")


@dataclass(frozen=True)
class RolloutStep:
    context: DecisionContext
    operation_index: int
    target_index: int
    log_prob: float
    value: float
    reward: float
    done: bool
    next_state_hash: str
    context_digest: str
    terminal_reward_added: bool = False
    policy_revision: int = 0
    temperature: float = 0.1
    policy_identity: str = ""

    def validate(self):
        if self.context.digest() != self.context_digest:
            raise ValueError("Rollout snapshot or mask changed")
        if not 0 <= self.operation_index < len(self.context.mask):
            raise ValueError("Invalid stored operation")
        if (
            not 0 <= self.target_index < len(self.context.object_ids)
            or not self.context.mask[self.operation_index][self.target_index]
        ):
            raise ValueError("Stored action is masked")
        if not all(math.isfinite(value) for value in (self.log_prob, self.value, self.reward)):
            raise ValueError("Rollout scalars must be finite")
        if not self.next_state_hash:
            raise ValueError("Transition requires a next-state hash")


def finalize_episode(steps: list[RolloutStep], terminal_reward: float) -> list[RolloutStep]:
    """Add supervised terminal reward exactly once, without modifying collected records."""
    if not steps or not steps[-1].done or any(step.done for step in steps[:-1]):
        raise ValueError("Expected one complete episode")
    if any(step.terminal_reward_added for step in steps):
        raise ValueError("The terminal reward has already been added")
    if not math.isfinite(terminal_reward):
        raise ValueError("Terminal reward must be finite")
    return [
        *steps[:-1],
        replace(steps[-1], reward=steps[-1].reward + terminal_reward, terminal_reward_added=True),
    ]


def validate_rollout(steps: list[RolloutStep], expected_specialists: str | None = None):
    if not steps:
        raise ValueError("Rollout batch is empty")
    signature = expected_specialists or steps[0].context.specialist_signature
    for index, step in enumerate(steps):
        step.validate()
        if step.context.specialist_signature != signature:
            raise ValueError("Specialist versions changed within rollout")
        if index and not steps[index - 1].done:
            if steps[index - 1].next_state_hash != step.context.state_hash:
                raise ValueError("Inconsistent state snapshots in rollout")


class PPOTrainer:
    def __init__(self, model, config: PPOConfig, optimizer=None):
        self.model = model
        self.config = config
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Controller has no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=0.0
        )
        self.updates = model.policy_revision

    def update(
        self,
        steps: list[RolloutStep],
        *,
        bootstrap_value=None,
        expected_specialists: str | None = None,
    ) -> dict[str, float]:
        validate_rollout(steps, expected_specialists)
        if any(step.policy_identity != self.model.policy_identity for step in steps):
            raise ValueError("Rollout policy identity differs from the collecting controller")
        if self.updates != self.model.policy_revision or any(
            step.policy_revision != self.updates for step in steps
        ):
            raise ValueError("Rollout policy revision differs from the collecting controller")
        if any(step.temperature != self.model.temperature for step in steps):
            raise ValueError("Rollout decoding temperature differs from the current policy")
        config = self.config
        advantages, returns = compute_gae(
            [step.reward for step in steps],
            [step.value for step in steps],
            [step.done for step in steps],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            bootstrap_value=bootstrap_value,
        )
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        self.model.optimization_mode()
        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        count = 0
        for _ in range(config.epochs):
            order = np.random.permutation(len(steps))
            for start in range(0, len(order), config.minibatch_size):
                indices = order[start : start + config.minibatch_size]
                self.optimizer.zero_grad(set_to_none=True)
                for index in indices:
                    step = steps[index]
                    distribution, value = self.model.distribution(step.context)
                    log_probability = distribution.log_prob(step.operation_index, step.target_index)
                    log_ratio = log_probability - step.log_prob
                    ratio = log_ratio.exp()
                    advantage = torch.as_tensor(
                        advantages[index], dtype=value.dtype, device=value.device
                    )
                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantage,
                    )
                    target_return = torch.as_tensor(
                        returns[index], dtype=value.dtype, device=value.device
                    )
                    value_loss = (value - target_return).square()
                    entropy = distribution.entropy()
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy
                    )
                    if not torch.isfinite(loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite PPO loss")
                    (loss / len(indices)).backward()
                    measured = {
                        "loss": loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": entropy,
                        "approx_kl": (ratio - 1) - log_ratio,
                        "clip_fraction": (torch.abs(ratio - 1) > config.clip_epsilon).float(),
                    }
                    for name, scalar in measured.items():
                        metrics[name] += float(scalar.detach().item())
                    count += 1
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    config.max_gradient_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
        self.updates += 1
        self.model.policy_revision = self.updates
        self.model.checkpoint_identity = None
        return {
            **{name: value / count for name, value in metrics.items()},
            "transitions": float(len(steps)),
            "updates": float(self.updates),
        }


def behavioral_cloning_update(model, optimizer, demonstrations, *, minibatch_size=16):
    """Fit both factors of annotated feasible actions under their original contexts."""
    if not demonstrations or minibatch_size < 1:
        raise ValueError("Behavioral cloning requires demonstrations and a positive minibatch size")
    model.optimization_mode()
    total_loss = 0.0
    for start in range(0, len(demonstrations), minibatch_size):
        minibatch = demonstrations[start : start + minibatch_size]
        optimizer.zero_grad(set_to_none=True)
        for context, operation, target in minibatch:
            distribution, _ = model.distribution(context)
            loss = -distribution.log_prob(operation, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite behavioral cloning loss")
            (loss / len(minibatch)).backward()
            total_loss += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True
        )
        optimizer.step()
    model.policy_revision += 1
    model.checkpoint_identity = None
    return total_loss / len(demonstrations)
