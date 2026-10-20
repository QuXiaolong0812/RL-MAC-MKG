from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rl_mac_mkg.controller.context import DecisionContext, build_context
from rl_mac_mkg.controller.policy import FactorizedPolicy, NeuralController
from rl_mac_mkg.training.ppo import PPOConfig, PPOTrainer, RolloutStep, finalize_episode


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 100 for char in text]


class TestEncoder(nn.Module):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 8)
        self.embedding.requires_grad_(False)
        self.adapter = nn.Linear(8, 8, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        base = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=base + self.adapter(base))


def context():
    return DecisionContext(
        input_ids=(1, 2, 3, 4),
        object_ids=(None, "candidate:1", "candidate:2"),
        object_positions=(0, 2, 3),
        mask=((False, True, True), (True, False, False)),
        operation_names=("verify", "stop"),
        state_hash="state-1",
        evidence_hash="evidence-1",
        specialist_signature="specialists-1",
    )


def test_factorization_and_exact_joint_entropy():
    operation_logits = torch.tensor([0.4, -0.2], requires_grad=True)
    pointer_logits = torch.tensor([[99.0, 0.2, 0.8], [2.0, 99.0, 99.0]], requires_grad=True)
    policy = FactorizedPolicy(operation_logits, pointer_logits, context().mask, temperature=0.7)
    assert policy.joint_probabilities().sum().item() == pytest.approx(1)
    assert policy.joint_probabilities()[0, 0].item() == 0
    expected = policy.operation.log_prob(torch.tensor(0)) + policy.pointer(0).log_prob(
        torch.tensor(2)
    )
    assert policy.log_prob(0, 2).item() == pytest.approx(expected.item())
    probs = policy.joint_probabilities()
    positive = probs[probs > 0]
    assert policy.entropy().item() == pytest.approx((-(positive * positive.log()).sum()).item())
    assert policy.pointer(1).entropy().item() == 0
    policy.entropy().backward()
    assert torch.isfinite(pointer_logits.grad).all()


def test_incompatible_or_all_masked_actions_are_rejected():
    policy = FactorizedPolicy(torch.zeros(2), torch.zeros(2, 3), context().mask)
    with pytest.raises(ValueError, match="masked"):
        policy.log_prob(0, 0)
    with pytest.raises(ValueError):
        FactorizedPolicy(torch.zeros(1), torch.zeros(1, 1), ((False,),))


def test_context_masks_targets_removed_by_token_limit():
    result = build_context(
        "budget=30",
        [(f"id:{i}", "x" * 250) for i in range(10)],
        [("verify", f"id:{i}") for i in range(10)] + [("stop", None)],
        CharacterTokenizer(),
        operation_names=("verify", "stop"),
        max_tokens=180,
        state_hash="s",
        evidence_hash="e",
        specialist_signature="v",
    )
    assert len(result.input_ids) <= 180
    assert len(result.object_ids) < 11
    assert result.mask[1][0]
    for index, object_id in enumerate(result.object_ids):
        if index:
            assert result.mask[0][index]
            assert object_id is not None
    assert all(0 <= position < len(result.input_ids) for position in result.object_positions)


def test_ppo_changes_heads_and_adapter_but_not_frozen_encoder():
    torch.manual_seed(8)
    model = NeuralController(TestEncoder(), hidden_size=8, n_operations=2, head_size=12)
    old_embedding = model.encoder.embedding.weight.detach().clone()
    old_adapter = model.encoder.adapter.weight.detach().clone()
    old_head = model.operation_head[0].weight.detach().clone()
    steps = []
    for index in range(4):
        obs = replace(context(), state_hash=f"s{index}")
        with torch.no_grad():
            dist, value = model.distribution(obs)
            log_prob = dist.log_prob(0, 1)
        steps.append(
            RolloutStep(
                context=obs,
                operation_index=0,
                target_index=1,
                log_prob=log_prob.item(),
                value=value.item(),
                reward=float(index % 2),
                done=index == 3,
                next_state_hash=f"s{index + 1}",
                context_digest=obs.digest(),
                policy_identity=model.policy_identity,
            )
        )
    trainer = PPOTrainer(model, PPOConfig(learning_rate=0.01, epochs=2, minibatch_size=2))
    report = trainer.update(steps)
    assert np.isfinite(report["loss"])
    assert torch.equal(old_embedding, model.encoder.embedding.weight)
    assert not torch.equal(old_adapter, model.encoder.adapter.weight)
    assert not torch.equal(old_head, model.operation_head[0].weight)


def test_stale_rollout_and_repeated_terminal_reward_are_rejected():
    model = NeuralController(TestEncoder(), hidden_size=8, n_operations=2, head_size=12)
    obs = context()
    step = RolloutStep(obs, 0, 1, -1, 0, 0, True, "s2", obs.digest())
    finalized = finalize_episode([step], 0.6)
    assert finalized[-1].reward == pytest.approx(0.6)
    with pytest.raises(ValueError, match="terminal reward"):
        finalize_episode(finalized, 0.6)
    tampered = replace(step, context=replace(obs, input_ids=(1, 2, 8, 9)))
    with pytest.raises(ValueError, match="snapshot"):
        PPOTrainer(model, PPOConfig()).update([tampered])


def test_ppo_rejects_reuse_and_changed_decoding_temperature():
    model = NeuralController(TestEncoder(), hidden_size=8, n_operations=2, head_size=12)
    obs = context()
    with torch.no_grad():
        policy, value = model.distribution(obs)
        step = RolloutStep(
            obs,
            0,
            1,
            policy.log_prob(0, 1).item(),
            value.item(),
            1,
            True,
            "s2",
            obs.digest(),
            policy_identity=model.policy_identity,
        )
    trainer = PPOTrainer(model, PPOConfig(epochs=1))
    trainer.update([step])
    with pytest.raises(ValueError, match="policy revision"):
        trainer.update([step])
    with pytest.raises(ValueError, match="policy revision"):
        PPOTrainer(model, PPOConfig(epochs=1)).update([step])
    model.temperature = 0.9
    with pytest.raises(ValueError, match="temperature"):
        PPOTrainer(model, PPOConfig()).update(
            [replace(step, policy_revision=model.policy_revision)]
        )


def test_small_qwen_lora_checkpoint_roundtrip(tmp_path):
    from transformers import Qwen3Config, Qwen3Model

    from rl_mac_mkg.controller.checkpoint import load_controller, save_controller
    from rl_mac_mkg.controller.policy import attach_lora

    torch.manual_seed(4)
    base = tmp_path / "base"
    config = Qwen3Config(
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
    )
    Qwen3Model(config).save_pretrained(base)
    encoder = attach_lora(Qwen3Model.from_pretrained(base), rank=16)
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = NeuralController(encoder, hidden_size=16, n_operations=2, head_size=8)
    layer = encoder.get_base_model().layers[0]
    original_checkpoint = layer._gradient_checkpointing_func
    checkpoint_calls = []

    def count_checkpoints(*args, **kwargs):
        checkpoint_calls.append(1)
        return original_checkpoint(*args, **kwargs)

    layer._gradient_checkpointing_func = count_checkpoints
    model.optimization_mode()
    _, checkpoint_value = model.distribution(context())
    checkpoint_value.backward()
    assert checkpoint_calls
    model.eval()
    before, value_before = model.distribution(context())
    save_controller(
        model,
        tmp_path / "checkpoint",
        base_path=base,
        metadata={"operation_names": ["verify", "stop"]},
    )
    restored, _ = load_controller(tmp_path / "checkpoint", device="cpu")
    restored.eval()
    after, value_after = restored.distribution(context())
    torch.testing.assert_close(before.joint_probabilities(), after.joint_probabilities())
    torch.testing.assert_close(value_before, value_after)
    assert all(
        "lora_" in name for name, p in restored.encoder.named_parameters() if p.requires_grad
    )
    different_base = tmp_path / "different-base"
    Qwen3Model(config).save_pretrained(different_base)
    with pytest.raises(ValueError, match="base model"):
        load_controller(tmp_path / "checkpoint", base_path=different_base)


def test_optimizer_restore_checks_integrity_and_checkpoint_pairing(tmp_path):
    from transformers import Qwen3Config, Qwen3Model

    from rl_mac_mkg.controller.checkpoint import load_controller, restore_trainer, save_controller
    from rl_mac_mkg.controller.policy import attach_lora

    base = tmp_path / "base"
    Qwen3Model(
        Qwen3Config(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).save_pretrained(base)
    model = NeuralController(attach_lora(Qwen3Model.from_pretrained(base)), 16, 2, 8)
    trainer = PPOTrainer(model, PPOConfig(epochs=1))
    save_controller(model, tmp_path / "one", base_path=base, trainer=trainer)
    with torch.no_grad():
        model.value_head[0].weight.add_(1)
    save_controller(model, tmp_path / "two", base_path=base, trainer=trainer)
    loaded, _ = load_controller(tmp_path / "two")
    with pytest.raises(ValueError, match="belong"):
        restore_trainer(PPOTrainer(loaded, trainer.config), tmp_path / "one")
    correctly_loaded, _ = load_controller(tmp_path / "one")
    restored_trainer = PPOTrainer(correctly_loaded, trainer.config)
    restore_trainer(restored_trainer, tmp_path / "one")
    training_file = tmp_path / "one" / "training.pt"
    saved = torch.load(training_file, weights_only=True)
    saved["updates"] = 999
    torch.save(saved, training_file)
    with pytest.raises(ValueError, match="integrity"):
        restore_trainer(restored_trainer, tmp_path / "one")
