import pytest
import torch

from rl_mac_mkg.training.supervised import completion_labels, validate_examples


def test_completion_labels_mask_only_prompt_tokens():
    labels = completion_labels(torch.tensor([[1, 2, 3, 4, 5]]), prefix_length=3)
    assert labels.tolist() == [[-100, -100, -100, 4, 5]]
    with pytest.raises(ValueError, match="completion"):
        completion_labels(torch.tensor([[1, 2]]), prefix_length=2)


def test_supervised_training_rejects_cross_partition_data():
    row = {
        "partition": "test",
        "entity_id": "e",
        "source_ids": ["s"],
        "system": "Extract",
        "user": "Evidence",
        "assistant": "{}",
    }
    with pytest.raises(ValueError, match="partition"):
        validate_examples(
            [row], partition="specialist", allowed_sources={"s"}, allowed_entities={"e"}
        )
    row["partition"] = "specialist"
    with pytest.raises(ValueError, match="source"):
        validate_examples(
            [row], partition="specialist", allowed_sources=set(), allowed_entities={"e"}
        )


def test_actual_small_qwen_specialist_fit_changes_lora_weights(tmp_path):
    from dataclasses import replace

    from safetensors.torch import load_file
    from test_workflows import make_environment
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from rl_mac_mkg.training.supervised import fit_specialist

    source = replace(next(iter(make_environment().state.sources.values())), partition="specialist")
    base = tmp_path / "base"
    tokenizer_engine = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "yes": 3,
                "no": 4,
                "system": 5,
                "user": 6,
                "assistant": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer_engine.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_engine, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ' ' + message['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            eos_token_id=2,
            pad_token_id=1,
        )
    ).save_pretrained(base)
    tokenizer.save_pretrained(base)
    output = tmp_path / "adapter"
    result = fit_specialist(
        base,
        [
            {
                "partition": "specialist",
                "entity_id": "a",
                "source_ids": ["s"],
                "system": "Judge",
                "user": "Evidence",
                "assistant": "yes",
            }
        ],
        {"s": source},
        output,
        partition="specialist",
        entity_ids={"a"},
        epochs=1,
    )
    assert result["examples"] == 1
    tensors = load_file(output / "adapter_model.safetensors")
    assert any(torch.count_nonzero(value) for key, value in tensors.items() if "lora_B" in key)
