import pytest

from rl_mac_mkg.backends.base import parse_json_object
from rl_mac_mkg.backends.http import HTTPBackend


def test_json_parser_rejects_partial_embedded_and_nonfinite_values():
    assert parse_json_object('{"triples": []}') == {"triples": []}
    assert parse_json_object('```json\n{"triples": []}\n```') == {"triples": []}
    for content in ('prefix {"triples": []}', '{"score": NaN}', "[1,2]", '{"a":1,"a":2}'):
        with pytest.raises(ValueError):
            parse_json_object(content)


def test_model_versions_include_fixed_inference_settings(tmp_path):
    from rl_mac_mkg.backends.local import LocalModelBackend

    (tmp_path / "config.json").write_text("{}")
    assert (
        LocalModelBackend(tmp_path, temperature=0.1).version
        != LocalModelBackend(tmp_path, temperature=0.2).version
    )
    assert (
        LocalModelBackend(tmp_path, max_new_tokens=32).version
        != LocalModelBackend(tmp_path, max_new_tokens=64).version
    )
    assert (
        HTTPBackend("http://localhost:8000/v1", "qwen", max_new_tokens=32).version
        != HTTPBackend("http://localhost:8000/v1", "qwen", max_new_tokens=64).version
    )


def test_http_backend_timeout_has_no_implicit_retries(monkeypatch):
    import requests

    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "post", post)
    backend = HTTPBackend("http://localhost:8000/v1", "qwen", timeout=2)
    with pytest.raises(requests.Timeout):
        backend.generate("system", "user")
    assert len(calls) == 1
    assert calls[0]["timeout"] == 2


def test_http_binary_scores_require_explicit_token_logprob_support(monkeypatch):
    backend = HTTPBackend("http://localhost:8000/v1", "qwen")
    with pytest.raises(ValueError, match="local"):
        backend.score_binary("Does evidence support the claim?")


def test_local_qwen_generation_and_label_likelihoods(tmp_path):
    import math

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from rl_mac_mkg.backends.local import LocalModelBackend

    base = tmp_path / "base"
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[EOS]": 2,
        "yes": 3,
        "no": 4,
        "system": 5,
        "user": 6,
        "assistant": 7,
    }
    tokenizer_engine = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
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
    backend = LocalModelBackend(base, temperature=0, max_new_tokens=2)
    response = backend.generate("Extract", "Evidence")
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert math.isfinite(backend.score_binary("Evidence supports the claim?"))
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())
