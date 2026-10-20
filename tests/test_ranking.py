import numpy as np
import pytest

from rl_mac_mkg.backends import ranking
from rl_mac_mkg.records import Entity


class Encoder:
    version = "frozen-sentence-test"

    def encode(self, texts):
        vectors = {
            "Panda": [1, 0],
            "Bear": [0.99, 0.1],
            "Family": [1, 0],
            "panda-like": [1, 0],
            "unrelated": [0, 1],
        }
        return np.asarray([vectors[text] for text in texts])


def test_canonicalizer_requires_entity_type_and_rejects_ambiguous_similarity():
    entities = [
        Entity("panda", "Panda", "animal"),
        Entity("bear", "Bear", "animal"),
        Entity("family", "Family", "family"),
    ]
    canonicalizer = ranking.FrozenCanonicalizer(
        entities, encoder=Encoder(), threshold=0.9, ambiguity_margin=0.03
    )
    assert canonicalizer("Panda") == "panda"
    assert canonicalizer("family") == "family"
    assert canonicalizer("hasImage") == "hasImage"
    assert canonicalizer.resolve("panda-like", types=("family",)) == "family"
    with pytest.raises(ValueError, match="[Aa]mbigu"):
        canonicalizer.resolve("panda-like", types=("animal",))
    with pytest.raises(ValueError, match="[Uu]nknown|threshold"):
        canonicalizer.resolve("unrelated", types=("animal",))
    with pytest.raises(ValueError, match="type"):
        canonicalizer.resolve("Panda", types=("family",))


def test_canonicalizer_never_semantically_merges_without_declared_types():
    canonicalizer = ranking.FrozenCanonicalizer(
        [Entity("panda", "Panda", "animal")], encoder=Encoder()
    )
    assert canonicalizer("panda-like") == "panda-like"
    assert canonicalizer.resolve("panda-like", types=("animal",)) == "panda"
    other = ranking.FrozenCanonicalizer(
        [Entity("panda", "Panda", "animal")], encoder=Encoder(), threshold=0.99
    )
    assert canonicalizer.version != other.version


def test_ranking_models_require_local_assets(tmp_path):
    with pytest.raises(FileNotFoundError):
        ranking.CLIPBackend(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        ranking.SentenceEncoder(tmp_path / "missing")


def test_clip_load_is_local_only_and_frozen_with_cosine_ranking(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch
    from PIL import Image

    (tmp_path / "config.json").write_text("{}")
    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            return cls()

        def forward(self, **inputs):
            assert not self.training and not self.weight.requires_grad
            return type(
                "Output",
                (),
                {
                    "text_embeds": torch.tensor([[1.0, 0.0]]),
                    "image_embeds": torch.tensor([[0.3, 0.4], [1.0, 0.0]]),
                },
            )()

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            return cls()

        def __call__(self, **kwargs):
            assert len(kwargs["images"]) == 2
            return {"pixel_values": torch.ones(2, 3, 4, 4)}

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(CLIPModel=Model, CLIPProcessor=Processor)
    )
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        Image.new("RGB", (8, 8)).save(path)
    backend = ranking.CLIPBackend(tmp_path)
    assert backend.score("panda", paths) == pytest.approx([0.6, 1.0])
    assert all(call["local_files_only"] and not call["trust_remote_code"] for call in calls)


def test_canonicalizer_rejects_nonfinite_and_malformed_vectors():
    class Broken:
        version = "broken"

        def encode(self, texts):
            return np.full((len(texts), 2), np.nan)

    with pytest.raises(ValueError, match="finite"):
        ranking.FrozenCanonicalizer([Entity("panda", "Panda", "animal")], encoder=Broken())


def test_clip_scores_with_real_tiny_local_transformers_checkpoint(tmp_path):
    import json

    import torch
    from PIL import Image
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer

    folder = tmp_path / "clip"
    folder.mkdir()
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "p": 2, "a": 3, "n": 4, "d": 5, "a</w>": 6}
    (folder / "vocab.json").write_text(json.dumps(vocab))
    (folder / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(
        str(folder / "vocab.json"), str(folder / "merges.txt"), model_max_length=16
    )
    processor = CLIPProcessor(
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
        ),
    )
    processor.save_pretrained(folder)
    config = CLIPConfig(
        projection_dim=4,
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 16,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
    )
    torch.manual_seed(7)
    CLIPModel(config).save_pretrained(folder)
    image = tmp_path / "red.png"
    Image.new("RGB", (8, 8), "red").save(image)
    backend = ranking.CLIPBackend(folder)
    scores = backend.score("panda", [image, image])
    assert len(scores) == 2 and np.isfinite(scores).all()
    assert scores[0] == pytest.approx(scores[1])
    assert not backend.model.training and all(
        not p.requires_grad for p in backend.model.parameters()
    )


def test_sentence_encoder_uses_local_only_and_freezes_model(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch

    (tmp_path / "config.json").write_text("{}")
    calls = []

    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            calls.append(kwargs)

        def encode(self, texts, **kwargs):
            assert not self.training and not self.weight.requires_grad
            return np.asarray([[3.0, 4.0]] * len(texts))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    encoder = ranking.SentenceEncoder(tmp_path)
    assert encoder.encode(["panda"])[0] == pytest.approx([0.6, 0.8])
    assert calls[0]["local_files_only"] and not calls[0]["trust_remote_code"]


def test_canonicalizer_version_is_frozen_model_and_settings_not_episode_vocabulary():
    train = ranking.FrozenCanonicalizer([Entity("panda", "Panda", "animal")], encoder=Encoder())
    test = ranking.FrozenCanonicalizer([Entity("bear", "Bear", "animal")], encoder=Encoder())
    assert train.version == test.version
    assert train.vocabulary_hash != test.vocabulary_hash
