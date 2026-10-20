import json
from dataclasses import replace
from hashlib import sha256

import pytest

from rl_mac_mkg.data import (
    assign_sources,
    audit_leakage,
    chunk_text,
    duplicate_groups,
    image_duplicate_groups,
    load_manifest,
    minhash_signature,
    perceptual_hash,
    split_entities,
)
from rl_mac_mkg.records import SourceRecord


def test_chunk_offsets_recover_exact_raw_text_without_normalizing():
    text = "熊猫\r\n  family Ursidae.\nNext paragraph: mammal.  "
    chunks = chunk_text(text, max_chars=20, overlap=3)
    assert chunks
    assert all(text[c.start : c.end] == c.text for c in chunks)
    assert max(c.end - c.start for c in chunks) <= 20
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    assert set().union(*(set(range(c.start, c.end)) for c in chunks)) == set(range(len(text)))


def test_manifest_resolves_paths_and_rejects_tampered_bytes(tmp_path):
    text = "Panda 熊猫\r\nUrsidae"
    (tmp_path / "source.txt").write_bytes(text.encode("utf-8"))
    row = dict(
        source_id="s1",
        entity_ids=["panda"],
        url="https://example.org/panda",
        retrieved_at="2026-10-19T10:00:00Z",
        license="CC-BY-4.0",
        content_hash=sha256(text.encode()).hexdigest(),
        kind="text",
        path="source.txt",
        document_id="d1",
        partition="test",
    )
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = load_manifest(manifest)
    assert records[0].text == text
    (tmp_path / "source.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_manifest(manifest)


def test_seeded_splits_have_exact_paper_ratios_and_train_subdivisions():
    ids = [f"e{i}" for i in range(1000)]
    splits = split_entities(ids, seed=42)
    assert splits == split_entities(ids, seed=42)
    assert splits != split_entities(ids, seed=43)
    assert {k: len(v) for k, v in splits.items()} == {
        "specialist": 210,
        "verifier": 150,
        "ppo": 240,
        "calibration": 100,
        "validation": 100,
        "test": 200,
    }
    assert len(set().union(*map(set, splits.values()))) == 1000
    with pytest.raises(ValueError, match="duplicate"):
        split_entities(["e1", "e1"])


def make_source(sid, entities, partition=None, group=None):
    text = "A source document " + sid
    return SourceRecord(
        sid,
        tuple(entities),
        "https://example.org/" + sid,
        "2026-10-19T10:00:00Z",
        "CC0",
        sha256(text.encode()).hexdigest(),
        "text",
        text=text,
        document_id=sid,
        partition=partition,
        duplicate_group=group,
    )


def test_sources_and_duplicate_groups_belong_to_earliest_partition():
    splits = {
        "specialist": ["a"],
        "verifier": [],
        "ppo": ["b"],
        "calibration": ["c"],
        "validation": [],
        "test": ["d"],
    }
    records = [
        make_source("s1", ["a", "d"]),
        make_source("s2", ["b"], group="near1"),
        make_source("s3", ["c"], group="near1"),
    ]
    assigned = assign_sources(records, splits)
    assert [s.partition for s in assigned] == ["specialist", "ppo", "ppo"]
    assert assigned[0].entity_ids == ("a",)
    assert assigned[2].entity_ids == ()
    assert audit_leakage(assigned, splits) == []


def test_audit_detects_source_image_and_duplicate_group_leakage():
    a = make_source("s1", ["a"], "specialist", "dup1")
    b = replace(a, entity_ids=("b",), partition="test")
    findings = audit_leakage([a, b], {"specialist": ["a"], "test": ["b"]})
    assert any("source" in item for item in findings)
    assert any("duplicate" in item for item in findings)


def test_duplicate_groups_capture_exact_and_near_text_transitively():
    groups = duplicate_groups(
        {
            "a": "Panda is a black and white bear.",
            "b": "Panda is a black and white bear.",
            "c": "Panda is a black and white bear!",
            "d": "Unrelated long description of a mineral.",
        },
        threshold=0.8,
    )
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_minhash_is_seeded_and_ignores_only_comparison_punctuation():
    a = minhash_signature("Panda is a bear.", seed=42)
    assert len(a) == 128
    assert a == minhash_signature("panda is a bear!", seed=42)
    assert a != minhash_signature("panda is a bear!", seed=43)


def test_perceptual_duplicate_images_are_grouped_without_affecting_originals(tmp_path):
    from PIL import Image

    first = Image.new("RGB", (32, 32))
    first.putdata([(x * 8, y * 8, 20) for y in range(32) for x in range(32)])
    first.save(tmp_path / "a.png")
    first.resize((64, 64)).save(tmp_path / "b.png")
    flipped = first.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flipped.save(tmp_path / "c.png")
    files = {k: tmp_path / f"{k}.png" for k in ("a", "b", "c")}
    assert perceptual_hash(files["a"]) == perceptual_hash(files["b"])
    groups = image_duplicate_groups(files, max_hamming=2)
    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]


def test_duplicate_embedding_grouping_accepts_numpy_and_rejects_nonfinite():
    import numpy as np

    texts = {"a": "Panda bear animal", "b": "A description with unrelated words"}
    vectors = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.01])}
    groups = duplicate_groups(texts, embeddings=vectors)
    assert groups["a"] == groups["b"]
    with pytest.raises(ValueError, match="finite"):
        duplicate_groups(texts, embeddings={"a": [1, 0], "b": [float("nan"), 0]})
