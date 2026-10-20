"""Portable adapter/head checkpoints with integrity and optimizer state."""

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..backends.local import asset_fingerprint
from .policy import NeuralController


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_hashes(path):
    path = Path(path).resolve()
    hashes = json.loads((path / "integrity.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path) or not file.is_file() or file_hash(file) != expected:
            raise ValueError(f"Checkpoint integrity check failed: {relative}")
    return hashes


def _identity(hashes):
    model_hashes = {name: digest for name, digest in hashes.items() if name != "training.pt"}
    return hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest()


def save_controller(model, path, *, base_path, metadata=None, trainer=None, tokenizer=None):
    from safetensors.torch import save_file

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    try:
        model.encoder.save_pretrained(temporary / "adapter", safe_serialization=True)
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.head_state_dict().items()
            },
            temporary / "heads.safetensors",
        )
        config = {
            "format_version": 1,
            "base_path": str(Path(base_path).resolve()),
            "hidden_size": model.hidden_size,
            "n_operations": model.n_operations,
            "head_size": model.head_size,
            "temperature": model.temperature,
            "dtype": str(next(model.encoder.parameters()).dtype).removeprefix("torch."),
            "base_fingerprint": asset_fingerprint(base_path),
            "gradient_checkpointing": bool(
                getattr(model.encoder, "is_gradient_checkpointing", False)
            ),
            "policy_identity": model.policy_identity,
            "policy_revision": model.policy_revision,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozen_parameters": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
            "metadata": metadata or {},
        }
        (temporary / "controller.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if tokenizer is not None:
            tokenizer.save_pretrained(temporary / "tokenizer")
        model_hashes = {
            file.relative_to(temporary).as_posix(): file_hash(file)
            for file in temporary.rglob("*")
            if file.is_file()
        }
        if trainer is not None:
            torch.save(
                {
                    "optimizer": trainer.optimizer.state_dict(),
                    "updates": trainer.updates,
                    "ppo_config": asdict(trainer.config),
                    "rng": capture_rng(),
                    "checkpoint_identity": _identity(model_hashes),
                    "policy_identity": model.policy_identity,
                },
                temporary / "training.pt",
            )
        hashes = {
            file.relative_to(temporary).as_posix(): file_hash(file)
            for file in temporary.rglob("*")
            if file.is_file()
        }
        (temporary / "integrity.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        if temporary.resolve().is_relative_to(path.parent) and temporary.name.startswith(
            ".checkpoint-"
        ):
            shutil.rmtree(temporary)
        raise


def load_controller(path, *, base_path=None, device="cpu", trainable=True):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModel

    path = Path(path)
    hashes = _checkpoint_hashes(path)
    config = json.loads((path / "controller.json").read_text(encoding="utf-8"))
    if config["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    resolved_base = base_path or config["base_path"]
    if asset_fingerprint(resolved_base) != config["base_fingerprint"]:
        raise ValueError("Frozen base model fingerprint differs from checkpoint")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        config["dtype"]
    ]
    backbone = AutoModel.from_pretrained(
        resolved_base, local_files_only=True, trust_remote_code=False, torch_dtype=dtype
    )
    backbone.config.use_cache = False
    encoder = PeftModel.from_pretrained(
        backbone, path / "adapter", is_trainable=trainable, local_files_only=True
    )
    if config.get("gradient_checkpointing"):
        encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = NeuralController(
        encoder,
        config["hidden_size"],
        config["n_operations"],
        config["head_size"],
        config["temperature"],
    )
    missing, unexpected = model.load_state_dict(load_file(path / "heads.safetensors"), strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError("Controller head checkpoint does not match the architecture")
    model.to(device)
    model.eval()
    model.policy_identity = config["policy_identity"]
    model.policy_revision = config["policy_revision"]
    model.checkpoint_identity = _identity(hashes)
    return model, config


def restore_trainer(trainer, path):
    hashes = _checkpoint_hashes(path)
    saved = torch.load(Path(path) / "training.pt", map_location="cpu", weights_only=True)
    if saved["ppo_config"] != asdict(trainer.config):
        raise ValueError("Resume PPO configuration differs from checkpoint")
    if (
        saved["checkpoint_identity"] != _identity(hashes)
        or trainer.model.checkpoint_identity != saved["checkpoint_identity"]
        or trainer.model.policy_identity != saved["policy_identity"]
        or trainer.model.policy_revision != saved["updates"]
    ):
        raise ValueError("Training state does not belong to the loaded controller checkpoint")
    trainer.optimizer.load_state_dict(saved["optimizer"])
    trainer.updates = saved["updates"]
    restore_rng(saved["rng"])
