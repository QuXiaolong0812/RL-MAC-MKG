"""Strict configuration and configuration-relative filesystem paths."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULTS = {
    "format_version": 1,
    "dataset": {"partition": "inference", "shared_entity_ids": []},
    "environment": {"budget": 30.0, "max_depth": 3, "max_tokens": 2048},
    "reward": {
        "lambda_c": 0.08,
        "lambda_u": 0.2,
        "factual_weight": 0.55,
        "visual_weight": 0.45,
        "abstention_threshold": 0.65,
    },
    "controller": {
        "device": "cpu",
        "dtype": "float32",
        "lora_rank": 16,
        "lora_alpha": 32,
        "head_size": 256,
        "temperature": 0.1,
        "gradient_checkpointing": False,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "ppo": {
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "batch_size": 128,
        "minibatch_size": 16,
        "epochs": 4,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "max_gradient_norm": 0.5,
    },
    "specialists": {},
    "media": {
        "minimum_resolution": 128,
        "allowed_licenses": ["CC0", "CC-BY-4.0", "CC-BY-SA-4.0"],
        "max_image_pixels": 40_000_000,
        "phash_distance": 4,
    },
    "schema": {},
    "run": {"seed": 42, "episodes": 1, "output": "../outputs/run"},
}

SECTION_KEYS = {
    "dataset": {
        "entities",
        "sources",
        "references",
        "demonstrations",
        "partition",
        "split_manifest",
        "shared_entity_ids",
    },
    "environment": set(DEFAULTS["environment"]),
    "reward": set(DEFAULTS["reward"]) | {"lambda_h", "lambda_p", "lambda_g", "costs_file"},
    "controller": set(DEFAULTS["controller"]) | {"model_path", "checkpoint"},
    "ppo": set(DEFAULTS["ppo"]),
    "specialists": {"textual", "visual", "verifier", "clip", "canonicalizer", "calibration_file"},
    "media": set(DEFAULTS["media"]),
    "schema": {"path"},
    "run": set(DEFAULTS["run"]),
}
MODEL_KEYS = {
    "backend",
    "model_path",
    "adapter_path",
    "device",
    "dtype",
    "base_url",
    "model",
    "api_key_env",
    "timeout",
    "max_new_tokens",
    "temperature",
    "multimodal",
    "threshold",
    "ambiguity_margin",
}


@dataclass(frozen=True)
class ProjectConfig:
    source: Path
    values: dict[str, Any]

    @property
    def environment(self):
        return self.values["environment"]

    @property
    def controller(self):
        return self.values["controller"]

    def path(self, dotted_key: str, *, required=True) -> Path | None:
        value: Any = self.values
        for key in dotted_key.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            if required:
                raise ValueError(f"Configuration requires {dotted_key}")
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid path for {dotted_key}")
        return (self.source.parent / value).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    if set(value) - set(DEFAULTS):
        raise ValueError(f"Unknown configuration keys: {sorted(set(value) - set(DEFAULTS))}")
    if value.get("format_version") != 1:
        raise ValueError("format_version must equal 1")
    result = copy.deepcopy(DEFAULTS)
    for section, entries in value.items():
        if section == "format_version":
            continue
        if not isinstance(entries, dict) or set(entries) - SECTION_KEYS[section]:
            raise ValueError(f"Unknown or invalid keys in {section}")
        result[section].update(entries)
    for name in ("lambda_h", "lambda_p", "lambda_g"):
        weight = result["reward"].get(name)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                f"reward.{name} must be explicitly configured as a finite nonnegative number"
            )
    for role, model in result["specialists"].items():
        if role == "calibration_file":
            continue
        if not isinstance(model, dict) or set(model) - MODEL_KEYS:
            raise ValueError(f"Unknown specialist keys for {role}")
        if model.get("backend", "local") not in {"local", "http"}:
            raise ValueError(f"Unsupported backend for {role}")
        if model.get("backend", "local") == "local" and not model.get("model_path"):
            raise ValueError(f"Local specialist {role} requires model_path")
    for name, number in result["environment"].items():
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"Invalid environment.{name}")
    if result["environment"]["max_tokens"] < 64:
        raise ValueError("environment.max_tokens must be at least 64")
    for key in ("max_tokens", "max_depth"):
        if type(result["environment"][key]) is not int:
            raise ValueError(f"environment.{key} must be an integer")
    if type(result["run"]["episodes"]) is not int or result["run"]["episodes"] < 1:
        raise ValueError("run.episodes must be a positive integer")
    if result["dataset"]["partition"] not in {
        "inference",
        "specialist",
        "verifier",
        "ppo",
        "calibration",
        "validation",
        "test",
    }:
        raise ValueError("Unknown dataset partition")
    return ProjectConfig(path, result)
