"""Frozen local Qwen3 / Qwen3-VL generation and conditional support likelihoods."""

import hashlib
import json
import math
import time
from pathlib import Path

from .base import ModelResponse


def asset_fingerprint(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Model directory is missing: {directory}")
    digest = hashlib.sha256()
    files = sorted(file for file in directory.rglob("*") if file.is_file())
    if not files:
        raise FileNotFoundError(f"Model directory is empty: {directory}")
    for file in files:
        digest.update(file.relative_to(directory).as_posix().encode("utf-8"))
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class LocalModelBackend:
    def __init__(
        self,
        model_path,
        *,
        adapter_path=None,
        multimodal=False,
        device="cpu",
        dtype="float32",
        temperature=0.1,
        max_new_tokens=1024,
    ):
        if (
            not math.isfinite(temperature)
            or temperature < 0
            or type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise ValueError("Generation temperature and token limit must be nonnegative/positive")
        self.model_path = Path(model_path)
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Local model config is missing: {self.model_path / 'config.json'}"
            )
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.multimodal, self.device, self.dtype = multimodal, device, dtype
        self.temperature, self.max_new_tokens = temperature, max_new_tokens
        self.version = asset_fingerprint(self.model_path)
        if self.adapter_path:
            self.version = hashlib.sha256(
                (self.version + asset_fingerprint(self.adapter_path)).encode()
            ).hexdigest()
        self.version = hashlib.sha256(
            json.dumps(
                {
                    "assets": self.version,
                    "multimodal": multimodal,
                    "dtype": dtype,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "backend_contract": "qwen3-local-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.model = self.processor = self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            Qwen3VLForConditionalGeneration,
        )

        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if self.dtype not in dtypes:
            raise ValueError("Unsupported model dtype")
        model_class = Qwen3VLForConditionalGeneration if self.multimodal else AutoModelForCausalLM
        self.model = model_class.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtypes[self.dtype],
            attn_implementation="sdpa",
        )
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path, is_trainable=False, local_files_only=True
            )
        self.model.to(self.device).eval().requires_grad_(False)
        if self.multimodal:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False
            )

    def _inputs(self, system, user, images=()):
        self._load()
        if images and not self.multimodal:
            raise ValueError("This model does not support image input")
        content = [{"type": "image", "image": str(Path(image).resolve())} for image in images]
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content if self.multimodal else user},
        ]
        if self.multimodal:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs.pop("token_type_ids", None)
        return {name: value.to(self.device) for name, value in inputs.items()}

    def generate(self, system, user, images=()):
        import torch

        start = time.perf_counter()
        inputs = self._inputs(system, user, images)
        count = inputs["input_ids"].shape[1]
        options = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            options["temperature"] = self.temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        completion = output[0, count:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return ModelResponse(text, count, len(completion), time.perf_counter() - start)

    def score_binary(self, prompt, images=()):
        """Return log P(yes|evidence) - log P(no|evidence), before temperature scaling."""
        import torch

        inputs = self._inputs(
            "Assess only the supplied evidence. Answer yes or no.", prompt, images
        )
        scores = []
        with torch.inference_mode():
            for label in ("yes", "no"):
                ids = self.tokenizer.encode(label, add_special_tokens=False)
                if not ids:
                    raise ValueError("Verifier label tokenization is empty")
                labels = torch.tensor([ids], device=self.device)
                combined = {**inputs, "input_ids": torch.cat([inputs["input_ids"], labels], dim=1)}
                combined["attention_mask"] = torch.ones_like(combined["input_ids"])
                output = self.model(**combined, use_cache=False, logits_to_keep=len(ids) + 1)
                logits = output.logits[0, -len(ids) - 1 : -1].float()
                log_probabilities = logits.log_softmax(dim=-1)
                scores.append(float(log_probabilities.gather(1, labels[0, :, None]).sum().item()))
        return scores[0] - scores[1]
