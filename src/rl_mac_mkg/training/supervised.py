"""Completion-only supervised LoRA fitting for textual, visual and verifier specialists."""

import json
from pathlib import Path

import numpy as np
import torch


def completion_labels(input_ids, *, prefix_length):
    if input_ids.ndim != 2 or not 0 <= prefix_length < input_ids.shape[1]:
        raise ValueError("A nonempty completion must follow the prompt")
    labels = input_ids.clone()
    labels[:, :prefix_length] = -100
    return labels


def validate_examples(rows, *, partition, allowed_sources, allowed_entities):
    if partition not in {"specialist", "verifier"}:
        raise ValueError("Supervised specialist fitting requires specialist or verifier partition")
    if not rows:
        raise ValueError("Supervised dataset is empty")
    allowed_keys = {
        "partition",
        "entity_id",
        "source_ids",
        "system",
        "user",
        "assistant",
        "image_source_ids",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed_keys:
            raise ValueError("Unknown supervised example fields")
        if row.get("partition") != partition:
            raise ValueError("Supervised example crosses the training partition")
        if row.get("entity_id") not in allowed_entities:
            raise ValueError("Supervised example references an entity outside its partition")
        if not row.get("source_ids") or not set(row["source_ids"]) <= set(allowed_sources):
            raise ValueError("Supervised example references an unavailable source")
        if not set(row.get("image_source_ids", [])) <= set(row["source_ids"]):
            raise ValueError("Image sources must be included in the supervised provenance")
        for name in ("system", "user", "assistant"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Supervised example requires {name}")
    return rows


def _prepare_example(row, tokenizer, processor, sources, device, max_tokens):
    image_ids = row.get("image_source_ids", [])
    content = [
        {"type": "image", "image": str(Path(sources[identifier].path).resolve())}
        for identifier in image_ids
    ]
    content.append({"type": "text", "text": row["user"]})
    if image_ids and processor is None:
        raise ValueError("Image examples require a multimodal model")
    prompt = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": content if processor else row["user"]},
    ]
    full = [*prompt, {"role": "assistant", "content": row["assistant"]}]
    if processor:
        inputs = processor.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
        )
        prefix = processor.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )["input_ids"]
    else:
        prefix_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        inputs = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        prefix = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if inputs["input_ids"].shape[1] > max_tokens:
        raise ValueError("Supervised example exceeds max_tokens; rechunk its source evidence")
    prefix_length = prefix.shape[1]
    if not torch.equal(inputs["input_ids"][:, :prefix_length], prefix):
        raise ValueError(
            "Chat template changed the prompt boundary; completion mask would be invalid"
        )
    inputs.pop("token_type_ids", None)
    inputs["labels"] = completion_labels(inputs["input_ids"], prefix_length=prefix_length)
    return {name: value.to(device) for name, value in inputs.items()}


def fit_specialist(
    base_path,
    rows,
    sources,
    output,
    *,
    partition,
    entity_ids,
    multimodal=False,
    device="cpu",
    dtype="float32",
    rank=16,
    alpha=32,
    learning_rate=1e-5,
    epochs=1,
    accumulation_steps=16,
    max_tokens=2048,
    seed=42,
):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )

    validate_examples(
        rows, partition=partition, allowed_sources=sources, allowed_entities=entity_ids
    )
    if any(source.partition != partition for source in sources.values()):
        raise ValueError("Source catalog must match the supervised partition")
    if epochs < 1 or accumulation_steps < 1 or learning_rate <= 0:
        raise ValueError("Invalid supervised optimization settings")
    output, base_path = Path(output), Path(base_path)
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_class = Qwen3VLForConditionalGeneration if multimodal else AutoModelForCausalLM
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    model = model_class.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()
    processor = (
        AutoProcessor.from_pretrained(base_path, local_files_only=True) if multimodal else None
    )
    tokenizer = (
        processor.tokenizer
        if processor
        else AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.0
    )
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), accumulation_steps):
            minibatch = order[start : start + accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            for index in minibatch:
                inputs = _prepare_example(
                    rows[index], tokenizer, processor, sources, device, max_tokens
                )
                loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite specialist supervised loss")
                (loss / len(minibatch)).backward()
                losses.append(float(loss.detach().item()))
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True
            )
            optimizer.step()
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    (processor or tokenizer).save_pretrained(output)
    metadata = {
        "partition": partition,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "examples": len(rows),
        "seed": seed,
        "rank": rank,
        "alpha": alpha,
        "mean_loss": float(np.mean(losses)),
        "source_ids": sorted(sources),
        "entity_ids": sorted(entity_ids),
        "multimodal": multimodal,
    }
    (output / "fitting.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
