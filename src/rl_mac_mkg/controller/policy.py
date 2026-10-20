"""Factorized masked controller with a frozen Qwen backbone and trainable LoRA."""

from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.distributions import Categorical

from .context import DecisionContext


class FactorizedPolicy:
    def __init__(self, operation_logits, pointer_logits, mask, temperature: float = 0.1):
        if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("temperature must be positive and finite")
        self.mask = torch.as_tensor(mask, dtype=torch.bool, device=operation_logits.device)
        if operation_logits.ndim != 1 or pointer_logits.shape != self.mask.shape:
            raise ValueError("Logits and mask shapes disagree")
        if self.mask.shape[0] != len(operation_logits) or not self.mask.any():
            raise ValueError("At least one action must be feasible")
        if not torch.isfinite(operation_logits).all() or not torch.isfinite(pointer_logits).all():
            raise ValueError("Policy logits must be finite")
        self.pointer_logits = pointer_logits.float() / temperature
        self.valid_operations = self.mask.any(dim=-1)
        self.operation = Categorical(
            logits=(operation_logits.float() / temperature).masked_fill(
                ~self.valid_operations, -torch.inf
            )
        )

    def pointer(self, operation_index: int) -> Categorical:
        if (
            not 0 <= operation_index < len(self.valid_operations)
            or not self.valid_operations[operation_index]
        ):
            raise ValueError("Operation is masked")
        return Categorical(
            logits=self.pointer_logits[operation_index].masked_fill(
                ~self.mask[operation_index], -torch.inf
            )
        )

    def log_prob(self, operation_index: int, target_index: int):
        if (
            not 0 <= operation_index < self.mask.shape[0]
            or not 0 <= target_index < self.mask.shape[1]
            or not self.mask[operation_index, target_index]
        ):
            raise ValueError("Action is masked")
        op = torch.tensor(operation_index, device=self.mask.device)
        target = torch.tensor(target_index, device=self.mask.device)
        return self.operation.log_prob(op) + self.pointer(operation_index).log_prob(target)

    def entropy(self):
        entropy = self.operation.entropy()
        for operation in torch.where(self.valid_operations)[0].tolist():
            entropy = entropy + self.operation.probs[operation] * self.pointer(operation).entropy()
        return entropy

    def joint_probabilities(self):
        rows = []
        for operation in range(len(self.valid_operations)):
            rows.append(
                self.operation.probs[operation] * self.pointer(operation).probs
                if self.valid_operations[operation]
                else torch.zeros_like(self.pointer_logits[operation])
            )
        return torch.stack(rows)

    def sample(self, deterministic: bool = False) -> tuple[int, int]:
        if deterministic:
            joint = self.joint_probabilities()
            flat = int(joint.argmax().item())
            return flat // joint.shape[1], flat % joint.shape[1]
        operation = int(self.operation.sample().item())
        return operation, int(self.pointer(operation).sample().item())


def attach_lora(encoder, rank: int = 16, alpha: int = 32, target_modules=None):
    from peft import LoraConfig, TaskType, get_peft_model

    encoder.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(encoder, config)


class NeuralController(nn.Module):
    def __init__(
        self,
        encoder,
        hidden_size: int,
        n_operations: int = 10,
        head_size: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.n_operations = n_operations
        self.head_size = head_size
        self.temperature = temperature
        self.policy_identity = uuid4().hex
        self.policy_revision = 0
        self.checkpoint_identity = None
        self.operation_head = nn.Sequential(
            nn.Linear(hidden_size, head_size), nn.Tanh(), nn.Linear(head_size, n_operations)
        )
        self.operation_embedding = nn.Embedding(n_operations, hidden_size)
        self.pointer_head = nn.Sequential(
            nn.Linear(3 * hidden_size, head_size), nn.Tanh(), nn.Linear(head_size, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, head_size), nn.Tanh(), nn.Linear(head_size, 1)
        )

    @classmethod
    def from_local(
        cls,
        base_path: str | Path,
        *,
        device="cpu",
        dtype="float32",
        rank=16,
        alpha=32,
        head_size=256,
        temperature=0.1,
        gradient_checkpointing=False,
        target_modules=None,
    ):
        from transformers import AutoModel

        base_path = Path(base_path)
        if not (base_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Controller model config is missing: {base_path / 'config.json'}"
            )
        allowed_dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if dtype not in allowed_dtypes:
            raise ValueError("Unsupported controller dtype")
        encoder = AutoModel.from_pretrained(
            base_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=allowed_dtypes[dtype],
            attn_implementation="sdpa",
        )
        if encoder.config.model_type != "qwen3":
            raise ValueError("The controller backbone must be Qwen3")
        encoder.config.use_cache = False
        encoder = attach_lora(encoder, rank=rank, alpha=alpha, target_modules=target_modules)
        if gradient_checkpointing:
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model = cls(
            encoder, encoder.config.hidden_size, head_size=head_size, temperature=temperature
        )
        return model.to(device)

    def forward(self, context: DecisionContext):
        if len(context.operation_names) != self.n_operations:
            raise ValueError("Controller operation count disagrees with context")
        device = next(self.encoder.parameters()).device
        input_ids = torch.tensor([context.input_ids], device=device, dtype=torch.long)
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[0].to(self.operation_head[0].weight.dtype)
        state = hidden[-1]
        objects = hidden[list(context.object_positions)]
        op_embeddings = self.operation_embedding.weight
        n_objects = len(objects)
        pointer_input = torch.cat(
            [
                state.expand(self.n_operations, n_objects, -1),
                objects.unsqueeze(0).expand(self.n_operations, -1, -1),
                op_embeddings.unsqueeze(1).expand(-1, n_objects, -1),
            ],
            dim=-1,
        )
        return (
            self.operation_head(state),
            self.pointer_head(pointer_input).squeeze(-1),
            self.value_head(state).squeeze(-1),
        )

    def distribution(self, context: DecisionContext):
        operation, pointer, value = self(context)
        return FactorizedPolicy(operation, pointer, context.mask, self.temperature), value

    def optimization_mode(self):
        """Enable checkpointing while keeping collection and update likelihoods deterministic."""
        self.eval()
        for module in self.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(getattr(module, "attention_dropout", None), (int, float)):
                module.attention_dropout = 0.0
        if getattr(self.encoder, "is_gradient_checkpointing", False):
            self.encoder.train()
        return self

    def head_state_dict(self):
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("encoder.")
        }
