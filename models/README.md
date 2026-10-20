# Local model files

All loading uses `local_files_only=True` and disables remote custom code. Obtain model files separately and preserve each model's license. No command downloads weights automatically.

```text
models/
  backbones/
    Qwen3-8B/
    Qwen3-VL-8B-Instruct/
    clip-vit-base-patch32/
    all-MiniLM-L6-v2/
  adapters/
    textual/
    visual/
    verifier/
  controller/
    bc/
    ppo/
  calibration/
    verifier.json
    action_costs.json
```

The directories exist in the repository. Base model directories must contain their original configuration, tokenizer/processor and complete weight shards. Qwen3-8B is shared as the controller and textual backbone, with separate trainable/frozen adapters. Qwen3-VL is shared as the visual and verifier architecture, with distinct fitted adapters. The controller loader uses `AutoModel`; the original causal-LM checkpoint is compatible, while its unused language-model head is excluded from control optimization.

Adapter directories use PEFT's `adapter_config.json` and `adapter_model.safetensors`. They can be created with `train-specialist` or supplied as compatible, previously fitted checkpoints. Missing assets raise an error. An unfitted adapter is never silently substituted.

`train-controller` checkpoints contain adapter weights, three decision heads, a tokenizer, integrity hashes and controller metadata. PPO checkpoints also include optimizer/RNG state and episode continuation metadata. Use the command's checkpoint directory directly with `infer --checkpoint`; copying it into `models/controller/ppo` is optional. A relocated backbone must have the same file fingerprint as the training backbone.

`score-verifier` followed by `calibrate` creates the verifier artifact. `fit-costs` creates the complete action-cost table from validation measurements. These files are generated from supplied annotations and measurements; they are not distributed as fitted assets.

`manifest.json` lists role-to-directory conventions. Weight and checkpoint files are excluded by `.gitignore`. For a model release, use a suitable model host or Git LFS in a separately configured release workflow.

Official model interfaces: [Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3), [Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl), [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
