# RL-MAC-MKG

Budget-constrained multimodal knowledge graph construction with a centralized PPO controller and five frozen specialists.

The controller chooses an operation and a compatible object. Deterministic provenance and schema gates protect graph acceptance; calibrated support, uncertainty, graph-quality changes and measured action costs guide policy learning. PPO updates the controller's rank-16 LoRA adapter and operation, pointer and value heads. Specialist parameters remain fixed.

## Installation

Use Python 3.10 or newer. Install the PyTorch build appropriate for your machine, then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[train,retrieval,dev]"
rl-mac-mkg --help
```

CPU tests instantiate small local Qwen3 and CLIP models. They do not download model weights or call external services. Full 8B training requires sufficient RAM/accelerator memory for the selected precision and sequence length; the configuration exposes devices separately for each role.

## Prepare models and evidence

1. Put the standard model files and fitted adapters in the directories described in [models/README.md](models/README.md).
2. Create entity, source and schema records using [the data format](docs/data-format.md).
3. Copy [configs/experiment.yaml](configs/experiment.yaml), set the paths and devices, and choose the three explicitly configured reward coefficients on validation episodes.
4. Create disjoint partitions and audit source hashes:

```bash
rl-mac-mkg split --entities data/processed/entities.jsonl --sources data/raw/sources.jsonl --output data/splits --seed 42
rl-mac-mkg validate --config configs/experiment.yaml
```

Point `dataset.sources` and `dataset.split_manifest` to the split command's outputs. Sources carry original text offsets or image identities, retrieval timestamps, content hashes, licenses and media safety review status. Images with unchecked safety or incompatible licenses are excluded. Original media are never deleted or renamed.

## Fit and run

See [training.md](docs/training.md) for data preparation and command details. The intended order is:

```bash
rl-mac-mkg train-specialist --config configs/experiment.yaml --role textual --examples data/processed/textual_sft.jsonl --output models/adapters/textual
rl-mac-mkg train-specialist --config configs/experiment.yaml --role visual --examples data/processed/visual_sft.jsonl --output models/adapters/visual
rl-mac-mkg train-specialist --config configs/experiment.yaml --role verifier --examples data/processed/verifier_sft.jsonl --output models/adapters/verifier
rl-mac-mkg score-verifier --config configs/experiment.yaml --candidates data/processed/calibration_candidates.jsonl --output outputs/calibration_scores.jsonl
rl-mac-mkg calibrate --scores outputs/calibration_scores.jsonl --output models/calibration/verifier.json
rl-mac-mkg fit-costs --measurements data/processed/validation_costs.jsonl --token-weight 0.001 --latency-weight 1 --retrieval-weight 1 --output models/calibration/action_costs.json
rl-mac-mkg train-controller --config configs/experiment.yaml --stage bc --output outputs/controller-bc
rl-mac-mkg train-controller --config configs/experiment.yaml --stage ppo --from-bc outputs/controller-bc --output outputs/controller-ppo
```

The three cost weights above are an explicit measurement-unit example. Select them on validation data. Every non-stop cost is fitted to [0.02, 1]; stopping costs zero.

For test inference, set `dataset.partition: test`, retain the same specialists/calibration/costs, and use the selected trained controller checkpoint:

```bash
rl-mac-mkg infer --config configs/test.yaml --checkpoint outputs/controller-ppo/update-000001 --seed-entity ENTITY_ID --output outputs/test/ENTITY_ID --deterministic
rl-mac-mkg infer --config configs/test.yaml --policy fixed --seed-entity ENTITY_ID --output outputs/fixed/ENTITY_ID
rl-mac-mkg evaluate --predictions outputs/test/ENTITY_ID/assertions.jsonl --references data/processed/test_references.jsonl
```

Each episode exports accepted assertions, unresolved/resolved candidates, provenance, conflicts, the full action trace, measured usage and run metadata. Stopping leaves unresolved candidates unasserted. Evaluation references should be scoped to the same episode or aggregated consistently across episodes.

## Implementation map

| Component | Module |
| --- | --- |
| PEA: anchored evidence and image eligibility/ranking | `agents/evidence.py` |
| TCA / VGA: structured candidate generation and revision | `agents/generation.py` |
| CVfA: independent calibrated factual and visual support | `agents/verification.py` |
| GIA: schema checks and atomic graph integration | `agents/integration.py`, `graph/`, `environment/` |
| Masked operation/object/value controller | `controller/` |
| Behavior cloning, GAE, PPO and supervised specialist LoRA | `training/` |
| Calibration, construction, graph, retrieval and statistics | `evaluation/` |

[Method mapping](docs/method-mapping.md) records manuscript constants, executable metric definitions and implementation choices for details not reported in the manuscript. [Reproducibility](docs/reproducibility.md) explains checkpoint integrity, partition isolation, perturbations and validation limits. [中文指南](docs/quickstart.zh-CN.md) provides a Chinese walkthrough.

## Development

```bash
python -m pytest -q
python -m ruff check src tests
python -m compileall -q src
python -m build
```

The MIT license covers this repository's code. Model checkpoints, evidence and third-party datasets retain their own licenses. Citation metadata is provided in [CITATION.cff](CITATION.cff).
