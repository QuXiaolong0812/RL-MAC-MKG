# Validation record

Date: 2026-10-20. Platform: Windows, Python 3.12.14, CPU.

The complete local test suite passes: **148 tests**. Ruff linting, Ruff formatting, source compilation and dependency checks pass.

Both wheel and source distributions build successfully. Archive inspection confirms that all Python modules and model-role directories are included, while weights, caches, runtime outputs and internal working notes are excluded. The wheel was installed into a separate clean environment; CLI help and reference-based construction evaluation passed outside the source checkout. CI repeats the installed-wheel smoke check.

```text
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m compileall -q src
python -m pip check
```

Coverage includes an actual locally instantiated Qwen3 model for controller LoRA training/checkpoint restoration, an actual small causal-LM specialist fitting step, local yes/no likelihood inference, a locally instantiated CLIP inference path, and BC → multiple PPO updates → resume → inference integration. These tests use small models created inside temporary test directories and make no external inference calls.

Regression tests cover unread retrieval chunks at the hop boundary, explicit inventory exhaustion, deterministic integration-version records, forged anchors, duplicate retrievals, entity partition isolation, literal/image identity preservation, revision invalidation, conditional action entropy, terminal reward placement, stale policy rejection and optimizer/checkpoint pairing.

The tested principal package versions are:

| Package | Version |
| --- | --- |
| NumPy | 2.5.4 |
| Pillow | 13.0.0 |
| PyTorch | 2.14.1+cpu |
| Transformers | 4.57.6 |
| PEFT | 0.21.2 |
| Sentence Transformers | 5.7.0 |
| pytest | 9.1.1 |
| Ruff | 0.17.1 |

Large Qwen3/Qwen3-VL checkpoints and the manuscript datasets are not included. Full 8B training, large-model image inference and the reported manuscript metrics require those assets and are outside this CPU validation record. Hosted GitHub Actions results are available only after the repository is pushed.
