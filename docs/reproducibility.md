# Reproducibility and validation boundaries

The repository implements the revised algorithm and its training/evaluation workflows. Reproducing the manuscript's numerical tables additionally requires its admissible evidence, adjudicated labels, source-disjoint splits, specialist adapters, selected reward/cost settings and trained controller initialization. No manuscript table values are embedded as evaluation output.

## Recorded identities

- Source URLs/timestamps, original content hashes, text character spans and image IDs.
- Entity/source partitions and duplicate groups.
- Frozen specialist asset, prompt, schema, calibration and settings signatures.
- Exact bounded controller tokens, visible target order, action masks and state/evidence hashes.
- Persistent controller identity/revision and sampling temperature.
- Adapter/head files, optimizer/RNG state, pending episode state/trace, seed schedule and integrity hashes.

Entity vocabularies are episode state. Frozen canonicalizer version identifies its model/settings rather than a particular training entity list, enabling held-out inference without exposing held-out names during training. Its separate vocabulary hash identifies the current allowed map.

## Robustness experiments

`perturb` supports missing text/images, irrelevant images, explicitly supplied sentence edits and supplied contradictory sources. Selection is seeded; original records remain immutable and a derivation manifest records changed hashes. Noise/conflict edits require explicit source-grounded transformations rather than a silent text fabrication rule. Test annotations remain evaluation-only.

For component/reward studies, keep budgets, seed partitions, experts and evidence fixed and record the exact changed configuration. Reward coefficients and uncertainty/cost terms are configurable. Fixed routing and BC-checkpoint inference provide internal coordination controls. External baseline implementations and provenance-disabled system variants are not bundled; their result tables must be produced by separately documented experimental implementations.

## Validation layers

Unit and integration tests cover anchors/hashes, partition isolation, action masks, failed-call recording, schema/type rules, literal preservation, conflicts, entity discovery, frontier depth, duplicate-call rejection, revision invalidation, reward resolution, terminal-once handling, GAE, joint policy entropy, genuine LoRA updates and checkpoint restoration.

Local small Qwen3 and CLIP models exercise the actual Transformers/PEFT paths without downloads. A CPU integration test runs behavioral cloning, multiple PPO updates, checkpoint resume and inference export. Large Qwen3-VL image inference and full 8B construction quality require the corresponding supplied model assets and execution hardware. CPU test success is not evidence of the manuscript's reported F1.

Use `python -m pip freeze` to record the installed environment for each experiment. CI checks Python 3.11/3.12 on Windows and Linux; actual hosted CI results become available after pushing the repository.

The completed local checks and tested principal dependency versions are recorded in [validation.md](validation.md).

## Outputs and publication

`outputs/`, raw/processed data, model weights, private credentials, local caches and training logs are ignored by Git. Episode outputs include supporting quotes for local inspection; review source licenses before redistributing them. A public evidence release can retain identifiers and anchors while distributing only permitted source content. Checkpoint integrity hashes detect changed files; they are consistency records, not a signature from an independent authority.
