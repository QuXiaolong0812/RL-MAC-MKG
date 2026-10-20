# Training and execution

## 1. Prepare disjoint assets

Create source and entity manifests, review image eligibility, then run `split`. Keep source files immutable after computing their hashes. Point the experiment configuration at the assigned source manifest and split manifest. Run `validate` before fitting any model.

The specialist and verifier partitions are distinct. The calibration partition fits temperatures, validation chooses coefficients/thresholds/costs, and test labels are reserved for evaluation. An explicit shared ontology may contain unpartitioned taxonomy nodes; held-out seed entity identities cannot be added to that shared list.

## 2. Fit specialists before the controller

Use `train-specialist --role textual|visual|verifier` with reviewed completion targets. Textual and visual fitting require the specialist partition; verifier fitting requires the verifier partition. The local model is frozen and a rank-16 LoRA adapter is optimized with completion-only causal language modeling. The CLI accepts explicit epochs, learning rate and accumulation steps because the manuscript does not report complete specialist fitting settings.

The verifier's target is an evidence-grounded yes/no support judgment. At execution, yes/no continuation log-likelihoods produce a binary logit, rather than accepting a generated confidence number as a calibrated probability. Its textual and visual questions are separate. Prepare training examples with the exact frozen prompt templates in `agents/verification.py`.

All adapters are fitted before controller training. During BC/PPO rollouts, local specialist models run with no gradients, and their asset/prompt/settings signatures must remain fixed. Compatible HTTP generation can be selected for textual/visual specialists; calibrated verification requires local conditional label likelihoods. HTTP calls use one attempt and explicit timeouts. Changing a remotely served checkpoint behind an unchanged endpoint is outside the local file-integrity mechanism, so use a versioned model deployment.

## 3. Calibrate and fit costs

```bash
rl-mac-mkg score-verifier --config configs/experiment.yaml --candidates data/processed/calibration_candidates.jsonl --output outputs/calibration_scores.jsonl
rl-mac-mkg calibrate --scores outputs/calibration_scores.jsonl --output models/calibration/verifier.json
```

This creates separate factual/visual temperatures bound to the frozen scoring model, prompts, schema and canonicalization settings. Temperatures are fitted by bounded one-dimensional NLL minimization in log-temperature space. The threshold `0.65` is a policy abstention feature/default fixed-route decision threshold, separate from deterministic provenance/schema masks.

Collect actual per-operation measurements on validation evidence and run `fit-costs` with explicit token/time/retrieval unit weights. The complete nine-operation table is normalized by its maximum and floored at 0.02; stopping is zero. Do not fill absent measurements with invented timings.

## 4. Behaviorally clone the controller

```bash
rl-mac-mkg train-controller --config configs/experiment.yaml --stage bc --demonstrations data/processed/controller_demonstrations.jsonl --epochs 4 --output outputs/controller-bc
```

The two factors of each annotated operation/object label receive supervised negative log-likelihood loss under their stored feasible mask. Qwen3 base weights remain frozen. The LoRA adapter, operation head, conditional pointer head and value head are represented in the checkpoint; the value head starts learning through subsequent PPO returns. Demonstrations belong to the PPO training entity partition and identify their sources and specialist versions.

Controller context has a maximum of 2,048 tokens. Packing prioritizes unresolved candidate facts, scores and conflicts, then frontier/entity/evidence objects. Only visible target IDs remain in the action mask. Full source documents in the environment are not exposed before retrieval.

## 5. PPO

```bash
rl-mac-mkg train-controller --config configs/experiment.yaml --stage ppo --from-bc outputs/controller-bc --output outputs/controller-ppo
```

Rollouts store exact token inputs, masks, object order, log probabilities, values, state/evidence hashes, specialist versions, and collecting controller identity/revision/temperature. PPO uses the joint operation-plus-pointer likelihood. Advantages are normalized within each rollout batch; entropy includes the conditional pointer entropy for every feasible operation.

Batches normally contain 128 transitions. A segment can end inside an episode; its final value is explicitly bootstrapped from the next state. Terminated episodes use zero bootstrap. A final shorter batch is processed when the requested seed schedule is exhausted. The annotated terminal reward is added once to the episode's final transition. No remaining segment from an old policy is reused after an update.

Optimization keeps dropout disabled. When gradient checkpointing is enabled, encoder training mode activates checkpoint computation without changing collection/recomputation probabilities. The optimizer contains only trainable controller parameters.

## 6. Resume

```bash
rl-mac-mkg train-controller --config configs/experiment.yaml --stage ppo --resume outputs/controller-ppo/update-000010 --output outputs/controller-resumed
```

Use a new output directory. The checkpoint includes adapter/heads, optimizer, Python/NumPy/Torch RNG states, policy identity/revision, pending environment and trace, seed schedule and data/settings signatures. Restoration rejects changed model bytes, temperatures, specialists, labels, partition assignments, environment limits, rewards or action costs. An identical backbone can be relocated and supplied by `controller.model_path`.

Checkpoint directories are written atomically and never overwritten. A completed seed schedule has no further episodes to resume; change the requested experiment through a new explicitly initialized training run.

## 7. Inference and evaluation

Set the configuration partition to `test` and select the validation-chosen checkpoint. `infer` executes the learned policy without updates. `--policy fixed` uses the same specialist implementations and hard acceptance gates with a fixed sequence. A supervised-control evaluation uses the BC checkpoint with the normal controller policy.

`evaluate` computes canonical relation/image-attribute P/R/F1. `evaluate-retrieval` scores fixed rankings. `statistics` performs paired entity bootstrap, permutation tests, Holm correction and paired directional effect size. Additional graph, agreement, selective and calibration metrics are available as Python functions in `evaluation/`.

Use the same five seeds and frozen evidence/specialists across compared systems. Report the evaluation unit and aggregation method: construction set scores, entity macro means, domain macro means and seed SD are different quantities.
