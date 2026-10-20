# Method mapping and executable definitions

This implementation follows the revised manuscript's centralized PPO formulation. The earlier GRPO bookkeeping and fixed nine-agent pipeline are independent prior work and are not dependencies of this package.

## State, actions and specialists

`ConstructionState` contains the accepted graph, frontier, acquired evidence ledger, candidates/history, verifications/conflicts, budget and depth. Immutable source metadata also reside in the environment for validation; controller serialization reveals only retrieved passage content. Reference annotations remain private to PPO reward calculation.

The ten operations are `retrieve_text`, `retrieve_image`, `generate_text`, `generate_visual`, `verify`, `revise`, `accept`, `reject`, `expand`, `stop`. Stop always has a null target and zero cost. Hard masks and the executor independently check object type, budget, source anchors, schema, duplicates and depth. Low learned support does not masquerade as a deterministic hard constraint.

PEA retrieves incrementally from a fixed admissible source bundle, preserving original offsets and image identity. Its image checks use file hash, integrity, dimensions, declared licensing/safety, CLIP ranking and perceptual duplicate suppression. TCA and VGA parse complete structured responses atomically. CVfA separates hard checks from calibrated likelihoods. GIA/environment apply deterministic acceptance, rejection and frontier updates. New entity proposals must be typed, source-associated and transactionally valid.

Repeated retrieval must introduce a new anchored span/image. Generation targets only successful retrieval records. Unchanged calls cannot be repeated through renamed ledger IDs. A revision increments candidate version, preserves prior versions/scores in the trace and invalidates current verification. Literal/image values are never treated as entity-hop edges.

## Controller

The frozen Qwen3 backbone has a rank-16 LoRA adapter. The operation and value heads each have two linear layers with tanh; the conditional pointer head has two linear layers over the state embedding, target embedding and operation embedding. Its hidden size defaults to 256, an explicit implementation choice because the manuscript does not specify this width. Object embeddings use the final token of each visible object description, and the state uses the final context token.

The joint action likelihood is `P(operation|state,mask) * P(target|operation,state,mask)`. Incompatible entries have exactly zero probability. Stop's sole pointer has probability one and entropy zero. PPO entropy is `H(operation) + sum_o P(o)*H(target|o)`. Deterministic inference chooses the maximum-probability feasible pair.

## Rewards

Step reward is `lambda_h*hard + lambda_p*proxy + lambda_g*delta_quality - lambda_c*normalized_cost`.

- `hard=+1` only when an admissible new assertion is committed; recorded execution/hard-rule failures use `-1`; other successful calls use zero.
- The proxy contributes only once at accepted resolution. Rejection and abstention have no acceptance proxy.
- Visual proxy: `.55*factual + .45*visual - .20*uncertainty`. Text-only proxy: `factual - .20*uncertainty`. All three quantities remain separately visible.
- Quality is `.35Q + .20P - .20C - .15D + .10I`, with clipped change in [-1,1].
- Terminal reward is `.60*F1 - .10*Contr - .10*Dup`, used only with PPO-partition annotations.

The manuscript does not fully specify the quality denominators. Here `Q` is the sum of clipped accepted support divided by the number of resolved candidates, so rejected candidates contribute zero. `P` is provenance-bearing accepted assertions divided by accepted assertion count. `C` is the fraction of accepted assertions involved in configured single-valued predicate contradictions. `D` is repeated canonical assertions divided by accepted assertion count. `I` is the number of newly discovered frontier entities divided by all discovered frontier/expanded entities, clipped to one. Empty component denominators contribute zero to the training quality potential. Full hard gates make constructed contradiction/duplicate rates zero for the configured deterministic rules; unresolved source disagreements remain explicit conflicts and affect uncertainty.

Graph-assertion stored support is the nonnegative risk-adjusted proxy. The raw component scores and uncertainty remain available in the verification ledger. The uncertainty term can make the un-clipped proxy negative.

The verifier uses normalized binary entropy as an uncertainty estimate. For two applicable channels, uncertainty is `min(1, mean(entropy_f,entropy_v) + .5*abs(p_f-p_v) + .25*unresolved_conflict_indicator)`; for text alone the channel-disagreement term is zero. This definition operationalizes a detail not specified by the manuscript. It is part of the frozen verifier signature.

## Reported constants and explicit choices

| Setting | Value | Basis |
| --- | --- | --- |
| Discount / GAE lambda | .99 / .95 | Manuscript |
| PPO clipping / learning rate | .20 / 1e-5 | Manuscript |
| Batch / minibatch / epochs | 128 / 16 / 4 | Manuscript |
| Value / entropy coefficients | .5 / .01 | Manuscript |
| Budget / maximum depth / input tokens | 30 / 3 / 2048 | Manuscript |
| LoRA rank / decoding temperature | 16 / .1 | Manuscript |
| Abstention / uncertainty / cost coefficient | .65 / .20 / .08 | Manuscript |
| LoRA alpha / head width / gradient norm | 32 / 256 / .5 | Configurable implementation choices |
| Controller adapter targets | q/k/v/o projections | Configurable; original target list not supplied |
| Hard / proxy / quality reward weights | Explicit configuration | Not reported completely |
| Action cost table and measurement-unit weights | Validation measurements | Original table not supplied |
| Specialist fitting hyperparameters | CLI arguments | Original full settings not supplied |
| Entity resolution threshold/margin | .95 / .05 defaults | Configurable implementation choices |

`configs/experiment.yaml` supplies explicit starting values of 1 for the three unreported reward coefficients. These values are not attributed to the manuscript; select and freeze them on validation episodes.

Checkpoint metadata records measured trainable and frozen parameter counts. The manuscript's rounded parameter totals cannot uniquely determine adapter target modules and head widths. Match the original adapter configuration when those assets are available; this repository does not hard-code a reported parameter total as a measured count.

## Evaluation definitions

Construction scores use sets of canonical `(head,predicate,tail)` keys, separated into relation and image records. Selective coverage is accepted annotated candidates / all annotated candidates; selective risk is incorrect accepted candidates / accepted annotated candidates. Undefined evaluation denominators return `None` rather than a fabricated score.

ECE uses 15 equal-width bins, with probability one included in the final bin. AUROC uses average ranks for ties and is unavailable for a single-class sample. Retrieval Recall@K averages the recovered fraction of each query's relevant set; MRR uses the first relevant rank. Queries without positive annotations are excluded and the evaluated query count is reported.

Bootstrap resamples paired entity observations. Permutation tests enumerate all sign assignments for at most 15 pairs and otherwise use seeded Monte Carlo with a plus-one correction. The reported `paired_cliffs_delta` is `(positive paired differences - negative paired differences)/number_of_pairs`, with ties contributing zero; it is a paired directional dominance measure. It differs from an unpaired all-cross-pairs Cliff's delta.
