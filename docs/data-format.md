# Data and annotation formats

Files use UTF-8 JSON Lines. Identifiers are stable strings; entity IDs and human-readable names are separate. Reject malformed records rather than omitting fields silently. The Python record constructors provide the authoritative field validation.

## Entities and schema

An entity row contains `entity_id`, `name`, `entity_type` and optional `aliases`:

```json
{"entity_id":"species:panda","name":"Giant panda","entity_type":"species","aliases":["Ailuropoda melanoleuca"]}
```

The schema is one JSON object. Predicates use exact configured IDs; entity aliases never rewrite a predicate or a literal attribute value.

```json
{"predicates":{"isA":{"head_types":["species"],"tail_types":["taxon"],"single_valued":false,"tail_is_entity":true},"hasImage":{"head_types":["species"],"tail_types":[],"single_valued":false,"tail_is_entity":false}}}
```

The complete entity catalog can include multiple partitions. Only the selected partition is exposed to its specialists and controller. Optional `dataset.shared_entity_ids` declares unpartitioned ontology nodes; it cannot include an entity assigned to any training, calibration, validation or test partition. Newly discovered entities must be returned with a type and supporting evidence; their retrieval is activated only through an accepted relation followed by `expand`.

## Sources

Required source fields:

| Field | Meaning |
| --- | --- |
| `source_id` | Stable source record ID |
| `entity_ids` | Associated seed/entity IDs; cross-partition associations are removed by splitting |
| `url` | Actual acquisition URL or permanent source identifier |
| `retrieved_at` | ISO-8601 timestamp including timezone |
| `license` | Recorded license identifier |
| `content_hash` | Lowercase SHA-256 of original bytes |
| `kind` | `text` or `image` |
| `partition` | `specialist`, `verifier`, `ppo`, `calibration`, `validation` or `test` |

Text records require `document_id` and either inline `text` or a `path`. If both are supplied they must match exactly. Source-file paths are relative to the source manifest. Read UTF-8 without newline normalization when computing hashes and offsets. Byte-order marks and CRLF characters belong to the original string and count toward character offsets.

Image records require `image_id`, `path`, original SHA-256, and `safety_status`. PEA admits only `safety_status: safe`, eligible sources, allowed licenses and images satisfying the configured resolution/integrity limits. Set that status through your media-review process; it is not inferred from image relevance. Optional width/height values are verified from the actual image during processing. Do not attach an eligible license to an image unless its source grants it.

Source anchors contain `source_id`, URL, retrieval time, hash and exactly one of:

- `document_id`, `start`, `end`, and optional exact `quote`;
- `image_id` and optional normalized region `[x0,y0,x1,y1]`.

Text spans are half-open character intervals in the complete original document. A supporting span must be contained in already retrieved evidence for that entity. Model outputs cannot cite an unretrieved page or manufacture a URL/hash. An image attribute's tail must equal its supporting image anchor's `image_id`.

## Partitions

`split` produces `splits.json` and a source manifest with explicit assignments. Outer proportions are 60% construction training, 10% calibration, 10% validation and 20% test. Training is further divided 35/25/40 into specialist, verifier and PPO entities. Integer sizes use deterministic largest-remainder allocation.

The tool groups duplicate text/images, connects source/page/hash/image duplicate identities, assigns each connected component to its earliest partition, and masks associations in other partitions. `validate` audits the resulting assignments. Review near-duplicate thresholds for the domain before freezing splits.

## Supervised specialist examples

Each JSONL row contains:

```json
{"partition":"specialist","entity_id":"species:panda","source_ids":["document:001"],"system":"Extract source-supported triples.","user":"The anchored passage and extraction instructions.","assistant":"{\"triples\":[]}","image_source_ids":[]}
```

`assistant` is the reviewed target response. Its structure must follow the generation contract in `agents/generation.py`. For visual examples, `image_source_ids` references actual image sources included in `source_ids`. Verifier examples use `partition: verifier` and the same evidence-question template as `CalibratedVerificationAgent`; the target is `yes` or `no`. Completion-only loss masks all prompt tokens.

## Calibration labels

`score-verifier` reads rows containing `partition: calibration`, `entity_id`, a full `candidate` record, `factual_label`, and `visual_label` where applicable. Candidate records contain `candidate_id`, `head`, `predicate`, `tail`, `anchors`, `modality` and optional version/lifecycle fields. For image-supported candidates, use `modality: visual` or `multimodal`.

Labels are binary integers. `score-verifier` uses the fixed verifier's conditional yes/no log-likelihoods and records its scoring signature. `calibrate` fits separate temperatures for factual and visual channels. Both channels need calibration examples for a complete multimodal run.

## Controller demonstrations

An annotated action-trace row contains `partition: ppo`, `entity_id`, `source_ids`, `context`, `operation` and `target_id`. `context` is the exact `DecisionContext.to_dict()` collected from the environment, including token IDs, object positions, feasible mask, specialist signature and snapshot hashes. Labels must select a visible, feasible operation/object pair. A stop label has a null target.

Do not build demonstrations by attaching arbitrary text to saved masks. The bounded context is the actual controller observation, and labels must be reviewed against that observation and evidence trace.

## Reference graphs and evaluation records

An annotated reference row uses `entity_id`, `partition`, `head`, `predicate`, `tail`, and optional `kind` (`relation` or `image`). PPO reference files must contain only PPO entities. Test references are supplied only to evaluation commands; they never enter controller observations or policy rewards.

Construction evaluation compares canonical IDs. Per-entity statistical rows use `comparison`, `entity_id`, `first` and `second` with paired observations. Retrieval rankings and relevance are JSON objects mapping query IDs to ordered item-ID arrays and relevant item-ID arrays respectively.

## Validation cost measurements

Rows contain `partition: validation`, `operation`, `prompt_tokens`, `completion_tokens`, `latency_seconds`, and `retrieval_calls`. Include deterministic integration operations with measured latency and zero model tokens. Unavailable usage is not equivalent to zero; incomplete measurements are rejected by `fit-costs`. The output records all selected unit weights and the measurement digest.
