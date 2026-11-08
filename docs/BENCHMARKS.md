# Benchmark integration

V1.6's completed **public Decision Index 0.3** run scored **49.83**. This is
the public suite score, not the organizers' full board score. The completed
run used the selected research checkpoint. The published package is pinned at
[HF revision `545549b8`](https://huggingface.co/kadirbekovvv/jevora-9b-v1/tree/545549b8e8104f1780966ef1c13136abb32712b4),
manifest SHA256 `483a5398d29b48f252d0aa6349946433b1c0b291bad81ce66d47f1f94ea3c6db`.
The operator reported 924 exact tensors, three cold CUDA research/public
requests with maximum probability difference 0.0, and remote verification
PASS. This is bounded package parity, not a repeat of the full benchmark.

## Similar-size model comparison

The README compares five selected roughly 8–10B decision models with Jevora.
It is not an exhaustive leaderboard or a matched training experiment. All
scores below are **Public skill points (0–100)**, with chance correction and
coverage adjustment. Missing measurements have not been estimated.

| Model | Reported total/served parameters | Public | Board engine / source |
| --- | ---: | ---: | --- |
| [GLiDE 9B](https://huggingface.co/fastino/GLiDE-9B) | 7,936,684,544 | 68.77 | `glide-9b` |
| [Bespoke Nimble 9B v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) | 9,650,000,000 | 57.19 | `nimble-v3` |
| [Sieve 9B v2](https://huggingface.co/sthanika-ai/Sieve-9B-v2) | 9,653,104,368 | 53.35 | `sieve-9b-plus` |
| Jevora V1.6 | 9,127,961,088 | 49.83 | [Saved research run](../assets/v16-evidence.json) |
| [JPT-9B](https://huggingface.co/kirp/jpt-9b) | 9,653,104,368 | 46.96 | `jpt-9b` |
| [vLLM-SR Decision 2.0 Lux 9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B) | 9,650,000,000 | 46.76 | `decision-2.0-lux-9b` |

### Model comparison sources

Snapshot: the organizers' `multimodalart/jev-decision-index` Space at commit
[`e05cd04cad2b4f8d6c34aaa486ff88f1a9584942`](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/e05cd04cad2b4f8d6c34aaa486ff88f1a9584942),
generated `2026-11-06T09:05:08Z`; checked November 7, 2026.

- [Public score records, `data/v03.json`](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/e05cd04cad2b4f8d6c34aaa486ff88f1a9584942/data/v03.json):
  match `models[].engine` to the table and read `pub`. Do not read `v`,
  `rank_v` or `parts`: those describe Full 0.4, not our comparison.
  SHA256 `461908473d85078e547ad99b621da22fe5549adc6fbbb0398b8455380674f39c`.
- [Parameter counts and public panel, `data/index.json`](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/e05cd04cad2b4f8d6c34aaa486ff88f1a9584942/data/index.json):
  read `meta.served_params` and `scores.public_skill` for the same engines.
  Every selected peer has `meta.params_estimated=false`; its Public score
  matches `v03.json`. SHA256
  `66d257218ef2f6d472fb65304b5afbe2ef7d5e3d5d26364478d56a0709b9be5c`.
- Jevora's count is the recorded total from its own training metadata,
  including adapters and head. Peers' counts are board-reported, not counted
  anew here. Counting conventions can differ, and a model's nominal “9B”
  label need not equal its served count. No memory or speed equivalence follows.
- Compatibility: `suite.panel_id=decision-index-0.3`, 37 counted benchmarks,
  identical area membership and weights, and matching base-row corpus hash
  `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`
  in the saved Jevora scores and board snapshot. The snapshot's 42-benchmark
  catalog includes five benchmarks outside the index. The
  [pinned official runner documentation](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/README.md#what-changed-in-04)
  confirms that the 0.4 board retains the public 0.3 suite unchanged.

Earlier editions are excluded: for example, our earlier archived Nimble v3
score of 56.88 belongs to 0.2.1, not this Public 0.3 comparison. Full 0.4 and
private scores are not substituted. These are separate runs with different
training data and runtime conditions; the table is not a causal comparison,
an inference-speed ranking or evidence of Jevora's leaderboard acceptance.

## Recorded V1.6 measurements

Selected weights: `train-a1`, update 71,327, temperature 1 (uncalibrated).
Official runner revision:
`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.

| Measurement | Value | Unit / denominator |
| --- | ---: | --- |
| Public balanced skill | 49.83 | Index points, 0–100; not accuracy |
| Balanced raw | 62.28 | Weighted native scores, scaled to 0–100 |
| Breadth skill | 48.47 | Index points, 0–100 |
| Summary requests | 140,620 | All submitted requests |
| Summary outcomes | 140,299 answered; 321 unsupported; 0 errors | All submitted requests |
| Scoreable outcomes | 140,172 OK; 6 unsupported | 140,178 completed scoreable requests |

The 442-request difference is the official suite's excluded set. Summary
coverage (99.77%) and scoreable coverage have different denominators. The
headline is taken from the official scorer, not reconstructed as a fraction
of correct requests. It combines 37 benchmarks in five areas, with fixed area
weights, benchmark weights, chance correction and coverage adjustment.

| Area | Skill (points, 0–100) |
| --- | ---: |
| Knowledge & Reasoning | 34.38 |
| Language Understanding | 53.85 |
| Retrieval & Classification | 54.53 |
| Tools & Automation | 69.23 |
| Arts & Human Taste | 34.54 |

The [four release figures](../README.md#v16-results) use only aggregate values:
Public area skill, window mean training cross-entropy, development accuracy,
and development NLL. The loss curve covers 36 windows: 35 of 2,000 updates
and a final 1,327. Its endpoints are 0.5458 and 0.2584 nats. Dev accuracy and
NLL trajectories use 16 in-training evaluations; separate saved-weight
measurements are not substituted into those trajectories. At T=1, reload
NLL is 1.0076 versus 0.8650 before training; lower is better. These training
and selection metrics do not demonstrate independent generalization or
calibrated confidence.

[Their evidence record](../assets/v16-evidence.json) identifies the source
export, input hashes, checkpoint and plotted values. The operator export
contains parsed/re-serialized source JSON; its hash authenticates that received
export, rather than reconstructing original JSON file bytes. Raw predictions
and private research reports are not included in this repository.

Dev352 is selection data: saved-weight accuracy is 254/352 (72.16%), compared
with 235/352 before training. The selected in-memory group macro score is
65.4599%, and its per-source counts imply 256/352. Raw accuracy and group macro
weight records differently. The reload discrepancy is documented in the
[model card](MODEL_CARD.md); neither development measurement is an independent
final test. No V1.6 final-set or JevBench result is included here.

The inference entry point is `jevora.decision_index:JevoraPackageEngine`.
It contains no training or private research imports. It accepts a verified
package path, the exact `manifest.json` SHA256, and `device=cpu|cuda`.
Package files are verified before loading. There is no implicit Hub download.

## Request contract

- `choice`: preserves the supplied criteria order. Each option is rendered as
  `<key>: <value>`; structured values use compact UTF-8 JSON.
- `noul`: scores ordered options `no`, `yes` and returns `P(yes)`.
- The input is `State:\n<state>\n\nQuestion:\n<instructions>` and uses the
  package's recorded Jevora prompt, answer vocabulary and temperature.
- At most 255 options and 8192 rendered input tokens for V1.6. Exceeding
  capacity returns the runner's `Unsupported`; inputs are never truncated.
- Each question uses one forward pass. No generation, retrieval or tools.
  CUDA is synchronized at runner measurement boundaries.

The mapping matches the completed V1.6 research run. `release.sh check` in
the research repository verifies three synthetic requests through separate
cold research and public loader processes, including 255 options and Unicode.
This bounded check is not a replacement for a benchmark evaluation.

## Official runner

The completed run used
[`apolinario/decision-index@25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`](https://github.com/apolinario/decision-index/tree/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d),
edition `0.3`. The runner and frozen suite are obtained separately using the
[official instructions](https://github.com/apolinario/decision-index).
The live board's private evaluation is performed by its organizers.

With the runner and this repository available in an existing compatible
environment, reproduction uses the official CLI:

```bash
export PYTHONPATH="$PUBLIC_REPO/src:$DECISION_INDEX_RUNNER"
python -m decision_index pipeline --edition 0.3 --suite-dir "$SUITE" \
  --engine jevora.decision_index:JevoraPackageEngine \
  --option package="$PACKAGE" \
  --option package_manifest_sha256="$PACKAGE_MANIFEST_SHA256" \
  --option device=cuda --compact --out "$FRESH_OUTPUT"
```

Use the package hash supplied with its publication receipt. Record the public
code commit, immutable Hub revision, runner commit, hardware, environment and
full output directory. A full evaluation is unnecessary for merely checking
that the package loads.

## Other evaluators

The public HTTP contract is documented in the [README](../README.md):
`POST /decision`, with `context` and ordered `options`. `choice` is zero-based;
`probabilities` has the same order as the options. `make api-check` exercises
valid requests, missing options and overlong input on the selected package.

This is a candidate scorer, not a free-text generation endpoint or a claim
of compatibility with another service's API. Organizers can load the package
with `jevora.model.load_release_package` and use `jevora.model.decision`.
The [JevBench submission form](https://benchmarkheaven.com/submit) accepts
code/weights links and evaluator notes; use its regular free queue.
