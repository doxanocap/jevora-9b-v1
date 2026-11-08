# Model card: Jevora 9B V1.6

## Model identity and use

Jevora 9B V1.6 selects among supplied answers in one forward pass. It reads
text and 2–255 ordered options, returning a zero-based choice and one softmax
probability per option. It does not generate a free-text answer, rationale or
tool call. Applications provide the candidate actions or answers themselves.

This card describes the selected weights from `jevora-9b-v1.6-01/train-a1`,
update 71,327. It is the fixed model description embedded in the inference
package. Package verification, availability and exact download revisions are
recorded separately in the release README and publication receipt. A research
checkpoint measurement alone does not establish public-package parity.

Public inference repository:
[doxanocap/jevora-9b-v1](https://github.com/doxanocap/jevora-9b-v1).
Project-owned inference code and derivative weights: Apache-2.0. Upstream
components and source data retain their own terms; see `LICENSES.md`.

## Architecture

- Base: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/ef3d031a90d340a92d71f83ec17d054e100ce713),
  revision `ef3d031a90d340a92d71f83ec17d054e100ce713`, the post-trained variant.
- Readout: hidden state at the last non-padding prompt token → trainable
  bias-free 4096 × 255 decision head → first K logits → softmax(logits / T).
  Head initialization uses pretrained LM-head answer tokens.
- LoRA: rank 64, alpha 128, dropout 0; attention/Gated DeltaNet/MLP projections.
  Frozen base weights, BF16 trainable adapters and head. 174,157,824 trainable
  parameters, including 173,113,344 adapter parameters; 9,127,961,088 total.
- Prompt: `jevora-v0.2-lm-head-codes`, state plus ordered letter-coded options
  ending in `Answer:`. All task text belongs in `context`.
- 2–255 non-empty options; zero-based `choice` and one probability per option.
  All K candidates share one prompt; there is no autoregressive answer decoding.
- Maximum rendered input: 8192 tokens. No silent truncation.
- As-is probability policy: temperature 1, **uncalibrated**. No held-out
  temperature fit or associated calibrated-confidence claim.

## Training

| Property | Value |
| --- | --- |
| Training records | 380,411 |
| Presentations / scheduled model tokens | 570,616 / 220,636,058 |
| Updates / selected step | 71,327 / 71,327 |
| Global batch | 8 = microbatch 1 × accumulation 8, one GPU |
| Optimizer / LR / weight decay | AdamW / 2e-5 / 0.01 |
| Schedule | 3,566 warmup updates; cosine to 0.1 × peak LR |
| Gradient clipping | 1.0 |
| Device / peak allocated memory | One H100 PCIe / 22.3 GiB |

Training uses K-way supervised decision loss, not generated chain-of-thought.
The fixed presentation schedule is approximately 1.5 presentations per record
on average; records need not have equal exposure. Training permutes option
order with the target adjusted accordingly (seed 17).
Tasks cover classification, evidence-based decisions, reasoning and action
selection. Training inputs retain research-only verification status; this
release does not claim exhaustive semantic verification of every record.
Raw records and training tooling are outside the inference release. Their
redistribution terms remain separate from the model's Apache-2.0 license.

## Evaluation

Decision Index **public 0.3**: 49.83 balanced skill, 62.28 balanced raw,
48.47 breadth skill. Official runner revision
`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`. Out of 140,620 requests,
140,299 were answered, 321 declared unsupported and none failed technically.
The public index is a fixed weighted aggregate across 37 benchmarks and five
areas, with chance correction and coverage adjustment. It is not accuracy.
No organizers' private Full score, JevBench result or leaderboard rank is
part of this evidence. The runner's recorded median request latency is
46.06 ms on this H100 run; this is not hosted API latency or the organizers' latency test.

| Area | Chance-corrected, coverage-adjusted skill (points, 0–100) |
| --- | ---: |
| Knowledge & Reasoning | 34.38 |
| Language Understanding | 53.85 |
| Retrieval & Classification | 54.53 |
| Tools & Automation | 69.23 |
| Arts & Human Taste | 34.54 |

Sources and aggregate values for the README's two figures are recorded in
`assets/v16-evidence.json` in the public repository; no raw predictions accompany the public release.
The 140,620-request summary above counts all submitted requests. The official
scorer separately reports 140,178 completed scoreable requests: 140,172 OK and
6 unsupported. The remaining 442 requests are excluded by the suite's scoring
contract. These denominators must not be substituted for one another.

Development consists of 352 cases and was used to select the checkpoint.
Saved-weight evaluation: 235/352 before training and 254/352 after reload
(66.76% → 72.16%, +5.40 percentage points). Paired outcomes: 42 corrected,
23 regressed, 75 wrong both times, 212 correct both times.

The in-training selection score at the last update was 65.4599% group macro
accuracy. It weights groups/families differently from raw accuracy. The
in-memory result implied 256 correct; saved-weight evaluation produced 254.
The difference is confined to one development slice (67/98 in memory,
65/98 after reload);
no per-case pre-save predictions are available to establish which decisions
changed. The cause is not established; the reload result is reported separately, without claiming exact numerical parity.
Public-package parity is a separate technical check against the source
checkpoint, not a replacement for either development measurement.

Probability quality on development worsened: NLL 0.8650 → 1.0076, Brier
0.4297 → 0.4481. ECE with ten equal-width bins rose from 6.60% to 19.35%.
These are descriptive selection-set measurements, not independent estimates.
There is no independent final-set result for V1.6 in this evidence and no
controlled causal comparison against the historical Full SFT model.

## Limitations

Measured weaknesses include full-state execution and multi-condition
instructions on development, and uneven sentiment, retrieval and arts
performance on the public benchmark. A strong aggregate Tools score does not
establish reliable execution of action sequences. Accuracy depends on option
count, ordering, wording and domain. High softmax confidence can be wrong. The model
is intended for decision research and applications with their own validation;
it does not supply unconstrained generation, vision or a calibrated guarantee.
A high probability is not a safety or correctness certificate. Validate the
model on the intended domain and option construction before consequential use.

## Identity and provenance

- Source run: `jevora-9b-v1.6-01/train-a1`, selected update 71,327.
- Training input manifest SHA256:
  `e2ad37143dcba6a709e2b34e57fb40c7d2c182fe75c841480cabb17dfa14012a`.
- Source checkpoint integrity SHA256:
  `2d9d995f47529c710169ecf102fe5db4adbbc89e823655029e077a63db930002`.
- Source weights SHA256:
  `3b101210ec32f8b6a212843c8fb55ed004b8a70fb8ec4c205e899361a4f3c7f9`.
- Official scores SHA256:
  `49cb4b9819946348723780deb6377ffb9de6cacc0a93263f6339136f1c1af6aa`.

The release exporter verifies weight identity, strips machine-specific metadata
and creates new package hashes. Source checkpoint hashes do not substitute for
a package manifest or an immutable publication revision.

## Project development

The 0.6B prototype tested supervised updates and checkpoint save/reload. The
4B stage expanded source-grounded tasks and the LM-head-initialized readout.
Earlier 9B work used Full SFT and separate development, calibration and final
splits. Those historical evaluation sets and fitted temperature do not carry
over to this LoRA model.

V1.6 expanded the training corpus and completed a LoRA run on one H100, then
measured the selected research checkpoint on Decision Index public 0.3. Models,
corpora and evaluations changed between stages. This history does not isolate
a benefit from model size, LoRA, data volume or any other single factor.

## Local inference contract

The public repository supplies the custom loader and environment lock. Given
a verified local package, `make example PACKAGE=... DEVICE=cpu` performs one
decision; `make serve PACKAGE=... DEVICE=cuda` exposes a local `POST /decision`.
All task text, including the question, goes in `context`; `options` is an
ordered list of non-empty strings. The response contains `choice` and
`probabilities` in that same order. The 8192-token limit includes all rendered
text and options. Over-capacity input is rejected, not truncated.

The service is local and has no authentication. No hosted endpoint or standard
Transformers text-generation pipeline is supplied. Download instructions and
immutable package revisions belong to the release README, outside this fixed
card. Retain earlier packages to reproduce the historical model independently.
