# Model card: jevora-9b-v1

## Summary

`jevora-9b-v1` is an open-weight research decision model. It selects one of
2–255 ordered textual options from model-visible context and returns its
zero-based `choice` plus `K` probabilities. It is not a general chat model and
is not represented as a fully reproducible training release. Project code and
derivative weights are licensed under Apache-2.0; the corpus and private
training and evaluation implementation are not published.

The public inference code is at
[doxanocap/jevora-9b-v1](https://github.com/doxanocap/jevora-9b-v1). The model
package is distributed through
[kadirbekovvv/jevora-9b-v1](https://huggingface.co/kadirbekovvv/jevora-9b-v1).

## Architecture and inference contract

- Base: `Qwen/Qwen3.5-9B-Base`, pinned revision
  `68c46c4b3498877f3ef123c856ecfde50c39f404`.
- Readout: `Qwen last hidden state -> Linear(4096 -> 255) -> first K logits
  -> softmax(logits / temperature)`. The decision head was initialized from
  pretrained LM-head rows; full fine-tuning updated the backbone and head.
- Prompt version: `jevora-v0.2-lm-head-codes`. The submitted `context` is
  rendered first, then options in submitted order under single-token letter
  codes, ending in `Answer:`. The API does not interpret a separate question
  field; include all task text visible to the model in `context`.

  ```text
  State:
  <context>

  Question:
  Choose the best matching option.

  Options:
  A: <options[0]>
  B: <options[1]>
  ...

  Return only the letter code of the best option.

  Answer:
  ```
- `max_options=255`; each option must be a non-empty string. `choice` is the
  zero-based index into the original option array.
- `max_length=1024` after rendering and tokenization. Inputs exceeding this
  limit return an error; determining text is not silently truncated.

## Intended use and limitations

Use for controlled multiple-choice research and applications that can validate
their own option set. Do not use as the sole basis for high-stakes decisions,
for unconstrained generation, or where option order may be changed without
consideration. Probabilities are model scores calibrated on one held-out split,
not guarantees of truth or human confidence. Accuracy can vary by source,
number of options, prompt, domain and option order.

## Training and data boundary

The released checkpoint was selected at step 4,000 of a 4,500-update
weights-only restart from an earlier step-500 checkpoint. This is not an exact
optimizer-state resume. Recipe: full SFT, BF16, AdamW, learning rate `2e-5`,
weight decay `0.01`, global batch `8`, gradient clipping `1.0`, gradient
checkpointing, seed `17`, two epochs and a 5% warmup schedule.

Training used a 20,000-record research corpus with high-level coverage of
intent routing, passage QA and natural-language inference, including public
source-derived examples and independently authored contrast cases. The corpus,
its source records, training pipeline and raw evaluation examples are not
published here. This repository makes no claim of rights to redistribute that
corpus.

Development, calibration and final were distinct frozen 600-case splits with
150 cases each from BANKING77, CLINC150, BoolQ and WANLI. Checkpoint selection
used unweighted macro development accuracy over those four sources. Calibration
and final were excluded from selection.

## Evaluation and calibration

- Reloaded development: 515/600, 85.83%, 0 technical failures.
- Independent final: 517/600, 86.17%, 0 technical failures.
- Temperature: `1.7484084899282697`, fitted only on calibration.
- On development, calibration changed no argmax choice. It reduced NLL from
  0.6945 to 0.4809, Brier from 0.2604 to 0.2309 and ECE from 0.1280 to 0.0640.
- The release package was assembled. `api_check --package` passed with
  `status: passed` for that package using the existing project runtime. A
  separate execution of this minimal public loader against the package is not
  yet recorded.

The README charts compare `jevora-9b-v1`, Jev and Qwen on the same
development IDs. This is a project-specific descriptive comparison, not an
official Qwen benchmark. Jev had 3 technical failures; they remain in the
600-case denominator. Qwen did not supply probability distributions,
so no NLL, Brier or ECE is claimed for it. Final is deliberately separate from
that dev comparison. No statistical superiority claim is made.

## Hugging Face repository and model-package format

The Hugging Face repository keeps its model card, license, notices and
aggregate figures at the root. The manifest-covered package is a distinct
subdirectory:

```text
README.md                   # HF model card, license: apache-2.0
LICENSE
NOTICE
LICENSES.md
assets/                     # aggregate figures only
jevora-9b-v1/
  checkpoint/
    model.pt                 # Qwen backbone plus decision head
    metadata.json
    integrity.json
    backbone_config/
    tokenizer/
  calibration.json           # checkpoint-bound temperature
  inference-config.json
  manifest.json              # SHA256 manifest for the complete package
  MODEL_CARD.md
```

The loader verifies the package manifest, checkpoint integrity, calibration
binding and inference metadata before loading weights. A tar archive is only a
portable backup or transfer format; it is not included in this Git repository.
The package is intentionally custom and is not advertised as compatible with
`AutoModel` or `pipeline`.

Download the repository into `MODEL_DIR` and pass
`PACKAGE="$MODEL_DIR/jevora-9b-v1"`; root-level HF files are not part of the
package manifest.

`MODEL_CARD.md` is covered by the package manifest. Replacing it inside an
already assembled package invalidates the package until the package builder (or
the same manifest-generation code) recalculates hashes and integrity is checked
again.

The BF16 weights are approximately 18 GB. Running inference also needs
additional system RAM or GPU VRAM beyond the raw weight file; no universal
minimum is asserted here.

## License and provenance status

Project code and derivative weights are licensed under
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The pinned
upstream Qwen model card separately declares Apache-2.0. See
[the exact upstream revision](https://huggingface.co/Qwen/Qwen3.5-9B-Base/tree/68c46c4b3498877f3ef123c856ecfde50c39f404)
and its [license file](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/LICENSE).
Tokenizer and backbone configuration are distributed as part of that upstream
model revision. Third-party runtime dependencies retain their own licenses.
The research corpus and source records are not included or relicensed. See
the root `LICENSE`, `LICENSES.md` and `NOTICE` files of the Hugging Face
repository for the complete project and upstream attribution notices.
