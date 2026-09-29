# Licensing and provenance

## Project code and derivative weights

The project-owned inference code in this repository and the `jevora-9b-v1`
derivative checkpoint are licensed under Apache License 2.0.
The complete license text is in [LICENSE](LICENSE). This grant applies only to
project-owned code and derivative weights, not to components listed below.

## Upstream base model, tokenizer and configuration

The base model is [`Qwen/Qwen3.5-9B-Base` at revision
`68c46c4b3498877f3ef123c856ecfde50c39f404`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/tree/68c46c4b3498877f3ef123c856ecfde50c39f404).
The upstream model card declares Apache-2.0, and the exact revision includes an
upstream [LICENSE](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/LICENSE).
The tokenizer and configuration in the package come from that revision. Their
license and notices remain those supplied by Qwen; this repository does not
relicense upstream materials.

## Runtime dependencies

Python packages installed through the locked environment, including PyTorch and
Transformers, are not bundled as project code and retain their respective
licenses. Consult each dependency's distributed license before redistributing
it.

## Non-published research materials

The full training corpus, raw source records, authored examples, evaluation
records, predictions and private training and evaluation implementation are not
published. Apache-2.0 does not grant redistribution rights for them.
