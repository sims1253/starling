# Higgs vendored sources

This directory contains code adapted from Boson AI sources. Starling changes
include local imports, reduced preprocessing dependencies, Transformers
compatibility fixes, attention-mask handling, and lint cleanup. The links below
identify upstream revisions checked against the vendored files; Starling's
original import commit did not record an upstream revision.

## Preprocessing

`higgs_audio_collator.py` and the `ChatMLDatasetSample` portion of
`chatml_dataset.py` come from `boson_multimodal` in
[Boson AI's higgs-audio repository](https://github.com/boson-ai/higgs-audio/tree/05a145bb490501b534563bf51bf2f7aa2326b271).
The original package/repository name also appears as `boson-multimodal` in
Starling's source comments.

Verified sources:

- [boson_multimodal/data_collator/higgs_audio_collator.py](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/boson_multimodal/data_collator/higgs_audio_collator.py)
- [boson_multimodal/dataset/chatml_dataset.py](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/boson_multimodal/dataset/chatml_dataset.py)
- [Upstream LICENSE](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/LICENSE), copied verbatim to [LICENSE.boson-multimodal](LICENSE.boson-multimodal).

The repository's root license is Apache License 2.0. These two source files
contain no copyright header at the linked revision. The repository separately
describes a research/noncommercial license for its v3 TTS model; that model
statement is not the source of the preprocessing code's license text.

## Remote modeling code

The Python files in `modeling/`, except Starling's `__init__.py`, derive from
[bosonai/higgs-audio-v3-stt](https://huggingface.co/bosonai/higgs-audio-v3-stt/tree/2ffd1aa39f5a1266931e405cba12e404a9f994b2).
Each upstream file has the same basename at the repository root:
`attention.py`, `common.py`, `configuration_higgs_audio.py`,
`cuda_graph_runner.py`, `custom_modules.py`, `modeling_higgs_audio.py`,
`modeling_higgs_audio_xcodec.py`, and `utils.py`.

The [model card](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/2ffd1aa39f5a1266931e405cba12e404a9f994b2/README.md)
declares `license: apache-2.0`. At that revision, the repository contains no
LICENSE or NOTICE file, and these Python files contain no copyright or license
headers. This records the upstream declaration without treating model metadata
as a separate, explicit code-license grant. The scope of that declaration for
redistributed Python code remains to be confirmed with upstream before package
publication.

Starling imported these sources in commit
`74f11584f6b5f5d7cb818b8c5797ab3fd321e2ed` and modified them afterward.
Model weights are downloaded separately; this notice covers the vendored source
files and does not change the terms of any model weights.
