# MOSS vendored sources

`chat_template_default.py`, `modeling_Moss.py`, and `processing_Moss.py` derive
from [OpenMOSS-Team/MOSS-Transcribe-preview-2B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/tree/c98175cb20e48bd9be4e95f6c85f2af18899f780).
Each file has the same basename at the upstream repository root. The linked
revision was checked against Starling's files; the original import commit did
not record an upstream revision.

The [model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/blob/c98175cb20e48bd9be4e95f6c85f2af18899f780/README.md)
declares `license: apache-2.0` and says the model is released under Apache-2.0.
At that revision, the repository contains no LICENSE or NOTICE file, and these
three Python files contain no copyright or license headers. This records the
upstream declaration without treating the model's license statement as a
separate, explicit code-license grant. The scope of that declaration for
redistributed Python code remains to be confirmed with upstream before package
publication.

Starling imported the files in commit
`61260b4584bbdd0a7f30125fcff1bdcf4f271f27`. Compared with the linked upstream revision,
Starling removes unused imports from the modeling and processing files. `chat_template_default.py` matches the linked upstream file exactly.

Model weights are downloaded separately. This notice attributes the vendored
source files and does not change the terms of any model weights.
