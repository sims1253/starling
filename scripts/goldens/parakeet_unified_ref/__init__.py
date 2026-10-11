"""NeMo-free PyTorch reference for nvidia/parakeet-unified-en-0.6b.

Mel frontend, Conformer encoder, RNN-T prediction net + joint, and the eager
greedy decode used to capture goldens without ``nemo_toolkit`` (see
``PORTING_NOTES.md``; native port: issue #358).
"""
