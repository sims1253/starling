"""SentencePiece BPE tokenizer wrapper for parakeet-unified-en-0.6b.

The model uses a 1024-piece BPE sentencepiece model; the RNNT blank (``<blk>``)
sits at index 1024 -- NOT a sentencepiece piece (it's appended after the spm
vocab by NeMo's RNNT decoding). So:

* ids 0..1023 -> sentencepiece pieces (``id_to_piece`` / ``decode_ids``).
* id 1024 -> blank, never emitted to the tokenizer; decode filters it out.

``ids_to_text`` takes a python list of int ids (the RNNT greedy output, blanks
included) and returns the detokenized string. sentencepiece's ``decode_ids``
ignores ids it doesn't know, but to be safe we filter blank + any out-of-range
id first.
"""

from __future__ import annotations

from typing import Iterable, List

import sentencepiece as spm

from . import config as C
from .loader import load_tokenizer_path


class ParakeetUnifiedTokenizer:
    """Wraps the sentencepiece model; blank handling lives at the decode layer."""

    def __init__(self, spm_path: str | None = None) -> None:
        path = spm_path or str(load_tokenizer_path())
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)
        if self.sp.get_piece_size() != C.VOCAB_SIZE:
            raise ValueError(
                f"tokenizer size {self.sp.get_piece_size()} != VOCAB_SIZE "
                f"{C.VOCAB_SIZE}; wrong tokenizer model?"
            )

    @property
    def vocab_size(self) -> int:
        # sentencepiece pieces (1024); blank is separate.
        return self.sp.get_piece_size()

    @property
    def blank_id(self) -> int:
        return C.BLANK_ID

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    def ids_to_text(self, ids: Iterable[int]) -> str:
        """Detokenize a sequence of ids (blank/out-of-range ids filtered)."""
        kept: List[int] = [
            int(i) for i in ids
            if 0 <= int(i) < C.VOCAB_SIZE  # drop blank (1024) and any junk
        ]
        if not kept:
            return ""
        return self.sp.decode_ids(kept)


__all__ = ["ParakeetUnifiedTokenizer"]
