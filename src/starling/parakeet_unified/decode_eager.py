"""Reference eager greedy RNN-T decode for parakeet-unified-en-0.6b.

Standard RNN-T greedy loop (the NeMo ``rnnt_greedy_decoding`` algorithm).
For each encoder frame, run the prediction net on the last emitted token (blank
= SOS on step 0), joint with the acoustic frame, argmax over the vocab; emit
non-blank tokens and repeat until a blank or ``max_symbols_per_step`` emissions,
then advance to the next encoder frame.

This is the byte-exact correctness ORACLE (the megakernel in ``decode_mega.py``
must reproduce it token-for-token). It also serves as the prefill step for the
graphed decoder (step 0 runs eager).

The loop is B=1 here; batching is the megakernel's job. Inputs are precomputed
encoder outputs (so the oracle can be exercised without the mel/encoder).
"""

from __future__ import annotations

from typing import List

import torch

from . import config as C


@torch.inference_mode()
def greedy_decode(
    encoder: torch.Tensor,
    enc_lengths: torch.Tensor,
    decoder: torch.nn.Module,
    joint: torch.nn.Module,
    *,
    blank_id: int = C.BLANK_ID,
    max_symbols_per_step: int = C.MAX_SYMBOLS_PER_STEP,
) -> List[List[int]]:
    """Greedy RNN-T decode over a batch of encoder outputs.

    Args:
        encoder: ``(B, T_enc, D)`` encoder hidden states.
        enc_lengths: ``(B,)`` valid encoder-frame counts.
        decoder: the :class:`~starling.parakeet_unified.modeling.RNNTDecoder`.
        joint: the :class:`~starling.parakeet_unified.modeling.RNNTJoint`.
        blank_id: the blank token id (1024).
        max_symbols_per_step: RNNT guard (10).

    Returns:
        list of ``B`` token-id lists (blanks excluded), one per utterance.
    """
    B = encoder.shape[0]
    device = encoder.device
    results: List[List[int]] = [[] for _ in range(B)]

    for b in range(B):
        T = int(enc_lengths[b].item())
        # LSTM state: (h, c), each (n_layers, 1, pred_hidden)
        n_layers = decoder.n_layers
        h = torch.zeros(n_layers, 1, decoder.pred_hidden, device=device,
                        dtype=encoder.dtype)
        c = torch.zeros_like(h)
        last_token = blank_id   # blank doubles as SOS on step 0
        for t in range(T):
            f = encoder[b:b + 1, t:t + 1]            # (1, 1, D) one acoustic frame
            symbols = 0
            not_blank = True
            while not_blank and symbols < max_symbols_per_step:
                tok = torch.tensor([[last_token]], device=device, dtype=torch.long)
                pred, (h, c) = decoder(tok, (h, c))  # pred (1,1,H)
                logits = joint(f, pred)              # (1,1,1,V+1)
                label = int(logits.argmax(-1).item())
                if label == blank_id:
                    not_blank = False                # advance acoustic frame
                else:
                    results[b].append(label)
                    last_token = label
                    symbols += 1
    return results


__all__ = ["greedy_decode"]
