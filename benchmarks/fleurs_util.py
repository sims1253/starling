"""Shared FLEURS streaming helpers for the quantization drivers.

`google/fleurs` is the one open, streamable dataset covering all 25
European languages parakeet-tdt-0.6b-v3 supports (including ru/uk, which
MLS and VoxPopuli lack), with human transcripts and train/validation/test
splits — calibrate on train, evaluate on test, never the twain.

Datasets >= 5 note: streaming audio arrives as a lazy ``AudioDecoder``;
``get_all_samples()`` returns an ``AudioSamples`` (torch tensor ``data``
[ch, n], int ``sample_rate``). The finalization crash its streaming thread
causes at interpreter exit is why the drivers hard-exit after use.
"""

from __future__ import annotations

import numpy as np

# parakeet-tdt-0.6b-v3's 25 languages -> FLEURS config ids.
PARAKEET_V3_LANGS = [
    "bg_bg", "hr_hr", "cs_cz", "da_dk", "nl_nl", "en_us", "et_ee", "fi_fi",
    "fr_fr", "de_de", "el_gr", "hu_hu", "it_it", "lv_lv", "lt_lt", "mt_mt",
    "pl_pl", "pt_br", "ro_ro", "sk_sk", "sl_si", "es_419", "sv_se", "ru_ru",
    "uk_ua",
]


def _decode(ex) -> tuple[np.ndarray, int]:
    aud = ex["audio"]
    if hasattr(aud, "get_all_samples"):  # datasets>=5 lazy decoder
        s = aud.get_all_samples()
        array, sr = s.data.numpy(), int(s.sample_rate)  # [ch, n]
    else:
        array, sr = aud["array"], int(aud["sampling_rate"])
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:  # mono arrives as [1, n]
        array = array[0] if array.shape[0] in (1, 2) else array.reshape(-1)
    return array, sr


def _to_16k(array: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return np.ascontiguousarray(array)
    import torch
    import torchaudio.functional as AF

    out = AF.resample(torch.from_numpy(array), sr, 16000).numpy()
    return np.ascontiguousarray(out)


def fleurs_clips(budget: dict[str, int], split: str = "train"):
    """Yield ``(label, audio16k, transcript)`` for each config's clip budget.

    ``budget`` maps FLEURS config ids (e.g. ``"de_de"``) to a clip count.
    Deterministic: takes the first N examples of each split shard order.
    """
    from datasets import load_dataset

    for cfg, n in budget.items():
        try:
            ds = load_dataset("google/fleurs", cfg, split=split, streaming=True)
            got = 0
            for i, ex in enumerate(ds):
                if got >= n:
                    break
                array, sr = _decode(ex)
                if array.size < 1600:  # <0.1 s: skip degenerate clips
                    continue
                text = ex.get("transcription") or ex.get("transcript") or ""
                if not text.strip():
                    continue
                yield f"fleurs:{cfg}:{i}", _to_16k(array, sr), text.strip()
                got += 1
        except Exception as e:  # a bad config must not sink the batch
            print(f"[fleurs] WARNING: {cfg} unavailable ({type(e).__name__}: "
                  f"{str(e)[:120]}) — skipping")
