"""Host check of export_prefill.py's graph math against the HF model: the
residual rows and per-layer K/V of a real prompt (padded to SB) vs the HF
Qwen3 text model's hidden states and KV cache. Runs the torch graph (not the
.tflite).  python -I check_prefill.py MODEL [SB] [S]"""
import glob
import os
import runpy
import sys

import torch

model = sys.argv[1]
SB = int(sys.argv[2]) if len(sys.argv) > 2 else 128
S = int(sys.argv[3]) if len(sys.argv) > 3 else 100
snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from transformers import Qwen3ASRForConditionalGeneration  # noqa: E402

torch.set_grad_enabled(False)
full = Qwen3ASRForConditionalGeneration.from_pretrained(snap, dtype=torch.float32).eval()
lm = full.model.language_model
ids = torch.randint(0, 150000, (1, S))
emb = lm.embed_tokens(ids)
out = lm(inputs_embeds=emb, use_cache=True, output_hidden_states=True)
NL = len(lm.layers)

# Build export_prefill's Segment without converting: run the module body up to the class.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_prefill.py")).read()
src = src[:src.index("seg = Segment().eval()")]
sys.argv = ["export_prefill.py", snap, str(SB), "0", str(NL), "/dev/null"]
g = runpy._run_module_code(src, {}, "__exp__")  # noqa: SLF001
seg = g["Segment"]().eval()
x = torch.zeros(1, SB, emb.shape[-1])
x[:, :S] = emb
h, kv = seg(x)
ref_h = out.hidden_states[-1]           # after the final norm
mine = lm.norm(h[:, :S])
rel = ((mine - ref_h).norm() / ref_h.norm()).item()
print(f"final hidden rows [0,{S}): rel-L2 {rel:.2e}")
cache = out.past_key_values
for l in (0, NL // 2, NL - 1):
    k_ref = cache.layers[l].keys[0, :, :S] if hasattr(cache, "layers") else cache.key_cache[l][0, :, :S]
    v_ref = cache.layers[l].values[0, :, :S] if hasattr(cache, "layers") else cache.value_cache[l][0, :, :S]
    k, v = kv[2 * l][:, :S], kv[2 * l + 1][:, :S]
    print(f"layer {l}: K rel {((k - k_ref).norm() / k_ref.norm()).item():.2e}  "
          f"V rel {((v - v_ref).norm() / v_ref.norm()).item():.2e}")
