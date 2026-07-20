"""Capture MOSS audio-encoder intermediate BF16 tensors for ggml diagnosis."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO/'src'))

def main():
 p=argparse.ArgumentParser(); p.add_argument('fixture',choices=['short','medium','long']); a=p.parse_args()
 from starling.moss.loader import load_model_and_processor
 from starling.parakeet.gpu_lock import with_gpu_lock
 outdir=REPO/'golden'/'raw'; outdir.mkdir(parents=True,exist_ok=True)
 with with_gpu_lock(session='ggml-rescue',model='MOSS-Transcribe-preview-2B',eta_min=10,note='encoder stage probe'):
  model,proc=load_model_and_processor(); enc=model.model.audio_model
  wav,sr=sf.read(REPO/'tests'/'fixtures'/f'{a.fixture}.wav'); assert sr==16000 and wav.ndim==1
  inp=proc(np.asarray(wav,dtype=np.float32)); saved={}; hooks=[]
  def hook(name):
   def f(_m,_i,o): saved[name]=(o[0] if isinstance(o,tuple) else o).detach()
   return f
  hooks.append(enc.conv2d1.register_forward_hook(hook('conv1_raw')))
  hooks.append(enc.conv2d2.register_forward_hook(hook('conv2_raw')))
  hooks.append(enc.conv2d3.register_forward_hook(hook('conv3_raw')))
  hooks.append(enc.conv_out.register_forward_pre_hook(lambda _m,i: saved.__setitem__('conv3_flat',i[0].detach())))
  hooks.append(enc.conv_out.register_forward_hook(hook('conv_out_padded')))
  hooks.append(enc.layers[0].register_forward_pre_hook(lambda _m,i: saved.__setitem__('post_conv',i[0].detach())))
  hooks.append(enc.layers[0].register_forward_hook(hook('encL0')))
  hooks.append(enc.layers[31].register_forward_hook(hook('encL31')))
  hooks.append(enc.ln_post.register_forward_hook(hook('ln_post')))
  with torch.inference_mode():
   audio=inp['audio_data'].cuda(); lens=inp['audio_data_seqlens'].cuda()
   # Public model helper follows the normative encoder path.
   final=enc(audio,lens).last_hidden_state
  for h in hooks:h.remove()
  saved['encoder_hidden']=final.detach()
  meta={}
  for name,t in saved.items():
   x=t.float().cpu().contiguous().numpy(); x.tofile(outdir/f'moss_{a.fixture}_{name}.f32');meta[name]=list(x.shape)
  (outdir/f'moss_{a.fixture}_stages.json').write_text(json.dumps(meta,indent=2)+'\n')
  print(meta)
if __name__=='__main__':main()
