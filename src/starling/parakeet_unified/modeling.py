"""Hand-built PyTorch modules for parakeet-unified-en-0.6b (NeMo-free port).

The .nemo checkpoint ships a bare ``state_dict``; this module reconstructs the
three networks (Conformer encoder, RNNT prediction net, joint) with matching
parameter names so ``load_state_dict(strict=True)`` is the byte-exact gate.
Architecture dims are in ``config.py`` (locked from the tensor shapes).

Reference: NVIDIA NeMo
``nemo/collections/asr/parts/submodules/{conformer_modules,multi_head_attention,
subsampling}.py`` and ``modules/conformer_encoder.py`` (the offline path).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# =========================================================================== #
# Conformer conv module (depthwise sep conv with BatchNorm, GeLU swish)
# =========================================================================== #
class ConformerConvModule(nn.Module):
    """Conformer convolution module: LN -> pointwise -> GLU/depthwise -> BN -> swish -> pointwise.

    Matches NeMo ``ConformerConvLayer`` (the layer conv module). The pointwise
    expansion uses a "swish-glu" half-gate (linear -> split -> glu-ish) -- NeMo
    calls it ``activation='swish'`` with a gated linear unit. We match the
    weight names ``pointwise_conv1`` (2048 out = 2x 1024 gate), ``depthwise_conv``
    (1024,1,9), ``batch_norm``, ``pointwise_conv2`` (1024 out).
    """

    def __init__(self, d_model: int = C.ENCODER_D_MODEL, kernel: int = C.CONV_KERNEL):
        super().__init__()
        # NOTE: the LayerNorm (``norm_conv``) lives at the LAYER level in NeMo
        # (key ``encoder.layers.N.norm_conv``), NOT inside this module. The conv
        # module receives an already-normalized tensor.
        # pointwise_conv1: d_model -> 2*d_model (the "GLU" expansion halves to d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        # GLU halves 2*d_model -> d_model; depthwise conv then runs on d_model
        # channels (checkpoint: depthwise_conv.weight (1024,1,9), group=1024).
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        # pointwise_conv2: d_model -> d_model (checkpoint (1024,1024,1))
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -- already LayerNorm'd by the owning ConformerLayer
        x = x.transpose(1, 2)              # (B, D, T)
        x = self.pointwise_conv1(x)        # (B, 2D, T)
        # Standard GLU (NeMo's glu_ activation): split halves, gate = sigmoid.
        x = F.glu(x, dim=1)                # (B, D, T)
        # depthwise conv with explicit (k//2, k//2) padding
        x = F.pad(x, (self.depthwise_conv.kernel_size[0] // 2,) * 2)
        x = self.depthwise_conv(x)         # (B, D, T)
        x = self.batch_norm(x)             # BatchNorm1d over channels-first (B,D,T)
        x = F.silu(x)                      # NeMo Swish
        x = self.pointwise_conv2(x)        # (B, D, T)
        return x.transpose(1, 2)           # (B, T, D)


# =========================================================================== #
# Relative-position multi-head self-attention (NeMo Conformer style)
# =========================================================================== #
class RelPosMultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative position bias (NeMo Conformer).

    Faithful port of NeMo ``MultiHeadAttention`` (rel-pos branch). Param names
    match the checkpoint: ``linear_q/k/v/out``, ``linear_pos``,
    ``pos_bias_u``/``pos_bias_v``. The ``q_with_bias_u``/``q_with_bias_v`` split
    and the ``rel_shift`` are taken verbatim from NeMo's
    ``nemo/collections/asr/parts/submodules/multi_head_attention.py``.
    """

    def __init__(self, d_model: int = C.ENCODER_D_MODEL, n_heads: int = C.ENCODER_N_HEADS):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads      # 128
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        # linear_pos: project the rel-pos embedding into the query space.
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        # pos_bias_u / pos_bias_v: (n_heads, d_head) -- NeMo inits to zeros.
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_head))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """NeMo rel_shift: x is (B, H, qlen, 2*qlen-1) -> (B, H, qlen, 2*qlen-1)
        relatively-shifted."""
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))           # (b,h,qlen,pos_len+1)
        x = x.view(b, h, -1, qlen)         # (b,h,pos_len+1,qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), pos_emb: (B, 2T-1, D)
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.linear_q(x).view(B, T, H, Dh)               # (B,T,H,Dh)
        k = self.linear_k(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        v = self.linear_v(x).view(B, T, H, Dh).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(B, -1, H, Dh).transpose(1, 2)  # (B,H,2T-1,Dh)
        # q + bias, then to (B,H,T,Dh)
        q_u = (q + self.pos_bias_u).transpose(1, 2)          # (B,H,T,Dh)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))   # (B,H,T,T)
        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))   # (B,H,T,2T-1)
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]  # (B,H,T,T)
        scores = (matrix_ac + matrix_bd) / math.sqrt(Dh)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                            # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.linear_out(out)


# =========================================================================== #
# One Conformer layer (macaron: FF1 + MHA + conv + FF2)
# =========================================================================== #
class ConformerLayer(nn.Module):
    """Macaron-net Conformer block: 0.5*FF1 + MHA + conv + 0.5*FF2 + residual+LN.

    Matches NeMo ``ConformerLayer`` with ``feed_forward_storage_multiplier=1``
    and two feed-forward blocks (feed_forward1 / feed_forward2) -- the
    "squeeze-and-excitation" / macaron variant the checkpoint uses. LayerNorms
    named ``norm_feed_forward1/2``, ``norm_self_att``, ``norm_out``.
    """

    def __init__(self, d_model: int = C.ENCODER_D_MODEL,
                 ff_expansion: int = C.FEED_FORWARD_EXPANSION,
                 n_heads: int = C.ENCODER_N_HEADS,
                 kernel: int = C.CONV_KERNEL,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = _PositionwiseFeedForward(d_model, ff_expansion)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPosMultiHeadAttention(d_model, n_heads)
        # NOTE: norm_conv lives at the LAYER level (not in the conv module) to
        # match the checkpoint's ``layers.N.norm_conv`` key path.
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = _PositionwiseFeedForward(d_model, ff_expansion)
        self.norm_out = nn.LayerNorm(d_model)
        self.ff_scale = 0.5

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        # macaron FF1
        x = x + self.ff_scale * self.feed_forward1(self.norm_feed_forward1(x))
        # self-attn
        x = x + self.self_attn(self.norm_self_att(x), pos_enc)
        # conv
        x = x + self.conv(self.norm_conv(x))
        # macaron FF2
        x = x + self.ff_scale * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class _PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NeMo uses SwiGLU-style? No -- standard FFN with swish activation.
        return self.linear2(F.silu(self.linear1(x)))


# =========================================================================== #
# Positional encoding (rel-pos sinusoidal for Conformer)
# =========================================================================== #
class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative-pos encoding producing (1, 2T-1, D).

    Faithful port of NeMo ``RelPositionalEncoding``: positions run from
    ``(L-1)`` DOWN to ``-(L-1)`` (descending), div_term uses log(INF_VAL=10000),
    and the center slice for length L is ``[center-L+1 : center+L]``.
    """

    INF_VAL = 10000.0

    def __init__(self, d_model: int = C.ENCODER_D_MODEL, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        # Build the full table once: positions (max_len-1) down to -(max_len-1).
        positions = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(2 * max_len - 1, d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(self.INF_VAL) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, 2*max_len-1, D)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        # NeMo: center_pos = pe.size(1)//2 + 1; slice [center-T : center+T-1]
        center = self.pe.size(1) // 2 + 1
        pos_emb = self.pe[:, center - T: center + T - 1]
        return pos_emb.to(device)        # (1, 2T-1, D)


# =========================================================================== #
# Pre-encode subsampling (ConvSubsampling: 2 strided convs + out Linear)
# =========================================================================== #
class ConvSubsampling(nn.Module):
    """x8-time Conv2D subsampling (3 strided stages) + Linear(4096 -> 1024).

    This is NeMo's streaming-capable ``StrideConformerConv``/``CovNar``-style
    subsampler used by parakeet-unified. It is a STRICTLY SEQUENTIAL chain of
    three stride-2 conv blocks (NOT the two-conv ConvSubsampling of older
    parakeets). Each block = (pointwise/strided-conv -> 1x1 mix -> relu), with
    the strided conv being depthwise (group=256). Confirmed against the sherpa
    ONNX node graph:

        conv.0: Conv2d(1 -> 256, 3x3, s2, pad=1)            # pointwise on raw input
        conv.1: ReLU
        conv.2: Conv2d(256 -> 256, 3x3, s2, pad=1, group=256)  # depthwise
        conv.3: Conv2d(256 -> 256, 1x1)                     # pointwise mix
        conv.4: ReLU  (ONNX conv.1_1/Relu)
        conv.5: Conv2d(256 -> 256, 3x3, s2, pad=1, group=256)  # depthwise
        conv.6: Conv2d(256 -> 256, 1x1)                     # pointwise mix
        conv.7: ReLU  (ONNX conv.1_2/Relu)

    Three stride-2 stages -> x8 on time. The feature (mel-bin) axis collapses
    F=128 -> 64 -> 32 -> 16, so the flattened (C, F') = (256, 16) = 4096 -> the
    ``out`` Linear(4096 -> 1024) input size. Matches the sherpa encoder IO
    (in (B,128,T) -> out (B,1024,T/8)).
    """

    def __init__(self, d_model: int = C.ENCODER_D_MODEL):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),         # conv.0
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),  # conv.2
            nn.Conv2d(256, 256, kernel_size=1),                             # conv.3
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),  # conv.5
            nn.Conv2d(256, 256, kernel_size=1),                             # conv.6
        ])
        self.out = nn.Linear(4096, d_model)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, F, T) -> (B, 1, T, F)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv[0](x); x = F.relu(x)        # conv.0 + conv.1/relu
        x = self.conv[1](x)                        # conv.2
        x = self.conv[2](x); x = F.relu(x)         # conv.3 + conv.4/relu
        x = self.conv[3](x)                        # conv.5
        x = self.conv[4](x); x = F.relu(x)         # conv.6 + conv.7/relu
        B, Cc, Tp, Fp = x.shape
        x = x.transpose(1, 2).contiguous().view(B, Tp, Cc * Fp)   # (B, T', 4096)
        x = self.out(x)
        # 3 stride-2 convs -> /8 on time (pad=1 keeps it exact).
        new_lens = torch.clamp((x_lengths + 7) // 8, min=1)
        return x, new_lens.to(torch.long)


# =========================================================================== #
# Full Conformer encoder
# =========================================================================== #
class ConformerEncoder(nn.Module):
    def __init__(self, n_layers: int = C.ENCODER_LAYERS,
                 d_model: int = C.ENCODER_D_MODEL,
                 n_heads: int = C.ENCODER_N_HEADS):
        super().__init__()
        self.pre_encode_conv = ConvSubsampling(d_model)
        # No LayerNorm after pre_encode: NeMo's ConvASREncoder feeds the raw
        # ConvSubsampling output straight into pos_enc (xscale) + layers. There
        # are no matching checkpoint weights, and adding one drifts the output.
        self.pre_encode_out = nn.Identity()
        self.layers = nn.ModuleList(
            [ConformerLayer(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.pos_enc = RelPositionalEncoding(d_model)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: (B, F, T) mel; lengths: (B,) sample-derived mel-frame counts.

        Returns (encoded (B, T_enc, D), encoded_lengths (B,))."""
        x, enc_lens = self.pre_encode_conv(features, lengths)
        x = self.pre_encode_out(x)
        # NeMo applies xscale = sqrt(d_model) to the encoder input (xscaling=True
        # for the Conformer family) before the layers; the rel-pos embeddings are
        # NOT scaled. We mirror that here.
        xscale = math.sqrt(self.layers[0].norm_out.normalized_shape[0])
        x = x * xscale
        pos = self.pos_enc(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, pos)
        return x, enc_lens

    def load_state_dict_prefixed(self, sd: dict, prefix: str = "encoder."):
        """Load only this encoder's keys from the flat checkpoint.

        The encoder weights live under ``encoder.pre_encode.conv.0...``,
        ``encoder.pre_encode.out...``, ``encoder.layers.N...``. We remap:
          ``encoder.pre_encode.conv.0`` -> ``pre_encode_conv.conv.0``
          ``encoder.pre_encode.out``   -> ``pre_encode_conv.out``
          ``encoder.layers.N``          -> ``layers.N``
        and drop the streaming-only conv.3/5/6 keys (offline path doesn't use
        them; verified by byte-exact match to the sherpa offline encoder).
        """
        own = {}
        # checkpoint conv indices -> our ModuleList indices
        conv_map = {"0": 0, "2": 1, "3": 2, "5": 3, "6": 4}
        for k, v in sd.items():
            if not k.startswith(prefix + "pre_encode.") and not k.startswith(prefix + "layers."):
                continue
            rel = k[len(prefix):]
            if rel.startswith("pre_encode.conv."):
                idx = rel[len("pre_encode.conv."):]
                cidx, rest = idx.split(".", 1)
                if cidx in conv_map:
                    own[f"pre_encode_conv.conv.{conv_map[cidx]}.{rest}"] = v
                # conv.1/4/7 are weightless ReLUs -- nothing to load
            elif rel.startswith("pre_encode.out."):
                own[f"pre_encode_conv.out.{rel[len('pre_encode.out.'):]}"] = v
            elif rel.startswith("layers."):
                own[rel] = v
        missing, unexpected = self.load_state_dict(own, strict=False)
        if missing:
            raise RuntimeError(f"encoder missing keys: {missing[:10]}")
        return unexpected


# =========================================================================== #
# RNNT prediction network (Embedding + 2-layer LSTM)
# =========================================================================== #
class RNNTDecoder(nn.Module):
    """Prediction net: Embedding(vocab+1, pred_hidden) + 2-layer LSTM(pred_hidden).

    Matches NeMo ``RNNTDecoder``: keys ``prediction.embed.weight`` (1025,640),
    ``prediction.dec_rnn.lstm.weight_ih_l{0,1}`` etc. (2560 = 4*640 LSTM gates).
    """

    def __init__(self, vocab_with_blank: int = C.NUM_TOKENS_WITH_BLANK,
                 pred_hidden: int = C.PRED_HIDDEN,
                 n_layers: int = C.PRED_RNN_LAYERS):
        super().__init__()
        self.pred_hidden = pred_hidden
        self.n_layers = n_layers
        # Names match the checkpoint: decoder.prediction.embed.*,
        # decoder.prediction.dec_rnn.lstm.*
        self.prediction = _PredictionBody(vocab_with_blank, pred_hidden, n_layers)

    def forward(self, tokens: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """tokens: (B, U) long (padded with blank); returns (B, U, pred_hidden)."""
        return self.prediction(tokens, state)


class _PredictionBody(nn.Module):
    """Matches the checkpoint's ``decoder.prediction.{embed,dec_rnn.lstm}`` names."""

    def __init__(self, vocab_with_blank: int, hidden: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_with_blank, hidden)
        self.dec_rnn = _LSTMWrapper(hidden, n_layers)

    def forward(self, tokens, state=None):
        x = self.embed(tokens)
        return self.dec_rnn(x, state)


class _LSTMWrapper(nn.Module):
    """nn.LSTM exposed with the checkpoint's ``dec_rnn.lstm.*`` naming."""

    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True)

    def forward(self, x, state=None):
        return self.lstm(x, state)


# =========================================================================== #
# RNNT joint network
# =========================================================================== #
class RNNTJoint(nn.Module):
    """Joint: enc_proj(enc) + pred_proj(pred) -> ReLU -> Linear -> (B,T,U,V+1).

    Matches NeMo ``RNNTJoint``: keys ``joint.enc`` (640,1024), ``joint.pred``
    (640,640), and ``joint.joint_net.2`` (1025,640) -- the only Linear in the
    ``joint_net`` Sequential sits at index 2 (indices 0/1 are an Identity/ReLU
    with no params). The activation between add and the final Linear is ReLU
    (NeMo default ``joint_activation='relu'``).
    """

    def __init__(self, d_model: int = C.ENCODER_D_MODEL,
                 joint_hidden: int = C.JOINT_HIDDEN,
                 vocab_with_blank: int = C.NUM_TOKENS_WITH_BLANK):
        super().__init__()
        self.enc = nn.Linear(d_model, joint_hidden)
        self.pred = nn.Linear(joint_hidden, joint_hidden)
        # Preserve NeMo's joint_net Sequential index so the checkpoint's
        # joint_net.2.* loads cleanly (index 0 = Identity, 1 = ReLU, 2 = Linear).
        self.joint_net = _JointNet(joint_hidden, vocab_with_blank)

    def forward(self, enc_out: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        # enc_out: (B, T, D), pred_out: (B, U, H)
        e = self.enc(enc_out).unsqueeze(2)        # (B, T, 1, H)
        p = self.pred(pred_out).unsqueeze(1)      # (B, 1, U, H)
        z = e + p                                  # (B, T, U, H) broadcast
        return self.joint_net(z)                   # (B, T, U, V+1)


class _JointNet(nn.Module):
    """NeMo joint_net with param at child index 2 (Identity, ReLU, Linear)."""

    def __init__(self, joint_hidden: int, vocab_with_blank: int):
        super().__init__()
        # child 0 and 1 are param-less; we keep them so the ``joint_net.2.*``
        # state-dict keys map straight onto self.linear via a name redirect.
        self.relu = nn.ReLU()
        self.linear = nn.Linear(joint_hidden, vocab_with_blank)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.relu(z))

    # make load_state_dict(joint_net.2.weight=...) work by remapping -> linear
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # incoming keys: {prefix}joint_net.2.{weight,bias} (set by parent)
        # our actual params live under {prefix}linear.*
        for w in ("weight", "bias"):
            ck = f"{prefix}2.{w}"
            nk = f"{prefix}linear.{w}"
            if ck in state_dict:
                state_dict[nk] = state_dict.pop(ck)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
