# export_kv_decoder.py
import torch
import torch.nn as nn
import numpy as np
import sys
# sys.path.insert(0, r"C:\Users\daniel\Desktop\whisper_group_project")

from my_model_config import RoPEWhisper, SMALL_DIMS
from whisper.model import ModelDimensions

CHECKPOINT = r"checkpoint_922000.pt"

N_LAYERS    = 4
N_HEADS     = 6
N_STATE     = 384
HEAD_DIM    = N_STATE // N_HEADS  # 64
N_AUDIO_CTX = 1500
N_TEXT_CTX  = 448

print("Loading checkpoint...")
model = RoPEWhisper(SMALL_DIMS)
ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()
print("Loaded.\n")

# ══════════════════════════════════════════════════════════════
# MODEL 1 — Cross-attention KV extractor
# Input:  audio_features (1, 1500, 384)
# Output: cross_k_0..3, cross_v_0..3 each (1, 1500, 384)
#         (before head split — simpler for NPU)
# ══════════════════════════════════════════════════════════════

class CrossKVExtractor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.blocks = decoder.blocks

    def forward(self, audio_features):
        outputs = []
        for block in self.blocks:
            k = block.cross_attn.key(audio_features)    # (1, 1500, 384)
            v = block.cross_attn.value(audio_features)  # (1, 1500, 384)
            outputs.extend([k, v])
        return tuple(outputs)

cross_kv_model = CrossKVExtractor(model.decoder)
cross_kv_model.eval()

af_dummy = torch.zeros(1, N_AUDIO_CTX, N_STATE)

print("Exporting whisper_cross_kv.onnx...")
torch.onnx.export(
    cross_kv_model,
    (af_dummy,),
    "whisper_cross_kv.onnx",
    input_names=["audio_features"],
    output_names=[f"cross_{t}_{i}" for i in range(N_LAYERS) for t in ["k", "v"]],
    opset_version=14,
    do_constant_folding=True,
)
print("  Saved: whisper_cross_kv.onnx")

# Verify
import onnxruntime as ort
sess = ort.InferenceSession("whisper_cross_kv.onnx")
out = sess.run(None, {"audio_features": np.zeros((1, N_AUDIO_CTX, N_STATE), dtype=np.float32)})
print(f"  Cross KV output shapes: {[o.shape for o in out]}\n")

# ══════════════════════════════════════════════════════════════
# MODEL 2 — Single decode step with KV cache
# 
# The model's kv_cache dict already handles accumulation.
# We export a fixed-T version by unrolling one step.
#
# Inputs:
#   token_embedding  (1, 1, 384)     — embedded + positional token
#   cross_k_0..3     (1, 1500, 384)  — cross-attention keys
#   cross_v_0..3     (1, 1500, 384)  — cross-attention values  
#   self_k_0..3      (1, T, 384)     — self-attention key cache
#   self_v_0..3      (1, T, 384)     — self-attention value cache
#
# Outputs:
#   logits           (1, 51865)
#   new_self_k_0..3  (1, T+1, 384)
#   new_self_v_0..3  (1, T+1, 384)
# ══════════════════════════════════════════════════════════════

class DecoderStepKV(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.blocks = decoder.blocks
        self.ln     = decoder.ln
        self.output_projection = decoder.output_projection

    def forward(self, token_embedding,
                cross_k_0, cross_v_0,
                cross_k_1, cross_v_1,
                cross_k_2, cross_v_2,
                cross_k_3, cross_v_3,
                self_k_0, self_v_0,
                self_k_1, self_v_1,
                self_k_2, self_v_2,
                self_k_3, self_v_3):

        cross_kvs = [
            (cross_k_0, cross_v_0),
            (cross_k_1, cross_v_1),
            (cross_k_2, cross_v_2),
            (cross_k_3, cross_v_3),
        ]
        self_k_caches = [self_k_0, self_k_1, self_k_2, self_k_3]
        self_v_caches = [self_v_0, self_v_1, self_v_2, self_v_3]

        x = token_embedding  # (1, 1, 384)
        new_self_ks = []
        new_self_vs = []

        for i, block in enumerate(self.blocks):
            cross_k, cross_v = cross_kvs[i]
            self_k_cache = self_k_caches[i]  # (1, T, 384)
            self_v_cache = self_v_caches[i]

            # ── Self attention ──────────────────────────────────────
            residual = x
            x_ln = block.attn_ln(x)

            q = block.attn.query(x_ln)   # (1, 1, 384)
            k = block.attn.key(x_ln)     # (1, 1, 384)
            v = block.attn.value(x_ln)   # (1, 1, 384)

            # Append new k,v to cache
            k_full = torch.cat([self_k_cache, k], dim=1)  # (1, T+1, 384)
            v_full = torch.cat([self_v_cache, v], dim=1)
            new_self_ks.append(k_full)
            new_self_vs.append(v_full)

            # Reshape for attention
            b, sq, _ = q.shape
            sk = k_full.shape[1]
            q_h = q.view(b, sq, N_HEADS, HEAD_DIM).transpose(1, 2)
            k_h = k_full.view(b, sk, N_HEADS, HEAD_DIM).transpose(1, 2)
            v_h = v_full.view(b, sk, N_HEADS, HEAD_DIM).transpos