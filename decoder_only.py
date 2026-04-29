# export_decoder_kv_fixed.py
import torch
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\daniel\Desktop\whisper_group_project")

from my_model_config import RoPEWhisper, SMALL_DIMS
import torch.nn as nn

CHECKPOINT = r"checkpoint_922000.pt"

N_LAYERS    = 4
N_HEADS     = 6
N_STATE     = 384
HEAD_DIM    = N_STATE // N_HEADS
N_AUDIO_CTX = 1500
MAX_SEQ     = 50  # fixed maximum decode length

print("Loading checkpoint...")
model = RoPEWhisper(SMALL_DIMS)
ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()
print("Loaded.")

class DecoderStepKVFixed(nn.Module):
    """
    Fixed-shape KV cache decoder for NPU compilation.
    Self-attention KV cache is fixed at MAX_SEQ length with a step counter
    to track how many tokens have been decoded so far.
    
    Inputs:
      token_embedding  (1, 1, 384)
      cross_k_0..3     (1, 1500, 384)
      cross_v_0..3     (1, 1500, 384)
      self_k_0..3      (1, MAX_SEQ, 384)  — padded cache
      self_v_0..3      (1, MAX_SEQ, 384)
      step             (1,)               — current decode step (int)
    
    Outputs:
      logits           (1, 51865)
      new_self_k_0..3  (1, MAX_SEQ, 384)  — updated cache
      new_self_v_0..3  (1, MAX_SEQ, 384)
    """
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
                self_k_3, self_v_3,
                step):

        cross_kvs     = [(cross_k_0, cross_v_0), (cross_k_1, cross_v_1),
                         (cross_k_2, cross_v_2), (cross_k_3, cross_v_3)]
        self_k_caches = [self_k_0, self_k_1, self_k_2, self_k_3]
        self_v_caches = [self_v_0, self_v_1, self_v_2, self_v_3]

        x = token_embedding  # (1, 1, 384)
        new_self_ks = []
        new_self_vs = []

        for i, block in enumerate(self.blocks):
            cross_k, cross_v = cross_kvs[i]
            self_k_cache = self_k_caches[i]  # (1, MAX_SEQ, 384)
            self_v_cache = self_v_caches[i]

            # Self attention
            residual = x
            x_ln = block.attn_ln(x)
            q = block.attn.query(x_ln)  # (1, 1, 384)
            k = block.attn.key(x_ln)    # (1, 1, 384)
            v = block.attn.value(x_ln)  # (1, 1, 384)

            # Write new k,v into cache at position `step`
            # Use scatter to write at the correct position
            step_idx = step[0]  # scalar
            new_k = self_k_cache.clone()
            new_v = self_v_cache.clone()
            new_k[:, step_idx:step_idx+1, :] = k
            new_v[:, step_idx:step_idx+1, :] = v
            new_self_ks.append(new_k)
            new_self_vs.append(new_v)

            # Attention over full cache (padding positions get low attention naturally)
            b, sq, _ = q.shape
            sk = new_k.shape[1]
            q_h = q.view(b, sq, N_HEADS, HEAD_DIM).transpose(1, 2)
            k_h = new_k.view(b, sk, N_HEADS, HEAD_DIM).transpose(1, 2)
            v_h = new_v.view(b, sk, N_HEADS, HEAD_DIM).transpose(1, 2)

            # RoPE on query only (position = step)
            half = HEAD_DIM // 2
            cos_q = block.attn.rope_cos[step_idx:step_idx+1].unsqueeze(0).unsqueeze(0)
            sin_q = block.attn.rope_sin[step_idx:step_idx+1].unsqueeze(0).unsqueeze(0)
            q1, q2 = q_h[..., :half], q_h[..., half:]
            q_h = torch.cat([q1*cos_q - q2*sin_q, q1*sin_q + q2*cos_q], dim=-1)

            # RoPE on keys for all positions 0..MAX_SEQ
            pos = torch.arange(MAX_SEQ, device=k_h.device)
            cos_k = block.attn.rope_cos[pos].unsqueeze(0).unsqueeze(0)
            sin_k = block.attn.rope_sin[pos].unsqueeze(0).unsqueeze(0)
            k1, k2 = k_h[..., :half], k_h[..., half:]
            k_h = torch.cat([k1*cos_k - k2*sin_k, k1*sin_k + k2*cos_k], dim=-1)

            scale = HEAD_DIM ** -0.5
            attn  = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            attn  = torch.softmax(attn, dim=-1)
            out   = torch.matmul(attn, v_h)
            out   = out.transpose(1, 2).contiguous().view(b, sq, N_STATE)
            out   = block.attn.out(out)
            x     = residual + out

            # Cross attention
            residual = x
            x_ln = block.cross_attn_ln(x)
            q    = block.cross_attn.query(x_ln)
            b, sq, _ = q.shape
            sk = cross_k.shape[1]
            q_h = q.view(b, sq, N_HEADS, HEAD_DIM).transpose(1, 2)
            k_h = cross_k.view(b, sk, N_HEADS, HEAD_DIM).transpose(1, 2)
            v_h = cross_v.view(b, sk, N_HEADS, HEAD_DIM).transpose(1, 2)
            attn = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)
            out  = torch.matmul(attn, v_h)
            out  = out.transpose(1, 2).contiguous().view(b, sq, N_STATE)
            out  = block.cross_attn.out(out)
            x    = residual + out

            # MLP
            x = x + block.mlp(block.mlp_ln(x))

        x = self.ln(x)
        logits = self.output_projection(x).squeeze(1)  # (1, 51865)

        return (logits,
                new_self_ks[0], new_self_vs[0],
                new_self_ks[1], new_self_vs[1],
                new_self_ks[2], new_self_vs[2],
                new_self_ks[3], new_self_vs[3])


decoder_step = DecoderStepKVFixed(model.decoder)
decoder_step.eval()

tok_dummy     = torch.zeros(1, 1, N_STATE)
cross_k_dummy = torch.zeros(1, N_AUDIO_CTX, N_STATE)
cross_v_dummy = torch.zeros(1, N_AUDIO_CTX, N_STATE)
self_k_dummy  = torch.zeros(1, MAX_SEQ, N_STATE)
self_v_dummy  = torch.zeros(1, MAX_SEQ, N_STATE)
step_dummy    = torch.zeros(1, dtype=torch.long)

input_names = ["token_embedding"] + \
    [f"cross_{t}_{i}" for i in range(N_LAYERS) for t in ["k", "v"]] + \
    [f"self_{t}_{i}"  for i in range(N_LAYERS) for t in ["k", "v"]] + \
    ["step"]

output_names = ["logits"] + \
    [f"new_self_{t}_{i}" for i in range(N_LAYERS) for t in ["k", "v"]]

print("Exporting whisper_decoder_kv_fixed.onnx...")
with torch.no_grad():
    torch.onnx.export(
        decoder_step,
        (tok_dummy,
         cross_k_dummy, cross_v_dummy,
         cross_k_dummy, cross_v_dummy,
         cross_k_dummy, cross_v_dummy,
         cross_k_dummy, cross_v_dummy,
         self_k_dummy, self_v_dummy,
         self_k_dummy, self_v_dummy,
         self_k_dummy, self_v_dummy,
         self_k_dummy, self_v_dummy,
         step_dummy),
        "whisper_decoder_kv_fixed.onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
print("Saved: whisper_decoder_kv_fixed.onnx")

import onnxruntime as ort
sess = ort.InferenceSession("whisper_decoder_kv_fixed.onnx")
print("ONNX valid. Inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
print("Outputs:")
for out in sess.get_outputs():
    print(f"  {out.name}: {out.shape}")

# Quick test
import onnx
orig = onnx.load("whisper_decoder_rank3.onnx")
for init in orig.graph.initializer:
    if init.name == "decoder.output_projection.weight":
        TOKEN_EMB = np.frombuffer(init.raw_data, dtype=np.float32).reshape(51865, N_STATE)
    if init.name == "decoder.positional_embedding":
        POS_EMB = np.frombuffer(init.raw_data, dtype=np.float32).reshape(448, N_STATE)

enc_sess      = ort.InferenceSession("whisper_encoder.onnx")
cross_kv_sess = ort.InferenceSession("whisper_cross_kv.onnx")

import librosa, glob, os
clips = sorted(glob.glob(r"cv-data\clips\*.mp3"))[:1]
if clips:
    array, sr = librosa.load(clips[0], sr=16000, mono=True)
    array = np.pad(array, (0, max(0, 480000-len(array))))[:480000].astype(np.float32)
    audio_t = torch.from_numpy(array).float()
    window = torch.hann_window(N_FFT := 400)
    stft = torch.stft(audio_t, N_FFT, 160, window=window, return_complex=True)
    mags = stft[..., :-1].abs() ** 2
    filt = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=80).astype(np.float32))
    mel = filt @ mags
    log = torch.clamp(mel, min=1e-10).log10()
    log = torch.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    mel_np = log.numpy().astype(np.float32)[np.newaxis]
else:
    mel_np = np.zeros((1, 80, 3000), dtype=np.float32)

audio_features = enc_sess.run(None, {"mel": mel_np})[0]
cross_kv = cross_kv_sess.run(None, {"audio_features": audio_features})
cross_feed = {f"cross_{t}_{i}": cross_kv[i*2+(0 if t=='k' else 1)]
              for i in range(N_LAYERS) for t in ["k","v"]}

self_kv = {f"self_{t}_{i}": np.zeros((1, MAX_SEQ, N_STATE), dtype=np.float32)
           for i in range(N_LAYERS) for t in ["k","v"]}

tokens = [50257]
print("\nTest decoding:")
for step in range(20):
    emb = (TOKEN_EMB[tokens[-1]] + POS_EMB[step]).astype(np.float32)
    tok = emb[np.newaxis, np.newaxis, :]
    feed = {"token_embedding": tok, "step": np.array([step], dtype=np.int64)}
    feed.update(cross_feed)
    feed.update(self_kv)
    outputs = sess.run(None, feed)
    logits = outputs[0][0].astype(np.float64)
    for prev in set(tokens): logits[prev] /= 2.0
    next_tok = int(np.argmax(logits))
    for i in range(N_LAYERS):
        self_kv[f"self_k_{i}"] = outputs[1+i*2]
        self_kv[f"self_v_{i}"] = outputs[1+i*2+1]
    print(f"  Step {step}: token {next_tok}")
    if next_tok == 50256: break
    tokens.append(next_tok)

import tiktoken
text = tiktoken.get_encoding("gpt2").decode([t for t in tokens[1:] if t < 50256])
print(f"Decoded: '{text}'")