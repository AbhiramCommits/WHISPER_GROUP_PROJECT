# test_kv_inference.py
# Tests KV cache decoder on CPU using ONNX Runtime
# Run this to verify correct decoding before NPU compilation

import numpy as np
import onnxruntime as ort
import librosa
import torch
import sys
sys.path.insert(0, r"C:\Users\daniel\Desktop\whisper_group_project")

N_LAYERS    = 4
N_HEADS     = 6
N_STATE     = 384
HEAD_DIM    = N_STATE // N_HEADS
N_AUDIO_CTX = 1500
N_MELS      = 80
N_SAMPLES   = 30 * 16000
SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
SOT_TOKEN   = 50257
EOT_TOKEN   = 50256
MAX_TOKENS  = 50

# ── Load embedding weights ──────────────────────────────────────────────────
import onnx
orig = onnx.load("whisper_decoder_rank3.onnx")
TOKEN_EMB = None
POS_EMB   = None
for init in orig.graph.initializer:
    if init.name == "decoder.output_projection.weight":
        TOKEN_EMB = np.frombuffer(init.raw_data, dtype=np.float32).reshape(51865, N_STATE)
    if init.name == "decoder.positional_embedding":
        POS_EMB = np.frombuffer(init.raw_data, dtype=np.float32).reshape(448, N_STATE)

print(f"Token embedding: {TOKEN_EMB.shape}")
print(f"Positional embedding: {POS_EMB.shape}")

# ── Load ONNX sessions ──────────────────────────────────────────────────────
print("Loading ONNX models...")
enc_sess      = ort.InferenceSession("whisper_encoder.onnx")
cross_kv_sess = ort.InferenceSession("whisper_cross_kv.onnx")
dec_sess      = ort.InferenceSession("whisper_decoder_kv.onnx")
print("Loaded.\n")

# ── Audio preprocessing ─────────────────────────────────────────────────────
def pad_or_trim(a, l):
    if len(a) > l: return a[:l]
    if len(a) < l: return np.pad(a, (0, l - len(a)))
    return a

def audio_to_mel(path):
    array, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    array = pad_or_trim(array.astype(np.float32), N_SAMPLES)
    audio_t = torch.from_numpy(array).float()
    window  = torch.hann_window(N_FFT)
    stft    = torch.stft(audio_t, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    mags    = stft[..., :-1].abs() ** 2
    filt    = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32))
    mel     = filt @ mags
    log     = torch.clamp(mel, min=1e-10).log10()
    log     = torch.maximum(log, log.max() - 8.0)
    log     = (log + 4.0) / 4.0
    return log.numpy().astype(np.float32)[np.newaxis, :, :]

# ── Pick a test audio file ──────────────────────────────────────────────────
import os, glob
clips_dir = r"cv-data/clips"
clips = sorted(glob.glob(os.path.join(clips_dir, "*.mp3")))[:5]

if not clips:
    print("No audio files found — using silence")
    mel = np.zeros((1, N_MELS, 3000), dtype=np.float32)
else:
    print(f"Testing on: {os.path.basename(clips[0])}")
    mel = audio_to_mel(clips[0])

# ── Step 1: Encoder ─────────────────────────────────────────────────────────
print("Running encoder...")
audio_features = enc_sess.run(None, {"mel": mel})[0]  # (1, 1500, 384)
print(f"Audio features shape: {audio_features.shape}")

# ── Step 2: Cross-attention KV (run once) ───────────────────────────────────
print("Computing cross-attention KV...")
cross_kv_outputs = cross_kv_sess.run(None, {"audio_features": audio_features})
cross_feed = {}
for i in range(N_LAYERS):
    cross_feed[f"cross_k_{i}"] = cross_kv_outputs[i * 2]
    cross_feed[f"cross_v_{i}"] = cross_kv_outputs[i * 2 + 1]
print(f"Cross KV shapes: {cross_kv_outputs[0].shape}")

# ── Step 3: Greedy decode with KV cache ────────────────────────────────────
print("Decoding with KV cache...")

# Initialize empty self-attention KV cache
self_kv = {}
for i in range(N_LAYERS):
    self_kv[f"self_k_{i}"] = np.zeros((1, 0, N_STATE), dtype=np.float32)
    self_kv[f"self_v_{i}"] = np.zeros((1, 0, N_STATE), dtype=np.float32)

tokens = [SOT_TOKEN]

for step in range(MAX_TOKENS):
    # Embed current token + positional
    emb = (TOKEN_EMB[tokens[-1]] + POS_EMB[step]).astype(np.float32)
    tok = emb[np.newaxis, np.newaxis, :]  # (1, 1, 384)

    # Build feed dict
    feed = {"token_embedding": tok}
    feed.update(cross_feed)
    feed.update(self_kv)

    # Run decoder step
    outputs = dec_sess.run(None, feed)

    logits = outputs[0]  # (1, 51865)
    l = logits[0].astype(np.float64)
    for prev in set(tokens):
        l[prev] /= 2.0
    next_tok = int(np.argmax(l))

    # Update self-attention KV cache with new outputs
    for i in range(N_LAYERS):
        self_kv[f"self_k_{i}"] = outputs[1 + i * 2]
        self_kv[f"self_v_{i}"] = outputs[1 + i * 2 + 1]

    print(f"  Step {step:3d}: token {next_tok:6d}  cache_len={self_kv['self_k_0'].shape[1]}")

    if next_tok == EOT_TOKEN:
        print("  EOT reached.")
        break
    tokens.append(next_tok)

# ── Decode tokens to text ───────────────────────────────────────────────────
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
valid = [t for t in tokens[1:] if t < EOT_TOKEN]
text = tokenizer.decode(valid)
print(f"\nDecoded text: '{text}'")
print(f"Token count: {len(tokens)}")