"""
export_whisper_tiny_onnx.py — Export OpenAI Whisper tiny to ONNX.

Produces:
  - whisper_tiny_encoder.onnx
  - whisper_tiny_decoder.onnx  ← takes pre-embedded float (1,1,384) — no GatherConstant
  - whisper_tiny_token_embedding.npy
  - whisper_tiny_positional_embedding.npy

Run:
    python export_whisper_tiny_onnx.py

Requirements:
    pip install openai-whisper onnx onnxruntime
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import whisper

OPSET       = 18
N_MELS      = 80
N_AUDIO_CTX = 1500
N_STATE     = 384   # Whisper tiny hidden size
N_VOCAB     = 51865

ENCODER_PATH = "whisper_tiny_encoder.onnx"
DECODER_PATH = "whisper_tiny_decoder.onnx"
TOKEN_EMB_PATH = "whisper_tiny_token_embedding.npy"
POS_EMB_PATH   = "whisper_tiny_positional_embedding.npy"

# ─── Load Whisper tiny ─────────────────────────────────────────────────────────

print("Loading Whisper tiny...")
model = whisper.load_model("tiny", device="cpu")
model.eval()
print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

# ─── Wrappers ──────────────────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel)


class DecoderEmbWrapper(nn.Module):
    """
    Decoder that takes pre-embedded float instead of token IDs.
    Removes GatherConstant so Aries2 NPU can run it fully.

    Input:
        token_emb      (1, 1, N_STATE) float32 — token + positional embedding (added in Python)
        audio_features (1, N_AUDIO_CTX, N_STATE) float32

    Output:
        logits (1, 1, N_VOCAB) float32
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, token_emb, audio_features):
        # token_emb already has positional embedding added in Python
        # Pass directly to transformer blocks
        x = token_emb  # (1, 1, N_STATE)

        for block in self.decoder.blocks:
            x = block(x, audio_features)

        x = self.decoder.ln(x)
        # Use token_embedding weight for output projection (tied weights)
        logits = x @ torch.transpose(self.decoder.token_embedding.weight, 0, 1)
        return logits   # (1, 1, N_VOCAB)


encoder_wrapper = EncoderWrapper(model.encoder).eval()
decoder_wrapper = DecoderEmbWrapper(model.decoder).eval()

# ─── Export Encoder ────────────────────────────────────────────────────────────

print("=" * 50)
print("Exporting encoder...")

dummy_mel = torch.zeros(1, N_MELS, 3000)

torch.onnx.export(
    encoder_wrapper,
    dummy_mel,
    ENCODER_PATH,
    opset_version=OPSET,
    input_names=["mel"],
    output_names=["audio_features"],
    dynamic_axes={
        "mel":            {0: "batch"},
        "audio_features": {0: "batch"},
    },
    do_constant_folding=True,
    verbose=False,
)
print(f"Saved: {ENCODER_PATH}")

onnx.checker.check_model(onnx.load(ENCODER_PATH))
enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
enc_out = enc_session.run(None, {"mel": dummy_mel.numpy()})[0]
print(f"  Output shape: {enc_out.shape}")
assert enc_out.shape == (1, N_AUDIO_CTX, N_STATE), f"Unexpected: {enc_out.shape}"
print("  Encoder OK.\n")

# ─── Export Decoder ────────────────────────────────────────────────────────────

print("=" * 50)
print("Exporting decoder (pre-embedded input, no GatherConstant)...")

# SOT token embedding as dummy input
sot_id  = model.decoder.token_embedding.weight.shape[0] - 5  # sot token
sot_emb = model.decoder.token_embedding(
    torch.tensor([[sot_id]])
).detach()   # (1, 1, N_STATE)

dummy_features = torch.from_numpy(enc_out)   # (1, N_AUDIO_CTX, N_STATE)

torch.onnx.export(
    decoder_wrapper,
    (sot_emb, dummy_features),
    DECODER_PATH,
    opset_version=OPSET,
    input_names=["token_emb", "audio_features"],
    output_names=["logits"],
    dynamic_axes={
        "audio_features": {0: "batch"},
        "logits":         {0: "batch"},
    },
    do_constant_folding=True,
    verbose=False,
)
print(f"Saved: {DECODER_PATH}")

onnx.checker.check_model(onnx.load(DECODER_PATH))
dec_session = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])
dec_out = dec_session.run(
    None,
    {"token_emb": sot_emb.numpy(), "audio_features": dummy_features.numpy()}
)[0]
print(f"  Output shape: {dec_out.shape}")
assert dec_out.shape == (1, 1, N_VOCAB), f"Unexpected: {dec_out.shape}"
print("  Decoder OK.\n")

# Check no GatherConstant in decoder ONNX
decoder_model = onnx.load(DECODER_PATH)
gather_ops = [n for n in decoder_model.graph.node if n.op_type == "Gather"]
print(f"  GatherConstant ops in decoder: {len(gather_ops)} (should be 0)")

# ─── Save embedding tables ─────────────────────────────────────────────────────

print("=" * 50)
token_emb = model.decoder.token_embedding.weight.detach().numpy()  # (51865, 384)
pos_emb   = model.decoder.positional_embedding.detach().numpy()     # (448, 384)

np.save(TOKEN_EMB_PATH, token_emb)
np.save(POS_EMB_PATH,   pos_emb)
print(f"Saved: {TOKEN_EMB_PATH}  {token_emb.shape}")
print(f"Saved: {POS_EMB_PATH}    {pos_emb.shape}")

# ─── Summary ───────────────────────────────────────────────────────────────────

import os
print(f"\n{'='*50}")
print("Export complete.")
for p in [ENCODER_PATH, DECODER_PATH, TOKEN_EMB_PATH, POS_EMB_PATH]:
    mb = os.path.getsize(p) / 1e6
    print(f"  {p}: {mb:.1f} MB")
print()
print("Next: copy all 4 files to Linux Docker and run onnx_to_mxq_whisper_tiny.py")