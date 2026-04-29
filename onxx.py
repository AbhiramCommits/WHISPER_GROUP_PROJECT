"""
Export the trained RoPEWhisper model to ONNX format.

Produces two files:
  - whisper_encoder.onnx  : takes mel spectrogram, outputs audio features
  - whisper_decoder.onnx  : takes token ids + audio features, outputs logits

Run:
    python export_onnx.py

Requirements:
    pip install onnx onnxruntime
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from my_model_config import get_model
from whisper.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# CONFIG — change checkpoint path here
# ---------------------------------------------------------------------------
CHECKPOINT   = "checkpoint_1000000.pt"
ENCODER_PATH = "whisper_encoder.onnx"
DECODER_PATH = "whisper_decoder.onnx"
OPSET        = 17

# Model dimensions — must match my_model_config.py
N_MELS      = 80
N_AUDIO_CTX = 1500   # frames after conv (3000 mel frames → 1500 after stride-2 conv)
N_TEXT_CTX  = 448
N_STATE     = 384
N_VOCAB     = 51865

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = "cpu"   # export on CPU — ONNX export doesn't require GPU
print(f"Loading checkpoint: {CHECKPOINT}")
model = get_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("Model loaded.\n")

tokenizer = get_tokenizer(multilingual=False)

# ---------------------------------------------------------------------------
# Wrapper classes for clean ONNX export
# We wrap encoder and decoder separately so ONNX sees simple
# input → output graphs with no Python control flow inside.
# ---------------------------------------------------------------------------

class EncoderWrapper(nn.Module):
    """
    Input:  mel  (1, 80, 3000)   — raw log-mel spectrogram
    Output: features (1, 1500, 384) — encoder hidden states
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel)


class DecoderWrapper(nn.Module):
    """
    Input:  tokens   (1, seq)         — token ids so far
            features (1, 1500, 384)   — encoder output (constant per utterance)
    Output: logits   (1, seq, 51865)  — next-token scores
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)


encoder_wrapper = EncoderWrapper(model.encoder)
decoder_wrapper = DecoderWrapper(model.decoder)
encoder_wrapper.eval()
decoder_wrapper.eval()

# ---------------------------------------------------------------------------
# Step 1 — Export Encoder
# ---------------------------------------------------------------------------
print("=" * 50)
print("Exporting encoder...")

# Dummy input: batch=1, n_mels=80, time=3000 (30 seconds at 100fps)
dummy_mel = torch.zeros(1, N_MELS, 3000)

torch.onnx.export(
    encoder_wrapper,
    dummy_mel,
    ENCODER_PATH,
    opset_version=OPSET,
    input_names=["mel"],
    output_names=["audio_features"],
    dynamic_axes={
        # batch dimension is dynamic so you can run batch inference later
        "mel":            {0: "batch"},
        "audio_features": {0: "batch"},
    },
    do_constant_folding=True,
    verbose=False,
)
print(f"Saved: {ENCODER_PATH}")

# Verify the exported encoder
print("Verifying encoder ONNX...")
onnx_model = onnx.load(ENCODER_PATH)
onnx.checker.check_model(onnx_model)
print("  ONNX check passed.")

enc_session = ort.InferenceSession(ENCODER_PATH,
                                    providers=["CPUExecutionProvider"])
enc_out = enc_session.run(
    None, {"mel": dummy_mel.numpy()}
)[0]
print(f"  Encoder output shape: {enc_out.shape}")   # expect (1, 1500, 384)
assert enc_out.shape == (1, N_AUDIO_CTX, N_STATE), \
    f"Unexpected encoder output shape: {enc_out.shape}"
print("  Encoder export OK.\n")

# ---------------------------------------------------------------------------
# Step 2 — Export Decoder
# ---------------------------------------------------------------------------
print("=" * 50)
print("Exporting decoder...")

# Dummy inputs: start-of-transcript token, plus encoder output
dummy_tokens   = torch.tensor([[tokenizer.sot]], dtype=torch.long)   # (1, 1)
dummy_features = torch.from_numpy(enc_out)                           # (1, 1500, 384)

torch.onnx.export(
    decoder_wrapper,
    (dummy_tokens, dummy_features),
    DECODER_PATH,
    opset_version=OPSET,
    input_names=["tokens", "audio_features"],
    output_names=["logits"],
    dynamic_axes={
        "tokens":         {0: "batch", 1: "seq_len"},
        "audio_features": {0: "batch"},
        "logits":         {0: "batch", 1: "seq_len"},
    },
    do_constant_folding=True,
    verbose=False,
)
print(f"Saved: {DECODER_PATH}")

# Verify the exported decoder
print("Verifying decoder ONNX...")
onnx_model = onnx.load(DECODER_PATH)
onnx.checker.check_model(onnx_model)
print("  ONNX check passed.")

dec_session = ort.InferenceSession(DECODER_PATH,
                                    providers=["CPUExecutionProvider"])
dec_out = dec_session.run(
    None,
    {
        "tokens":         dummy_tokens.numpy(),
        "audio_features": dummy_features.numpy(),
    }
)[0]
print(f"  Decoder output shape: {dec_out.shape}")   # expect (1, 1, 51865)
assert dec_out.shape[2] == N_VOCAB, \
    f"Unexpected vocab size in decoder output: {dec_out.shape}"
print("  Decoder export OK.\n")

# ---------------------------------------------------------------------------
# Step 3 — Sanity check: run greedy decode through ONNX and compare to PyTorch
# ---------------------------------------------------------------------------
print("=" * 50)
print("Running greedy decode comparison (PyTorch vs ONNX)...")

test_mel = torch.zeros(1, N_MELS, 3000)

# PyTorch decode
with torch.no_grad():
    pt_features = model.encoder(test_mel)
    pt_tokens   = [tokenizer.sot]
    for _ in range(20):
        tok_tensor = torch.tensor([pt_tokens], dtype=torch.long)
        pt_logits  = model.decoder(tok_tensor, pt_features)
        next_tok   = pt_logits[0, -1].argmax().item()
        if next_tok == tokenizer.eot:
            break
        pt_tokens.append(next_tok)

# ONNX decode
onnx_features = enc_session.run(None, {"mel": test_mel.numpy()})[0]
onnx_tokens   = [tokenizer.sot]
for _ in range(20):
    tok_array  = np.array([onnx_tokens], dtype=np.int64)
    onnx_logits = dec_session.run(
        None,
        {"tokens": tok_array, "audio_features": onnx_features}
    )[0]
    next_tok = int(onnx_logits[0, -1].argmax())
    if next_tok == tokenizer.eot:
        break
    onnx_tokens.append(next_tok)

pt_text   = tokenizer.decode(pt_tokens[1:])
onnx_text = tokenizer.decode(onnx_tokens[1:])
print(f"  PyTorch output : '{pt_text}'")
print(f"  ONNX output    : '{onnx_text}'")

if pt_tokens == onnx_tokens:
    print("  Outputs match exactly. Export successful.")
else:
    print("  WARNING: outputs differ. Check for non-deterministic ops.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
encoder_mb = os.path.getsize(ENCODER_PATH) / 1e6
decoder_mb = os.path.getsize(DECODER_PATH) / 1e6
print(f"\n{'='*50}")
print(f"Export complete.")
print(f"  {ENCODER_PATH}: {encoder_mb:.1f} MB")
print(f"  {DECODER_PATH}: {decoder_mb:.1f} MB")
print(f"  Total         : {encoder_mb + decoder_mb:.1f} MB")
print(f"{'='*50}")
print("\nTo use the ONNX models at inference:")
print("  1. Load whisper_encoder.onnx with onnxruntime")
print("  2. Run encoder on mel spectrogram to get audio_features")
print("  3. Loop: run decoder with growing token list until EOT")