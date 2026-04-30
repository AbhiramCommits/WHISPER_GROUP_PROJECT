"""
Export the trained RoPEWhisper model to ONNX format.
 
Produces two files:
  - whisper_encoder.onnx
  - whisper_decoder.onnx
 
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
# CONFIG
# ---------------------------------------------------------------------------
CHECKPOINT   = os.path.join("checkpoints", "checkpoint_922000.pt")
ENCODER_PATH = "whisper_encoder.onnx"
DECODER_PATH = "whisper_decoder.onnx"
OPSET        = 18   # use 18 — torch will upgrade anyway, cleaner to be explicit
 
N_MELS      = 80
N_AUDIO_CTX = 1500
N_STATE     = 384
N_VOCAB     = 51865
 
# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
device = "cpu"
print(f"Loading checkpoint: {CHECKPOINT}")
model = get_model().to(device)
 
checkpoint = torch.load(CHECKPOINT, map_location=device)
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
 
expected_missing    = {"decoder.output_projection.weight"}
expected_unexpected = {k for k in unexpected if k.endswith(".bias")}
real_missing        = [k for k in missing    if k not in expected_missing]
real_unexpected     = [k for k in unexpected if k not in expected_unexpected]
 
if real_missing:
    print(f"WARNING — unexpected missing keys: {real_missing}")
if real_unexpected:
    print(f"WARNING — unexpected extra keys: {real_unexpected}")
 
model.decoder.output_projection.weight = model.decoder.token_embedding.weight
print(f"Checkpoint loaded. Skipped {len(expected_unexpected)} old bias keys.")
model.eval()
print("Model ready.\n")
 
tokenizer = get_tokenizer(multilingual=False)
 
# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
 
class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
 
    def forward(self, mel):
        return self.encoder(mel)
 
 
class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
 
    def forward(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)
 
 
encoder_wrapper = EncoderWrapper(model.encoder).eval()
decoder_wrapper = DecoderWrapper(model.decoder).eval()
print(decoder_wrapper)
 
# ---------------------------------------------------------------------------
# Export Encoder
# ---------------------------------------------------------------------------
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
    external_data=False,
)
print(f"Saved: {ENCODER_PATH}")
 
print("Verifying encoder...")
onnx.checker.check_model(onnx.load(ENCODER_PATH))
enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
enc_out = enc_session.run(None, {"mel": dummy_mel.numpy()})[0]
print(f"  Output shape: {enc_out.shape}")
assert enc_out.shape == (1, N_AUDIO_CTX, N_STATE), f"Bad shape: {enc_out.shape}"
print("  Encoder OK.\n")
 
# ---------------------------------------------------------------------------
# Export Decoder — fixed seq_len=1 for NPU (Aries2) compatibility.
#
# Dynamic seq_len causes qubee to produce width=-1 in Adding layers which
# the Aries2 backend cannot handle. Since we do greedy decoding one token
# at a time on the NPU, seq_len=1 is all we need at inference time.
# The full token history is tracked in Python; only the latest token is
# fed to the decoder each step.
# ---------------------------------------------------------------------------
print("=" * 50)
print("Exporting decoder (fixed seq_len=1 for NPU)...")
 
dummy_tokens   = torch.tensor([[[tokenizer.sot]]], dtype=torch.int64)  # (1, 1)
dummy_features = torch.from_numpy(enc_out)                          # (1, 1500, 384)
 
torch.onnx.export(
    decoder_wrapper,
    (dummy_tokens, dummy_features),
    DECODER_PATH,
    opset_version=OPSET,
    input_names=["tokens", "audio_features"],
    output_names=["logits"],
    dynamic_axes={
        # batch is still dynamic, but seq_len is fixed at 1
        "audio_features": {0: "batch"},
        "logits":         {0: "batch"},
    },
    do_constant_folding=True,
    verbose=False,
    external_data=False,
)
print(f"Saved: {DECODER_PATH}")
 
print("Verifying decoder...")
onnx.checker.check_model(onnx.load(DECODER_PATH))
dec_session = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])
 
dec_out_1 = dec_session.run(
    None,
    {
        "tokens":         np.array([[[tokenizer.sot]]], dtype=np.int64),
        "audio_features": dummy_features.numpy(),
    }
)[0]
print(f"  Output shape (seq=1): {dec_out_1.shape}")
assert dec_out_1.shape == (1, 1, N_VOCAB)
print("  Decoder OK.\n")
 
# ---------------------------------------------------------------------------
# Greedy decode comparison: PyTorch vs ONNX
# ---------------------------------------------------------------------------
print("=" * 50)
print("Greedy decode comparison (PyTorch vs ONNX)...")
 
test_mel = torch.zeros(1, N_MELS, 3000)
 
# PyTorch
with torch.no_grad():
    pt_features = model.encoder(test_mel)
    pt_tokens   = [tokenizer.sot]
    for _ in range(20):
        tok_t     = torch.tensor([pt_tokens], dtype=torch.long)
        pt_logits = model.decoder(tok_t, pt_features)
        next_tok  = pt_logits[0, -1].argmax().item()
        if next_tok == tokenizer.eot:
            break
        pt_tokens.append(next_tok)
 
# ONNX — pass only the latest token each step (seq_len=1 fixed)
onnx_feats  = enc_session.run(None, {"mel": test_mel.numpy()})[0]
onnx_tokens = [tokenizer.sot]
for _ in range(20):
    # Feed only the most recent token — decoder is stateless so we track
    # position implicitly via the growing token list in Python
    last_tok    = np.array([[[onnx_tokens[-1]]]], dtype=np.int64)  # (1, 1)
    onnx_logits = dec_session.run(
        None, {"tokens": last_tok, "audio_features": onnx_feats}
    )[0]
    next_tok = int(onnx_logits[0, 0].argmax())
    if next_tok == tokenizer.eot:
        break
    onnx_tokens.append(next_tok)
 
print(f"  PyTorch: '{tokenizer.decode(pt_tokens[1:])}'")
print(f"  ONNX   : '{tokenizer.decode(onnx_tokens[1:])}'")
print("  Match!" if pt_tokens == onnx_tokens else "  WARNING: outputs differ.")
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
enc_mb = os.path.getsize(ENCODER_PATH) / 1e6
dec_mb = os.path.getsize(DECODER_PATH) / 1e6
print(f"\n{'='*50}")
print(f"Export complete.")
print(f"  {ENCODER_PATH}: {enc_mb:.1f} MB")
print(f"  {DECODER_PATH}: {dec_mb:.1f} MB")
print(f"  Total         : {enc_mb + dec_mb:.1f} MB")
print(f"{'='*50}")
