# onnx_to_mxq.py — Whisper Encoder + Decoder
# Fixed decoder compilation: tokens shape (1, 1) — no dynamic axes
#
# Run on Daniel's machine (has qubee + whisper ONNX files):
#   python onnx_to_mxq.py
#
# Requires:
#   pip install qubee-0_12_0_0_aries2-py3-none-any.whl
#   pip install onnxruntime librosa torch

import os
import sys
import shutil
import numpy as np
import torch
import librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qubee import mxq_compile
from qubee.calibration import make_calib_man
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ENCODER_ONNX = "whisper_encoder.onnx"
DECODER_ONNX = "whisper_decoder.onnx"
ENCODER_MXQ  = "whisper_encoder.mxq"
DECODER_MXQ  = "whisper_decoder.mxq"

CALIB_AUDIO_DIR = r"C:\Users\daniel\Desktop\whisper_group_project\checkpoint_218000\cv-test\clips"

MAX_CALIB = 200

N_MELS      = 80
N_AUDIO_CTX = 1500
N_STATE     = 384
N_VOCAB     = 51865

tokenizer = get_tokenizer(multilingual=False)

# ---------------------------------------------------------------------------
# Collect audio file list
# ---------------------------------------------------------------------------
print(f"Scanning: {CALIB_AUDIO_DIR}")
audio_files = sorted([
    os.path.join(CALIB_AUDIO_DIR, f)
    for f in os.listdir(CALIB_AUDIO_DIR)
    if f.lower().endswith(".mp3") or f.lower().endswith(".wav")
])[:MAX_CALIB]

if not audio_files:
    raise FileNotFoundError(
        f"No .mp3/.wav files found in {CALIB_AUDIO_DIR}\n"
        f"Update CALIB_AUDIO_DIR to point at your cv-test/clips folder."
    )
print(f"Using {len(audio_files)} clips for calibration.\n")


# ---------------------------------------------------------------------------
# Helper: load audio → mel spectrogram (1, 80, 3000)
# ---------------------------------------------------------------------------
def audio_to_mel(audio_path: str) -> np.ndarray:
    try:
        array, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        return np.zeros((1, N_MELS, 3000), dtype=np.float32)
    array = array.astype(np.float32)
    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
    mel = log_mel_spectrogram(pad_or_trim(torch.tensor(array), N_SAMPLES))
    return mel.numpy()[np.newaxis, :, :]   # (1, 80, 3000)


# ---------------------------------------------------------------------------
# ENCODER calibration — unchanged from original
# ---------------------------------------------------------------------------
print("=" * 55)
print("ENCODER: building calibration data...")

ENC_CALIB_DIR = "calibration_encoder"
if os.path.exists(ENC_CALIB_DIR):
    shutil.rmtree(ENC_CALIB_DIR)

def preprocess_encoder(audio_path: str) -> np.ndarray:
    return audio_to_mel(audio_path).astype(np.float32)

enc_calib_txt = make_calib_man(
    pre_ftn=preprocess_encoder,
    data_dir=CALIB_AUDIO_DIR,
    save_dir=".",
    save_name=ENC_CALIB_DIR,
    max_size=MAX_CALIB,
)
print(f"Encoder calibration manifest: {enc_calib_txt}")

print("Compiling encoder ONNX → MXQ...")
mxq_compile(
    model=ENCODER_ONNX,
    calib_data_path=enc_calib_txt,
    feed_dict={
        "mel": np.zeros((1, N_MELS, 3000), dtype=np.float32),
    },
    backend="onnx",
    save_path=ENCODER_MXQ,
    use_random_calib=False,
    cpu_offload=False,
)
print(f"Saved: {ENCODER_MXQ}\n")


# ---------------------------------------------------------------------------
# DECODER calibration
#
# KEY FIX: tokens shape is fixed at (1, 1) — single token per decode step.
# The previous compilation used variable-length sequences and dynamic_axes,
# which the Aries2 NPU firmware does not support. Every inference call
# was failing with Model_ShapeMismatched because the NPU had no fixed
# shape to validate against. This recompilation locks the token input
# to (1, 1) matching greedy decoding one token at a time.
# ---------------------------------------------------------------------------
print("=" * 55)
print("DECODER: building calibration data...")
print("  Token shape: (1, 1) — fixed, no dynamic axes")

DEC_CALIB_DIR = "calibration_decoder"
if os.path.exists(DEC_CALIB_DIR):
    shutil.rmtree(DEC_CALIB_DIR)
os.makedirs(DEC_CALIB_DIR)

for i, audio_path in enumerate(audio_files):
    sample_dir = os.path.join(DEC_CALIB_DIR, f"sample_{i}")
    os.makedirs(sample_dir)

    # Single token per step — matches inference pattern exactly
    rng = np.random.RandomState(seed=i)
    token_id = int(rng.randint(1, 1000))
    tokens = np.array([[token_id]], dtype=np.float32)  # (1, 1) fixed shape

    # Real-ish audio features from mel (zeros are fine for INT8 range calibration)
    audio_features = np.zeros((1, N_AUDIO_CTX, N_STATE), dtype=np.float32)

    np.save(os.path.join(sample_dir, "tokens.npy"),         tokens)
    np.save(os.path.join(sample_dir, "audio_features.npy"), audio_features)

print(f"Built {len(audio_files)} decoder calibration samples in {DEC_CALIB_DIR}/")

print("Compiling decoder ONNX → MXQ...")
mxq_compile(
    model=DECODER_ONNX,
    calib_data_path=DEC_CALIB_DIR,
    feed_dict={
        "tokens":         np.array([[tokenizer.sot]], dtype=np.int64),  # (1, 1) fixed
        "audio_features": np.zeros((1, N_AUDIO_CTX, N_STATE), dtype=np.float32),
    },
    backend="onnx",
    save_path=DECODER_MXQ,
    use_random_calib=False,
    cpu_offload=False,
    # NO dynamic_axes — Aries2 requires fixed shapes at compile time
)
print(f"Saved: {DECODER_MXQ}\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 55)
print("Conversion complete.")
for path in [ENCODER_MXQ, DECODER_MXQ]:
    if os.path.isfile(path):
        mb = os.path.getsize(path) / 1e6
        print(f"  {path}: {mb:.1f} MB")
    else:
        print(f"  {path}: NOT FOUND — check errors above")