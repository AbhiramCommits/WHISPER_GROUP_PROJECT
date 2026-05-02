# onnx_to_mxq.py — Whisper Encoder + Decoder
# Run: python onnx_to_mxq.py

import os
import shutil
import numpy as np
import torch
import librosa

from qubee import mxq_compile
from qubee.calibration.utils_calib import list_np_files_in_txt

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 80
N_SAMPLES   = 30 * SAMPLE_RATE


def pad_or_trim(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) > length:
        return array[:length]
    if len(array) < length:
        return np.pad(array, (0, length - len(array)))
    return array


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    audio_t = torch.from_numpy(audio).float()
    window  = torch.hann_window(N_FFT)
    stft    = torch.stft(audio_t, N_FFT, HOP_LENGTH, window=window,
                         return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters    = torch.from_numpy(
        librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    )
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)   # (80, 3000)


def audio_to_mel(audio_path: str) -> np.ndarray:
    """Returns (1, 80, 3000) — matches ONNX model input shape."""
    try:
        array, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        return np.zeros((1, N_MELS, 3000), dtype=np.float32)
    array = array.astype(np.float32)
    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
    array = pad_or_trim(array, N_SAMPLES)
    mel   = log_mel_spectrogram(array)
    return mel[np.newaxis, :, :]  # (1, 80, 3000)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ENCODER_ONNX = "whisper_encoder.onnx"
DECODER_ONNX = "whisper_decoder.onnx"
ENCODER_MXQ  = "whisper_encoder.mxq"
DECODER_MXQ  = "whisper_decoder.mxq"

CALIB_AUDIO_DIR = "clips"
MAX_CALIB       = 200

N_AUDIO_CTX = 1500
N_STATE     = 384
SOT_TOKEN   = 50257

# ---------------------------------------------------------------------------
# Collect audio files
# ---------------------------------------------------------------------------
print(f"Scanning calibration audio: {CALIB_AUDIO_DIR}")
audio_files = sorted([
    os.path.join(CALIB_AUDIO_DIR, f)
    for f in os.listdir(CALIB_AUDIO_DIR)
    if f.lower().endswith(".mp3") or f.lower().endswith(".wav")
])[:MAX_CALIB]

if not audio_files:
    raise FileNotFoundError(f"No .mp3/.wav files found in {CALIB_AUDIO_DIR}")
print(f"Using {len(audio_files)} clips.\n")

# ---------------------------------------------------------------------------
# ENCODER calibration
#
# Two shapes are involved and must NOT be confused:
#   feed_dict shape: (1, 80, 3000)   — ONNX model input, validated at parse time
#   npy calib shape: (1, 1, 80, 3000) — qubee's internal NHWC layout after conversion,
#                                        validated at quantisation time
# The warning "shape mismatch [1,80,3000] VS [1,1,80,3000]" in previous runs was
# because both were set to the same value. They must be different.
# ---------------------------------------------------------------------------
print("=" * 55)
print("ENCODER: building calibration data...")

ENC_CALIB_NPY = os.path.abspath("calibration_encoder_npy")
ENC_CALIB_TXT = os.path.abspath("calibration_encoder.txt")

if os.path.exists(ENC_CALIB_NPY):
    shutil.rmtree(ENC_CALIB_NPY)
os.makedirs(ENC_CALIB_NPY)

for i, audio_path in enumerate(audio_files):
    mel = audio_to_mel(audio_path)               # (1, 80, 3000)
    mel_nhwc = mel[np.newaxis, :, :, :]          # (1, 1, 80, 3000) for qubee
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    np.save(os.path.join(ENC_CALIB_NPY, f"{stem}.npy"), mel_nhwc)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(audio_files)} npy files saved...")

print(f"  Saved {len(audio_files)} npy files → shape (1, 1, 80, 3000)")
list_np_files_in_txt(ENC_CALIB_NPY, ENC_CALIB_TXT)
print(f"  Manifest: {ENC_CALIB_TXT}")

print("Compiling encoder ONNX → MXQ...")
mxq_compile(
    model=os.path.abspath(ENCODER_ONNX),
    calib_data_path=ENC_CALIB_TXT,
    feed_dict={
        "mel": np.zeros((1, N_MELS, 3000), dtype=np.float32),
    },
    backend="onnx",
    save_path=os.path.abspath(ENCODER_MXQ),
    use_random_calib=False,
    cpu_offload=True,
    optimize_option=0,
)
print(f"Saved: {ENCODER_MXQ}\n")

# ---------------------------------------------------------------------------
# DECODER calibration
#
# KEY FIX: tokens shape fixed at (1, 1) — one token ID per decode step.
#
# Previous compilations were broken:
#   - dynamic_axes: Aries2 NPU does not support dynamic shapes →
#     Model_ShapeMismatched on every call
#   - shape (1, 1, 384): treated token as embedding vector instead of
#     integer ID → decoder output garbage ("sher sher", "at at at")
#
# This uses (1, 1) fixed token shape with use_random_calib=True so
# qubee uses the feed_dict shapes directly without needing sample folders.
# ---------------------------------------------------------------------------
print("=" * 55)
print("DECODER: building calibration data...")

DEC_CALIB_DIR = os.path.abspath("calibration_decoder")
if os.path.exists(DEC_CALIB_DIR):
    shutil.rmtree(DEC_CALIB_DIR)
os.makedirs(DEC_CALIB_DIR)

for i, audio_path in enumerate(audio_files):
    sample_dir = os.path.join(DEC_CALIB_DIR, f"sample_{i}")
    os.makedirs(sample_dir)

    # Fixed seq_len=1 — one token per decoder step on NPU
    rng    = np.random.RandomState(seed=i)
    tokens = np.array([[[int(rng.randint(1, 51865))]]], dtype=np.float32)  # (1, 1)
    audio_features = np.zeros((1, N_AUDIO_CTX, N_STATE), dtype=np.float32)

    np.save(os.path.join(sample_dir, "tokens.npy"),         tokens)
    np.save(os.path.join(sample_dir, "audio_features.npy"), audio_features)

print(f"  Built {len(audio_files)} samples in {DEC_CALIB_DIR}/")

print("Compiling decoder ONNX → MXQ...")
mxq_compile(
    model=os.path.abspath(DECODER_ONNX),
    calib_data_path="",          # unused when use_random_calib=True
    feed_dict={
        "tokens":         np.array([[[SOT_TOKEN]]], dtype=np.int64),  # (1, 1, 1) fixed
        "audio_features": np.zeros((1, N_AUDIO_CTX, N_STATE), dtype=np.float32),
    },
    backend="onnx",
    save_path=os.path.abspath(DECODER_MXQ),
    use_random_calib=True,   # uses feed_dict shapes directly
    cpu_offload=True,
    optimize_option=0,
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