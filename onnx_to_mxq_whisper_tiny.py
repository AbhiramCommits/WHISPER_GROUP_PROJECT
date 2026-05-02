# onnx_to_mxq_whisper_tiny.py — Compile Whisper tiny ONNX → MXQ
# Run on Linux Docker: python onnx_to_mxq_whisper_tiny.py

import os
import shutil
import numpy as np
import torch
import librosa

from qubee import mxq_compile
from qubee.calibration.utils_calib import list_np_files_in_txt

SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 80
N_SAMPLES   = 30 * SAMPLE_RATE
N_AUDIO_CTX = 1500
N_STATE     = 384

ENCODER_ONNX = "whisper_tiny_encoder.onnx"
DECODER_ONNX = "whisper_tiny_decoder.onnx"
ENCODER_MXQ  = "whisper_tiny_encoder.mxq"
DECODER_MXQ  = "whisper_tiny_decoder.mxq"

CALIB_AUDIO_DIR = "cv-test/clips"
MAX_CALIB       = 50

def pad_or_trim(array, length):
    if len(array) > length: return array[:length]
    if len(array) < length: return np.pad(array, (0, length - len(array)))
    return array

def log_mel_spectrogram(audio):
    audio_t = torch.from_numpy(audio).float()
    window  = torch.hann_window(N_FFT)
    stft    = torch.stft(audio_t, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(
        librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    )
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)

def audio_to_mel(audio_path):
    try:
        array, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        return np.zeros((1, N_MELS, 3000), dtype=np.float32)
    array = array.astype(np.float32)
    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
    array = pad_or_trim(array, N_SAMPLES)
    return log_mel_spectrogram(array)[np.newaxis, :, :]  # (1, 80, 3000)

# ─── Collect audio files ───────────────────────────────────────────────────────

print(f"Scanning: {CALIB_AUDIO_DIR}")
audio_files = sorted([
    os.path.join(CALIB_AUDIO_DIR, f)
    for f in os.listdir(CALIB_AUDIO_DIR)
    if f.lower().endswith(".mp3") or f.lower().endswith(".wav")
])[:MAX_CALIB]

if not audio_files:
    raise FileNotFoundError(f"No audio files found in {CALIB_AUDIO_DIR}")
print(f"Using {len(audio_files)} clips.\n")

# ─── ENCODER ──────────────────────────────────────────────────────────────────

print("=" * 55)
print("ENCODER: building calibration data...")

ENC_CALIB_NPY = os.path.abspath("calib_whisper_tiny_encoder_npy")
ENC_CALIB_TXT = os.path.abspath("calib_whisper_tiny_encoder.txt")

if os.path.exists(ENC_CALIB_NPY):
    shutil.rmtree(ENC_CALIB_NPY)
os.makedirs(ENC_CALIB_NPY)

for i, audio_path in enumerate(audio_files):
    mel      = audio_to_mel(audio_path)          # (1, 80, 3000)
    mel_nhwc = mel[np.newaxis, :, :, :]          # (1, 1, 80, 3000)
    stem     = os.path.splitext(os.path.basename(audio_path))[0]
    np.save(os.path.join(ENC_CALIB_NPY, f"{stem}.npy"), mel_nhwc)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(audio_files)} npy files saved...")

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

# ─── DECODER ──────────────────────────────────────────────────────────────────

print("=" * 55)
print("DECODER: compiling...")
print("  Input: token_emb (1,1,384) float32 — no GatherConstant")

mxq_compile(
    model=os.path.abspath(DECODER_ONNX),
    calib_data_path="",
    feed_dict={
        "token_emb":      np.zeros((1, 1, N_STATE), dtype=np.float32),
        "audio_features": np.zeros((1, N_AUDIO_CTX, N_STATE), dtype=np.float32),
    },
    backend="onnx",
    save_path=os.path.abspath(DECODER_MXQ),
    use_random_calib=True,
    cpu_offload=True,
    optimize_option=0,
)
print(f"Saved: {DECODER_MXQ}\n")

# ─── Summary ──────────────────────────────────────────────────────────────────

print("=" * 55)
print("Conversion complete.")
for path in [ENCODER_MXQ, DECODER_MXQ]:
    if os.path.isfile(path):
        mb = os.path.getsize(path) / 1e6
        print(f"  {path}: {mb:.1f} MB")
    else:
        print(f"  {path}: NOT FOUND — check errors above")
print()
print("Copy to NPU machine:")
print(f"  {ENCODER_MXQ}, {DECODER_MXQ}")
print("  whisper_tiny_token_embedding.npy, whisper_tiny_positional_embedding.npy")
