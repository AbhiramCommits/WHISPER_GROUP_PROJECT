"""
NPU inference script — Whisper Encoder + Decoder on Mobilint ARIES NPU
Measures WER and latency using maccel runtime.

Usage:
    python npu_infer.py

Requirements:
    pip install maccel-0_30_1-cp312-cp312-win_amd64.whl
    pip install jiwer librosa

Files needed in the same folder:
    whisper_encoder.mxq
    whisper_decoder.mxq
    cv-test/test.tsv
    cv-test/clips/
"""

import os
import csv
import time
import numpy as np
import librosa
import jiwer

import maccel

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ENCODER_MXQ = "whisper_encoder.mxq"
DECODER_MXQ = "whisper_decoder.mxq"

CV_ROOT      = r"C:\Users\daniel\Desktop\whisper_group_project\checkpoint_218000\cv-test"
CV_TEST_TSV  = os.path.join(CV_ROOT, "test.tsv")
CV_CLIPS_DIR = os.path.join(CV_ROOT, "clips")

# Set to None to run all 16k examples, or an int to run a subset (faster)
MAX_EXAMPLES = 500

# Greedy decode stops at EOT or this many tokens
MAX_DECODE_TOKENS = 200

# Whisper audio constants (no whisper package needed)
SAMPLE_RATE  = 16000
N_FFT        = 400
HOP_LENGTH   = 160
N_MELS       = 80
N_SAMPLES    = 30 * SAMPLE_RATE   # 480000 = 30 seconds of audio

# Whisper English-only tokenizer constants
SOT_TOKEN    = 50257   # <|startoftranscript|>
EOT_TOKEN    = 50256   # <|endoftext|>

# ---------------------------------------------------------------------------
# Self-contained audio preprocessing (no whisper package import)
# ---------------------------------------------------------------------------

def pad_or_trim(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) > length:
        return array[:length]
    if len(array) < length:
        return np.pad(array, (0, length - len(array)))
    return array


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Returns (80, 3000) log-mel spectrogram matching Whisper preprocessing."""
    import torch
    audio_t = torch.from_numpy(audio).float()
    window  = torch.hann_window(N_FFT)
    stft    = torch.stft(audio_t, N_FFT, HOP_LENGTH, window=window,
                         return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters    = _mel_filters()
    mel_spec   = torch.from_numpy(filters) @ magnitudes
    log_spec   = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec   = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec   = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)   # (80, 3000)


def _mel_filters() -> np.ndarray:
    filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    return filters.astype(np.float32)


def audio_to_mel(audio_path: str) -> np.ndarray:
    """Load audio and return (1, 80, 3000) mel spectrogram."""
    array, sr = librosa.load(audio_path, sr=None, mono=True)
    array = array.astype(np.float32)
    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
    array = pad_or_trim(array, N_SAMPLES)
    mel   = log_mel_spectrogram(array)
    return mel[np.newaxis, :, :]   # (1, 80, 3000)


# ---------------------------------------------------------------------------
# Minimal tokenizer decoder (maps token IDs → text without openai-whisper)
# Uses the tiktoken BPE used by Whisper English model.
# ---------------------------------------------------------------------------

def load_whisper_tokenizer():
    """Load the Whisper English tokenizer via tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        raise ImportError(
            "pip install tiktoken  — needed for token → text decoding"
        )


def decode_tokens(enc, token_ids: list) -> str:
    """Decode a list of integer token IDs to a string."""
    # Filter out special tokens (>= 50256)
    valid = [t for t in token_ids if t < 50256]
    try:
        return enc.decode(valid)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Normalisation for WER — matches evaluate.py
# ---------------------------------------------------------------------------

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

# ---------------------------------------------------------------------------
# Load NPU models
# ---------------------------------------------------------------------------

print("Connecting to NPU...")
acc = maccel.Accelerator()
print(f"  Available cores: {acc.get_available_cores()}")

print(f"Loading encoder: {ENCODER_MXQ}")
encoder = maccel.Model(ENCODER_MXQ)
encoder.launch(acc)
print(f"  Input shape:  {encoder.get_model_input_shape()}")
print(f"  Output shape: {encoder.get_model_output_shape()}")

print(f"Loading decoder: {DECODER_MXQ}")
decoder = maccel.Model(DECODER_MXQ)
decoder.launch(acc)
print(f"  Input shape:  {decoder.get_model_input_shape()}")
print(f"  Output shape: {decoder.get_model_output_shape()}")
print()

# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------
print("Loading tokenizer...")
enc = load_whisper_tokenizer()
print("  Tokenizer ready.\n")

# ---------------------------------------------------------------------------
# Load test examples
# ---------------------------------------------------------------------------

def iter_test_rows(tsv_path, limit=None):
    count = 0
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if limit and count >= limit:
                break
            audio_path = os.path.join(CV_CLIPS_DIR, row["path"])
            sentence   = row["sentence"].strip()
            if sentence and os.path.isfile(audio_path):
                yield audio_path, sentence
                count += 1

all_rows = list(iter_test_rows(CV_TEST_TSV, limit=MAX_EXAMPLES))
print(f"Test examples: {len(all_rows)}\n")

# ---------------------------------------------------------------------------
# Inference loop — greedy decode on NPU
# ---------------------------------------------------------------------------

references  = []
hypotheses  = []

# Latency tracking (milliseconds)
enc_latencies = []
dec_latencies = []
total_latencies = []

print("Running NPU inference...\n")

for i, (audio_path, sentence) in enumerate(all_rows):
    try:
        # ── Audio preprocessing (CPU) ────────────────────────────────────
        mel = audio_to_mel(audio_path)   # (1, 80, 3000) float32

        # ── Encoder (NPU) ───────────────────────────────────────────────
        t0 = time.perf_counter()
        enc_outputs = encoder.infer([mel])
        t1 = time.perf_counter()

        audio_features = enc_outputs[0]   # (1, 1500, 384) float32
        enc_ms = (t1 - t0) * 1000
        enc_latencies.append(enc_ms)

        # ── Greedy decode loop (NPU) ─────────────────────────────────────
        tokens  = [SOT_TOKEN]
        dec_ms_total = 0.0

        for _ in range(MAX_DECODE_TOKENS):
            # Decoder input: tokens as float32 (NPU quantised model expects float)
            tok_arr = np.array([tokens], dtype=np.float32)   # (1, seq_len)

            t0 = time.perf_counter()
            dec_outputs = decoder.infer([tok_arr, audio_features])
            t1 = time.perf_counter()
            dec_ms_total += (t1 - t0) * 1000

            logits    = dec_outputs[0]          # (1, seq_len, vocab_size)
            next_tok  = int(np.argmax(logits[0, -1]))

            if next_tok == EOT_TOKEN:
                break
            tokens.append(next_tok)

        dec_latencies.append(dec_ms_total)
        total_latencies.append(enc_ms + dec_ms_total)

        # ── Decode tokens → text ─────────────────────────────────────────
        predicted = normalise(decode_tokens(enc, tokens[1:]))
        reference = normalise(sentence)

        if not reference:
            continue

        hypotheses.append(predicted)
        references.append(reference)

        if i % 50 == 0:
            print(f"[{i}/{len(all_rows)}] REF: {reference}")
            print(f"[{i}/{len(all_rows)}] HYP: {predicted}")
            print(f"         Encoder: {enc_ms:.1f}ms | "
                  f"Decoder: {dec_ms_total:.1f}ms | "
                  f"Total: {enc_ms + dec_ms_total:.1f}ms")
            print()

    except Exception as e:
        print(f"Skipping {i} ({os.path.basename(audio_path)}): {e}")
        continue

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

enc_latencies   = np.array(enc_latencies)
dec_latencies   = np.array(dec_latencies)
total_latencies = np.array(total_latencies)

wer = jiwer.wer(references, hypotheses)
cer = jiwer.cer(references, hypotheses)

print(f"\n{'='*55}")
print(f"NPU Inference Results")
print(f"{'='*55}")
print(f"Encoder MXQ  : {ENCODER_MXQ}")
print(f"Decoder MXQ  : {DECODER_MXQ}")
print(f"Examples     : {len(references)}")
print()
print(f"  WER : {wer * 100:.2f}%")
print(f"  CER : {cer * 100:.2f}%")
print()
print(f"Latency (ms)          Mean     P50      P95      P99")
print(f"  Encoder NPU    {enc_latencies.mean():8.1f} "
      f"{np.percentile(enc_latencies, 50):8.1f} "
      f"{np.percentile(enc_latencies, 95):8.1f} "
      f"{np.percentile(enc_latencies, 99):8.1f}")
print(f"  Decoder NPU    {dec_latencies.mean():8.1f} "
      f"{np.percentile(dec_latencies, 50):8.1f} "
      f"{np.percentile(dec_latencies, 95):8.1f} "
      f"{np.percentile(dec_latencies, 99):8.1f}")
print(f"  Total          {total_latencies.mean():8.1f} "
      f"{np.percentile(total_latencies, 50):8.1f} "
      f"{np.percentile(total_latencies, 95):8.1f} "
      f"{np.percentile(total_latencies, 99):8.1f}")
print(f"{'='*55}")

# Clean up
encoder.dispose()
decoder.dispose()
