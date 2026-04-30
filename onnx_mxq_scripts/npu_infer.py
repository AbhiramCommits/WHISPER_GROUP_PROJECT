"""
npu_infer_rank3.py — NPU inference using original MXQ files
whisper_encoder.mxq + whisper_decoder_rank3.mxq
Measures WER and latency on Mobilint Aries2.
"""

import os
import csv
import time
import numpy as np
import librosa
import jiwer
import tiktoken
import torch
import onnx
import maccel

# ─── CONFIG ────────────────────────────────────────────────────────────────

ENCODER_MXQ  = "whisper_encoder.mxq"
DECODER_MXQ  = "whisper_decoder_rank3.mxq"
DECODER_ONNX = "whisper_decoder_rank3.onnx"  # for embedding extraction only

CV_ROOT      = os.path.dirname(os.path.abspath(__file__))
CV_TEST_TSV  = os.path.join(CV_ROOT, "test.tsv")
CV_CLIPS_DIR = os.path.join(CV_ROOT, "clips")

MAX_EXAMPLES      = 500
MAX_DECODE_TOKENS = 200

SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 80
N_SAMPLES   = 30 * SAMPLE_RATE
N_STATE     = 384
N_AUDIO_CTX = 1500

SOT_TOKEN = 50257
EOT_TOKEN = 50256

REPETITION_PENALTY = 2.0

# ─── LOAD EMBEDDING WEIGHTS FROM ONNX ─────────────────────────────────────

print("Loading embedding weights from ONNX...")
orig = onnx.load(DECODER_ONNX)
TOKEN_EMBEDDING = None
POS_EMBEDDING   = None
for init in orig.graph.initializer:
    if init.name == "decoder.output_projection.weight":
        TOKEN_EMBEDDING = np.frombuffer(
            init.raw_data, dtype=np.float32
        ).reshape(list(init.dims)).copy()
    if init.name == "decoder.positional_embedding":
        POS_EMBEDDING = np.frombuffer(
            init.raw_data, dtype=np.float32
        ).reshape(list(init.dims)).copy()

print(f"  Token embedding:     {TOKEN_EMBEDDING.shape}")
print(f"  Positional embedding: {POS_EMBEDDING.shape}\n")

def embed_token(token_id, position):
    """CPU embedding lookup → (1, 1, 384) float32."""
    emb = TOKEN_EMBEDDING[token_id] + POS_EMBEDDING[position]
    return emb.astype(np.float32)[np.newaxis, np.newaxis, :]  # (1, 1, 384)

# ─── NPU INITIALISATION ────────────────────────────────────────────────────

print("Connecting to NPU...")
acc = maccel.Accelerator()
print(f"  Available cores: {len(acc.get_available_cores())}")

enc_cfg = maccel.ModelConfig()
enc_cfg.set_single_core_mode(4)

dec_cfg = maccel.ModelConfig()
dec_cfg.set_single_core_mode(4)

print(f"Loading encoder: {ENCODER_MXQ}")
encoder = maccel.Model(ENCODER_MXQ, enc_cfg)

print(f"Loading decoder: {DECODER_MXQ}")
decoder = maccel.Model(DECODER_MXQ, dec_cfg)

print("Launching models...")
encoder.launch(acc)
print(f"  Encoder input:  {encoder.get_model_input_shape()}")
print(f"  Encoder output: {encoder.get_model_output_shape()}")

decoder.launch(acc)
print(f"  Decoder input:  {decoder.get_model_input_shape()}")
print(f"  Decoder output: {decoder.get_model_output_shape()}")
print("NPU ready.\n")

# ─── AUDIO PREPROCESSING ───────────────────────────────────────────────────

def pad_or_trim(array, length):
    if len(array) > length: return array[:length]
    if len(array) < length: return np.pad(array, (0, length - len(array)))
    return array

def log_mel_spectrogram(audio):
    audio_t = torch.from_numpy(audio).float()
    window  = torch.hann_window(N_FFT)
    stft    = torch.stft(audio_t, N_FFT, HOP_LENGTH, window=window,
                         return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(
        librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS
        ).astype(np.float32)
    )
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)

def audio_to_mel(path):
    array, sr = librosa.load(path, sr=None, mono=True)
    array = array.astype(np.float32)
    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
    array = pad_or_trim(array, N_SAMPLES)
    mel = log_mel_spectrogram(array)
    return mel[np.newaxis, :, :]  # (1, 80, 3000)

# ─── TOKENIZER ─────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")

def decode_tokens(ids):
    valid = [t for t in ids if t < EOT_TOKEN]
    try:
        return tokenizer.decode(valid)
    except:
        return ""

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

# ─── LOAD TEST SET ─────────────────────────────────────────────────────────

def load_test_rows(tsv_path, clips_dir, limit=None):
    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            path = os.path.join(clips_dir, row["path"])
            sent = row["sentence"].strip()
            if sent and os.path.isfile(path):
                rows.append((path, sent))
            if limit and len(rows) >= limit:
                break
    return rows

print("Loading test set...")
rows = load_test_rows(CV_TEST_TSV, CV_CLIPS_DIR, limit=MAX_EXAMPLES)
print(f"  {len(rows)} examples\n")

# ─── INFERENCE LOOP ────────────────────────────────────────────────────────

references  = []
hypotheses  = []
enc_lat     = []
dec_lat     = []
total_lat   = []

print("Running NPU inference...\n")

for i, (audio_path, sentence) in enumerate(rows):
    try:
        mel = audio_to_mel(audio_path)  # (1, 80, 3000)

        # ── Encoder (NPU) ─────────────────────────────────────────────
        t0 = time.perf_counter()
        enc_out = encoder.infer([mel])
        t1 = time.perf_counter()
        enc_ms = (t1 - t0) * 1000
        enc_lat.append(enc_ms)

        # Encoder output: (1, 1, 1500, 384) NHWC → reshape to (1, 1500, 384)
        audio_features = enc_out[0].reshape(1, N_AUDIO_CTX, N_STATE)

        # ── Greedy decode (NPU) ───────────────────────────────────────
        tokens   = [SOT_TOKEN]
        dec_ms   = 0.0

        for step in range(MAX_DECODE_TOKENS):
            # CPU embedding lookup → (1, 1, 384)
            tok = embed_token(tokens[-1], step)

            t0 = time.perf_counter()
            dec_out = decoder.infer([tok, audio_features])
            t1 = time.perf_counter()
            dec_ms += (t1 - t0) * 1000

            if not dec_out or len(dec_out) == 0:
                break

            # Output: (1, 1, 51865) → flatten
            logits = dec_out[0].flatten().astype(np.float64)

            # Repetition penalty
            for prev in set(tokens):
                logits[prev] /= REPETITION_PENALTY

            next_tok = int(np.argmax(logits))

            if next_tok == EOT_TOKEN:
                break
            tokens.append(next_tok)

            # Early stop if stuck in repetition
            if len(tokens) > 10 and len(set(tokens[-10:])) == 1:
                break

        dec_lat.append(dec_ms)
        total_lat.append(enc_ms + dec_ms)

        pred = normalise(decode_tokens(tokens[1:]))
        ref  = normalise(sentence)
        if not ref:
            continue
        hypotheses.append(pred)
        references.append(ref)

        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] REF: {ref}")
            print(f"[{i}/{len(rows)}] HYP: {pred}")
            print(f"  enc={enc_ms:.1f}ms  dec={dec_ms:.1f}ms  "
                  f"total={enc_ms+dec_ms:.1f}ms\n")

    except Exception as e:
        print(f"  [SKIP {i}] {e}")
        continue

# ─── RESULTS ───────────────────────────────────────────────────────────────

ea = np.array(enc_lat)
da = np.array(dec_lat)
ta = np.array(total_lat)

if references:
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)

    print(f"\n{'='*56}")
    print(f"  NPU Inference Results  (maccel / Aries2)")
    print(f"{'='*56}")
    print(f"  Encoder MXQ  : {ENCODER_MXQ}")
    print(f"  Decoder MXQ  : {DECODER_MXQ}")
    print(f"  Examples     : {len(references)}")
    print()
    print(f"  WER : {wer * 100:.2f}%")
    print(f"  CER : {cer * 100:.2f}%")
    print()
    print(f"  Latency (ms)      Mean    P50     P95     P99")
    print(f"  Encoder NPU    {ea.mean():7.1f} {np.percentile(ea,50):7.1f} "
          f"{np.percentile(ea,95):7.1f} {np.percentile(ea,99):7.1f}")
    print(f"  Decoder NPU    {da.mean():7.1f} {np.percentile(da,50):7.1f} "
          f"{np.percentile(da,95):7.1f} {np.percentile(da,99):7.1f}")
    print(f"  Total          {ta.mean():7.1f} {np.percentile(ta,50):7.1f} "
          f"{np.percentile(ta,95):7.1f} {np.percentile(ta,99):7.1f}")
    print(f"{'='*56}")
else:
    print("No valid results — all samples skipped.")

encoder.dispose()
decoder.dispose()