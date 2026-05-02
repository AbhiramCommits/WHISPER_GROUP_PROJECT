

import os
import io
import time
import numpy as np
import librosa
import jiwer
import tiktoken
import torch
import soundfile as sf
import maccel
from datasets import load_dataset, Audio as HFAudio


ENCODER_MXQ = "whisper_encoder.mxq"
DECODER_MXQ = "whisper_decoder.mxq"

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



TOKEN_EMBEDDING = np.load("token_embedding.npy")      
POS_EMBEDDING   = np.load("positional_embedding.npy") 
print(f"  Token embedding:      {TOKEN_EMBEDDING.shape}")
print(f"  Positional embedding: {POS_EMBEDDING.shape}\n")

def embed_token(token_id, position):
    emb = TOKEN_EMBEDDING[token_id] + POS_EMBEDDING[position]
    return emb.astype(np.float32)[np.newaxis, np.newaxis, :]  # (1, 1, 384)

#NPU INITIALISATION 


acc = maccel.Accelerator()

enc_cfg = maccel.ModelConfig()
enc_cfg.set_single_core_mode(4)

dec_cfg = maccel.ModelConfig()
dec_cfg.set_single_core_mode(4)

encoder = maccel.Model(ENCODER_MXQ, enc_cfg)

decoder = maccel.Model(DECODER_MXQ, dec_cfg)

encoder.launch(acc)


decoder.launch(acc)

#AUDIO PREPROCESSING 

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
        librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    )
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)

# TOKENIZER 

tokenizer = tiktoken.get_encoding("gpt2")

def decode_tokens(ids):
    valid = [t for t in ids if t < EOT_TOKEN]
    try:    return tokenizer.decode(valid)
    except: return ""

normalise = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
    jiwer.Strip(), jiwer.RemoveMultipleSpaces(),
])

# LOAD DATASET 

print("Loading LibriSpeech test-clean (streaming)...")
ls_dataset = load_dataset(
    "openslr/librispeech_asr", "clean",
    split="test", streaming=True
)
ls_dataset = ls_dataset.cast_column("audio", HFAudio(decode=False))

# INFERENCE LOOP

references, hypotheses = [], []
enc_lat, dec_lat, total_lat = [], [], []


for i, item in enumerate(ls_dataset):
    if i >= MAX_EXAMPLES:
        break
    try:
        raw_audio, sr = sf.read(io.BytesIO(item["audio"]["bytes"]))
        raw_audio = raw_audio.astype(np.float32)
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raw_audio = librosa.resample(raw_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        raw_audio = pad_or_trim(raw_audio, N_SAMPLES)
        mel = log_mel_spectrogram(raw_audio)[np.newaxis, :, :]  # (1, 80, 3000)
        sentence = item["text"]

   
        t0 = time.perf_counter()
        enc_out = encoder.infer([mel])
        t1 = time.perf_counter()
        enc_ms = (t1 - t0) * 1000
        enc_lat.append(enc_ms)

        audio_features = enc_out[0].reshape(1, N_AUDIO_CTX, N_STATE)

        tokens, dec_ms = [SOT_TOKEN], 0.0

        for step in range(MAX_DECODE_TOKENS):
            tok = embed_token(tokens[-1], step)  # (1, 1, 384)

            t0 = time.perf_counter()
            dec_out = decoder.infer([tok, audio_features])
            t1 = time.perf_counter()
            dec_ms += (t1 - t0) * 1000

            if not dec_out or len(dec_out) == 0:
                break

            logits = dec_out[0].flatten().astype(np.float64)

            for prev in set(tokens):
                logits[prev] /= REPETITION_PENALTY

            next_tok = int(np.argmax(logits))
            if next_tok == EOT_TOKEN:
                break
            tokens.append(next_tok)


        dec_lat.append(dec_ms)
        total_lat.append(enc_ms + dec_ms)

        pred = normalise(decode_tokens(tokens[1:]))
        ref  = normalise(sentence)
        if not ref:
            continue
        hypotheses.append(pred)
        references.append(ref)

        if i % 50 == 0:
            print(f"[{i}/{MAX_EXAMPLES}] REF: {ref}")
            print(f"[{i}/{MAX_EXAMPLES}] HYP: {pred}")
            print(f"  enc={enc_ms:.1f}ms  dec={dec_ms:.1f}ms  "
                  f"total={enc_ms+dec_ms:.1f}ms\n")

    except Exception as e:
        print(f"  [SKIP {i}] {e}")
        continue




ea = np.array(enc_lat)
da = np.array(dec_lat)
ta = np.array(total_lat)

wer = jiwer.wer(references, hypotheses)

print(f"  NPU Inference Results")
print()
print(f"  Encoder MXQ : {ENCODER_MXQ}")
print(f"  Decoder MXQ : {DECODER_MXQ}")
print(f"  Examples    : {len(references)}")
print()
print(f"  WER : {wer * 100:.2f}%")
print()
print(f"  Latency (ms)      Mean ")
print(f"  Encoder NPU    {ea.mean():7.1f}")
print(f"  Decoder NPU    {da.mean():7.1f}")
print(f"  Total          {ta.mean():7.1f}")

encoder.dispose()
decoder.dispose()