"""
evaluate_voxpopuli.py — Evaluate on VoxPopuli English test set.

VoxPopuli is one of the datasets used to benchmark Whisper in the original paper.
It contains European Parliament speech recordings — spontaneous formal speech,
very different domain from LibriSpeech (audiobooks) and Common Voice (crowdsourced).

Dataset: facebook/voxpopuli, config "en", split "test"
~1800 utterances, streams directly from HuggingFace — no large download needed.

Usage:
    python evaluate_voxpopuli.py

Requirements:
    pip install jiwer datasets soundfile librosa
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import numpy as np
import torch
import jiwer
from datasets import load_dataset
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer
from my_model_config import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on : {device}")

# ← CHANGE THIS to whichever checkpoint you want to evaluate
CHECKPOINT = r"checkpoint_922000.pt"

model = get_model().to(device)
ckpt  = torch.load(CHECKPOINT, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"Loaded        : {CHECKPOINT}")

tokenizer = get_tokenizer(multilingual=False)

# VoxPopuli English test set — streams from HuggingFace, no bulk download
print("Loading VoxPopuli English test set (streaming)...")
test_set = load_dataset(
    "facebook/voxpopuli",
    "en",
    split="test",
    streaming=True,
    trust_remote_code=False,
)
print("  Ready.\n")

# ─── TEXT NORMALISATION ────────────────────────────────────────────────────────

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

# ─── EVALUATION LOOP ───────────────────────────────────────────────────────────

references = []
hypotheses = []

print("Running NPU inference...\n")

with torch.no_grad():
    for i, example in enumerate(test_set):
        try:
            # VoxPopuli audio field: {"array": np.array, "sampling_rate": int}
            audio_array = example["audio"]["array"]
            sr           = example["audio"]["sampling_rate"]
            audio        = torch.tensor(np.array(audio_array, dtype=np.float32))

            if sr != SAMPLE_RATE:
                import torchaudio
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

            mel = log_mel_spectrogram(
                pad_or_trim(audio, N_SAMPLES)
            ).unsqueeze(0).to(device)

            audio_features = model.encoder(mel)

            tokens = [tokenizer.sot]
            for _ in range(200):
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                logits       = model.decoder(token_tensor, audio_features)
                next_token   = logits[0, -1].argmax().item()
                if next_token == tokenizer.eot:
                    break
                tokens.append(next_token)

            # VoxPopuli has both raw_text and normalized_text — use normalized
            ref_text  = example.get("normalized_text") or example.get("raw_text", "")
            predicted = normalise(tokenizer.decode(tokens[1:]))
            reference = normalise(ref_text)

            if not reference:
                continue

            hypotheses.append(predicted)
            references.append(reference)

            if i % 50 == 0:
                print(f"[{i}] REF: {reference}")
                print(f"[{i}] HYP: {predicted}")
                print()

        except Exception as e:
            print(f"  [SKIP {i}] {e}")
            continue

# ─── RESULTS ───────────────────────────────────────────────────────────────────

wer = jiwer.wer(references, hypotheses)
cer = jiwer.cer(references, hypotheses)

print(f"\n{'='*40}")
print(f"Checkpoint    : {CHECKPOINT}")
print(f"Dataset       : VoxPopuli English / test")
print(f"Total examples: {len(references)}")
print(f"WER : {wer*100:.2f}%")
print(f"CER : {cer*100:.2f}%")
print(f"{'='*40}")