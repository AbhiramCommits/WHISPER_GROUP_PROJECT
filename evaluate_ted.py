"""
evaluate_tedlium.py — Evaluate on TED-LIUM v3 test set via torchaudio.

TED-LIUM is one of the datasets used to benchmark Whisper in the original paper.
Uses torchaudio.datasets.TEDLIUM which downloads directly — no HuggingFace needed.

Usage:
    python evaluate_tedlium.py

Requirements:
    pip install torchaudio jiwer
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import torch
import torchaudio
import jiwer
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer
from my_model_config import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on : {device}")

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

# Downloads to ./tedlium_data/ on first run (~20 GB for release3).
# Use release1 if disk space is tight (~14 GB).
TEDLIUM_ROOT = "./tedlium_data"
if not os.path.isdir(TEDLIUM_ROOT):
    os.mkdir(TEDLIUM_ROOT)
print(f"Loading TED-LIUM release3 test set (downloads to {TEDLIUM_ROOT} if needed)...")
dataset = torchaudio.datasets.TEDLIUM(
    root=TEDLIUM_ROOT,
    release="release3",
    subset="test",
    download=True,
)
print(f"  {len(dataset)} utterances\n")

# ─── TEXT NORMALISATION ────────────────────────────────────────────────────────

def clean_transcript(text: str) -> str:
    """Remove TED-LIUM specific markers like <unk> and tidy whitespace."""
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", "", text)       # remove <unk> etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

# ─── EVALUATION LOOP ───────────────────────────────────────────────────────────

references = []
hypotheses = []

print("Running evaluation...\n")

with torch.no_grad():
    for i in range(len(dataset)):
        try:
            # torchaudio TEDLIUM returns:
            # waveform (Tensor), sample_rate (int), transcript (str),
            # talk_id, speaker_id, identifier
            waveform, sr, transcript, *_ = dataset[i]

            # waveform shape: (channels, samples) — convert to mono float32
            audio = waveform.mean(dim=0).float()

            if sr != SAMPLE_RATE:
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

            predicted = normalise(tokenizer.decode(tokens[1:]))
            reference = normalise(clean_transcript(transcript))

            if not reference:
                continue

            hypotheses.append(predicted)
            references.append(reference)

            if i % 50 == 0:
                print(f"[{i}/{len(dataset)}] REF: {reference}")
                print(f"[{i}/{len(dataset)}] HYP: {predicted}")
                print()

        except Exception as e:
            print(f"  [SKIP {i}] {e}")
            continue

# ─── RESULTS ───────────────────────────────────────────────────────────────────

wer = jiwer.wer(references, hypotheses)
cer = jiwer.cer(references, hypotheses)

print(f"\n{'='*40}")
print(f"Checkpoint    : {CHECKPOINT}")
print(f"Dataset       : TED-LIUM release3 / test")
print(f"Total examples: {len(references)}")
print(f"WER : {wer*100:.2f}%")
print(f"CER : {cer*100:.2f}%")
print(f"{'='*40}")