import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import torch
import jiwer
import numpy as np
import librosa
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer
from my_model_config import get_model
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# CONFIGURE THESE before running
# ---------------------------------------------------------------------------

CV_ROOT    = r"cv-data"
CHECKPOINT = os.path.join("checkpoint_922000.pt")

# Repetition penalty — how much to penalise tokens already generated.
# 0.0 = no penalty (old behaviour), 1.0 = strong penalty. Start at 0.5.
REPETITION_PENALTY = 0.5

# ---------------------------------------------------------------------------

CV_CLIPS_DIR = os.path.join(CV_ROOT, "clips")
CV_TEST_TSV  = os.path.join(CV_ROOT, "test.tsv")

if not os.path.isfile(CV_TEST_TSV):
    sys.exit(f"ERROR: test.tsv not found at {CV_TEST_TSV}\n"
             f"Run extract_cv.py first.")
if not os.path.isdir(CV_CLIPS_DIR):
    sys.exit(f"ERROR: clips/ folder not found at {CV_CLIPS_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on : {device}")

model = get_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Loaded        : {CHECKPOINT}")
print(f"Test TSV      : {CV_TEST_TSV}")
print(f"Clips dir     : {CV_CLIPS_DIR}")
print()

# FIX 1: use get_tokenizer properly so sot_sequence is available
tokenizer = get_tokenizer(multilingual=False, language="en", task="transcribe")

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

def iter_test_rows(tsv_path):
    test_num_rows = 16398 - 1

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc=tsv_path, total=test_num_rows):
            audio_path = os.path.join(CV_CLIPS_DIR, row["path"])
            sentence   = row["sentence"].strip()
            if sentence and os.path.isfile(audio_path):
                yield audio_path, sentence

all_rows = list(iter_test_rows(CV_TEST_TSV))
print(f"Test examples found: {len(all_rows)}\n")

# FIX 1: print what sot_sequence looks like so we can confirm it's correct
print(f"SOT sequence tokens: {tokenizer.sot_sequence}")
print(f"  (should include sot + language + task tokens, not just sot alone)\n")

print("Running evaluation...\n")

references = []
hypotheses = []

with torch.no_grad():
    for i, (audio_path, sentence) in enumerate(all_rows):
        try:
            array, sr = librosa.load(audio_path, sr=None, mono=True)
            array = array.astype("float32")

            if sr != SAMPLE_RATE:
                array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)

            if np.abs(array).max() < 1e-5:
                continue

            mel = log_mel_spectrogram(
                pad_or_trim(torch.tensor(array), N_SAMPLES)
            ).unsqueeze(0).to(device)

            audio_features = model.encoder(mel)

            # FIX 1: start with full sot_sequence instead of bare [sot]
            # This gives the decoder [sot, lang_token, task_token, no_timestamps]
            # which is what it was pre-trained to expect before any output tokens
            tokens = list(tokenizer.sot_sequence)

            for _ in range(200):
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model.decoder(token_tensor, audio_features)

                # FIX 5: repetition penalty — subtract score for already-seen tokens
                # This reduces hallucination loops where the model repeats itself
                if REPETITION_PENALTY > 0.0:
                    for tok in set(tokens):
                        logits[0, -1, tok] -= REPETITION_PENALTY

                next_token = logits[0, -1].argmax().item()
                if next_token == tokenizer.eot:
                    break
                tokens.append(next_token)

            # Strip the sot_sequence prefix before decoding to text
            predicted = normalise(tokenizer.decode(tokens[len(tokenizer.sot_sequence):]))
            reference = normalise(sentence)

            if not reference:
                continue

            hypotheses.append(predicted)
            references.append(reference)

            if i % 50 == 0:
                print(f"[{i}/{len(all_rows)}] REF: {reference}")
                print(f"[{i}/{len(all_rows)}] HYP: {predicted}")
                print()

        except Exception as e:
            print(f"Skipping example {i} ({os.path.basename(audio_path)}): {e}")
            continue

if not references:
    print("No examples evaluated — check CV_ROOT and that clips/ is populated.")
else:
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    print(f"\n{'='*40}")
    print(f"Checkpoint    : {CHECKPOINT}")
    print(f"Dataset       : Common Voice Scripted Speech 25.0 / test")
    print(f"Repetition penalty: {REPETITION_PENALTY}")
    print(f"Total examples: {len(references)}")
    print(f"WER : {wer * 100:.2f}%")
    print(f"CER : {cer * 100:.2f}%")
    print(f"{'='*40}")