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

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------

CV_ROOT    = r"C:\Users\daniel\Desktop\whisper_group_project\checkpoint_218000\cv-test"
CHECKPOINT = "checkpoints/checkpoint_200000.pt"

# Set to None to evaluate all 16k examples (slow).
# Set to 500 for a fast ~3 minute checkpoint comparison run.
MAX_EXAMPLES = 500

REPETITION_PENALTY = 0.5

# ---------------------------------------------------------------------------

CV_CLIPS_DIR = os.path.join(CV_ROOT, "clips")
CV_TEST_TSV  = os.path.join(CV_ROOT, "test.tsv")

if not os.path.isfile(CV_TEST_TSV):
    sys.exit(f"ERROR: test.tsv not found at {CV_TEST_TSV}")
if not os.path.isdir(CV_CLIPS_DIR):
    sys.exit(f"ERROR: clips/ not found at {CV_CLIPS_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on : {device}")

model = get_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Loaded        : {CHECKPOINT}")
print(f"Max examples  : {MAX_EXAMPLES if MAX_EXAMPLES else 'ALL'}")
print()

tokenizer = get_tokenizer(multilingual=False)

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

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
print(f"Examples to evaluate: {len(all_rows)}\n")

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
            tokens = [tokenizer.sot]
            for _ in range(200):
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model.decoder(token_tensor, audio_features)

                if REPETITION_PENALTY > 0.0:
                    for tok in set(tokens):
                        logits[0, -1, tok] -= REPETITION_PENALTY

                next_token = logits[0, -1].argmax().item()
                if next_token == tokenizer.eot:
                    break
                tokens.append(next_token)

            predicted = normalise(tokenizer.decode(tokens[1:]))
            reference = normalise(sentence)

            if not reference:
                continue

            hypotheses.append(predicted)
            references.append(reference)

            if i % 100 == 0:
                print(f"[{i}/{len(all_rows)}] REF: {reference}")
                print(f"[{i}/{len(all_rows)}] HYP: {predicted}")
                print()

        except Exception as e:
            print(f"Skipping {i}: {e}")
            continue

if not references:
    print("No examples evaluated.")
else:
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    print(f"\n{'='*40}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Examples   : {len(references)}")
    print(f"WER : {wer * 100:.2f}%")
    print(f"CER : {cer * 100:.2f}%")
    print(f"{'='*40}")