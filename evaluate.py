import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import jiwer
import soundfile as sf
import io
from datasets import load_dataset, Audio
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer
from my_model_config import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on: {device}")

# ← CHANGE THIS to whichever checkpoint you want to evaluate
CHECKPOINT = "checkpoint_922000.pt"

model = get_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Loaded {CHECKPOINT}")

tokenizer = get_tokenizer(multilingual=False)

test_set = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="test",
    streaming=True,
)
test_set = test_set.cast_column("audio", Audio(decode=False))

references = []
hypotheses = []

print("Running evaluation...")

with torch.no_grad():
    for i, example in enumerate(test_set):
        try:
            audio_bytes = example["audio"]["bytes"]
            array, sr = sf.read(io.BytesIO(audio_bytes))
            array = array.astype("float32")

            if sr != SAMPLE_RATE:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)

            mel = log_mel_spectrogram(
                pad_or_trim(torch.tensor(array), N_SAMPLES)
            ).unsqueeze(0).to(device)

            audio_features = model.encoder(mel)

            tokens = [tokenizer.sot]
            for _ in range(200):
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model.decoder(token_tensor, audio_features)
                next_token = logits[0, -1].argmax().item()
                if next_token == tokenizer.eot:
                    break
                tokens.append(next_token)

            predicted = tokenizer.decode(tokens[1:]).lower().strip()
            reference = example["text"].lower().strip()

            hypotheses.append(predicted)
            references.append(reference)

            if i % 50 == 0:
                print(f"[{i}] REF: {reference}")
                print(f"[{i}] HYP: {predicted}")
                print()

        except Exception as e:
            print(f"Skipping example {i}: {e}")
            continue

wer = jiwer.wer(references, hypotheses)
print(f"\n{'='*40}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Total examples: {len(references)}")
print(f"Final WER: {wer*100:.2f}%")
print(f"{'='*40}")