import sys, os
import time  # New import
import numpy as np # For statistics
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

# --- Latency tracking lists ---
enc_latencies = []
dec_latencies = []
total_latencies = []

print("Running evaluation...")

with torch.no_grad():
    for i, example in enumerate(test_set):
        # Limit to match your NPU script if desired (e.g., 500)
        if i >= 500: break 
        
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

            # --- Measure Encoder ---
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            audio_features = model.encoder(mel)
            
            if device == "cuda": torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t0) * 1000
            enc_latencies.append(enc_ms)

            # --- Measure Decoder Loop ---
            tokens = [tokenizer.sot]
            dec_ms_total = 0.0
            
            for _ in range(200):
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                
                if device == "cuda": torch.cuda.synchronize()
                t_dec_start = time.perf_counter()
                
                logits = model.decoder(token_tensor, audio_features)
                
                if device == "cuda": torch.cuda.synchronize()
                dec_ms_total += (time.perf_counter() - t_dec_start) * 1000
                
                next_token = logits[0, -1].argmax().item()
                if next_token == tokenizer.eot:
                    break
                tokens.append(next_token)

            dec_latencies.append(dec_ms_total)
            total_latencies.append(enc_ms + dec_ms_total)

            predicted = tokenizer.decode(tokens[1:]).lower().strip()
            reference = example["text"].lower().strip()

            hypotheses.append(predicted)
            references.append(reference)

            if i % 50 == 0:
                print(f"[{i}] REF: {reference}")
                print(f"[{i}] HYP: {predicted}")
                print(f"      Encoder: {enc_ms:.1f}ms | Decoder: {dec_ms_total:.1f}ms")

        except Exception as e:
            print(f"Skipping example {i}: {e}")
            continue

# --- Results Reporting ---
enc_latencies = np.array(enc_latencies)
dec_latencies = np.array(dec_latencies)
total_latencies = np.array(total_latencies)

wer = jiwer.wer(references, hypotheses)

print(f"\n{'='*55}")
print(f"PyTorch Inference Results ({device})")
print(f"{'='*55}")
print(f"Total examples: {len(references)}")
print(f"Final WER: {wer*100:.2f}%")
print()
print(f"Latency (ms)          Mean     P50      P95      P99")
print(f"  Encoder        {enc_latencies.mean():8.1f} "
      f"{np.percentile(enc_latencies, 50):8.1f} "
      f"{np.percentile(enc_latencies, 95):8.1f} "
      f"{np.percentile(enc_latencies, 99):8.1f}")
print(f"  Decoder        {dec_latencies.mean():8.1f} "
      f"{np.percentile(dec_latencies, 50):8.1f} "
      f"{np.percentile(dec_latencies, 95):8.1f} "
      f"{np.percentile(dec_latencies, 99):8.1f}")
print(f"  Total          {total_latencies.mean():8.1f} "
      f"{np.percentile(total_latencies, 50):8.1f} "
      f"{np.percentile(total_latencies, 95):8.1f} "
      f"{np.percentile(total_latencies, 99):8.1f}")
print(f"{'='*55}")