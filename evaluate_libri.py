import os
import io
import sys
import time
import torch
import jiwer
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer
from my_model_config import get_model

# Setup pathing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoint_922000.pt"


def run_evaluation():
    print(f"--- Starting Eval on {DEVICE} ---")

    # Init model and pull in weights
    model = get_model().to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Successfully loaded: {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find {CHECKPOINT_PATH}. Check your file path.")
        return

    model.eval()
    tokenizer = get_tokenizer(multilingual=False)

    # We're using streaming to avoid nuking the RAM
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    hyps, refs = [], []

    # Latency tracking
    latencies_total = []    # end-to-end per sample (load + encode + decode)
    latencies_encode = []   # encoder only
    latencies_decode = []   # decoder only
    audio_durations = []    # actual audio length in seconds

    with torch.no_grad():
        for i, item in enumerate(ds):
            try:
                t_start = time.perf_counter()

                # Manual audio loading from bytes
                raw_audio, sr = sf.read(io.BytesIO(item["audio"]["bytes"]))
                raw_audio = raw_audio.astype("float32")

                audio_duration = len(raw_audio) / sr  # seconds of audio

                # Prep the mel spectrogram
                mel_data = pad_or_trim(torch.tensor(raw_audio), N_SAMPLES)
                mel = log_mel_spectrogram(mel_data).unsqueeze(0).to(DEVICE)

                # --- Encoder latency ---
                t_enc_start = time.perf_counter()
                features = model.encoder(mel)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t_enc_end = time.perf_counter()

                # --- Decoder latency ---
                tokens = [tokenizer.sot]
                t_dec_start = time.perf_counter()
                for _ in range(200):
                    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
                    out = model.decoder(input_ids, features)

                    next_id = out[0, -1].argmax().item()
                    if next_id == tokenizer.eot:
                        break
                    tokens.append(next_id)

                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t_dec_end = time.perf_counter()

                t_end = time.perf_counter()

                # Clean up text
                pred_text = tokenizer.decode(tokens[1:]).lower().strip()
                ref_text = item["text"].lower().strip()

                hyps.append(pred_text)
                refs.append(ref_text)

                # Record latencies
                latencies_total.append(t_end - t_start)
                latencies_encode.append(t_enc_end - t_enc_start)
                latencies_decode.append(t_dec_end - t_dec_start)
                audio_durations.append(audio_duration)

                if i % 50 == 0:
                    rtf = latencies_total[-1] / audio_duration if audio_duration > 0 else float("inf")
                    print(f"Ex {i} | Ref: {ref_text[:60]}...")
                    print(f"Ex {i} | Hyp: {pred_text[:60]}...")
                    print(
                        f"Ex {i} | Latency: total={latencies_total[-1]*1000:.1f}ms  "
                        f"encode={latencies_encode[-1]*1000:.1f}ms  "
                        f"decode={latencies_decode[-1]*1000:.1f}ms  "
                        f"RTF={rtf:.3f}"
                    )
                    print()

            except Exception as err:
                print(f"Oops, skipped index {i} because: {err}")
                continue

    # ---- Final stats ----
    error_rate = jiwer.wer(refs, hyps)
    num_params = sum(p.numel() for p in model.parameters())

    lat  = np.array(latencies_total)  * 1000  # ms
    enc  = np.array(latencies_encode) * 1000
    dec  = np.array(latencies_decode) * 1000
    durs = np.array(audio_durations)
    rtfs = np.array(latencies_total) / durs

    print("-" * 55)
    print(f"Results for {CHECKPOINT_PATH} ({DEVICE}):")
    print(f"  Samples : {len(refs)}")
    print(f"  Params  : {num_params / 1e6:.2f}M")
    print(f"  WER     : {error_rate * 100:.2f}%")
    print(f"  End-to-End Latency (ms)")
    print(f"    Mean   : {lat.mean():.1f}")
    print(f"    Median : {np.median(lat):.1f}")
    print(f"    P90    : {np.percentile(lat, 90):.1f}")
    print(f"    P99    : {np.percentile(lat, 99):.1f}")
    print(f"    Min    : {lat.min():.1f}")
    print(f"    Max    : {lat.max():.1f}")
    print(f"  Encoder Latency (ms)")
    print(f"    Mean   : {enc.mean():.1f}    Median : {np.median(enc):.1f}")
    print(f"  Decoder Latency (ms)")
    print(f"    Mean   : {dec.mean():.1f}    Median : {np.median(dec):.1f}")
    print(f"  Real-Time Factor (RTF)  [lower = faster than real-time]")
    print(f"    Mean   : {rtfs.mean():.4f}")
    print(f"    Median : {np.median(rtfs):.4f}")
    print("-" * 55)


if __name__ == "__main__":
    run_evaluation()
