import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from whisper.tokenizer import get_tokenizer
from my_model_config import get_model
from whisper_dataset import (
    get_librispeech, load_cv_examples,
    preprocess, preprocess_cv,
    mixed_epoch_iter
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

model     = get_model().to(device)
tokenizer = get_tokenizer(multilingual=False)

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
WARMUP     = 2000
TOTAL      = 1200000  # adjust based on how long you want to train
BATCH_SIZE = 16

# How much of each batch comes from Common Voice vs LibriSpeech.
# 0.5 = 50/50 mix. Increase toward 0.7 if you want to bias more toward CV.
CV_RATIO = 0.5

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

def lr_lambda(s):
    if s < WARMUP:
        return s / max(1, WARMUP)
    progress = (s - WARMUP) / max(1, TOTAL - WARMUP)
    return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------------------------------------------------------------------
# Resume from checkpoint (set to None to start from scratch)
# ---------------------------------------------------------------------------
RESUME_CHECKPOINT = None   # e.g. "checkpoints/checkpoint_50000.pt"

if RESUME_CHECKPOINT and os.path.isfile(RESUME_CHECKPOINT):
    model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device))
    step = int(os.path.splitext(os.path.basename(RESUME_CHECKPOINT))[0].split("_")[-1])
    for _ in range(step):
        scheduler.step()
    print(f"Resumed from {RESUME_CHECKPOINT} at step {step}")
else:
    step = 0
    print("Starting from scratch...")

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
print("Loading LibriSpeech...")
librispeech = get_librispeech(streaming=False)
print(f"  LibriSpeech: {len(librispeech)} examples")

print("Loading Common Voice...")
cv_examples = load_cv_examples(splits=("train.tsv", "dev.tsv"))
print(f"  Common Voice: {len(cv_examples)} examples")

print(f"\nMix ratio: {int((1-CV_RATIO)*100)}% LibriSpeech / {int(CV_RATIO*100)}% Common Voice")
print(f"Total steps: {TOTAL}\n")

os.makedirs("checkpoints", exist_ok=True)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
data_iter    = mixed_epoch_iter(librispeech, cv_examples, cv_ratio=CV_RATIO)
model.train()
running_loss = 0.0
batch_mels   = []
batch_tokens = []

def run_batch(mels, tokens_list):
    dec_inputs = [t[:-1] for t in tokens_list]
    targets    = [t[1:]  for t in tokens_list]

    max_len = max(t.shape[0] for t in dec_inputs)
    dec_pad = torch.zeros(len(dec_inputs), max_len, dtype=torch.long, device=device)
    tgt_pad = torch.full((len(targets), max_len), -100, dtype=torch.long, device=device)

    for i, (d, t) in enumerate(zip(dec_inputs, targets)):
        dec_pad[i, :d.shape[0]] = d.to(device)
        tgt_pad[i, :t.shape[0]] = t.to(device)

    mel_batch      = torch.stack(mels).to(device)
    audio_features = model.encoder(mel_batch)
    logits         = model.decoder(dec_pad, audio_features)

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tgt_pad.reshape(-1),
        ignore_index=-100,
        label_smoothing=0.0,
    )

for source, example in data_iter:
    try:
        if source == "cv":
            mel, tokens = preprocess_cv(example, tokenizer)
        else:
            mel, tokens = preprocess(example, tokenizer)
    except Exception:
        continue

    batch_mels.append(mel)
    batch_tokens.append(tokens)

    if len(batch_mels) < BATCH_SIZE:
        continue

    optimizer.zero_grad()
    loss = run_batch(batch_mels, batch_tokens)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    running_loss += loss.item()
    batch_mels   = []
    batch_tokens = []
    step        += 1

    if step % 100 == 0:
        avg          = running_loss / 100
        running_loss = 0.0
        print(f"Step {step} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if step % 1000 == 0 and step > 0:
        with torch.no_grad():
            test_mel = torch.randn(1, 80, 3000).to(device)
            enc_out  = model.encoder(test_mel)
            print(f"  Encoder std: {enc_out.std():.4f}")
        ckpt = f"checkpoints/checkpoint_{step}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"  Saved {ckpt}")

    if step >= TOTAL:
        break

torch.save(model.state_dict(), "checkpoints/checkpoint_final.pt")
print("Training done.")