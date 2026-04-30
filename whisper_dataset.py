from datasets import load_dataset, Audio, interleave_datasets
import torch
import soundfile as sf
import librosa
import io
import os
import csv
import random
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, SAMPLE_RATE
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 448

# Path to extracted Common Voice data folder (output of extract_cv.py)
CV_DATA_ROOT = r"cv-data"

# LibriSpeech — same as before
def get_librispeech(streaming=False):
    libri_100   = load_dataset("openslr/librispeech_asr", "clean",
                               split="train.100", streaming=streaming)
    libri_360   = load_dataset("openslr/librispeech_asr", "clean",
                               split="train.360", streaming=streaming)
    libri_other = load_dataset("openslr/librispeech_asr", "other",
                               split="train.500", streaming=streaming)

    libri_100   = libri_100.cast_column("audio", Audio(decode=False))
    libri_360   = libri_360.cast_column("audio", Audio(decode=False))
    libri_other = libri_other.cast_column("audio", Audio(decode=False))

    combined = interleave_datasets(
        [libri_100, libri_360, libri_other],
        probabilities=[0.2, 0.4, 0.4],
        stopping_strategy="all_exhausted",
    )
    return combined

# Common Voice — load train + dev splits from disk
def get_commonvoice(splits=("train.tsv", "dev.tsv")):
    """
    Returns a list of dicts: {"audio_path": ..., "text": ...}
    Only includes rows where the audio file actually exists on disk.
    """
    clips_dir = os.path.join(CV_DATA_ROOT, "clips")
    examples  = []
    for split in splits:
        tsv_path = os.path.join(CV_DATA_ROOT, split)
        if not os.path.isfile(tsv_path):
            print(f"  Warning: {tsv_path} not found, skipping.")
            continue
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_path = os.path.join(clips_dir, row["path"].strip())
                sentence   = row["sentence"].strip()
                if sentence and os.path.isfile(audio_path):
                    examples.append({"audio_path": audio_path, "text": sentence})
    print(f"  Loaded {len(examples)} Common Voice examples from {splits}")
    return examples

# SpecAugment
def spec_augment(mel):
    for _ in range(2):
        f  = torch.randint(0, 15, (1,)).item()
        f0 = torch.randint(0, mel.shape[0] - f, (1,)).item()
        mel[f0:f0+f, :] = 0
    for _ in range(2):
        t  = torch.randint(0, 50, (1,)).item()
        t0 = torch.randint(0, mel.shape[1] - t, (1,)).item()
        mel[:, t0:t0+t] = 0
    return mel

def preprocess_ls(example, tokenizer):
    audio_bytes = example["audio"]["bytes"]
    if audio_bytes is None:
        raise ValueError("No audio bytes")

    array, sr = sf.read(io.BytesIO(audio_bytes))
    array = array.astype("float32")

    if array.ndim > 1:
        array = array.mean(axis=1)

    array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)

    audio_tensor = torch.tensor(array, dtype=torch.float32)
    mel = log_mel_spectrogram(pad_or_trim(audio_tensor, N_SAMPLES))
    mel = spec_augment(mel)

    text = example["text"].lower().strip()
    if not text:
        raise ValueError("Empty text")

    tokens = (
        [tokenizer.sot]
        + tokenizer.encode(" " + text)
        + [tokenizer.eot]
    )
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"Token sequence too long: {len(tokens)} > {MAX_TOKENS}")
    return mel, torch.tensor(tokens, dtype=torch.long)

# Preprocess — Common Voice example (file path based, MP3)
def preprocess_cv(example, tokenizer):
    array, sr = librosa.load(example["audio_path"], sr=None, mono=True)
    array = array.astype("float32")

    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)

    import numpy as np
    if np.abs(array).max() < 1e-5:
        raise ValueError("Silent audio")

    audio_tensor = torch.tensor(array, dtype=torch.float32)
    mel = log_mel_spectrogram(pad_or_trim(audio_tensor, N_SAMPLES))
    mel = spec_augment(mel)

    text = example["text"].lower().strip()
    if not text:
        raise ValueError("Empty text")

    tokens = (
        [tokenizer.sot]
        + tokenizer.encode(" " + text)
        + [tokenizer.eot]
    )
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"Token sequence too long: {len(tokens)} > {MAX_TOKENS}")
    return mel, torch.tensor(tokens, dtype=torch.long)

# Mixed iterator: yields examples from LibriSpeech and Common Voice
# interleaved at a configurable ratio.
def mixed_epoch_iter(librispeech_ds, cv_examples, cv_ratio=0.5):
    """
    Each epoch:
    - Shuffle LibriSpeech indices
    - Shuffle CV examples
    - For each example, randomly pick CV or LibriSpeech based on cv_ratio
    """
    ls_indices = list(range(len(librispeech_ds)))
    epoch = 0

    while True:
        random.shuffle(ls_indices)
        random.shuffle(cv_examples)
        epoch += 1
        print(f"  Starting epoch {epoch} "
              f"(LibriSpeech: {len(ls_indices)}, CV: {len(cv_examples)})...")

        ls_iter = iter(ls_indices)
        cv_iter = iter(cv_examples)
        ls_done = False
        cv_done = False

        while not (ls_done and cv_done):
            use_cv = (random.random() < cv_ratio) and not cv_done

            if use_cv:
                try:
                    yield ("cv", next(cv_iter))
                except StopIteration:
                    cv_done = True
                    if not ls_done:
                        try:
                            yield ("ls", librispeech_ds[next(ls_iter)])
                        except StopIteration:
                            ls_done = True
            else:
                try:
                    yield ("ls", librispeech_ds[next(ls_iter)])
                except StopIteration:
                    ls_done = True
                    if not cv_done:
                        try:
                            yield ("cv", next(cv_iter))
                        except StopIteration:
                            cv_done = True
