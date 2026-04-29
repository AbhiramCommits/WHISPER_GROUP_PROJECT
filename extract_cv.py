"""
Extracts Common Voice train, dev, and test splits from the tar.gz.
Run this once before training. Puts everything in cv-data/ folder.

Usage:
    python extract_cv.py
"""

import tarfile
import csv
import io
import os

TAR_PATH   = r"common-voice-scripted-speech-25-0-englis-0c0b9a16.tar.gz"
OUTPUT_DIR = r"cv-data"

os.makedirs(os.path.join(OUTPUT_DIR, "clips"), exist_ok=True)

# ── Pass 1: extract all TSV files and collect clip filenames ──────────────────
print("Pass 1: reading TSV files...")

splits_to_extract = ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv"]
tsv_data   = {}   # split name → raw bytes
all_clips  = set()

with tarfile.open(TAR_PATH, "r:gz") as tar:
    for member in tar:
        name = member.name
        for split in splits_to_extract:
            if name.endswith(f"/en/{split}"):
                f = tar.extractfile(member)
                raw = f.read()
                tsv_data[split] = raw
                reader = csv.DictReader(
                    io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"),
                    delimiter="\t"
                )
                count = 0
                for row in reader:
                    all_clips.add(row["path"].strip())
                    count += 1
                print(f"  Found {split}: {count} rows")

# Write TSV files
for split, raw in tsv_data.items():
    out_path = os.path.join(OUTPUT_DIR, split)
    with open(out_path, "wb") as f:
        f.write(raw)
    print(f"  Wrote {out_path}")

print(f"\nTotal unique clips needed: {len(all_clips)}")

# ── Pass 2: extract clips ─────────────────────────────────────────────────────
print(f"\nPass 2: extracting {len(all_clips)} clips (this will take a while)...")
print("Progress printed every 5000 clips.\n")

clips_dir = os.path.join(OUTPUT_DIR, "clips")
extracted  = 0
skipped    = 0

with tarfile.open(TAR_PATH, "r:gz") as tar:
    for member in tar:
        if "/en/clips/" not in member.name:
            continue
        basename = os.path.basename(member.name)
        if basename not in all_clips:
            continue
        out_path = os.path.join(clips_dir, basename)
        if os.path.isfile(out_path):   # skip already extracted
            extracted += 1
            continue
        with tar.extractfile(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        extracted += 1
        if extracted % 5000 == 0:
            print(f"  {extracted}/{len(all_clips)} clips extracted...")

print(f"\nDone. Extracted {extracted} clips, skipped {skipped}.")
print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
print(f"\nNext step: update CV_DATA_ROOT in whisper_dataset.py to:")
print(f"  {os.path.abspath(OUTPUT_DIR)}")
