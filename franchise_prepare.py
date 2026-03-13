"""
One-time data preparation for autoresearch using a local FDD franchise corpus.

Reads FDD extracted text files from the network corpus directory, trains a
franchise-specific BPE tokenizer, and writes binary token shards that
prepare.py's make_dataloader can read transparently.

Usage:
    uv run franchise_prepare.py                    # full prep
    uv run franchise_prepare.py --retrain-tokenizer  # force tokenizer retrain
    uv run franchise_prepare.py --corpus-dir "D:\\other\\path"  # override corpus path
    uv run franchise_prepare.py --sample-only 50   # process only 50 brands (testing)

Output (all local, network never touched again after this):
    ~/.cache/autoresearch/tokenizer/tokenizer.pkl
    ~/.cache/autoresearch/tokenizer/token_bytes.pt
    ~/.cache/autoresearch/data/shard_00000.bin  ...  shard_NNNNN.bin
    ~/.cache/autoresearch/data/shard_val.bin
    ~/.cache/autoresearch/val_brands.txt
    ~/.cache/autoresearch/progress.json
"""

import os
import sys
import re
import time
import math
import pickle
import json
import argparse
import random
import struct
import array
from html.parser import HTMLParser

import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Configuration — edit CORPUS_DIR if the network path changes
# ---------------------------------------------------------------------------

CORPUS_DIR = r"Z:\Core Data\FDD Extraction\OCR_MCD Extraction Files"

CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR      = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_BRANDS_FILE  = os.path.join(CACHE_DIR, "val_brands.txt")
PROGRESS_FILE    = os.path.join(CACHE_DIR, "progress.json")

VAL_FRACTION        = 0.05   # hold out last 5% of brands alphabetically
TOKENIZER_SAMPLE    = 200    # brands sampled for tokenizer training
TOKENS_PER_SHARD    = 50_000_000  # ~50M tokens per train shard (~200MB on disk)
VOCAB_SIZE          = 8192   # must match prepare.py

# BPE split pattern — identical to prepare.py so tokenizer is compatible
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Strips HTML tags, keeping inner text with whitespace normalization."""
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self):
        return " ".join(self._parts)


def clean_text(raw: str) -> str:
    """
    Clean a raw FDD extracted text:
    1. Strip '--- Page N ---' markers
    2. Strip HTML tags (from table markup), keep inner text
    3. Normalize whitespace
    """
    # Remove page markers like "--- Page 1 ---" or "--- Page 123 ---"
    text = re.sub(r"-{2,}\s*Page\s+\d+\s*-{2,}", " ", raw)

    # Strip HTML blocks — find any content containing < > and run through parser
    # Only process segments that look like HTML to avoid mangling URLs etc.
    def strip_html_block(m):
        stripper = _HTMLStripper()
        try:
            stripper.feed(m.group(0))
            return " " + stripper.get_text() + " "
        except Exception:
            return " " + m.group(0) + " "

    text = re.sub(r"<[^>]+>.*?</[^>]+>|<[^>]+/>", strip_html_block, text, flags=re.DOTALL)

    # Collapse excess whitespace (keep single newlines as sentence breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Corpus scanning
# ---------------------------------------------------------------------------

def scan_corpus(corpus_dir: str) -> list[str]:
    """
    Walk corpus_dir and return sorted list of *_extracted.txt file paths.
    Expected layout:
        corpus_dir/
            {fruns}_{year}/
                {fruns}_{year}_extracted.txt
    """
    if not os.path.isdir(corpus_dir):
        print(f"ERROR: Corpus directory not found: {corpus_dir!r}")
        print("Check that the Z: drive is mapped and the path is correct.")
        sys.exit(1)

    txt_files = []
    for entry in os.scandir(corpus_dir):
        if not entry.is_dir():
            continue
        folder_name = entry.name
        candidate = os.path.join(entry.path, f"{folder_name}_extracted.txt")
        if os.path.isfile(candidate):
            txt_files.append(candidate)
        else:
            # Fallback: find any *_extracted.txt in the folder
            try:
                for f in os.scandir(entry.path):
                    if f.name.endswith("_extracted.txt"):
                        txt_files.append(f.path)
                        break
            except PermissionError:
                pass

    txt_files.sort()
    return txt_files


def compute_val_split(all_files: list[str], val_fraction: float = VAL_FRACTION) -> tuple[list[str], list[str]]:
    """
    Split files into train and val.
    Val = last val_fraction of brands (alphabetically by file path).
    If val_brands.txt already exists, honours those pinned brands exactly.
    """
    if os.path.exists(VAL_BRANDS_FILE):
        with open(VAL_BRANDS_FILE, "r", encoding="utf-8") as f:
            val_set = set(line.strip() for line in f if line.strip())
        val_files   = [p for p in all_files if os.path.basename(os.path.dirname(p)) in val_set]
        train_files = [p for p in all_files if os.path.basename(os.path.dirname(p)) not in val_set]
        print(f"Corpus: loaded pinned val split — {len(val_files)} val, {len(train_files)} train brands")
        return train_files, val_files

    n_val = max(1, int(len(all_files) * val_fraction))
    val_files   = all_files[-n_val:]
    train_files = all_files[:-n_val]

    # Persist the val split so it stays stable across re-runs
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(VAL_BRANDS_FILE, "w", encoding="utf-8") as f:
        for p in val_files:
            f.write(os.path.basename(os.path.dirname(p)) + "\n")

    print(f"Corpus: created val split — {len(val_files)} val, {len(train_files)} train brands")
    print(f"Corpus: val brands pinned to {VAL_BRANDS_FILE}")
    return train_files, val_files


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def read_text(path: str) -> str:
    """Read a txt file with fallback encoding."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def train_tokenizer(train_files: list[str], retrain: bool = False, sample_size: int = TOKENIZER_SAMPLE):
    """
    Train a franchise-specific BPE tokenizer using rustbpe.
    Saves tokenizer.pkl and token_bytes.pt to TOKENIZER_DIR.
    Identical save format to prepare.py so Tokenizer.from_directory() works.
    """
    tokenizer_pkl    = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if not retrain and os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR} (use --retrain-tokenizer to redo)")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # Sample a subset of train brands for tokenizer training to limit network reads
    sample = random.sample(train_files, min(sample_size, len(train_files)))
    print(f"Tokenizer: training on {len(sample)} sampled brands (of {len(train_files)} train brands)...")

    def doc_iterator():
        for path in sample:
            raw = read_text(path)
            if raw:
                yield clean_text(raw)

    t0 = time.time()
    tok = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tok.train_from_iterator(doc_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding
    pattern         = tok.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    tokens_offset   = len(mergeable_ranks)
    special_tokens  = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup (used by evaluate_bpb in prepare.py)
    print("Tokenizer: building token_bytes lookup...")
    special_set      = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test    = "Franchise royalty fee: 6% of gross sales. Initial investment: $250,000."
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Binary shard writer
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_train": [], "completed_val": False, "train_shard_tokens": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


class ShardWriter:
    """
    Writes int32 token IDs into fixed-size binary shard files.
    Each file is a flat array of int32 values (little-endian).
    A header uint64 at byte 0 stores the token count in the shard.
    """
    HEADER_BYTES = 8  # uint64 token count

    def __init__(self, path: str):
        self.path     = path
        self.tmp_path = path + ".tmp"
        self._buf     = array.array("i")  # signed int32
        self._count   = 0

    def write(self, token_ids: list[int]):
        self._buf.extend(token_ids)
        self._count += len(token_ids)

    def flush(self):
        """Write tmp file then atomically rename."""
        with open(self.tmp_path, "wb") as f:
            f.write(struct.pack("<Q", self._count))  # uint64 header
            self._buf.tofile(f)
        os.replace(self.tmp_path, self.path)

    @property
    def token_count(self):
        return self._count


def write_shards(
    files: list[str],
    enc: tiktoken.Encoding,
    split: str,
    progress: dict,
    tokens_per_shard: int = TOKENS_PER_SHARD,
    total_brands: int = 0,
):
    """
    Stream txt files from the network, tokenize, and write binary shards.
    Resumes from where it left off using progress.json.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    bos_id       = enc.encode_single_token(BOS_TOKEN)
    completed    = set(progress.get("completed_train", []) if split == "train" else [])

    if split == "val":
        if progress.get("completed_val"):
            val_path = os.path.join(DATA_DIR, "shard_val.bin")
            if os.path.exists(val_path):
                print(f"Shards ({split}): already written at {val_path}, skipping.")
                return
        shard_path = os.path.join(DATA_DIR, "shard_val.bin")
        writer     = ShardWriter(shard_path)
        n_written  = 0
        n_total    = len(files)
        t0         = time.time()

        for idx, path in enumerate(files):
            brand = os.path.basename(os.path.dirname(path))
            raw   = read_text(path)
            if not raw:
                print(f"  [{idx+1}/{n_total}] {brand} — EMPTY, skipping")
                continue
            text   = clean_text(raw)
            tokens = [bos_id] + enc.encode_ordinary(text)
            writer.write(tokens)
            n_written += len(tokens)
            elapsed = time.time() - t0
            rate    = n_written / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{n_total}] {brand} — {len(tokens):,} tokens  |  total: {n_written:,}  |  {rate/1e6:.2f}M tok/s")

        writer.flush()
        progress["completed_val"] = True
        save_progress(progress)
        print(f"Shards (val): wrote {n_written:,} tokens → {shard_path}")
        return

    # --- Train shards ---
    shard_idx    = len([p for p in os.listdir(DATA_DIR) if re.match(r"shard_\d{5}\.bin", p)])
    # Find highest existing complete shard to resume properly
    existing_shards = sorted(
        p for p in os.listdir(DATA_DIR) if re.match(r"shard_\d{5}\.bin", p)
    )
    shard_idx = len(existing_shards)

    writer       = ShardWriter(os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.bin"))
    n_total      = len(files)
    n_processed  = 0
    total_tokens = 0
    t0           = time.time()

    for idx, path in enumerate(files):
        brand = os.path.basename(os.path.dirname(path))

        if brand in completed:
            n_processed += 1
            continue

        raw = read_text(path)
        if not raw:
            print(f"  [{idx+1}/{n_total}] {brand} — EMPTY, skipping")
            completed.add(brand)
            progress["completed_train"] = list(completed)
            save_progress(progress)
            continue

        text   = clean_text(raw)
        tokens = [bos_id] + enc.encode_ordinary(text)
        writer.write(tokens)
        total_tokens += len(tokens)
        n_processed  += 1

        elapsed = time.time() - t0
        rate    = total_tokens / elapsed if elapsed > 0 else 0
        pct     = 100 * n_processed / n_total
        print(f"  [{n_processed}/{n_total}] {brand} — {len(tokens):,} tokens  |  shard {shard_idx} ({writer.token_count/1e6:.1f}M)  |  {pct:.1f}%  |  {rate/1e6:.2f}M tok/s")

        # Roll to next shard when current is full
        if writer.token_count >= tokens_per_shard:
            writer.flush()
            print(f"  → Closed shard_{shard_idx:05d}.bin ({writer.token_count/1e6:.1f}M tokens)")
            shard_idx += 1
            writer = ShardWriter(os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.bin"))

        # Checkpoint progress every 50 brands
        completed.add(brand)
        if n_processed % 50 == 0:
            progress["completed_train"] = list(completed)
            save_progress(progress)

    # Flush final partial shard (only if it has tokens)
    if writer.token_count > 0:
        writer.flush()
        print(f"  → Closed shard_{shard_idx:05d}.bin ({writer.token_count/1e6:.1f}M tokens)")

    progress["completed_train"] = list(completed)
    save_progress(progress)
    print(f"Shards (train): wrote {total_tokens:,} tokens across {shard_idx+1} shards")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare franchise corpus for autoresearch")
    parser.add_argument("--corpus-dir",        type=str, default=CORPUS_DIR,
                        help="Path to the FDD extraction corpus folder")
    parser.add_argument("--retrain-tokenizer", action="store_true",
                        help="Force tokenizer retraining even if one already exists")
    parser.add_argument("--sample-only",       type=int, default=None,
                        help="Process only N brands total (for quick testing)")
    parser.add_argument("--tokenizer-sample",  type=int, default=TOKENIZER_SAMPLE,
                        help=f"Number of brands to sample for tokenizer training (default: {TOKENIZER_SAMPLE})")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    print(f"Cache directory : {CACHE_DIR}")
    print(f"Corpus directory: {args.corpus_dir}")
    print()

    # Step 1: Scan corpus
    print("Step 1: Scanning corpus...")
    all_files = scan_corpus(args.corpus_dir)
    if not all_files:
        print("ERROR: No *_extracted.txt files found in corpus directory.")
        sys.exit(1)
    print(f"Corpus: found {len(all_files)} brand documents")

    if args.sample_only:
        print(f"Corpus: --sample-only {args.sample_only} — limiting to {args.sample_only} brands")
        all_files = all_files[:args.sample_only]

    train_files, val_files = compute_val_split(all_files)
    print()

    # Step 2: Train tokenizer
    print("Step 2: Tokenizer...")
    train_tokenizer(train_files, retrain=args.retrain_tokenizer, sample_size=args.tokenizer_sample)
    print()

    # Load the trained tokenizer
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    with open(tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)

    # Step 3: Write shards
    progress = load_progress()

    print("Step 3: Writing train shards...")
    write_shards(train_files, enc, "train", progress, total_brands=len(train_files))
    print()

    print("Step 4: Writing val shard...")
    write_shards(val_files, enc, "val", progress, total_brands=len(val_files))
    print()

    # Summary
    train_shards = sorted(p for p in os.listdir(DATA_DIR) if re.match(r"shard_\d{5}\.bin", p))
    val_shard    = os.path.join(DATA_DIR, "shard_val.bin")
    print("=" * 60)
    print(f"Done! Ready to run: uv run train.py")
    print(f"  Train shards : {len(train_shards)} files in {DATA_DIR}")
    print(f"  Val shard    : {val_shard}")
    print(f"  Tokenizer    : {tokenizer_pkl}")
    print(f"  Val brands   : {VAL_BRANDS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
