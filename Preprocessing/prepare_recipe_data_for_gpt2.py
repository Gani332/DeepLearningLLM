"""
FULL PIPELINE:
- Load 5M dataset
- Clean + filter + sample (Option A)
- Convert to GPT-2 format
- Create HuggingFace Arrow datasets
- Save train/val datasets to disk

Author: AI Assistant
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm
import random

# =============================================================================
# CONFIG
# =============================================================================

INPUT_CSV = "datasets/Cleaned/clean_recipes.csv"
OUTPUT_DIR = "datasets/Cleaned/recipe_gpt2"
TARGET_SIZE = 50_000
MAX_RESPONSE_WORDS = 500
MIN_RESPONSE_WORDS = 20
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print("=" * 80)
print("GPT-2 RECIPE DATASET: CLEANING → CONVERSION → ARROW EXPORT")
print("=" * 80)

# =============================================================================
# STEP 1 — LOAD RAW DATASET
# =============================================================================

print("\n[1/6] Loading dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"      ✓ Loaded {len(df):,} rows")
print(f"      Columns found: {list(df.columns)}")

if "prompt" not in df.columns or "response" not in df.columns:
    raise ValueError("CSV must contain columns: 'prompt' and 'response'")

# Remove NaN
df = df.dropna(subset=["prompt", "response"])
print(f"      ✓ After NaN removal: {len(df):,} rows")


# =============================================================================
# STEP 2 — BASIC CLEANING
# =============================================================================

print("\n[2/6] Cleaning formatting...")

df["prompt"] = df["prompt"].astype(str).str.strip()
df["response"] = df["response"].astype(str).str.strip()

df = df[(df["prompt"] != "") & (df["response"] != "")]
print(f"      ✓ After empty-row removal: {len(df):,} rows")


# =============================================================================
# STEP 3 — FILTERING HIGH-QUALITY EXAMPLES
# =============================================================================

print("\n[3/6] Filtering long/short/irrelevant recipes...")

# Restrict recipe length
df["response_word_count"] = df["response"].apply(lambda x: len(x.split()))
df = df[
    (df["response_word_count"] >= MIN_RESPONSE_WORDS) &
    (df["response_word_count"] <= MAX_RESPONSE_WORDS)
]
print(f"      ✓ After length filtering: {len(df):,} rows")

# Keep only cooking prompts
COOKING_KEYS = [
    "i have these ingredients",
    "what can i make",
    "how do i cook",
    "recipe",
    "make",
    "cook",
]

def is_relevant_prompt(p):
    low = p.lower()
    return any(k in low for k in COOKING_KEYS)

df = df[df["prompt"].apply(is_relevant_prompt)]
print(f"      ✓ After relevance filtering: {len(df):,} rows")


# =============================================================================
# STEP 4 — SAMPLE DOWN TO 50K
# =============================================================================

print("\n[4/6] Sampling dataset...")

if len(df) > TARGET_SIZE:
    df = df.sample(n=TARGET_SIZE, random_state=RANDOM_SEED)
    print(f"      ✓ Sampled down to {TARGET_SIZE:,} rows")
else:
    print(f"      ⚠ Using full {len(df):,} rows (less than target)")


# Rename for GPT-2 script
df = df.rename(columns={"prompt": "input", "response": "output"})
df = df[["input", "output"]]


# =============================================================================
# STEP 5 — CONVERT TO GPT-2 FORMAT
# =============================================================================

print("\n[5/6] Converting to GPT-2 training format...")

def to_gpt2_format(row):
    return f"<|user|> {row['input']} <|assistant|> {row['output']} <|endoftext|>"

tqdm.pandas(desc="      formatting")
df["text"] = df.progress_apply(to_gpt2_format, axis=1)

print(f"      ✓ Converted {len(df):,} examples")


# Train/val split
df = df.sample(frac=1, random_state=RANDOM_SEED)
split = int(len(df) * TRAIN_SPLIT)

train_df = df[:split]
val_df = df[split:]

print(f"\n      Train size: {len(train_df):,}")
print(f"      Val size:   {len(val_df):,}")


# =============================================================================
# STEP 6 — SAVE AS HUGGINGFACE ARROW DATASETS
# =============================================================================

print("\n[6/6] Creating HuggingFace Arrow datasets...")

train_dataset = Dataset.from_pandas(train_df[["text"]])
val_dataset = Dataset.from_pandas(val_df[["text"]])

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_dataset.save_to_disk(f"{OUTPUT_DIR}/train")
val_dataset.save_to_disk(f"{OUTPUT_DIR}/val")

print(f"      ✓ Saved Arrow dataset to {OUTPUT_DIR}/train")
print(f"      ✓ Saved Arrow dataset to {OUTPUT_DIR}/val")


print("\n" + "="*80)
print("✓ COMPLETE — Dataset is ready for GPT-2 fine-tuning!")
print("="*80)
