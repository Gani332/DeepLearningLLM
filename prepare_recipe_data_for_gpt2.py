"""
Data Preprocessing for GPT-2 Stage 2 Fine-Tuning
Converts conversational_training_data.csv to GPT-2 format

Author: AI Assistant
Purpose: Prepare recipe data for Stage 2 fine-tuning
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
import random
from tqdm import tqdm

print("=" * 80)
print("PREPARING RECIPE DATA FOR GPT-2 FINE-TUNING (STAGE 2)")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "datasets/Cleaned/conversational_training_data.csv"
OUTPUT_DIR = "datasets/Cleaned/recipe_gpt2"
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n[1/5] Loading conversational training data...")
df = pd.read_csv(INPUT_CSV)
print(f"      ✓ Loaded {len(df):,} training pairs")
print(f"      Columns: {list(df.columns)}")

# Show sample
print("\n      Sample (before conversion):")
print(f"      Input:  {df.iloc[0]['input'][:100]}...")
print(f"      Output: {df.iloc[0]['output'][:100]}...")

# ============================================================================
# STEP 2: Convert to GPT-2 Format
# ============================================================================

print("\n[2/5] Converting to GPT-2 format...")

def convert_to_gpt2_format(row):
    """
    Convert input/output pair to GPT-2 training format.
    
    Format: <|user|> {input} <|assistant|> {output} <|endoftext|>
    
    This format:
    - Clearly separates user and assistant messages
    - Uses special tokens for structure
    - Ends with <|endoftext|> to mark completion
    """
    input_text = row['input'].strip()
    output_text = row['output'].strip()
    
    # GPT-2 training format
    formatted = f"<|user|> {input_text} <|assistant|> {output_text} <|endoftext|>"
    
    return formatted

# Apply conversion
tqdm.pandas(desc="      Converting")
df['text'] = df.progress_apply(convert_to_gpt2_format, axis=1)

print(f"      ✓ Converted {len(df):,} examples")

# Show sample after conversion
print("\n      Sample (after conversion):")
print(f"      {df.iloc[0]['text'][:200]}...")

# ============================================================================
# STEP 3: Split Train/Val
# ============================================================================

print(f"\n[3/5] Splitting into train ({TRAIN_SPLIT*100:.0f}%) and validation ({(1-TRAIN_SPLIT)*100:.0f}%)...")

# Shuffle
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Split
split_idx = int(len(df) * TRAIN_SPLIT)
train_df = df[:split_idx]
val_df = df[split_idx:]

print(f"      ✓ Train: {len(train_df):,} examples")
print(f"      ✓ Val:   {len(val_df):,} examples")

# ============================================================================
# STEP 4: Create HuggingFace Datasets
# ============================================================================

print("\n[4/5] Creating HuggingFace datasets...")

# Create datasets (only keep 'text' column)
train_dataset = Dataset.from_pandas(train_df[['text']])
val_dataset = Dataset.from_pandas(val_df[['text']])

# Create dataset dict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

print(f"      ✓ Train dataset: {len(train_dataset):,} examples")
print(f"      ✓ Val dataset:   {len(val_dataset):,} examples")

# ============================================================================
# STEP 5: Save to Disk
# ============================================================================

print(f"\n[5/5] Saving datasets to {OUTPUT_DIR}...")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Save train and val separately (format expected by finetune_llm)
train_dataset.save_to_disk(f"{OUTPUT_DIR}/train")
val_dataset.save_to_disk(f"{OUTPUT_DIR}/val")

print(f"      ✓ Saved train dataset to: {OUTPUT_DIR}/train")
print(f"      ✓ Saved val dataset to:   {OUTPUT_DIR}/val")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

# Load back to verify
from datasets import load_from_disk

train_loaded = load_from_disk(f"{OUTPUT_DIR}/train")
val_loaded = load_from_disk(f"{OUTPUT_DIR}/val")

print(f"\n✓ Train dataset loaded: {len(train_loaded):,} examples")
print(f"✓ Val dataset loaded:   {len(val_loaded):,} examples")

print("\nSample from train dataset:")
print("-" * 80)
print(train_loaded[0]['text'][:300] + "...")
print("-" * 80)

print("\nSample from val dataset:")
print("-" * 80)
print(val_loaded[0]['text'][:300] + "...")
print("-" * 80)

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

# Calculate statistics
train_lengths = [len(text.split()) for text in train_df['text']]
val_lengths = [len(text.split()) for text in val_df['text']]

print(f"\nTrain dataset:")
print(f"  Total examples: {len(train_df):,}")
print(f"  Avg length: {sum(train_lengths)/len(train_lengths):.1f} words")
print(f"  Min length: {min(train_lengths)} words")
print(f"  Max length: {max(train_lengths)} words")

print(f"\nValidation dataset:")
print(f"  Total examples: {len(val_df):,}")
print(f"  Avg length: {sum(val_lengths)/len(val_lengths):.1f} words")
print(f"  Min length: {min(val_lengths)} words")
print(f"  Max length: {max(val_lengths)} words")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ DATA PREPARATION COMPLETE!")
print("=" * 80)

print(f"""
Output files created:
  {OUTPUT_DIR}/train/  (HuggingFace dataset)
  {OUTPUT_DIR}/val/    (HuggingFace dataset)

Format:
  <|user|> [input] <|assistant|> [output] <|endoftext|>
""")
