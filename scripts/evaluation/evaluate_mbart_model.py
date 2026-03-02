#!/usr/bin/env python3
"""
mBART-50 Model Evaluation on Paper's Test Set
Evaluates the best mBART checkpoint (step 2000, loss 1.2666) using exact paper methodology
"""

import json
import string
import sys
import os
from pathlib import Path

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import evaluate

print("="*80)
print("mBART-50 MODEL EVALUATION - BEST CHECKPOINT")
print("="*80)
print()

# mBART language codes
LANG_TO_MBART_CODE = {
    "ea": "ar_AR",  # Ancient Egyptian -> Arabic
    "tnt": "th_TH",  # Transliteration -> Thai
    "en": "en_XX",
    "de": "de_DE",
}

# Best checkpoint path
CHECKPOINT_PATH = "mbart50_checkpoints_a6000/checkpoint_step2000_loss1.2666"
TEST_DATA_PATH = "../hiero-transformer/test_and_validation_data/test_data.json"

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Test data: {TEST_DATA_PATH}")
print()

# Load test data
print("Loading test data...")
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    raw_test_data = json.load(f)

print(f"✓ Loaded {len(raw_test_data)} test samples")
print()

# Process data (same as paper)
def extract_data_standard(data, src_lang, tgt_lang):
    """Extract standard source->target pairs"""
    filtered = [
        {
            "source": d["source"],
            "target": d["target"],
            "metadata": d["metadata"],
        }
        for d in data
        if (d["metadata"]["source_lang"] == src_lang
            and d["metadata"]["target_lang"] == tgt_lang
            and d["source"] != ""
            and d["target"] != "")
    ]
    print(f"  {src_lang} -> {tgt_lang}: {len(filtered)} datapoints")
    return filtered

def extract_data_transliteration_target(data, src_lang):
    """Extract hieroglyphs->transliteration pairs"""
    filtered = [
        {
            "source": d["source"],
            "target": d["transliteration"],
            "metadata": d["metadata"],
        }
        for d in data
        if (d["metadata"]["source_lang"] == src_lang
            and d["source"] != ""
            and d.get("transliteration", "") != "")
    ]
    print(f"  {src_lang} -> transliteration: {len(filtered)} datapoints")
    return filtered

def extract_data_transliteration_source(data, tgt_lang):
    """Extract transliteration->translation pairs"""
    filtered = [
        {
            "source": d["transliteration"],
            "target": d["target"],
            "metadata": d["metadata"],
        }
        for d in data
        if (d["metadata"]["target_lang"] == tgt_lang
            and d["target"] != ""
            and d.get("transliteration", "") != "")
    ]
    print(f"  transliteration -> {tgt_lang}: {len(filtered)} datapoints")
    return filtered

print("Processing test data...")
test_data = {
    "ea": {
        "de": extract_data_standard(raw_test_data, "ea", "de"),
        "en": extract_data_standard(raw_test_data, "ea", "en"),
        "tnt": extract_data_transliteration_target(raw_test_data, "ea"),
    },
    "tnt": {
        "de": extract_data_transliteration_source(raw_test_data, "de"),
        "en": extract_data_transliteration_source(raw_test_data, "en"),
    },
}
print()

# Load the mBART model
print("="*80)
print("Loading mBART-50 Best Checkpoint")
print("="*80)
print()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    print(f"Loading from: {CHECKPOINT_PATH}")
    model = MBartForConditionalGeneration.from_pretrained(CHECKPOINT_PATH).to(device).eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(CHECKPOINT_PATH)
    print("✓ Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

print()

# Generate predictions
print("="*80)
print("Generating predictions...")
print("="*80)
print()

total_samples = sum(len(data) for lang_data in test_data.values() for data in lang_data.values())
pbar = tqdm(total=total_samples, desc="Translating")

for src_lang, values in test_data.items():
    for tgt_lang, data in values.items():
        print(f"\nTranslating {src_lang} -> {tgt_lang} ({len(data)} samples)...")

        for element in data:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # Set source and target languages for mBART
                    tokenizer.src_lang = LANG_TO_MBART_CODE[src_lang]
                    tgt_code = LANG_TO_MBART_CODE[tgt_lang]

                    # Tokenize input
                    model_inputs = tokenizer(
                        element["source"],
                        return_tensors="pt",
                        max_length=128,
                        truncation=True
                    ).to(device)

                    # Generate (same parameters as paper: num_beams=10)
                    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_code]
                    generated_tokens = model.generate(
                        **model_inputs,
                        num_beams=10,
                        max_length=128,
                        forced_bos_token_id=forced_bos_token_id
                    )

                    # Decode
                    element["prediction"] = tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )[0]

            pbar.update(1)

pbar.close()
print("\n✓ All predictions generated")
print()

# Calculate metrics
print("="*80)
print("Calculating metrics...")
print("="*80)
print()

# Load metrics
metrics = {
    src_lang: {
        tgt_lang: {
            "sacrebleu": evaluate.load("sacrebleu"),
            "rouge": evaluate.load("rouge")
        }
        for tgt_lang in values.keys()
    }
    for src_lang, values in test_data.items()
}

# Add predictions to metrics
for src_lang, values in test_data.items():
    for tgt_lang, data in values.items():
        for element in data:
            for metric_name, metric in metrics[src_lang][tgt_lang].items():
                # Same preprocessing as paper
                pred = element["prediction"].strip(string.punctuation).lower()
                ref = element["target"].strip(string.punctuation).lower()

                metric.add_batch(
                    predictions=[pred],
                    references=[ref]
                )

# Compute final metrics
results = {}
for src_lang, values in metrics.items():
    results[src_lang] = {}
    for tgt_lang, metric_dict in values.items():
        sacrebleu_result = metric_dict["sacrebleu"].compute()
        rouge_result = metric_dict["rouge"].compute()

        results[src_lang][tgt_lang] = {
            "sacrebleu": sacrebleu_result["score"],
            "rougeL": 100 * rouge_result["rougeL"]
        }

# Display results
print("\n" + "="*80)
print("RESULTS - mBART-50 (Step 2000, Loss 1.2666)")
print("="*80)
print()

print("SacreBLEU Scores:")
print("-" * 60)
print(f"{'Source':<15} {'Target':<15} {'BLEU':>10}")
print("-" * 60)
for src_lang in ["ea", "tnt"]:
    for tgt_lang in results[src_lang].keys():
        bleu = results[src_lang][tgt_lang]["sacrebleu"]
        print(f"{src_lang:<15} {tgt_lang:<15} {bleu:>10.1f}")
print()

print("RougeL Scores:")
print("-" * 60)
print(f"{'Source':<15} {'Target':<15} {'RougeL':>10}")
print("-" * 60)
for src_lang in ["ea", "tnt"]:
    for tgt_lang in results[src_lang].keys():
        rouge = results[src_lang][tgt_lang]["rougeL"]
        print(f"{src_lang:<15} {tgt_lang:<15} {rouge:>10.1f}")
print()

# Compare with paper and Hugging Face model
print("="*80)
print("COMPARISON WITH PAPER & HUGGING FACE MODEL")
print("="*80)
print()

paper_results = {
    "Paper (DE+ENB)": {"de_bleu": 61.5, "en_bleu": 36.4, "de_rouge": 67.7, "en_rouge": 38.1},
    "HuggingFace": {"de_bleu": 35.4, "en_bleu": 12.4, "de_rouge": 64.2, "en_rouge": 35.5},
}

mbart_results = {
    "de_bleu": results["ea"]["de"]["sacrebleu"],
    "en_bleu": results["ea"]["en"]["sacrebleu"],
    "de_rouge": results["ea"]["de"]["rougeL"],
    "en_rouge": results["ea"]["en"]["rougeL"],
}

print(f"{'Model':<20} {'DE BLEU':<12} {'EN BLEU':<12} {'DE RougeL':<12} {'EN RougeL':<12}")
print("-" * 80)
print(f"{'Paper (DE+ENB)':<20} {paper_results['Paper (DE+ENB)']['de_bleu']:<12.1f} {paper_results['Paper (DE+ENB)']['en_bleu']:<12.1f} {paper_results['Paper (DE+ENB)']['de_rouge']:<12.1f} {paper_results['Paper (DE+ENB)']['en_rouge']:<12.1f}")
print(f"{'HuggingFace':<20} {paper_results['HuggingFace']['de_bleu']:<12.1f} {paper_results['HuggingFace']['en_bleu']:<12.1f} {paper_results['HuggingFace']['de_rouge']:<12.1f} {paper_results['HuggingFace']['en_rouge']:<12.1f}")
print(f"{'mBART-50 (Ours)':<20} {mbart_results['de_bleu']:<12.1f} {mbart_results['en_bleu']:<12.1f} {mbart_results['de_rouge']:<12.1f} {mbart_results['en_rouge']:<12.1f}")
print()

print("Difference from HuggingFace:")
print(f"  DE BLEU:  {mbart_results['de_bleu'] - paper_results['HuggingFace']['de_bleu']:+.1f}")
print(f"  EN BLEU:  {mbart_results['en_bleu'] - paper_results['HuggingFace']['en_bleu']:+.1f}")
print(f"  DE RougeL: {mbart_results['de_rouge'] - paper_results['HuggingFace']['de_rouge']:+.1f}")
print(f"  EN RougeL: {mbart_results['en_rouge'] - paper_results['HuggingFace']['en_rouge']:+.1f}")
print()

# Save results
output_file = "mbart50_evaluation_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "mBART-50",
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_info": {
            "step": 2000,
            "validation_loss": 1.2666,
            "status": "Best checkpoint (overfitting occurred after this)"
        },
        "test_data": TEST_DATA_PATH,
        "results": results,
        "comparison": {
            "paper_best": paper_results["Paper (DE+ENB)"],
            "paper_huggingface": paper_results["HuggingFace"],
            "mbart50_ours": mbart_results
        }
    }, f, indent=2)

print(f"✓ Results saved to: {output_file}")
print()

# Save example translations
examples_file = "mbart50_examples.txt"
with open(examples_file, 'w', encoding='utf-8') as f:
    f.write("EXAMPLE TRANSLATIONS FROM mBART-50 MODEL (Step 2000)\n")
    f.write("="*80 + "\n\n")

    for src_lang in ["ea", "tnt"]:
        for tgt_lang, data in test_data[src_lang].items():
            f.write(f"\n{src_lang} -> {tgt_lang}\n")
            f.write("-"*80 + "\n")

            # Show first 5 examples
            for i, element in enumerate(data[:5]):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"Source:     {element['source'][:100]}...\n" if len(element['source']) > 100 else f"Source:     {element['source']}\n")
                f.write(f"Reference:  {element['target']}\n")
                f.write(f"Prediction: {element['prediction']}\n")

print(f"✓ Example translations saved to: {examples_file}")
print()
print("="*80)
print("EVALUATION COMPLETE")
print("="*80)
