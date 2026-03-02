#!/usr/bin/env python3
"""
Exact replication of the paper's evaluation methodology
Tests the Hugging Face model (mattiadc/hiero-transformer) on their exact test set
"""

import json
import string
import sys
import os
from pathlib import Path

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import evaluate

print("="*80)
print("PAPER MODEL EVALUATION - EXACT REPLICATION")
print("="*80)
print()

# Language ID mapping (exactly as in their utils.py)
lang_to_m2m_lang_id = {
    "ea": "ar",  # Ancient Egyptian -> Arabic code
    "tnt": "lo",  # Transliteration -> Lao code
    "en": "en",
    "de": "de",
    "lKey": "my",
    "wordClass": "th",
}

# Load their test data
TEST_DATA_PATH = "../hiero-transformer/test_and_validation_data/test_data.json"
print(f"Loading test data from: {TEST_DATA_PATH}")

with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    raw_test_data = json.load(f)

print(f"✓ Loaded {len(raw_test_data)} test samples")
print()

# Process data exactly as they do in utils.py
def extract_data_standard(data, src_lang, tgt_lang):
    """Extract standard source->target pairs (exactly as in utils.py)"""
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

# Process test data (exactly as in their inference.py)
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

# Load the Hugging Face model
print("="*80)
print("Loading Hugging Face model: mattiadc/hiero-transformer")
print("="*80)
print()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    print("Downloading model from Hugging Face...")
    model = M2M100ForConditionalGeneration.from_pretrained("mattiadc/hiero-transformer").to(device).eval()
    tokenizer = M2M100Tokenizer.from_pretrained("mattiadc/hiero-transformer")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model from Hugging Face: {e}")
    print("\nTrying to load from facebook/m2m100_418M instead...")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device).eval()
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    print("✓ Loaded base M2M-100 model (not fine-tuned)")

print()

# Generate predictions (EXACTLY as in their inference.py)
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
                with torch.cuda.amp.autocast():
                    # Set source and target languages
                    tokenizer.src_lang = lang_to_m2m_lang_id[src_lang]
                    tokenizer.tgt_lang = lang_to_m2m_lang_id[tgt_lang]

                    # Tokenize input
                    model_inputs = tokenizer(
                        [element["source"]],
                        return_tensors="pt"
                    ).to(device)

                    # Generate (EXACTLY their parameters: num_beams=10)
                    generated_tokens = model.generate(
                        **model_inputs,
                        num_beams=10,  # CRITICAL: Same as paper
                        forced_bos_token_id=tokenizer.get_lang_id(
                            lang_to_m2m_lang_id[tgt_lang]
                        )
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

# Calculate metrics (EXACTLY as in their inference.py)
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

# Add predictions to metrics (EXACTLY their preprocessing)
for src_lang, values in test_data.items():
    for tgt_lang, data in values.items():
        for element in data:
            for metric_name, metric in metrics[src_lang][tgt_lang].items():
                # CRITICAL: Same preprocessing as paper
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
            "rougeL": 100 * rouge_result["rougeL"]  # New API returns float directly
        }

# Display results in paper format
print("\n" + "="*80)
print("RESULTS (Paper Table 4 format)")
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

# Compare with paper's reported results
print("="*80)
print("COMPARISON WITH PAPER")
print("="*80)
print()

paper_results = {
    "ea": {
        "de": {"bleu": 61.5, "rouge": 67.7},  # DE+ENB model (best)
        "en": {"bleu": 36.4, "rouge": 38.1},  # DE+ENB model (best)
    }
}

print(f"{'Metric':<20} {'Paper (DE+ENB)':<20} {'HuggingFace':<20} {'Difference':<20}")
print("-" * 80)

for tgt_lang in ["de", "en"]:
    if tgt_lang in results["ea"]:
        paper_bleu = paper_results["ea"][tgt_lang]["bleu"]
        hf_bleu = results["ea"][tgt_lang]["sacrebleu"]
        diff_bleu = hf_bleu - paper_bleu

        paper_rouge = paper_results["ea"][tgt_lang]["rouge"]
        hf_rouge = results["ea"][tgt_lang]["rougeL"]
        diff_rouge = hf_rouge - paper_rouge

        print(f"ea->de BLEU        {paper_bleu:>10.1f}         {hf_bleu:>10.1f}         {diff_bleu:>+10.1f}" if tgt_lang == "de" else f"ea->en BLEU        {paper_bleu:>10.1f}         {hf_bleu:>10.1f}         {diff_bleu:>+10.1f}")
        print(f"ea->de RougeL      {paper_rouge:>10.1f}         {hf_rouge:>10.1f}         {diff_rouge:>+10.1f}" if tgt_lang == "de" else f"ea->en RougeL      {paper_rouge:>10.1f}         {hf_rouge:>10.1f}         {diff_rouge:>+10.1f}")
        print()

# Save results
output_file = "paper_model_evaluation_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "mattiadc/hiero-transformer",
        "test_data": TEST_DATA_PATH,
        "results": results,
        "paper_comparison": {
            "paper_best_model": "DE+ENB",
            "paper_results": paper_results,
            "differences": {
                "ea_de_bleu": results["ea"]["de"]["sacrebleu"] - 61.5,
                "ea_en_bleu": results["ea"]["en"]["sacrebleu"] - 36.4,
            }
        }
    }, f, indent=2)

print(f"✓ Results saved to: {output_file}")
print()

# Save example translations
examples_file = "paper_model_examples.txt"
with open(examples_file, 'w', encoding='utf-8') as f:
    f.write("EXAMPLE TRANSLATIONS FROM HUGGING FACE MODEL\n")
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
