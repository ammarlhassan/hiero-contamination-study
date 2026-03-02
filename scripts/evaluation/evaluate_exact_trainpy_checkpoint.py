#!/usr/bin/env python3
"""
Evaluate the exact train.py checkpoint with contamination splits
Checkpoint: ../hiero-transformer/checkpoint_total_steps=43040_loss=1.60
"""

import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sacrebleu.metrics import BLEU, CHRF
import re

# Checkpoint path
CHECKPOINT_PATH = "../hiero-transformer/checkpoint_total_steps=43040_loss=1.60"

def load_test_data_with_contamination():
    """Load test data and mark contamination status"""
    
    # Load training targets for contamination check
    training_path = Path("../hiero-transformer/training_data/training_data.json")
    with open(training_path) as f:
        training_data = json.load(f)
    
    # Get all German training targets (normalized)
    training_targets = set()
    for sample in training_data:
        target = sample.get('target', '') or ''
        if target.strip():
            target_norm = target.strip().lower()
            target_norm = re.sub(r'[^\w\s]', '', target_norm)
            training_targets.add(target_norm)
    
    print(f"Loaded {len(training_targets)} unique training targets")
    
    # Load test data
    test_path = Path("../hiero-transformer/test_and_validation_data/test_data.json")
    with open(test_path) as f:
        test_data = json.load(f)
    
    # Filter for German samples with non-empty source and target
    samples = []
    for sample in test_data:
        meta = sample.get('metadata', {})
        if isinstance(meta, dict) and meta.get('target_lang') == 'de':
            target = sample.get('target', '') or ''
            source = sample.get('source', '') or ''
            if target.strip() and source.strip():
                target_norm = target.strip().lower()
                target_norm = re.sub(r'[^\w\s]', '', target_norm)
                
                contaminated = target_norm in training_targets
                
                samples.append({
                    'source': source.strip(),
                    'target': target.strip(),
                    'target_normalized': target_norm,
                    'contaminated': contaminated
                })
    
    print(f"Loaded {len(samples)} German test samples")
    contam_count = sum(1 for s in samples if s['contaminated'])
    print(f"Contaminated: {contam_count}, Clean: {len(samples) - contam_count}")
    
    return samples

def translate_batch(model, tokenizer, sources, batch_size=8):
    """Translate a batch of sources"""
    predictions = []
    
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        
        # Set source language to Arabic proxy for hieroglyphics
        tokenizer.src_lang = "ar"
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id("de"),
                max_length=128,
                num_beams=5
            )
        
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(decoded)
    
    return predictions

def compute_corpus_bleu(predictions, references):
    """Compute corpus BLEU"""
    bleu = BLEU()
    return bleu.corpus_score(predictions, [references]).score

def compute_corpus_chrf(predictions, references):
    """Compute corpus chrF++"""
    chrf = CHRF(word_order=2)
    return chrf.corpus_score(predictions, [references]).score

def main():
    print("="*80)
    print("EXACT TRAIN.PY CHECKPOINT EVALUATION")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()
    
    # Load data
    samples = load_test_data_with_contamination()
    
    # Load model
    print(f"\nLoading model from {CHECKPOINT_PATH}...")
    model = M2M100ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH).to(device)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model.eval()
    print("✓ Model loaded")
    
    # Split by contamination status
    all_sources = [s['source'] for s in samples]
    all_targets = [s['target'] for s in samples]
    
    clean_indices = [i for i, s in enumerate(samples) if not s['contaminated']]
    contam_indices = [i for i, s in enumerate(samples) if s['contaminated']]
    
    clean_sources = [all_sources[i] for i in clean_indices]
    clean_targets = [all_targets[i] for i in clean_indices]
    
    contam_sources = [all_sources[i] for i in contam_indices]
    contam_targets = [all_targets[i] for i in contam_indices]
    
    print(f"\nSplit sizes: All={len(samples)}, Clean={len(clean_sources)}, Contaminated={len(contam_sources)}")
    
    # Translate all samples
    print("\nTranslating all samples...")
    all_predictions = translate_batch(model, tokenizer, all_sources)
    
    # Extract by subset
    clean_predictions = [all_predictions[i] for i in clean_indices]
    contam_predictions = [all_predictions[i] for i in contam_indices]
    
    # Compute metrics
    print("\nComputing metrics...")
    
    results = {
        'model': 'Exact train.py (M2M-100)',
        'checkpoint': CHECKPOINT_PATH,
        'all': {
            'n': len(samples),
            'bleu': compute_corpus_bleu(all_predictions, all_targets),
            'chrf': compute_corpus_chrf(all_predictions, all_targets)
        },
        'contaminated': {
            'n': len(contam_sources),
            'bleu': compute_corpus_bleu(contam_predictions, contam_targets),
            'chrf': compute_corpus_chrf(contam_predictions, contam_targets)
        },
        'clean': {
            'n': len(clean_sources),
            'bleu': compute_corpus_bleu(clean_predictions, clean_targets),
            'chrf': compute_corpus_chrf(clean_predictions, clean_targets)
        }
    }
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Exact train.py Checkpoint")
    print("="*80)
    print(f"\n{'Subset':<15} {'N':>5} {'BLEU':>10} {'chrF++':>10}")
    print("-"*45)
    print(f"{'All':<15} {results['all']['n']:>5} {results['all']['bleu']:>10.1f} {results['all']['chrf']:>10.1f}")
    print(f"{'Contaminated':<15} {results['contaminated']['n']:>5} {results['contaminated']['bleu']:>10.1f} {results['contaminated']['chrf']:>10.1f}")
    print(f"{'Clean':<15} {results['clean']['n']:>5} {results['clean']['bleu']:>10.1f} {results['clean']['chrf']:>10.1f}")
    print("-"*45)
    gap = results['contaminated']['bleu'] - results['clean']['bleu']
    print(f"\nContamination gap: +{gap:.1f} BLEU points")
    
    # Save results
    output_path = Path("exact_trainpy_evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Print in paper table format
    print("\n" + "="*80)
    print("FOR PAPER TABLE 2:")
    print("="*80)
    print(f"Exact train.py & All & {results['all']['n']} & {results['all']['bleu']:.1f} & {results['all']['chrf']:.1f} \\\\")
    print(f"         & Contaminated & {results['contaminated']['n']} & {results['contaminated']['bleu']:.1f} & {results['contaminated']['chrf']:.1f} \\\\")
    print(f"         & Clean & {results['clean']['n']} & {results['clean']['bleu']:.1f} & {results['clean']['chrf']:.1f} \\\\")

if __name__ == "__main__":
    main()
