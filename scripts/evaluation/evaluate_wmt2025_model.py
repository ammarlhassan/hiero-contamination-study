#!/usr/bin/env python3
"""
Evaluate the WMT2025 trained M2M-100 model using SacreBLEU and RougeL
Same methodology as the paper for fair comparison
"""

import json
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer

# Configuration
CHECKPOINT_PATH = "m2m100_checkpoints_WMT2025/checkpoint_step11000_loss3.3394"
TEST_DATA_PATH = "../hiero-transformer/test_and_validation_data/test_data.json"

# Language mapping (same as training)
LANG_TO_M2M_CODE = {
    "ea": "ar",   # Egyptian hieroglyphics -> Arabic (proxy)
    "tnt": "lo",  # Transliteration -> Lao (proxy)
    "de": "de",   # German
    "en": "en",   # English
}

print("="*80)
print("WMT2025 M2M-100 MODEL EVALUATION")
print("="*80)
print()

# Load test data
print("Loading test data...")
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Organize by language pair
test_pairs = {}
for item in test_data:
    src_lang = item['metadata']['source_lang']
    tgt_lang = item['metadata']['target_lang']
    pair = f"{src_lang}->{tgt_lang}"
    if pair not in test_pairs:
        test_pairs[pair] = []
    test_pairs[pair].append(item)

for pair, items in test_pairs.items():
    print(f"  {pair}: {len(items)} samples")

print()

# Load model
print(f"Loading model from checkpoint: {CHECKPOINT_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = M2M100ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)
tokenizer = M2M100Tokenizer.from_pretrained(CHECKPOINT_PATH)
model = model.to(device)
model.eval()
print("Model loaded!")
print()

# Initialize metrics
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_pair(items, src_lang, tgt_lang):
    """Evaluate a language pair"""
    predictions = []
    references = []
    
    # Set language codes
    src_code = LANG_TO_M2M_CODE.get(src_lang, src_lang)
    tgt_code = LANG_TO_M2M_CODE.get(tgt_lang, tgt_lang)
    tokenizer.src_lang = src_code
    
    for item in tqdm(items, desc=f"{src_lang}->{tgt_lang}"):
        source = item['source']
        target = item['target']
        
        # Tokenize
        inputs = tokenizer(source, return_tensors="pt", max_length=112, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        forced_bos_token_id = tokenizer.get_lang_id(tgt_code)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=112,
                num_beams=10,
                early_stopping=True,
            )
        
        # Decode
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(target)
    
    # Calculate SacreBLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    # Calculate RougeL
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(score['rougeL'].fmeasure * 100)
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    
    return {
        'sacrebleu': bleu.score,
        'rougeL': avg_rouge,
        'samples': len(items),
        'predictions': predictions,
        'references': references,
    }

# Evaluate relevant pairs
print("Generating predictions...")
print()
results = {}

# Focus on main pairs: ea->de and ea->en
for pair, items in test_pairs.items():
    src_lang, tgt_lang = pair.split('->')
    if src_lang == 'ea' and tgt_lang in ['de', 'en']:
        result = evaluate_pair(items, src_lang, tgt_lang)
        results[pair] = result
        print(f"  SacreBLEU: {result['sacrebleu']:.2f}")
        print(f"  RougeL: {result['rougeL']:.2f}")
        print()

# Summary
print("="*80)
print("WMT2025 MODEL RESULTS")
print("="*80)
print(f"{'Language Pair':<20} {'Samples':<10} {'SacreBLEU':<15} {'RougeL':<15}")
print("-"*60)
for pair, result in results.items():
    print(f"{pair:<20} {result['samples']:<10} {result['sacrebleu']:<15.2f} {result['rougeL']:<15.2f}")

# Sample translations
print()
print("="*80)
print("SAMPLE TRANSLATIONS")
print("="*80)
for pair, result in results.items():
    print(f"\n{pair}:")
    for i in range(min(3, len(result['predictions']))):
        print(f"  Source: {test_pairs[pair][i]['source'][:80]}...")
        print(f"  Target: {result['references'][i][:80]}...")
        print(f"  Prediction: {result['predictions'][i][:80]}...")
        print()

# Save results
output = {
    'model': 'WMT2025 M2M-100',
    'checkpoint': CHECKPOINT_PATH,
    'results': {pair: {'sacrebleu': r['sacrebleu'], 'rougeL': r['rougeL'], 'samples': r['samples']} 
                for pair, r in results.items()}
}
with open('wmt2025_evaluation_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to wmt2025_evaluation_results.json")
