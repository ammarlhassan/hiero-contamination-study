#!/usr/bin/env python3
"""
Causal source-side analysis:

For each exact-target test sample, find the training samples where the TARGET
matches (≥70% 8-gram overlap), then check whether THOSE contaminating training
samples have sources similar to the test source.

The question: do the training samples that cause target contamination come
from similar source contexts, or do they come from entirely different sources?
- If contaminating training targets come from SIMILAR sources → source pattern-matching
- If contaminating training targets come from DIFFERENT sources → pure target memorization
"""

import json, unicodedata, re
from pathlib import Path
from collections import defaultdict

_HERE     = Path(__file__).parent.resolve()
DATA_DIR  = _HERE / "../hiero-transformer"
BAND_FILE = _HERE / "intermediate_band_results.json"

def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def char_ngrams(text, n=8):
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def ngram_cov(a, b, n=8):
    a_ng = char_ngrams(normalize(a), n)
    b_ng = char_ngrams(normalize(b), n)
    return len(a_ng & b_ng) / len(a_ng) if a_ng else 0.0

# Load data
with open(DATA_DIR / "test_and_validation_data/test_data.json", encoding="utf-8") as f:
    raw_test = json.load(f)
with open(DATA_DIR / "training_data/training_data.json", encoding="utf-8") as f:
    raw_train = json.load(f)
with open(BAND_FILE, encoding="utf-8") as f:
    band_data = json.load(f)

test_de = [s for s in raw_test
           if s["metadata"]["source_lang"]=="ea" and s["metadata"]["target_lang"]=="de"
           and s["source"].strip() and s["target"].strip()]
train_de = [s for s in raw_train
            if s["metadata"].get("target_lang")=="de"
            and s["source"].strip() and s["target"].strip()]

exact_idx = band_data["exact_indices"]

print(f"Test: {len(test_de)}, Train: {len(train_de)}")
print("\n=== For each exact-target sample: source similarity of contaminating training samples ===")
print(f"{'Idx':>4}  {'Test target':40}  #Match  AvgSrcSim  MinSrcSim  MaxSrcSim  Category")

summary = []
for i in sorted(exact_idx):
    test_tgt  = test_de[i]["target"]
    test_src  = test_de[i]["source"]
    # Find all training samples with ≥70% target overlap (the contaminating ones)
    contam_sources = []
    for tr in train_de:
        if ngram_cov(test_tgt, tr["target"]) >= 0.70:
            src_sim = ngram_cov(test_src, tr["source"])
            contam_sources.append(src_sim)
    if contam_sources:
        avg_sim = sum(contam_sources) / len(contam_sources)
        min_sim = min(contam_sources)
        max_sim = max(contam_sources)
        n_high  = sum(1 for s in contam_sources if s >= 0.70)
        n_zero  = sum(1 for s in contam_sources if s < 0.05)
        # Classify: if ALL contaminating training samples have low source sim → memorization
        cat = "MEMORIZE" if n_high == 0 else ("MIXED" if n_zero > 0 else "SRC-MATCH")
    else:
        avg_sim = min_sim = max_sim = 0.0
        n_high = n_zero = 0
        cat = "NO-MATCH?"
    print(f"  {i:2d}  {test_tgt[:40]:40}  {len(contam_sources):5d}  {avg_sim:.3f}     {min_sim:.3f}     {max_sim:.3f}  {cat}  [≥70%src:{n_high}, <5%src:{n_zero}]")
    summary.append({"idx": i, "n_contam": len(contam_sources),
                    "avg_src_sim": round(avg_sim, 3),
                    "n_high_src": n_high, "n_low_src": n_zero, "cat": cat})

# Summary
cats = defaultdict(int)
for s in summary:
    cats[s["cat"]] += 1
print(f"\n=== Classification of 16 exact-target samples ===")
for cat, n in cats.items():
    print(f"  {cat}: {n}/16")

memorize_cases = [s for s in summary if s["n_low_src"] > 0 or s["n_high_src"] == 0]
print(f"\nCases where at least one contaminating training sample has LOW source similarity (<5%): {sum(1 for s in summary if s['n_low_src']>0)}/16")
print("→ These prove target memorization independent of source context")
