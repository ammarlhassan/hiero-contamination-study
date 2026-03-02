#!/usr/bin/env python3
"""
Source-side KNN retrieval baseline (non-oracle).

For each test sample, finds the NEAREST TRAINING SAMPLE by source-side
character 8-gram overlap, then returns that training sample's TARGET as the
prediction. No access to test targets during prediction.

This is contrasted with the (oracle) target-side retrieval baseline, which
computes overlap against test targets.
"""

import json, unicodedata, re
from pathlib import Path
from collections import Counter
from sacrebleu.metrics import BLEU

_HERE = Path(__file__).parent.resolve()
DATA_DIR = _HERE / "../hiero-transformer"
BAND_FILE = _HERE / "intermediate_band_results.json"

def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def char_ngrams(text, n=8):
    return {text[i:i+n] for i in range(len(text) - n + 1)}

with open(DATA_DIR / "test_and_validation_data/test_data.json", encoding="utf-8") as f:
    raw_test = json.load(f)
with open(DATA_DIR / "training_data/training_data.json", encoding="utf-8") as f:
    raw_train = json.load(f)
with open(BAND_FILE, encoding="utf-8") as f:
    band_data = json.load(f)

test_de = [s for s in raw_test
           if s["metadata"]["source_lang"] == "ea" and s["metadata"]["target_lang"] == "de"
           and s["source"].strip() and s["target"].strip()]
train_de = [s for s in raw_train
            if s["metadata"].get("target_lang") == "de"
            and s["source"].strip() and s["target"].strip()]

exact_idx  = band_data["exact_indices"]
soft_idx   = band_data["soft_indices"]
clean_idx  = band_data["clean_indices"]
contam_idx = exact_idx + soft_idx

# Pre-normalize training sources
print("Pre-normalizing training sources...")
train_norm_src = [char_ngrams(normalize(s["source"]), 8) for s in train_de]
train_targets  = [s["target"] for s in train_de]
# Most frequent training target (as a global fallback)
most_freq_tgt = Counter(train_targets).most_common(1)[0][0]

# Source-KNN: for each test sample, find closest training source
print("Computing source-KNN predictions for all 50 test samples...")
preds = []
source_sims = []
for i, s in enumerate(test_de):
    test_ng = char_ngrams(normalize(s["source"]), 8)
    best_sim, best_tgt = 0.0, most_freq_tgt
    for j, tr_ng in enumerate(train_norm_src):
        sim = len(test_ng & tr_ng) / len(test_ng) if test_ng else 0.0
        if sim > best_sim:
            best_sim = sim
            best_tgt = train_targets[j]
        if best_sim >= 1.0:
            break
    preds.append(best_tgt)
    source_sims.append(best_sim)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/50")

refs = [s["target"] for s in test_de]
bleu = BLEU(lowercase=True)

def score(idxs):
    p = [preds[i] for i in idxs]
    r = [refs[i]  for i in idxs]
    return round(bleu.corpus_score(p, [r]).score, 1) if idxs else 0.0

b_all    = score(list(range(50)))
b_exact  = score(exact_idx)
b_soft   = score(soft_idx)
b_clean  = score(clean_idx)
b_contam = score(contam_idx)

print(f"\n=== Source-KNN Baseline (non-oracle) ===")
print(f"All (n=50):      BLEU={b_all}")
print(f"Exact (n=16):    BLEU={b_exact}")
print(f"Soft  (n=9):     BLEU={b_soft}")
print(f"Clean (n=25):    BLEU={b_clean}")
print(f"Contam (n=25):   BLEU={b_contam}")

out = {
    "source_knn": {"all": b_all, "exact": b_exact, "soft": b_soft,
                   "clean": b_clean, "contaminated": b_contam},
    "source_sims": [round(s, 3) for s in source_sims],
}
with open(_HERE / "source_knn_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved source_knn_results.json")
