#!/usr/bin/env python3
"""
English-side contamination check.

Checks whether the English references in the ea→en test set (50 samples)
appear in the English-language training data.

Outputs:
  - english_contamination_results.json
"""

import json
import re
from collections import Counter

DATA_DIR   = "../hiero-transformer"
TEST_FILE  = f"{DATA_DIR}/test_and_validation_data/test_data.json"
TRAIN_FILE = f"{DATA_DIR}/training_data/training_data.json"

print("Loading data...")
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)
with open(TRAIN_FILE, encoding="utf-8") as f:
    raw_train = json.load(f)

# ── Filter test (ea→en) ───────────────────────────────────────────────────────
test_en = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "en"
    and s["source"].strip()
    and s["target"].strip()
]
print(f"Test samples (ea→en): {len(test_en)}")

# ── Build training English target sets ───────────────────────────────────────
train_en = [s for s in raw_train if s["metadata"]["target_lang"] == "en" and s["target"].strip()]
print(f"Training samples with English targets: {len(train_en)}")

def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'[^\w\s]', '', t)
    return t

train_en_exact  = set(s["target"] for s in train_en)
train_en_norm   = set(normalize(s["target"]) for s in train_en)

print(f"Unique training English targets (exact): {len(train_en_exact)}")
print(f"Unique training English targets (normalized): {len(train_en_norm)}")

# ── Check test targets ────────────────────────────────────────────────────────
exact_matches = []
norm_matches  = []

for s in test_en:
    tgt = s["target"]
    if tgt in train_en_exact:
        exact_matches.append(tgt)
    elif normalize(tgt) in train_en_norm:
        norm_matches.append(tgt)

total = len(test_en)
n_exact = len(exact_matches)
n_norm  = len(norm_matches)
n_clean = total - n_exact - n_norm

print(f"\n{'='*50}")
print(f"English contamination results ({total} test samples):")
print(f"  Exact match:      {n_exact}/{total} = {n_exact/total*100:.1f}%")
print(f"  Normalized match: {n_exact+n_norm}/{total} = {(n_exact+n_norm)/total*100:.1f}%")
print(f"  Clean:            {n_clean}/{total} = {n_clean/total*100:.1f}%")

if exact_matches:
    print(f"\nExactly contaminated English targets:")
    for t in exact_matches[:10]:
        print(f"  - {t!r}")

if norm_matches:
    print(f"\nNorm-only contaminated English targets:")
    for t in norm_matches[:10]:
        print(f"  - {t!r}")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "total_test_en": total,
    "n_training_en": len(train_en),
    "unique_training_en_exact": len(train_en_exact),
    "unique_training_en_norm":  len(train_en_norm),
    "exact_matches":    n_exact,
    "norm_only_matches": n_norm,
    "total_contaminated": n_exact + n_norm,
    "clean":             n_clean,
    "contamination_rate_exact": f"{n_exact/total*100:.1f}%",
    "contamination_rate_normalized": f"{(n_exact+n_norm)/total*100:.1f}%",
    "contaminated_targets_exact": exact_matches,
    "contaminated_targets_norm":  norm_matches,
}

with open("english_contamination_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved: english_contamination_results.json")
