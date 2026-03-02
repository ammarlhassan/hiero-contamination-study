#!/usr/bin/env python3
"""
Bootstrap CIs for new evaluation numbers (lowercase_only setting).
"""

import json
import re
import string

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

np.random.seed(42)
N_BOOT = 1000

# Load data
with open("../hiero-transformer/training_data/training_data.json", "r") as f:
    raw_train = json.load(f)
with open("../hiero-transformer/test_and_validation_data/test_data.json", "r") as f:
    raw_test = json.load(f)

train_de = [s for s in raw_train if s["metadata"]["target_lang"] == "de" and s["target"].strip()]
train_targets = set(s["target"] for s in train_de)

test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip()
    and s["target"].strip()
]

def normalize(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

train_targets_norm = set(normalize(t) for t in train_targets)

for s in test_de:
    s["_is_contaminated"] = normalize(s["target"]) in train_targets_norm

contaminated_idx = [i for i, s in enumerate(test_de) if s["_is_contaminated"]]
clean_idx = [i for i, s in enumerate(test_de) if not s["_is_contaminated"]]

references = [s["target"] for s in test_de]

with open("cached_predictions.json", "r") as f:
    all_predictions = json.load(f)

results = {}

for model_name, predictions in all_predictions.items():
    print(f"\nBootstrapping {model_name}...")

    bleu = BLEU(lowercase=True)

    # Point estimates
    all_bleu = bleu.corpus_score(predictions, [references]).score
    cont_preds = [predictions[i] for i in contaminated_idx]
    cont_refs = [references[i] for i in contaminated_idx]
    cont_bleu = bleu.corpus_score(cont_preds, [cont_refs]).score
    clean_preds = [predictions[i] for i in clean_idx]
    clean_refs = [references[i] for i in clean_idx]
    clean_bleu = bleu.corpus_score(clean_preds, [clean_refs]).score

    # Bootstrap
    boot_all, boot_cont, boot_clean = [], [], []

    for b in range(N_BOOT):
        # Sample with replacement for each subset
        idx_all = np.random.choice(len(test_de), size=len(test_de), replace=True)
        b_preds = [predictions[i] for i in idx_all]
        b_refs = [references[i] for i in idx_all]
        boot_all.append(bleu.corpus_score(b_preds, [b_refs]).score)

        idx_cont = np.random.choice(contaminated_idx, size=len(contaminated_idx), replace=True)
        b_preds = [predictions[i] for i in idx_cont]
        b_refs = [references[i] for i in idx_cont]
        boot_cont.append(bleu.corpus_score(b_preds, [b_refs]).score)

        idx_clean = np.random.choice(clean_idx, size=len(clean_idx), replace=True)
        b_preds = [predictions[i] for i in idx_clean]
        b_refs = [references[i] for i in idx_clean]
        boot_clean.append(bleu.corpus_score(b_preds, [b_refs]).score)

    results[model_name] = {
        "all": {
            "bleu": round(all_bleu, 1),
            "ci_95": [round(np.percentile(boot_all, 2.5), 1), round(np.percentile(boot_all, 97.5), 1)],
        },
        "contaminated": {
            "bleu": round(cont_bleu, 1),
            "ci_95": [round(np.percentile(boot_cont, 2.5), 1), round(np.percentile(boot_cont, 97.5), 1)],
        },
        "clean": {
            "bleu": round(clean_bleu, 1),
            "ci_95": [round(np.percentile(boot_clean, 2.5), 1), round(np.percentile(boot_clean, 97.5), 1)],
        },
    }

    print(f"  All:    {all_bleu:.1f} [{np.percentile(boot_all, 2.5):.1f}, {np.percentile(boot_all, 97.5):.1f}]")
    print(f"  Contam: {cont_bleu:.1f} [{np.percentile(boot_cont, 2.5):.1f}, {np.percentile(boot_cont, 97.5):.1f}]")
    print(f"  Clean:  {clean_bleu:.1f} [{np.percentile(boot_clean, 2.5):.1f}, {np.percentile(boot_clean, 97.5):.1f}]")

with open("bootstrap_new_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote bootstrap_new_results.json")
