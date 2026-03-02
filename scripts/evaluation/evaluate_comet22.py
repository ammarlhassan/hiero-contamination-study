#!/usr/bin/env python3
"""
COMET-22 evaluation on all 5 models × 3 subsets (all / contaminated / clean).

Uses cached_predictions.json (50 translations per model) + test_data.json for
references and contamination labels. Downloads Unbabel/wmt22-comet-da once and
caches it locally.

Run with:
    conda run -n comet_eval python evaluate_comet22.py
"""

import json
import os
import re
import sys

# ── Data loading ─────────────────────────────────────────────────────────────

DATA_DIR   = "../hiero-transformer"
TEST_FILE  = f"{DATA_DIR}/test_and_validation_data/test_data.json"
TRAIN_FILE = f"{DATA_DIR}/training_data/training_data.json"
PREDS_FILE = "cached_predictions.json"
OUT_FILE   = "comet22_results.json"

print("Loading test data...")
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)

print("Loading training data (for contamination labels)...")
with open(TRAIN_FILE, encoding="utf-8") as f:
    raw_train = json.load(f)

# Filter to ea→de test pairs (same filter as inference script)
test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip()
    and s["target"].strip()
]
print(f"Test samples (ea→de): {len(test_de)}")

# Build normalised training target set
def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

train_de = [s for s in raw_train if s["metadata"]["target_lang"] == "de" and s["target"].strip()]
train_targets_norm = set(normalize(s["target"]) for s in train_de)

# Label contamination
for s in test_de:
    s["_is_contaminated"] = normalize(s["target"]) in train_targets_norm

contaminated_idx = [i for i, s in enumerate(test_de) if s["_is_contaminated"]]
clean_idx        = [i for i, s in enumerate(test_de) if not s["_is_contaminated"]]
print(f"Contaminated: {len(contaminated_idx)}  Clean: {len(clean_idx)}")

# Ground-truth references and sources (MDC Gardiner notation)
references = [s["target"] for s in test_de]
sources    = [s["source"] for s in test_de]

# Load cached model predictions
with open(PREDS_FILE, encoding="utf-8") as f:
    cached_preds = json.load(f)

MODEL_NAMES = ["Released", "Exact_trainpy", "M2M100_Hybrid", "M2M100_WMT2025", "mBART50"]

# ── COMET loading ─────────────────────────────────────────────────────────────

from comet import download_model, load_from_checkpoint

MODEL_ID = "Unbabel/wmt22-comet-da"
print(f"\nDownloading / loading COMET model: {MODEL_ID}")
comet_path  = download_model(MODEL_ID)
comet_model = load_from_checkpoint(comet_path)
print("COMET model ready.")

# ── Helper ────────────────────────────────────────────────────────────────────

def comet_score(predictions, references, sources):
    """Return mean COMET score (0-1) for a list of (src, mt, ref) triples."""
    if not predictions:
        return None
    data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    output = comet_model.predict(data, batch_size=8, gpus=1 if _has_gpu else 0)
    return float(output.system_score)

import torch
_has_gpu = torch.cuda.is_available()
print(f"GPU available: {_has_gpu}")

# ── Evaluation ────────────────────────────────────────────────────────────────

results = {}

for model_name in MODEL_NAMES:
    if model_name not in cached_preds:
        print(f"[WARN] {model_name} not in cached_predictions.json — skipping")
        continue

    preds = cached_preds[model_name]
    assert len(preds) == len(test_de), f"Length mismatch for {model_name}"

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")

    # Subsets
    preds_all    = preds
    refs_all     = references
    srcs_all     = sources

    preds_contam = [preds[i] for i in contaminated_idx]
    refs_contam  = [references[i] for i in contaminated_idx]
    srcs_contam  = [sources[i] for i in contaminated_idx]

    preds_clean  = [preds[i] for i in clean_idx]
    refs_clean   = [references[i] for i in clean_idx]
    srcs_clean   = [sources[i] for i in clean_idx]

    score_all    = comet_score(preds_all,    refs_all,    srcs_all)
    score_contam = comet_score(preds_contam, refs_contam, srcs_contam)
    score_clean  = comet_score(preds_clean,  refs_clean,  srcs_clean)

    print(f"  All          : {score_all:.4f}")
    print(f"  Contaminated : {score_contam:.4f}")
    print(f"  Clean        : {score_clean:.4f}")

    results[model_name] = {
        "all":          round(score_all,    4),
        "contaminated": round(score_contam, 4),
        "clean":        round(score_clean,  4),
        "n_all":        len(preds_all),
        "n_contaminated": len(preds_contam),
        "n_clean":      len(preds_clean),
    }

# ── Save results ──────────────────────────────────────────────────────────────

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to {OUT_FILE}")
print(json.dumps(results, indent=2))
