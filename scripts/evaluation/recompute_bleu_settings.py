#!/usr/bin/env python3
"""
Recompute BLEU/chrF++ under three settings using cached predictions:
  1. Raw: case-sensitive, original text
  2. Lowercase-only: BLEU(lowercase=True), original text (likely matches paper)
  3. Full normalized: lowercase + punct strip text, BLEU(lowercase=True)

This helps reconcile paper numbers with our new results.
"""

import json
import re
import string

from sacrebleu.metrics import BLEU, CHRF

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

contaminated = [s for s in test_de if s["_is_contaminated"]]
clean = [s for s in test_de if not s["_is_contaminated"]]
print(f"Contaminated: {len(contaminated)}, Clean: {len(clean)}")

# Load cached predictions
with open("cached_predictions.json", "r") as f:
    all_predictions = json.load(f)

references = [s["target"] for s in test_de]

def strip_punct(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

results = {}

for model_name, predictions in all_predictions.items():
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    model_results = {}

    for setting_name, bleu_lc, do_strip in [
        ("raw", False, False),
        ("lowercase_only", True, False),
        ("full_normalized", True, True),
    ]:
        bleu = BLEU(lowercase=bleu_lc)
        chrf = CHRF(word_order=2)

        if do_strip:
            preds = [strip_punct(p) for p in predictions]
            refs = [strip_punct(r) for r in references]
        else:
            preds = predictions
            refs = references

        # All
        b_all = bleu.corpus_score(preds, [refs])
        c_all = chrf.corpus_score(preds, [refs])

        # Contaminated
        p_cont = [preds[i] for i, s in enumerate(test_de) if s["_is_contaminated"]]
        r_cont = [refs[i] for i, s in enumerate(test_de) if s["_is_contaminated"]]
        b_cont = bleu.corpus_score(p_cont, [r_cont])
        c_cont = chrf.corpus_score(p_cont, [r_cont])

        # Clean
        p_clean = [preds[i] for i, s in enumerate(test_de) if not s["_is_contaminated"]]
        r_clean = [refs[i] for i, s in enumerate(test_de) if not s["_is_contaminated"]]
        b_clean = bleu.corpus_score(p_clean, [r_clean])
        c_clean = chrf.corpus_score(p_clean, [r_clean])

        sig = str(bleu.get_signature())

        print(f"\n  Setting: {setting_name} (lc={bleu_lc}, strip={do_strip})")
        print(f"  Signature: {sig}")
        print(f"  {'Subset':<15} {'BLEU':>8} {'chrF++':>8}")
        print(f"  {'-'*35}")
        print(f"  {'All (n=50)':<15} {b_all.score:>8.1f} {c_all.score:>8.1f}")
        print(f"  {'Contam (n=16)':<15} {b_cont.score:>8.1f} {c_cont.score:>8.1f}")
        print(f"  {'Clean (n=34)':<15} {b_clean.score:>8.1f} {c_clean.score:>8.1f}")

        model_results[setting_name] = {
            "signature": sig,
            "all": {"bleu": round(b_all.score, 1), "chrf": round(c_all.score, 1)},
            "contaminated": {"bleu": round(b_cont.score, 1), "chrf": round(c_cont.score, 1)},
            "clean": {"bleu": round(b_clean.score, 1), "chrf": round(c_clean.score, 1)},
        }

    results[model_name] = model_results

with open("bleu_three_settings.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote bleu_three_settings.json")
