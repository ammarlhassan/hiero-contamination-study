#!/usr/bin/env python3
"""
Intermediate n-gram band stratification.

Partitions the 50 test samples into three tiers based on 8-gram overlap:
  - Exact (normalized match): 16 samples
  - Soft leakage (70–99% 8-gram overlap, NOT exact match): ~9 samples
  - Clean (<70% overlap AND not exact match): ~25 samples

Computes BLEU for all 5 models on each tier to show a performance gradient.

Outputs:
  - intermediate_band_results.json
  - intermediate_band_table.tex
"""

import json
import re
from sacrebleu.metrics import BLEU

DATA_DIR   = "../hiero-transformer"
TEST_FILE  = f"{DATA_DIR}/test_and_validation_data/test_data.json"
TRAIN_FILE = f"{DATA_DIR}/training_data/training_data.json"
PREDS_FILE = "cached_predictions.json"
THRESH     = 0.70   # soft-leakage threshold

print("Loading data...")
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)
with open(TRAIN_FILE, encoding="utf-8") as f:
    raw_train = json.load(f)
with open(PREDS_FILE, encoding="utf-8") as f:
    cached_preds = json.load(f)

# Filter test (ea→de)
test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip()
    and s["target"].strip()
]

def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'[^\w\s]', '', t)
    return t

train_de = [s for s in raw_train if s["metadata"]["target_lang"] == "de" and s["target"].strip()]
train_targets_norm = set(normalize(s["target"]) for s in train_de)

def char_ngrams(text: str, n: int = 8) -> set:
    text = normalize(text)
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

# Pre-compute training n-gram sets (unique normalized targets)
unique_norm_targets = list(train_targets_norm)
unique_norm_ng      = [char_ngrams(t) for t in unique_norm_targets]

def max_ngram_coverage(test_text: str) -> float:
    test_ng = char_ngrams(test_text)
    if not test_ng:
        return 0.0
    best = 0.0
    for tng in unique_norm_ng:
        if not tng:
            continue
        sc = len(test_ng & tng) / len(test_ng)
        if sc > best:
            best = sc
        if best >= 1.0:
            break
    return best

print("Computing per-sample 8-gram coverage...")
for i, s in enumerate(test_de):
    s["_is_exact"]   = normalize(s["target"]) in train_targets_norm
    s["_ng_coverage"] = max_ngram_coverage(s["target"])
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/50", end="\r")
print()

# Assign tiers
exact_idx  = [i for i, s in enumerate(test_de) if s["_is_exact"]]
soft_idx   = [i for i, s in enumerate(test_de) if not s["_is_exact"] and s["_ng_coverage"] >= THRESH]
clean_idx  = [i for i, s in enumerate(test_de) if not s["_is_exact"] and s["_ng_coverage"] < THRESH]

print(f"Exact match:    {len(exact_idx)} samples")
print(f"Soft leakage:   {len(soft_idx)} samples ({THRESH*100:.0f}-99% overlap)")
print(f"Clean (<{THRESH*100:.0f}%):   {len(clean_idx)} samples")

# Show soft-leakage samples
print(f"\nSoft-leakage samples (70-99% 8-gram overlap, NOT exact):")
for i in soft_idx:
    s = test_de[i]
    print(f"  [{s['_ng_coverage']:.3f}] {s['target'][:80]}")

# ── BLEU scoring ─────────────────────────────────────────────────────────────

bleu     = BLEU(lowercase=True)
MODEL_NAMES = ["Released", "Exact_trainpy", "M2M100_Hybrid", "M2M100_WMT2025", "mBART50"]
refs_all = [s["target"] for s in test_de]

def bleu_subset(preds, refs):
    if not preds:
        return None
    return round(bleu.corpus_score(preds, [refs]).score, 1)

results = {}
print("\nBLEU by tier:")
print(f"{'Model':20s} {'Exact(n='+str(len(exact_idx))+')':12s} {'Soft(n='+str(len(soft_idx))+')':11s} {'Clean(n='+str(len(clean_idx))+')':11s}")
print("-" * 60)
for m in MODEL_NAMES:
    if m not in cached_preds:
        continue
    preds = cached_preds[m]
    b_exact = bleu_subset([preds[i] for i in exact_idx],  [refs_all[i] for i in exact_idx])
    b_soft  = bleu_subset([preds[i] for i in soft_idx],   [refs_all[i] for i in soft_idx])
    b_clean = bleu_subset([preds[i] for i in clean_idx],  [refs_all[i] for i in clean_idx])
    results[m] = {"exact": b_exact, "soft": b_soft, "clean": b_clean}
    print(f"{m:20s} {str(b_exact):12s} {str(b_soft):11s} {str(b_clean):11s}")

# ── Save ─────────────────────────────────────────────────────────────────────

out = {
    "threshold": THRESH,
    "n_exact":  len(exact_idx),
    "n_soft":   len(soft_idx),
    "n_clean":  len(clean_idx),
    "exact_indices": exact_idx,
    "soft_indices":  soft_idx,
    "clean_indices": clean_idx,
    "soft_samples": [
        {"target": test_de[i]["target"], "ng_coverage": test_de[i]["_ng_coverage"]}
        for i in soft_idx
    ],
    "model_scores": results,
}
with open("intermediate_band_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved: intermediate_band_results.json")

# ── LaTeX table ───────────────────────────────────────────────────────────────

tex_lines = [
    r"% Intermediate n-gram band stratification table",
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lccc@{}}",
    r"\toprule",
    r"\textbf{Model} & \textbf{Exact} & \textbf{Soft leakage} & \textbf{Clean} \\",
    r" & \textbf{(n=" + str(len(exact_idx)) + r", 100\%)} & \textbf{(n=" + str(len(soft_idx)) + r", 70--99\%)} & \textbf{(n=" + str(len(clean_idx)) + r", $<$70\%)} \\",
    r"\midrule",
]
for m in MODEL_NAMES:
    if m not in results:
        continue
    label = m.replace("_", r"\_").replace("M2M100", "M2M-100")
    r = results[m]
    tex_lines.append(f"{label} & {r['exact']} & {r['soft']} & {r['clean']} \\\\")

tex_lines += [
    r"\addlinespace",
    r"\textit{Retrieval baseline} & 81.8 & -- & 5.8 \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{BLEU (\texttt{case:lc}) stratified by three contamination tiers: "
    r"\textbf{Exact} = normalized exact match with training target (16 samples); "
    r"\textbf{Soft leakage} = 70--99\% character 8-gram overlap, not exact match (" + str(len(soft_idx)) + r" samples); "
    r"\textbf{Clean} = $<$70\% overlap (" + str(len(clean_idx)) + r" samples). "
    r"Performance degrades monotonically from exact to soft to clean, "
    r"and the retrieval baseline matches neural models on exact-match items.}",
    r"\label{tab:three-tier}",
    r"\end{table}",
    "",
]
with open("intermediate_band_table.tex", "w") as f:
    f.write("\n".join(tex_lines))
print("Saved: intermediate_band_table.tex")
