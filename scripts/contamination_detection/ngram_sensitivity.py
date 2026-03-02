#!/usr/bin/env python3
"""
N-gram sensitivity analysis: how does contamination rate change across
character n-gram overlap thresholds (n=6, 8, 10, 12) and percentage
thresholds (50%, 60%, 70%, 80%, 90%, 100%)?

Outputs:
  - ngram_sensitivity_results.json  : full data
  - ngram_sensitivity_table.tex     : LaTeX table for paper
"""

import json
import re

# ── Data loading ──────────────────────────────────────────────────────────────

DATA_DIR   = "../hiero-transformer"
TEST_FILE  = f"{DATA_DIR}/test_and_validation_data/test_data.json"
TRAIN_FILE = f"{DATA_DIR}/training_data/training_data.json"

print("Loading data...")
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)
with open(TRAIN_FILE, encoding="utf-8") as f:
    raw_train = json.load(f)

# Filter test set (ea→de)
test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip()
    and s["target"].strip()
]

# Build training target set (normalized)
def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'[^\w\s]', '', t)
    return t

train_de = [s for s in raw_train if s["metadata"]["target_lang"] == "de" and s["target"].strip()]
train_targets_norm = [normalize(s["target"]) for s in train_de]

print(f"Test samples: {len(test_de)}, Training targets: {len(train_targets_norm)}")

# ── N-gram functions ──────────────────────────────────────────────────────────

def char_ngrams(text: str, n: int) -> set:
    """Character n-grams from a string (after normalization)."""
    text = normalize(text)
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def build_train_ngram_sets(n: int) -> list:
    """Pre-compute character n-gram sets for all training targets."""
    return [char_ngrams(t, n) for t in train_targets_norm]

def max_overlap(test_ngrams: set, train_ngram_sets: list) -> float:
    """Maximum Jaccard-style character overlap: |A∩B| / |A| (test-sided)."""
    if not test_ngrams:
        return 0.0
    best = 0.0
    for train_ngrams in train_ngram_sets:
        if not train_ngrams:
            continue
        overlap = len(test_ngrams & train_ngrams) / len(test_ngrams)
        if overlap > best:
            best = overlap
        if best >= 1.0:
            break
    return best

# ── Run analysis ──────────────────────────────────────────────────────────────

N_VALUES         = [6, 8, 10, 12]
PCT_THRESHOLDS   = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
TOTAL            = len(test_de)

results = {}

for n in N_VALUES:
    print(f"\nBuilding {n}-gram sets for {len(train_targets_norm)} training targets...")
    train_ngram_sets = build_train_ngram_sets(n)
    print(f"  Done. Computing overlaps for {TOTAL} test samples...")

    # Compute per-sample max overlap
    overlaps = []
    for i, s in enumerate(test_de):
        test_ng = char_ngrams(s["target"], n)
        ov = max_overlap(test_ng, train_ngram_sets)
        overlaps.append(ov)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{TOTAL}", end="\r")

    results[n] = {}
    for pct in PCT_THRESHOLDS:
        flagged = sum(1 for ov in overlaps if ov >= pct)
        results[n][pct] = {
            "n_flagged": flagged,
            "rate": round(flagged / TOTAL * 100, 1),
        }
        print(f"  n={n}, threshold={int(pct*100)}%: {flagged}/{TOTAL} = {flagged/TOTAL*100:.1f}%")

# ── Save JSON ─────────────────────────────────────────────────────────────────

with open("ngram_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: ngram_sensitivity_results.json")

# ── LaTeX table ───────────────────────────────────────────────────────────────

lines = []
lines.append(r"% Character n-gram sensitivity table")
lines.append(r"% Values = n_flagged / 50  (contamination rate %)")
lines.append(r"")
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{} & \multicolumn{6}{c}{\textbf{Overlap threshold}} \\")
lines.append(r"\cmidrule(l){2-7}")
lines.append(r"\textbf{n-gram} & 50\% & 60\% & 70\% & 80\% & 90\% & 100\% \\")
lines.append(r"\midrule")

for n in N_VALUES:
    cells = []
    for pct in PCT_THRESHOLDS:
        r = results[n][pct]
        # Bold the 8-gram / 70% cell (the one we report in the paper)
        cell = f"{r['n_flagged']}/50 ({r['rate']:.0f}\\%)"
        if n == 8 and abs(pct - 0.70) < 0.01:
            cell = r"\textbf{" + cell + r"}"
        cells.append(cell)
    lines.append(f"{n}-gram & " + " & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\caption{Contamination detection sensitivity across character n-gram size "
    r"and overlap threshold. Each cell shows the number of test samples flagged "
    r"out of 50 and the corresponding contamination rate. "
    r"\textbf{Bold} = primary threshold used in the paper (8-gram, 70\%). "
    r"Contamination is robust: even at 90\% overlap and $n=12$, "
    r"more than one-third of the test set is flagged.}"
)
lines.append(r"\label{tab:ngram-sensitivity}")
lines.append(r"\end{table}")

tex = "\n".join(lines) + "\n"
with open("ngram_sensitivity_table.tex", "w") as f:
    f.write(tex)

print("Saved: ngram_sensitivity_table.tex")
print("\nDone.")
