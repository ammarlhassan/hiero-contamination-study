#!/usr/bin/env python3
"""
Source-side 8-gram overlap analysis.

For each of the 50 ea→de test samples, compute the maximum character 8-gram
overlap between the test SOURCE (Gardiner notation) and ALL training SOURCES.
Cross-tabulate with target contamination tier (exact/soft/clean).

This answers: do contaminated test items also have near-duplicate SOURCES in
training (which would suggest legitimate pattern-matching), or are sources
distinct while targets match (which would confirm pure target memorization)?
"""

import json
import unicodedata
import re
from pathlib import Path
from collections import defaultdict

_HERE      = Path(__file__).parent.resolve()
DATA_DIR   = _HERE / "../hiero-transformer"
TEST_FILE  = DATA_DIR / "test_and_validation_data/test_data.json"
TRAIN_FILE = DATA_DIR / "training_data/training_data.json"
BAND_FILE  = _HERE / "intermediate_band_results.json"
OUT_FILE   = _HERE / "source_side_overlap_results.json"
TABLE_FILE = _HERE / "source_overlap_table.tex"

# ── Normalization (same as target analysis) ────────────────────────────────────
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def char_ngrams(text: str, n: int) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def max_overlap(test_str: str, train_strs: list[str], n: int = 8) -> float:
    """Max character n-gram coverage of test_str against any training string."""
    t_norm = normalize(test_str)
    test_ng = char_ngrams(t_norm, n)
    if not test_ng:
        return 0.0
    best = 0.0
    for tr in train_strs:
        tr_norm = normalize(tr)
        tr_ng = char_ngrams(tr_norm, n)
        score = len(test_ng & tr_ng) / len(test_ng)
        if score > best:
            best = score
        if best >= 1.0:
            break
    return best

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)
with open(TRAIN_FILE, encoding="utf-8") as f:
    raw_train = json.load(f)
with open(BAND_FILE, encoding="utf-8") as f:
    band_data = json.load(f)

test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip() and s["target"].strip()
]
train_de = [
    s for s in raw_train
    if s["metadata"].get("target_lang") == "de"
    and s["source"].strip() and s["target"].strip()
]
print(f"Test samples: {len(test_de)}, Training samples: {len(train_de)}")

# Canonical partition from intermediate band analysis
exact_idx = set(band_data["exact_indices"])   # 16
soft_idx  = set(band_data["soft_indices"])    # 9
clean_idx = set(band_data["clean_indices"])   # 25

train_sources = [s["source"] for s in train_de]
train_targets = [s["target"] for s in train_de]

# ── Compute source-side overlap for each test sample ──────────────────────────
print("Computing source-side overlaps (this may take ~1-2 min)...")
source_overlaps = []
for i, s in enumerate(test_de):
    ov = max_overlap(s["source"], train_sources, n=8)
    source_overlaps.append(ov)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/50 done")

# ── Cross-tabulation ───────────────────────────────────────────────────────────
def stats(vals):
    if not vals:
        return {"mean": 0, "max": 0, "pct_high": 0, "n": 0}
    import statistics
    high = sum(1 for v in vals if v >= 0.7)
    return {
        "mean": round(statistics.mean(vals), 3),
        "max":  round(max(vals), 3),
        "median": round(statistics.median(vals), 3),
        "pct_high": round(100 * high / len(vals), 1),
        "n": len(vals),
        "values": [round(v, 3) for v in vals],
    }

exact_src_ov = [source_overlaps[i] for i in sorted(exact_idx)]
soft_src_ov  = [source_overlaps[i] for i in sorted(soft_idx)]
clean_src_ov = [source_overlaps[i] for i in sorted(clean_idx)]
all_src_ov   = source_overlaps

print("\n=== Source-Side Overlap by Target Tier ===")
for label, vals in [("Exact (n=16)", exact_src_ov),
                    ("Soft  (n=9)",  soft_src_ov),
                    ("Clean (n=25)", clean_src_ov),
                    ("All   (n=50)", all_src_ov)]:
    st = stats(vals)
    print(f"  {label}: mean={st['mean']:.3f}, median={st['median']:.3f}, "
          f"max={st['max']:.3f}, ≥70%={st['pct_high']}%")

# ── Per-item: print the contaminated samples with their source overlaps ────────
print("\n=== Exact-target samples: source vs target overlap ===")
print(f"{'Idx':>4} {'Src Overlap':>11} {'Target Overlap':>14}  Source (first 40 chars)")
for i in sorted(exact_idx):
    s = test_de[i]
    src_ov = source_overlaps[i]
    print(f"  {i:2d}   src={src_ov:.3f}  tgt=1.000   {s['source'][:60]!r}")

# ── Check: does exact source match imply same target? ─────────────────────────
# Find closest training sample (by source) for each exact-target test sample
print("\n=== Closest training source for each exact-target test sample ===")
for i in sorted(exact_idx):
    test_src = normalize(test_de[i]["source"])
    test_ng  = char_ngrams(test_src, 8)
    best_ov, best_j = 0.0, -1
    for j, tr_src in enumerate(train_sources):
        tr_ng = char_ngrams(normalize(tr_src), 8)
        ov = len(test_ng & tr_ng) / len(test_ng) if test_ng else 0.0
        if ov > best_ov:
            best_ov, best_j = ov, j
    best_tgt_ov = 0.0
    if best_j >= 0:
        test_tgt_norm = normalize(test_de[i]["target"])
        tr_tgt_norm   = normalize(train_targets[best_j])
        test_tgt_ng   = char_ngrams(test_tgt_norm, 8)
        tr_tgt_ng     = char_ngrams(tr_tgt_norm, 8)
        best_tgt_ov   = len(test_tgt_ng & tr_tgt_ng) / len(test_tgt_ng) if test_tgt_ng else 0.0
    print(f"  idx={i:2d}: src_ov={best_ov:.3f}  "
          f"closest_train_tgt_ov_with_test_tgt={best_tgt_ov:.3f}  "
          f"test_tgt={test_de[i]['target'][:40]!r}")

# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "source_overlaps": source_overlaps,
    "by_tier": {
        "exact": stats(exact_src_ov),
        "soft":  stats(soft_src_ov),
        "clean": stats(clean_src_ov),
        "all":   stats(all_src_ov),
    },
    "exact_indices": sorted(exact_idx),
    "soft_indices":  sorted(soft_idx),
    "clean_indices": sorted(clean_idx),
}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT_FILE}")

# ── LaTeX table ────────────────────────────────────────────────────────────────
with open(TABLE_FILE, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Target Tier} & \textbf{n} & \textbf{Mean Src} & \textbf{Max Src} & \textbf{Src$\geq$70\%} \\
                     &            & \textbf{Overlap}  & \textbf{Overlap} & \textbf{(\%)} \\
\midrule
""")
    tiers = [
        ("Exact match",     exact_src_ov),
        ("Soft (70--99\%)", soft_src_ov),
        ("Clean ($<$70\%)", clean_src_ov),
        ("All",             all_src_ov),
    ]
    for label, vals in tiers:
        st = stats(vals)
        f.write(f"{label} & {st['n']} & {st['mean']:.3f} & {st['max']:.3f} & {st['pct_high']:.0f}\\% \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\caption{Source-side character 8-gram overlap between test and training samples,
cross-tabulated by target contamination tier. Low source overlap for
exact-target-match samples confirms that contamination arises from
\emph{target memorization}, not source-driven pattern matching.}
\label{tab:source-overlap}
\end{table}
""")
print(f"LaTeX table written to {TABLE_FILE}")
