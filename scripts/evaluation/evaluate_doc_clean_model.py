#!/usr/bin/env python3
"""
Evaluate the document-level-clean M2M-100 model on the 50 ea→de test samples.

Reports BLEU (sacrebleu) for:
  - All 50 test samples
  - Exact contaminated (16 samples)
  - Soft-leakage (9 samples, 70-99% 8-gram overlap)
  - Clean (25 samples, <70% 8-gram overlap)

Also compares against the WMT2025 baseline (from cached_predictions.json).
"""

import json
import unicodedata
import re
from pathlib import Path
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sacrebleu.metrics import BLEU, CHRF

_HERE     = Path(__file__).parent.resolve()
DATA_DIR  = _HERE / "../hiero-transformer"
TEST_FILE = DATA_DIR / "test_and_validation_data/test_data.json"
TRAIN_FILE = DATA_DIR / "training_data/training_data.json"  # only used for reference
DOC_CLEAN_DIR = _HERE / "m2m100_checkpoints_DOC_CLEAN/best"
CACHED    = _HERE / "cached_predictions.json"
OUT_FILE  = _HERE / "doc_clean_evaluation_results.json"

SRC_LANG  = "ar"
TGT_LANG  = "de"
MAX_SRC   = 256
MAX_TGT   = 128
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load test data ─────────────────────────────────────────────────────────────
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)

test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip() and s["target"].strip()
]
print(f"Test samples: {len(test_de)}")

# ── Normalization (matches contamination analysis) ─────────────────────────────
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def char_ngrams(text: str, n: int) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def ngram_overlap(test_tgt: str, train_tgt: str, n: int = 8) -> float:
    t_norm = normalize(test_tgt)
    tr_norm = normalize(train_tgt)
    test_ng = char_ngrams(t_norm, n)
    if not test_ng:
        return 0.0
    train_ng = char_ngrams(tr_norm, n)
    return len(test_ng & train_ng) / len(test_ng)

# ── Load canonical contamination partition from intermediate band analysis ─────
# Use pre-computed indices for consistency with paper (16 exact, 9 soft, 25 clean)
with open(_HERE / "intermediate_band_results.json", encoding="utf-8") as f:
    band_data = json.load(f)
exact_idx = band_data["exact_indices"]
soft_idx  = band_data["soft_indices"]
clean_idx = band_data["clean_indices"]
labels    = [0.0] * len(test_de)  # placeholder
for i in exact_idx:
    labels[i] = 1.0
for i in soft_idx:
    labels[i] = 0.85
print(f"Partition (from paper): Exact={len(exact_idx)}, Soft={len(soft_idx)}, Clean={len(clean_idx)}")

# ── Generate doc-clean model predictions ──────────────────────────────────────
print(f"\nLoading doc-clean model from: {DOC_CLEAN_DIR}")
tokenizer = M2M100Tokenizer.from_pretrained(DOC_CLEAN_DIR)
model     = M2M100ForConditionalGeneration.from_pretrained(DOC_CLEAN_DIR).to(DEVICE)
model.eval()

tokenizer.src_lang = SRC_LANG
forced_bos = tokenizer.get_lang_id(TGT_LANG)

srcs = [s["source"] for s in test_de]
refs = [s["target"] for s in test_de]
preds = []

BATCH = 16
for i in range(0, len(srcs), BATCH):
    batch_srcs = srcs[i:i+BATCH]
    enc = tokenizer(batch_srcs, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_SRC).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=forced_bos,
                             max_new_tokens=MAX_TGT, num_beams=4)
    preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    print(f"  Generated {min(i+BATCH, len(srcs))}/{len(srcs)}")

# ── Compute BLEU ──────────────────────────────────────────────────────────────
bleu_metric = BLEU(lowercase=True)
chrf_metric = CHRF(word_order=2)

def subset_bleu(idxs):
    if not idxs:
        return 0.0, 0.0
    sub_preds = [preds[i] for i in idxs]
    sub_refs  = [refs[i]  for i in idxs]
    b = bleu_metric.corpus_score(sub_preds, [sub_refs]).score
    c = chrf_metric.corpus_score(sub_preds, [sub_refs]).score
    return round(b, 1), round(c, 1)

all_bleu, all_chrf         = subset_bleu(list(range(len(test_de))))
exact_bleu, exact_chrf     = subset_bleu(exact_idx)
soft_bleu, soft_chrf       = subset_bleu(soft_idx)
clean_bleu, clean_chrf     = subset_bleu(clean_idx)
contam_bleu, contam_chrf   = subset_bleu(exact_idx + soft_idx)

print(f"\n=== Doc-Clean Model Results ===")
print(f"All (n=50):      BLEU={all_bleu:.1f}  chrF++={all_chrf:.1f}")
print(f"Exact (n={len(exact_idx)}): BLEU={exact_bleu:.1f}  chrF++={exact_chrf:.1f}")
print(f"Soft  (n={len(soft_idx)}):  BLEU={soft_bleu:.1f}  chrF++={soft_chrf:.1f}")
print(f"Clean (n={len(clean_idx)}): BLEU={clean_bleu:.1f}  chrF++={clean_chrf:.1f}")
print(f"Contam (n={len(exact_idx)+len(soft_idx)}): BLEU={contam_bleu:.1f}")

# Compare against WMT2025 from cache
with open(CACHED, encoding="utf-8") as f:
    cached = json.load(f)
wmt_preds = cached.get("M2M100_WMT2025", [])
if wmt_preds:
    wmt_all   = bleu_metric.corpus_score(wmt_preds, [refs]).score
    wmt_exact = bleu_metric.corpus_score([wmt_preds[i] for i in exact_idx],
                                          [[refs[i] for i in exact_idx]]).score
    wmt_soft  = bleu_metric.corpus_score([wmt_preds[i] for i in soft_idx],
                                          [[refs[i] for i in soft_idx]]).score if soft_idx else 0.0
    wmt_clean = bleu_metric.corpus_score([wmt_preds[i] for i in clean_idx],
                                          [[refs[i] for i in clean_idx]]).score
    print(f"\n=== WMT2025 Baseline (from cache) ===")
    print(f"All (n=50):      BLEU={wmt_all:.1f}")
    print(f"Exact (n={len(exact_idx)}): BLEU={wmt_exact:.1f}")
    print(f"Soft  (n={len(soft_idx)}):  BLEU={wmt_soft:.1f}")
    print(f"Clean (n={len(clean_idx)}): BLEU={wmt_clean:.1f}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "doc_clean_model": {
        "all": {"bleu": all_bleu, "chrf": all_chrf, "n": len(test_de)},
        "exact": {"bleu": exact_bleu, "chrf": exact_chrf, "n": len(exact_idx)},
        "soft": {"bleu": soft_bleu, "chrf": soft_chrf, "n": len(soft_idx)},
        "clean": {"bleu": clean_bleu, "chrf": clean_chrf, "n": len(clean_idx)},
        "contaminated": {"bleu": contam_bleu, "chrf": contam_chrf,
                         "n": len(exact_idx)+len(soft_idx)},
    },
    "predictions": preds,
    "labels": labels,
}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT_FILE}")
