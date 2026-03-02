#!/usr/bin/env python3
"""
English direction evaluation: run Released and Exact_trainpy models on the
50 ea→en test samples (only 1/50 = 2% contaminated).

Provides a "clean control" showing how models perform when contamination
is virtually absent, contrasting with the 32% contaminated German results.

Outputs:
  - english_direction_results.json
"""

import json
import torch
from sacrebleu.metrics import BLEU, CHRF

DATA_DIR  = "../hiero-transformer"
TEST_FILE = f"{DATA_DIR}/test_and_validation_data/test_data.json"

print("Loading test data...")
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)

test_en = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "en"
    and s["source"].strip()
    and s["target"].strip()
]
print(f"Test samples (ea→en): {len(test_en)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODELS = {
    "Released": {
        "path": "mattiadc/hiero-transformer",
        "tokenizer_path": "mattiadc/hiero-transformer",
        "is_hf": True,
    },
    "Exact_trainpy": {
        "path": "../hiero-transformer/checkpoint_total_steps=43040_loss=1.60",
        "tokenizer_path": "facebook/m2m100_418M",
        "is_hf": False,
    },
}

# M2M-100 doesn't have an "ea" language code; the hiero-transformer uses "ar" as proxy
SRC_LANG = "ar"
TGT_LANG = "en"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 16

bleu_metric = BLEU(lowercase=True)
chrf_metric = CHRF(word_order=2)

results = {}

for model_name, cfg in MODELS.items():
    print(f"\n{'='*50}\nModel: {model_name}")

    tokenizer = M2M100Tokenizer.from_pretrained(cfg["tokenizer_path"])
    model     = M2M100ForConditionalGeneration.from_pretrained(cfg["path"]).to(device)
    model.eval()

    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.get_lang_id(TGT_LANG)

    sources = [s["source"] for s in test_en]
    refs    = [s["target"] for s in test_en]
    preds   = []

    for i in range(0, len(sources), BATCH_SIZE):
        batch = sources[i:i+BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(decoded)
        print(f"  {min(i+BATCH_SIZE, len(sources))}/{len(sources)}", end="\r")
    print()

    bleu_score = bleu_metric.corpus_score(preds, [refs]).score
    chrf_score = chrf_metric.corpus_score(preds, [refs]).score

    print(f"  BLEU: {bleu_score:.1f}, chrF++: {chrf_score:.1f}")

    results[model_name] = {
        "bleu": round(bleu_score, 1),
        "chrf": round(chrf_score, 1),
        "n_test": len(test_en),
        "n_contaminated": 1,
        "predictions": preds,
        "references":  refs,
    }

    del model
    torch.cuda.empty_cache()

with open("english_direction_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved: english_direction_results.json")
print(json.dumps({m: {"bleu": r["bleu"], "chrf": r["chrf"]} for m, r in results.items()}, indent=2))
