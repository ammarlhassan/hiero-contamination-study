#!/usr/bin/env python3
"""
Document-level retrain starting from the WMT2025 checkpoint.

Filters training data to exclude ALL sentences from source documents
(id_datapoint) that appear in the test set, then continues fine-tuning
from the WMT2025 checkpoint (which achieves 47.3 BLEU on all / 39.2 BLEU clean).

This tests whether removing source-document overlap reduces test performance,
and by how much contamination persists from targets appearing in non-test docs.

Key finding from analysis:
  - 33 unique test id_datapoints removed → 5,304 samples removed (28.4%)
  - 8/16 contaminated targets STILL appear in training (from other source docs)
  - 8/16 contaminated targets fully removed (only appeared in test docs)

Saves checkpoint to: m2m100_checkpoints_DOC_CLEAN/

Run: python document_level_retrain.py
"""

import json
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sacrebleu.metrics import BLEU

_HERE       = Path(__file__).parent.resolve()
DATA_DIR    = _HERE / "../hiero-transformer"
TRAIN_FILE  = DATA_DIR / "training_data/training_data.json"
TEST_FILE   = DATA_DIR / "test_and_validation_data/test_data.json"
VAL_FILE    = DATA_DIR / "test_and_validation_data/validation_data.json"
OUT_DIR     = _HERE / "m2m100_checkpoints_DOC_CLEAN"

# Resolve best WMT2025 checkpoint by lowest validation loss
_wmt_dir    = _HERE / "m2m100_checkpoints_WMT2025"
_ckpts      = [d for d in os.listdir(_wmt_dir) if d.startswith("checkpoint_")]
_best_ckpt  = min(_ckpts, key=lambda x: float(x.split("loss")[1]) if "loss" in x else 999)
BASE_MODEL  = _wmt_dir / _best_ckpt

os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE      = 16
GRAD_ACCUM      = 4
LR              = 3e-6     # lower LR for continued fine-tuning
WARMUP_STEPS    = 100
MAX_STEPS       = 3000     # fewer steps needed from a good checkpoint
EVAL_EVERY      = 300
MAX_SRC_LEN     = 256
MAX_TGT_LEN     = 128
SRC_LANG        = "ar"
TGT_LANG        = "de"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(TRAIN_FILE, encoding="utf-8") as f:
    raw_train = json.load(f)
with open(TEST_FILE, encoding="utf-8") as f:
    raw_test = json.load(f)
with open(VAL_FILE, encoding="utf-8") as f:
    raw_val = json.load(f)

test_de = [
    s for s in raw_test
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip() and s["target"].strip()
]
test_doc_ids = set(s["metadata"]["id_datapoint"] for s in test_de)
print(f"Test documents to exclude: {len(test_doc_ids)}")

train_de_all = [
    s for s in raw_train
    if s["metadata"]["target_lang"] == "de"
    and s["source"].strip() and s["target"].strip()
]
train_de_clean = [
    s for s in train_de_all
    if s["metadata"].get("id_datapoint", "") not in test_doc_ids
]
print(f"Training: {len(train_de_all)} → {len(train_de_clean)} (removed {len(train_de_all)-len(train_de_clean)}, {(len(train_de_all)-len(train_de_clean))/len(train_de_all)*100:.1f}%)")

val_de = [
    s for s in raw_val
    if s["metadata"]["source_lang"] == "ea"
    and s["metadata"]["target_lang"] == "de"
    and s["source"].strip() and s["target"].strip()
]
print(f"Validation: {len(val_de)} samples")

# ── Dataset ───────────────────────────────────────────────────────────────────

class HieroDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["source"], s["target"]

def make_collate_fn(tokenizer):
    def collate(batch):
        srcs, tgts = zip(*batch)
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = TGT_LANG
        enc = tokenizer(
            list(srcs), text_target=list(tgts),
            return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SRC_LEN,
        )
        labels = enc["labels"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
        }
    return collate

# ── Training ──────────────────────────────────────────────────────────────────

print(f"\nLoading from WMT2025 checkpoint: {BASE_MODEL}")
tokenizer = M2M100Tokenizer.from_pretrained(BASE_MODEL)
model     = M2M100ForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)

train_ds  = HieroDataset(train_de_clean)
val_ds    = HieroDataset(val_de)
collate   = make_collate_fn(tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32,         shuffle=False,
                          collate_fn=collate, num_workers=2)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
)

bleu_metric = BLEU(lowercase=True)

def evaluate(model, val_de_list, tokenizer):
    """Evaluate directly on val_de list for clean BLEU computation."""
    model.eval()
    preds, refs = [], []
    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.get_lang_id(TGT_LANG)
    srcs = [s["source"] for s in val_de_list]
    refs = [s["target"] for s in val_de_list]
    for i in range(0, len(srcs), 32):
        enc = tokenizer(
            srcs[i:i+32], return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SRC_LEN
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_TGT_LEN,
                num_beams=4,
            )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    model.train()
    return bleu_metric.corpus_score(preds, [refs]).score

step       = 0
best_bleu  = 0.0
best_step  = 0
accum_loss = 0.0

# Evaluate starting point
print("Evaluating WMT2025 starting point on validation set...")
init_bleu = evaluate(model, val_de, tokenizer)
print(f"WMT2025 starting Val BLEU: {init_bleu:.2f}")

print(f"\nStarting fine-tuning: {MAX_STEPS} steps, lr={LR}")
model.train()

import itertools
train_iter = itertools.cycle(train_loader)

while step < MAX_STEPS:
    batch = next(train_iter)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    out  = model(**batch)
    loss = out.loss / GRAD_ACCUM
    loss.backward()
    accum_loss += loss.item()

    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    step += 1

    if step % 100 == 0:
        print(f"Step {step}/{MAX_STEPS} — loss: {accum_loss/100:.4f}", flush=True)
        accum_loss = 0.0

    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        val_bleu = evaluate(model, val_de, tokenizer)
        print(f"Step {step} — Val BLEU: {val_bleu:.2f}", flush=True)
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            best_step = step
            save_path = f"{OUT_DIR}/best"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  ✓ Saved best model (BLEU={best_bleu:.2f}) → {save_path}")

print(f"\nTraining complete. Best BLEU={best_bleu:.2f} at step {best_step}")
print(f"WMT2025 starting Val BLEU was: {init_bleu:.2f}")
