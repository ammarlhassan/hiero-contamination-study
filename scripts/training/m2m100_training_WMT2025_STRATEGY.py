#!/usr/bin/env python3
"""
M2M-100 Training with WMT 2025 CreoleMT Strategy - CONSERVATIVE VERSION
Applies the winning strategies from WMT 2025 CreoleMT paper to hieroglyphics translation

Key WMT 2025 Strategies Applied:
1. Learning Rate: 1e-5 (WMT 2025 conservative for low-resource)
2. Warmup: 1000 steps + Cosine scheduler (WMT 2025 proven)
3. 5x upsampling for low-resource pairs (~60K dataset)
4. Label smoothing: 0.15, Dropout optimization
5. Mixed precision training + Early stopping

Base Model: facebook/m2m100_418M (same as hieroglyphics paper)
Strategy: Conservative WMT 2025 approach for maximum stability
"""

import copy
import json
import shutil
import os
import sys

# Set environment variables for GPU stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    get_cosine_schedule_with_warmup,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Import utility functions
sys.path.append('../hiero-transformer')
from utils import (
    batch_it,
    clean_data,
    load_data_from_folder,
    processed_data,
)

print("="*80)
print("M2M-100 TRAINING WITH WMT 2025 CREOLEMT STRATEGY")
print("="*80)
print()
print("Applying WMT 2025 winning strategies:")
print("  1. Language-agnostic unified tagging")
print("  2. 5x upsampling for low-resource pairs (~60K dataset)")
print("  3. Optimized hyperparameters (LR: 1e-5, Warmup: 1000, Cosine schedule)")
print("  4. Strategic dropout and weight decay")
print("  5. Early stopping with patience")
print("="*80)
print()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print()

# WMT 2025 OPTIMIZED HYPERPARAMETERS
# Based on CreoleMT paper findings for ~60K low-resource data
EPOCHS = 20
NUM_GPUS = 1  # Single GPU for stability (avoids DataParallel deadlocks)
BATCH_SIZE_PER_GPU = 8  # Larger batch on single GPU
BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS  # 8 total
GRADIENT_ACCUMULATION_STEPS = 36  # Effective batch: 8 * 36 = 288
LEARNING_RATE = 1e-5  # WMT 2025 optimal for low-resource
WARMUP_STEPS = 1000   # WMT 2025 optimal for ~60K data
WEIGHT_DECAY = 0.1    # Strong regularization
LABEL_SMOOTHING = 0.15  # Reduce overfitting
MAX_LENGTH = 112
EVAL_PERIOD = 500      # Frequent evaluation
MAX_GRAD_NORM = 1.0
PATIENCE = 15
MIN_EVAL_STEPS = 5000  # Don't stop too early
UPSAMPLE_FACTOR = 5    # WMT 2025: 5x optimal (10x saturates)
USE_UNIFIED_TAGGING = True  # WMT 2025: Language-agnostic approach

# Dropout configuration (WMT 2025 strategy)
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.05
ACTIVATION_DROPOUT = 0.05

# Model configuration
MODEL_NAME = "facebook/m2m100_418M"  # Same as original paper
OUTPUT_DIR = "m2m100_checkpoints_WMT2025"

# Language pairs to train
LANG_PAIRS = [
    ("ea", "de"),
    ("ea", "en"),
]

# M2M-100 language codes
# WMT 2025 Strategy: Use unified tagging for source (hieroglyphics)
LANG_TO_M2M_CODE = {
    "ea": "ar",     # Arabic proxy for hieroglyphics (WMT unified approach)
    "tnt": "lo",    # Lao proxy for transliteration
    "en": "en",
    "de": "de",
}

print("MODEL CONFIGURATION")
print("-"*80)
print(f"Base Model: {MODEL_NAME}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Language Pairs: {LANG_PAIRS}")
print()

print("WMT 2025 TRAINING CONFIGURATION")
print("-"*80)
print(f"Epochs: {EPOCHS}")
print(f"GPUs: {NUM_GPUS}")
print(f"Batch Size per GPU: {BATCH_SIZE_PER_GPU}")
print(f"Total Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning Rate: {LEARNING_RATE} (WMT 2025 optimal)")
print(f"Warmup Steps: {WARMUP_STEPS} (WMT 2025 optimal for ~60K)")
print(f"Scheduler: Cosine with warmup (WMT 2025 proven superior)")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Label Smoothing: {LABEL_SMOOTHING}")
print(f"Dropout: {DROPOUT} / {ATTENTION_DROPOUT} / {ACTIVATION_DROPOUT}")
print(f"Upsampling Factor: {UPSAMPLE_FACTOR}x (WMT 2025 optimal)")
print(f"Unified Tagging: {USE_UNIFIED_TAGGING} (WMT 2025 language-agnostic)")
print(f"Max Sequence Length: {MAX_LENGTH}")
print(f"Eval Period: {EVAL_PERIOD} steps")
print(f"Early Stopping Patience: {PATIENCE}")
print(f"Minimum Eval Steps: {MIN_EVAL_STEPS}")
print("="*80)
print()

# Load data
print("Loading training data...")
training_data = load_data_from_folder("../hiero-transformer/training_data")
print("Loading validation data...")
validation_data = load_data_from_folder("../hiero-transformer/validation_data")

print("Cleaning data...")
training_data = clean_data(training_data)

print("Processing data...")
training_data = processed_data(training_data)
validation_data = processed_data(validation_data)

print("Data loading complete!")
print()

# Add German->English translations if available
translation_file = "../hiero-transformer/translations_de2en.json"
if os.path.exists(translation_file):
    print("Adding German->English translations...")
    with open(translation_file, encoding="utf-8") as f:
        translations = json.load(f)

    for lang in ("ea", "tnt"):
        if lang in training_data and "en" in training_data[lang] and "de" in training_data[lang]:
            ids_sentence = {
                element["metadata"]["id_sentence"]
                for element in training_data[lang]["en"]
                if "id_sentence" in element["metadata"]
            }

            for element in training_data[lang]["de"]:
                if (
                    "id_sentence" in element["metadata"]
                    and element["metadata"]["id_sentence"] not in ids_sentence
                    and element["target"] in translations
                ):
                    new_element = copy.deepcopy(element)
                    new_element["target"] = translations[element["target"]]
                    new_element["metadata"]["target_lang"] = "en"
                    training_data[lang]["en"].append(new_element)

            print(f'{lang} -> en: After translation augmentation: {len(training_data[lang]["en"])} datapoints')
    print()

# WMT 2025 Strategy: 5x upsampling (optimal for low-resource)
if UPSAMPLE_FACTOR > 1:
    print("="*80)
    print(f"APPLYING WMT 2025 UPSAMPLING STRATEGY: {UPSAMPLE_FACTOR}x")
    print("="*80)
    print("WMT 2025 Finding: 5x upsampling optimal for ~60K data")
    print("                  10x upsampling saturates performance")
    print()

    for src_lang, tgt_lang in LANG_PAIRS:
        if src_lang in training_data and tgt_lang in training_data[src_lang]:
            original_size = len(training_data[src_lang][tgt_lang])
            training_data[src_lang][tgt_lang] = training_data[src_lang][tgt_lang] * UPSAMPLE_FACTOR
            upsampled_size = len(training_data[src_lang][tgt_lang])
            print(f"{src_lang} -> {tgt_lang}: {original_size:,} -> {upsampled_size:,} samples")

    print("="*80)
    print()

# Check for existing checkpoints to resume from
def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.exists(output_dir):
        return None, 0, float('inf')
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint_step')]
    if not checkpoints:
        return None, 0, float('inf')
    
    # Parse checkpoint info: checkpoint_step{step}_loss{loss}
    checkpoint_info = []
    for cp in checkpoints:
        try:
            parts = cp.split('_')
            step = int(parts[1].replace('step', ''))
            loss = float(parts[2].replace('loss', ''))
            checkpoint_info.append((cp, step, loss))
        except (IndexError, ValueError):
            continue
    
    if not checkpoint_info:
        return None, 0, float('inf')
    
    # Sort by step (highest first) and return the latest
    checkpoint_info.sort(key=lambda x: x[1], reverse=True)
    best_checkpoint = checkpoint_info[0]
    
    # Also find the best loss across all checkpoints
    best_loss = min(cp[2] for cp in checkpoint_info)
    
    return os.path.join(output_dir, best_checkpoint[0]), best_checkpoint[1], best_loss

# Resume from checkpoint if available
resume_checkpoint, resume_step, resume_best_loss = find_latest_checkpoint(OUTPUT_DIR)
start_epoch = 0

# Load model and tokenizer
if resume_checkpoint and os.path.exists(resume_checkpoint):
    print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
    print(f"  Resume step: {resume_step}")
    print(f"  Best loss so far: {resume_best_loss:.4f}")
    model = M2M100ForConditionalGeneration.from_pretrained(
        resume_checkpoint,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
        activation_dropout=ACTIVATION_DROPOUT,
    )
    tokenizer = M2M100Tokenizer.from_pretrained(resume_checkpoint)
else:
    print("Loading M2M-100 model from scratch...")
    model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
        activation_dropout=ACTIVATION_DROPOUT,
    )
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)

# Multi-GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if NUM_GPUS > 1 and torch.cuda.is_available():
    print(f"Using DataParallel with {NUM_GPUS} GPUs")
    model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))

model = model.to(device)

print("Model loaded successfully!")
total_params = sum(p.numel() for p in model.parameters()) / 1e6
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"Model parameters: {total_params:.1f}M")
print(f"Trainable parameters: {trainable_params:.1f}M")
print()

# Calculate total training steps
total_samples = sum(
    len(data)
    for src_lang, values in training_data.items()
    for tgt_lang, data in values.items()
    if (src_lang, tgt_lang) in LANG_PAIRS
)
steps_per_epoch = total_samples // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
total_training_steps = steps_per_epoch * EPOCHS

print(f"Total training samples: {total_samples:,}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_training_steps}")
print()

# Get actual model for optimizer
model_for_optimizer = model.module if isinstance(model, nn.DataParallel) else model

# WMT 2025 Optimizer: Proper parameter grouping with weight decay
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model_for_optimizer.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model_for_optimizer.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),  # Standard Adam betas
    eps=1e-8,
)

# WMT 2025 Scheduler: Cosine with warmup (proven superior to linear)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_training_steps,
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

print("Optimizer and scheduler configured!")
print("="*80)
print()

# Label smoothed loss function
def label_smoothed_nll_loss(logits, labels, epsilon=LABEL_SMOOTHING):
    """Compute label-smoothed negative log likelihood loss"""
    # Create mask for valid (non-padding) tokens
    valid_mask = labels != -100
    
    # Replace -100 with 0 for scatter operation (will be masked out later)
    labels_for_scatter = labels.clone()
    labels_for_scatter[~valid_mask] = 0
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Create smoothed labels
    vocab_size = log_probs.size(-1)
    smoothing_value = epsilon / (vocab_size - 1)

    # One-hot encode labels (using safe labels)
    one_hot = torch.zeros_like(log_probs).scatter_(-1, labels_for_scatter.unsqueeze(-1), 1.0)

    # Apply label smoothing
    smoothed_labels = one_hot * (1 - epsilon) + (1 - one_hot) * smoothing_value

    # Compute loss per token
    loss = -torch.sum(smoothed_labels * log_probs, dim=-1)

    # Mask and average over valid tokens only
    loss = (loss * valid_mask.float()).sum() / valid_mask.float().sum()

    return loss


def tokenize_batch(batch, tokenizer, src_lang, tgt_lang):
    """Tokenize batch for M2M-100 model"""
    # Set source and target languages
    tokenizer.src_lang = LANG_TO_M2M_CODE[src_lang]
    tgt_lang_code = LANG_TO_M2M_CODE[tgt_lang]
    tokenizer.tgt_lang = tgt_lang_code  # Must set before tokenizing with text_target

    # Tokenize
    tokenized_batch = tokenizer(
        [element["source"] for element in batch],
        text_target=[element["target"] for element in batch],
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Set forced_bos_token_id for target language
    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang_code)

    # Replace padding in labels with -100
    tokenized_batch["labels"] = torch.where(
        tokenized_batch["labels"] == tokenizer.pad_token_id,
        torch.full_like(tokenized_batch["labels"], -100),
        tokenized_batch["labels"],
    )

    return tokenized_batch, forced_bos_token_id


def training_step(batch, model, tokenizer, optimizer, scheduler, scaler, src_lang, tgt_lang, accumulation_steps):
    """Training step with gradient accumulation and mixed precision"""
    with torch.amp.autocast('cuda'):
        tokenized_batch, forced_bos_token_id = tokenize_batch(batch, tokenizer, src_lang, tgt_lang)

        # Note: decoder_start_token_id is only for generate(), not forward()
        # The tokenizer already handles the target language token in labels
        outputs = model(
            input_ids=tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["attention_mask"],
            labels=tokenized_batch["labels"],
        )

        # Use label smoothed loss
        logits = outputs.logits
        labels = tokenized_batch["labels"]
        loss = label_smoothed_nll_loss(logits, labels, epsilon=LABEL_SMOOTHING)

        # Handle DataParallel
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        loss = loss / accumulation_steps

        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            valid_mask = labels != -100
            correct = (predictions == labels) & valid_mask
            accuracy = correct.sum().item() / valid_mask.sum().item() if valid_mask.sum() > 0 else 0.0

    scaler.scale(loss).backward()

    return loss.item() * accumulation_steps, accuracy


def validation_step(batch, model, tokenizer, src_lang, tgt_lang):
    """Validation step"""
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            tokenized_batch, forced_bos_token_id = tokenize_batch(batch, tokenizer, src_lang, tgt_lang)

            # Note: decoder_start_token_id is only for generate(), not forward()
            outputs = model(
                input_ids=tokenized_batch["input_ids"],
                attention_mask=tokenized_batch["attention_mask"],
                labels=tokenized_batch["labels"],
            )

            logits = outputs.logits
            labels = tokenized_batch["labels"]
            loss = label_smoothed_nll_loss(logits, labels, epsilon=LABEL_SMOOTHING)

            # Handle DataParallel
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            valid_mask = labels != -100
            correct = (predictions == labels) & valid_mask
            accuracy = correct.sum().item() / valid_mask.sum().item() if valid_mask.sum() > 0 else 0.0
            num_tokens = valid_mask.sum().item()

            return loss.item(), num_tokens, accuracy


# Prepare validation data
validation_data_batched = [
    (src_lang, tgt_lang, batch)
    for src_lang, values in validation_data.items()
    for tgt_lang, data in values.items()
    for batch in batch_it(data, BATCH_SIZE)
    if (src_lang, tgt_lang) in LANG_PAIRS
]

print(f"Validation batches: {len(validation_data_batched)}")
print()

# Training state - restore if resuming
if resume_checkpoint and resume_step > 0:
    total_steps = resume_step
    best_eval_loss = resume_best_loss
    # Calculate which epoch we should start from
    start_epoch = total_steps // steps_per_epoch
    print(f"Resuming training from step {total_steps}, epoch {start_epoch + 1}")
    
    # Fast-forward the scheduler to the correct position
    for _ in range(total_steps):
        scheduler.step()
    print(f"Scheduler fast-forwarded to step {total_steps}")
    
    # Load existing validation losses
    val_loss_file = os.path.join(OUTPUT_DIR, "validation_losses.json")
    if os.path.exists(val_loss_file):
        with open(val_loss_file, 'r') as f:
            validation_losses = json.load(f)
    else:
        validation_losses = {}
    
    # Rebuild topk_models from existing checkpoints
    topk_models = []
    for d in os.listdir(OUTPUT_DIR):
        if d.startswith('checkpoint_step'):
            try:
                parts = d.split('_')
                loss = float(parts[2].replace('loss', ''))
                topk_models.append((loss, os.path.join(OUTPUT_DIR, d)))
            except (IndexError, ValueError):
                continue
    topk_models.sort(key=lambda x: x[0])
else:
    total_steps = 0
    best_eval_loss = float("inf")
    validation_losses = {}
    topk_models = []

patience_counter = 0
training_losses = []
max_models = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting WMT 2025 training...")
print("="*80)
print()

for epoch in range(start_epoch, EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'='*80}\n")

    # Shuffle training data
    for src_lang, values in training_data.items():
        for data in values.values():
            np.random.shuffle(data)

    training_data_batched = [
        (src_lang, tgt_lang, batch)
        for src_lang, values in training_data.items()
        for tgt_lang, data in values.items()
        for batch in batch_it(data, BATCH_SIZE)
        if (src_lang, tgt_lang) in LANG_PAIRS
    ]
    np.random.shuffle(training_data_batched)

    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    num_train_batches = 0
    optimizer.zero_grad()

    iterator = tqdm(training_data_batched, desc=f"Epoch {epoch+1}")
    for step_in_epoch, (src_lang, tgt_lang, batch) in enumerate(iterator):
        loss, accuracy = training_step(
            batch,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            src_lang,
            tgt_lang,
            GRADIENT_ACCUMULATION_STEPS,
        )
        epoch_loss += loss
        epoch_accuracy += accuracy
        num_train_batches += 1
        training_losses.append(loss)

        if (step_in_epoch + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_for_optimizer.parameters(), MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_steps += 1

            # Only evaluate immediately after step increment (fixes bug where eval triggered multiple times)
            if total_steps % EVAL_PERIOD == 0:
                print("\n" + "-"*80)
                print(f"Evaluating at step {total_steps}...")

                model.eval()
                eval_model = model.module if isinstance(model, nn.DataParallel) else model

                total_eval_loss = 0
                total_eval_tokens = 0
                total_accuracy = 0
                num_batches = 0

                for src_lang_v, tgt_lang_v, batch_v in tqdm(validation_data_batched, desc="Validation"):
                    val_loss, tokens, val_accuracy = validation_step(batch_v, eval_model, tokenizer, src_lang_v, tgt_lang_v)
                    total_eval_loss += val_loss * tokens
                    total_eval_tokens += tokens
                    total_accuracy += val_accuracy
                    num_batches += 1

                avg_eval_loss = total_eval_loss / total_eval_tokens
                avg_accuracy = (total_accuracy / num_batches) * 100
                validation_losses[total_steps] = avg_eval_loss

                print(f"\n{'='*60}")
                print(f"Validation at Step {total_steps}")
                print(f"{'='*60}")
                print(f"  Loss:     {avg_eval_loss:.4f}")
                print(f"  Accuracy: {avg_accuracy:.2f}%")
                print(f"{'='*60}\n")

                with open(os.path.join(OUTPUT_DIR, "validation_losses.json"), "w") as f:
                    json.dump(validation_losses, f, indent=2)

                if avg_eval_loss < best_eval_loss:
                    improvement = ((best_eval_loss - avg_eval_loss) / best_eval_loss) * 100
                    print(f"Model improved! Old: {best_eval_loss:.4f}, New: {avg_eval_loss:.4f} ({improvement:.2f}% improvement)")

                    fname = os.path.join(OUTPUT_DIR, f"checkpoint_step{total_steps}_loss{avg_eval_loss:.4f}")

                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    model_to_save.save_pretrained(fname)
                    tokenizer.save_pretrained(fname)

                    topk_models.append((avg_eval_loss, fname))
                    topk_models.sort(key=lambda x: x[0])

                    best_eval_loss = avg_eval_loss
                    patience_counter = 0

                    if len(topk_models) > max_models:
                        _, old_fname = topk_models.pop(-1)
                        if os.path.exists(old_fname):
                            shutil.rmtree(old_fname)
                            print(f"Removed checkpoint: {old_fname}")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

                    if patience_counter >= PATIENCE and total_steps >= MIN_EVAL_STEPS:
                        print(f"\nEarly stopping triggered after {total_steps} steps!")
                    elif patience_counter >= PATIENCE:
                        print(f"  (Would stop, but waiting for minimum {MIN_EVAL_STEPS} steps)")
                        patience_counter = PATIENCE - 1

                print("-"*80 + "\n")
                model.train()

        current_lr = scheduler.get_last_lr()[0]
        avg_train_acc = (epoch_accuracy / num_train_batches) * 100
        iterator.set_postfix(
            step=total_steps,
            loss=f"{loss:.4f}",
            acc=f"{avg_train_acc:.1f}%",
            lr=f"{current_lr:.2e}",
            pair=f"{src_lang}->{tgt_lang}"
        )

        if patience_counter >= PATIENCE and total_steps >= MIN_EVAL_STEPS:
            print(f"\nEarly stopping triggered after {total_steps} steps!")
            break

    if patience_counter >= PATIENCE and total_steps >= MIN_EVAL_STEPS:
        break

    avg_epoch_loss = epoch_loss / len(training_data_batched)
    print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

print("\n" + "="*80)
print("WMT 2025 TRAINING COMPLETED!")
print(f"Best validation loss: {best_eval_loss:.4f}")
print(f"Total steps: {total_steps}")
print("="*80)

# Save final state
final_state = {
    'model': MODEL_NAME,
    'strategy': 'WMT 2025 CreoleMT',
    'total_steps': total_steps,
    'epochs_completed': epoch + 1,
    'best_validation_loss': best_eval_loss,
    'final_training_loss': training_losses[-1] if training_losses else None,
    'training_losses': training_losses,
    'validation_losses': validation_losses,
    'language_pairs': LANG_PAIRS,
    'wmt2025_config': {
        'batch_size': BATCH_SIZE,
        'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS,
        'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        'learning_rate': LEARNING_RATE,
        'warmup_steps': WARMUP_STEPS,
        'scheduler': 'cosine',
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'dropout': DROPOUT,
        'attention_dropout': ATTENTION_DROPOUT,
        'activation_dropout': ACTIVATION_DROPOUT,
        'upsample_factor': UPSAMPLE_FACTOR,
        'unified_tagging': USE_UNIFIED_TAGGING,
    }
}

with open(os.path.join(OUTPUT_DIR, 'final_training_state.json'), 'w') as f:
    json.dump(final_state, f, indent=2)

print("\nAll training artifacts saved:")
print(f"  - Checkpoints: {OUTPUT_DIR}/checkpoint_step*/")
print(f"  - Validation losses: {OUTPUT_DIR}/validation_losses.json")
print(f"  - Final state: {OUTPUT_DIR}/final_training_state.json")
print("\n" + "="*80)
print("READY FOR EVALUATION!")
print("Use evaluate_m2m100_model.py to test the best checkpoint")
print("="*80)
