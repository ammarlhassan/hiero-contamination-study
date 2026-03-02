#!/usr/bin/env python3
"""
mBART-50 Training OPTIMIZED for 4x NVIDIA RTX A6000 (48GB each)
Uses GPUs 2,3,4,5 with CUDA_VISIBLE_DEVICES
Includes: Mixed Precision, Gradient Checkpointing, Optimized Batch Size
"""

import copy
import json
import shutil
import os
import sys

# ============================================================================
# GPU CONFIGURATION - SET BEFORE IMPORTING TORCH
# ============================================================================
# Use GPUs 3,4,5 (the ones with most free memory - GPU 2 is busy)
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # Updated env var name
os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P for stability

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    get_cosine_schedule_with_warmup,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Import utility functions from parent directory
sys.path.insert(0, '../hiero-transformer')
from utils import (
    batch_it,
    clean_data,
    load_data_from_folder,
    processed_data,
)

print("="*80)
print("mBART-50 Training - Optimized for 4x RTX A6000 (48GB)")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Visible GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

# ============================================================================
# TRAINING HYPERPARAMETERS - OPTIMIZED FOR 4x A6000 (48GB each)
# ============================================================================
# With 48GB per GPU, we can be MUCH more aggressive with batch size
# A6000 can handle batch_size=8 per GPU easily for mBART (610M params)
# Using gradient checkpointing allows even larger batches

EPOCHS = 20
NUM_GPUS = 3  # GPUs 3,4,5 (GPU 2 is busy)

# CONSERVATIVE SETTINGS - Reduced to avoid OOM
BATCH_SIZE_PER_GPU = 4  # 4 samples per GPU (safe for shared GPUs)
BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS  # 12 total
GRADIENT_ACCUMULATION_STEPS = 24  # Effective batch = 12 * 24 = 288 (same as before)
MAX_LENGTH = 112  # Reduced from 128 for safety

# Learning rate and optimization
LEARNING_RATE = 3e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
LABEL_SMOOTHING = 0.15
MAX_GRAD_NORM = 1.0

# Dropout (mBART specific)
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.05
ACTIVATION_DROPOUT = 0.05

# Training control
EVAL_PERIOD = 500  # More frequent eval with faster training
PATIENCE = 15
MIN_EVAL_STEPS = 5000  # Can reach faster with larger batches

# Data augmentation
UPSAMPLE_FACTOR = 5
USE_UNIFIED_TAGGING = True

# Disable gradient checkpointing - it's too slow (49s/iter is unusable)
# We have enough GPU memory without it
USE_GRADIENT_CHECKPOINTING = False

# Model configuration
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "mbart50_checkpoints_a6000"

LANG_PAIRS = [
    ("ea", "de"),
    ("ea", "en"),
]

LANG_TO_MBART_CODE = {
    "ea": "ar_AR",
    "tnt": "lo_LA",
    "en": "en_XX",
    "de": "de_DE",
    "lKey": "my_MM",
    "wordClass": "th_TH",
}

print(f"\n{'='*60}")
print("CONFIGURATION")
print(f"{'='*60}")
print(f"Model: {MODEL_NAME}")
print(f"GPUs: 4x RTX A6000 (CUDA_VISIBLE_DEVICES=2,3,4,5)")
print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
print(f"Total batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
print(f"Mixed precision: Enabled (AMP)")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================================
# DATA LOADING
# ============================================================================
print("Loading training data...")
training_data = load_data_from_folder("../hiero-transformer/training_data")
print("Loading validation data...")
validation_data = load_data_from_folder("../hiero-transformer/validation_data")

print("Cleaning data...")
training_data = clean_data(training_data)

print("Processing data...")
training_data = processed_data(training_data)
validation_data = processed_data(validation_data)

print("\nData loading complete!")

# Add German->English translations
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

# WMT 2025 Strategy: 5x upsampling
if UPSAMPLE_FACTOR > 1:
    print(f"\n=== Applying WMT 2025 upsampling strategy ({UPSAMPLE_FACTOR}x) ===")
    for src_lang, tgt_lang in LANG_PAIRS:
        if src_lang in training_data and tgt_lang in training_data[src_lang]:
            original_size = len(training_data[src_lang][tgt_lang])
            training_data[src_lang][tgt_lang] = training_data[src_lang][tgt_lang] * UPSAMPLE_FACTOR
            print(f"{src_lang}->{tgt_lang}: {original_size} -> {len(training_data[src_lang][tgt_lang])} samples")
    print("=" * 60)

# ============================================================================
# MODEL LOADING
# ============================================================================
import glob
checkpoint_pattern = os.path.join(OUTPUT_DIR, "checkpoint_step*_loss*")
existing_checkpoints = sorted(glob.glob(checkpoint_pattern),
                              key=lambda x: int(x.split("_step")[-1].split("_")[0]),
                              reverse=True)

if existing_checkpoints:
    latest_checkpoint = existing_checkpoints[0]
    checkpoint_step = int(latest_checkpoint.split("_step")[-1].split("_")[0])
    print(f"\n🔄 Found checkpoint at step {checkpoint_step}: {os.path.basename(latest_checkpoint)}")
    print(f"Resuming training from checkpoint...")
    model = MBartForConditionalGeneration.from_pretrained(latest_checkpoint)
    tokenizer = MBart50TokenizerFast.from_pretrained(latest_checkpoint)
    resume_from_step = checkpoint_step
    print(f"✓ Model loaded from checkpoint (will resume from step {checkpoint_step})")
else:
    print(f"\nLoading mBART-50 model: {MODEL_NAME}")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)

    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
        activation_dropout=ACTIVATION_DROPOUT,
    )
    resume_from_step = 0
    print("✓ Model loaded fresh")

# Enable gradient checkpointing for memory efficiency
if USE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")

# Multi-GPU with DataParallel
device = torch.device("cuda:0")
if torch.cuda.device_count() > 1:
    print(f"\n🚀 Using DataParallel with {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

print(f"Model loaded successfully!")
total_params = sum(p.numel() for p in model.parameters()) / 1e6
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"Model parameters: {total_params:.1f}M")
print(f"Trainable parameters: {trainable_params:.1f}M")

# ============================================================================
# TRAINING SETUP
# ============================================================================
total_samples = sum(
    len(data)
    for src_lang, values in training_data.items()
    for tgt_lang, data in values.items()
    if (src_lang, tgt_lang) in LANG_PAIRS
)
steps_per_epoch = total_samples // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
total_training_steps = steps_per_epoch * EPOCHS

print(f"\nTotal training samples: {total_samples}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_training_steps}")

# Get actual model for optimizer
model_for_optimizer = model.module if isinstance(model, nn.DataParallel) else model

# Optimizer with parameter grouping
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
    betas=(0.9, 0.999),
    eps=1e-8,
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_training_steps,
)

# Resume scheduler if needed
if resume_from_step > 0:
    print(f"\n🔄 Fast-forwarding scheduler to step {resume_from_step}...")
    for _ in range(resume_from_step):
        scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"✓ Scheduler restored - Current learning rate: {current_lr:.2e}")

# Mixed precision scaler
scaler = torch.amp.GradScaler('cuda')

print("Optimizer and scheduler configured!")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def tokenize_batch(model, batch, tokenizer, src_lang, tgt_lang):
    """Tokenize batch for mBART-50 model"""
    tokenizer.src_lang = LANG_TO_MBART_CODE[src_lang]
    tokenizer.tgt_lang = LANG_TO_MBART_CODE[tgt_lang]

    inputs = tokenizer(
        [element["source"] for element in batch],
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            [element["target"] for element in batch],
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels["input_ids"].to(device)

    labels = torch.where(
        labels == tokenizer.pad_token_id,
        torch.full_like(labels, -100),
        labels,
    )

    tgt_lang_id = tokenizer.lang_code_to_id[LANG_TO_MBART_CODE[tgt_lang]]

    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    decoder_input_ids = actual_model.prepare_decoder_input_ids_from_labels(labels)
    decoder_input_ids[:, 0] = tgt_lang_id

    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["labels"] = labels

    return inputs


def training_step(batch, model, tokenizer, optimizer, scheduler, scaler, src_lang, tgt_lang, accumulation_steps, max_retries=3):
    """Training step with multi-GPU support and error recovery"""
    for retry in range(max_retries):
        try:
            with torch.amp.autocast('cuda'):
                tokenized_batch = tokenize_batch(model, batch, tokenizer, src_lang, tgt_lang)
                outputs = model(**tokenized_batch)
                loss = outputs.loss

                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()

                loss = loss / accumulation_steps

                with torch.no_grad():
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    labels = tokenized_batch["labels"]
                    valid_mask = labels != -100
                    correct = (predictions == labels) & valid_mask
                    accuracy = correct.sum().item() / valid_mask.sum().item() if valid_mask.sum() > 0 else 0.0

            scaler.scale(loss).backward()

            return loss.item() * accumulation_steps, accuracy

        except RuntimeError as e:
            if "NCCL" in str(e) or "out of memory" in str(e).lower():
                if retry < max_retries - 1:
                    print(f"\n⚠️  Error on retry {retry + 1}/{max_retries}: {str(e)[:100]}")
                    print("Attempting to recover...")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    continue
            raise


def validation_step(batch, model, tokenizer, src_lang, tgt_lang):
    """Validation step with accuracy and F1 metrics"""
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            tokenized_batch = tokenize_batch(model, batch, tokenizer, src_lang, tgt_lang)
            outputs = model(**tokenized_batch)
            loss = outputs.loss

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = tokenized_batch["labels"]

            valid_mask = labels != -100
            correct = (predictions == labels) & valid_mask
            accuracy = correct.sum().item() / valid_mask.sum().item() if valid_mask.sum() > 0 else 0.0

            tp = correct.sum().item()
            fp = ((predictions != labels) & valid_mask).sum().item()
            fn = fp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            num_tokens = valid_mask.sum().item()

            return loss.item(), num_tokens, accuracy, f1


def plot_training_progress(training_losses, validation_losses, output_dir):
    """Generate comprehensive training analysis plots"""
    print("\n" + "="*80)
    print("Generating training analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('mBART-50 Training Analysis (4x A6000)', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    if len(training_losses) > 0:
        window_size = min(50, len(training_losses) // 10)
        if window_size > 1:
            smoothed = np.convolve(training_losses, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(len(smoothed)), smoothed, 'b-', linewidth=2, label='Smoothed')
        ax1.plot(range(len(training_losses)), training_losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if len(validation_losses) > 0:
        steps = sorted([int(k) if isinstance(k, str) else k for k in validation_losses.keys()])
        losses = [validation_losses.get(str(step), validation_losses.get(step)) for step in steps]
        ax2.plot(steps, losses, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Over Time')
        ax2.grid(True, alpha=0.3)
        best_step = min(steps, key=lambda s: validation_losses.get(str(s), validation_losses.get(s)))
        best_loss = validation_losses.get(str(best_step), validation_losses.get(best_step))
        ax2.axhline(y=best_loss, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_loss:.4f}')
        ax2.legend()

    ax3 = axes[1, 0]
    if len(training_losses) > 0:
        total_steps_plot = len(training_losses)
        lr_schedule = []
        for step in range(total_steps_plot):
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step / WARMUP_STEPS)
            else:
                progress = (step - WARMUP_STEPS) / (total_training_steps - WARMUP_STEPS)
                lr = LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
            lr_schedule.append(lr)

        ax3.plot(range(len(lr_schedule)), lr_schedule, 'g-', linewidth=2)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = "Training Statistics\n" + "="*40 + "\n\n"
    stats_text += f"Model: mBART-50 (610M)\n"
    stats_text += f"GPUs: 4x RTX A6000 (48GB)\n\n"
    stats_text += f"Training Samples: {total_samples:,}\n"
    stats_text += f"Epochs Completed: {epoch + 1}/{EPOCHS}\n"
    stats_text += f"Total Steps: {total_steps:,}\n\n"

    if len(training_losses) > 0:
        stats_text += f"Final Training Loss: {training_losses[-1]:.4f}\n"

    if len(validation_losses) > 0:
        stats_text += f"Best Validation Loss: {best_eval_loss:.4f}\n"

    stats_text += f"\nBatch Size: {BATCH_SIZE}\n"
    stats_text += f"Grad Accum: {GRADIENT_ACCUMULATION_STEPS}\n"
    stats_text += f"Effective Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved: {plot_path}")
    plt.close()

    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("mBART-50 TRAINING REPORT (4x A6000)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"GPUs: 4x NVIDIA RTX A6000 (48GB each)\n")
        f.write(f"CUDA_VISIBLE_DEVICES: 2,3,4,5\n\n")
        f.write(f"Epochs: {epoch + 1}/{EPOCHS}\n")
        f.write(f"Total Steps: {total_steps:,}\n")
        f.write(f"Batch Size: {BATCH_SIZE} (per GPU: {BATCH_SIZE_PER_GPU})\n")
        f.write(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n\n")
        if len(validation_losses) > 0:
            f.write(f"Best Validation Loss: {best_eval_loss:.4f}\n")
        f.write("\n" + "="*80 + "\n")

    print(f"✓ Training report saved: {report_path}")
    print("="*80 + "\n")


# ============================================================================
# TRAINING LOOP
# ============================================================================
validation_data_batched = [
    (src_lang, trg_lang, batch)
    for src_lang, values in validation_data.items()
    for trg_lang, data in values.items()
    for batch in batch_it(data, BATCH_SIZE)
    if (src_lang, trg_lang) in LANG_PAIRS
]

print(f"Validation batches: {len(validation_data_batched)}")

total_steps = resume_from_step
best_eval_loss = float("inf")
patience_counter = 0
validation_losses = {}
training_losses = []
topk_models = []
max_models = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("Starting Training on 4x RTX A6000...")
print("="*80)

start_epoch = resume_from_step // steps_per_epoch if resume_from_step > 0 else 0

for epoch in range(start_epoch, EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'='*80}\n")

    for src_lang, values in training_data.items():
        for data in values.values():
            np.random.shuffle(data)

    training_data_batched = [
        (src_lang, trg_lang, batch)
        for src_lang, values in training_data.items()
        for trg_lang, data in values.items()
        for batch in batch_it(data, BATCH_SIZE)
        if (src_lang, trg_lang) in LANG_PAIRS
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

        current_lr = scheduler.get_last_lr()[0]
        avg_train_acc = (epoch_accuracy / num_train_batches) * 100
        iterator.set_postfix(
            step=total_steps,
            loss=f"{loss:.4f}",
            acc=f"{avg_train_acc:.1f}%",
            lr=f"{current_lr:.2e}",
            pair=f"{src_lang}->{tgt_lang}"
        )

        if total_steps % EVAL_PERIOD == 0 and total_steps != 0:
            print("\n" + "-" * 80)
            print(f"Evaluating at step {total_steps}...")

            model.eval()
            eval_model = model.module if isinstance(model, nn.DataParallel) else model

            total_eval_loss = 0
            total_eval_tokens = 0
            total_accuracy = 0
            total_f1 = 0
            num_batches = 0

            for src_lang_v, tgt_lang_v, batch_v in tqdm(validation_data_batched, desc="Validation"):
                loss_v, tokens, accuracy_v, f1 = validation_step(batch_v, eval_model, tokenizer, src_lang_v, tgt_lang_v)
                total_eval_loss += loss_v * tokens
                total_eval_tokens += tokens
                total_accuracy += accuracy_v
                total_f1 += f1
                num_batches += 1

            avg_eval_loss = total_eval_loss / total_eval_tokens
            avg_accuracy = (total_accuracy / num_batches) * 100
            avg_f1 = (total_f1 / num_batches) * 100
            validation_losses[total_steps] = avg_eval_loss

            print(f"\n{'='*60}")
            print(f"Validation Metrics at Step {total_steps}")
            print(f"{'='*60}")
            print(f"  Loss:     {avg_eval_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.2f}%")
            print(f"  F1 Score: {avg_f1:.2f}%")
            print(f"{'='*60}\n")

            with open(os.path.join(OUTPUT_DIR, "validation_losses.json"), "w") as f:
                json.dump(validation_losses, f, indent=2)

            if avg_eval_loss < best_eval_loss:
                improvement = ((best_eval_loss - avg_eval_loss) / best_eval_loss) * 100
                print(f"✓ Model improved! Old loss: {best_eval_loss:.4f}, New loss: {avg_eval_loss:.4f} ({improvement:.2f}% improvement)")

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
                print(f"✗ No improvement. Patience: {patience_counter}/{PATIENCE}")

                if patience_counter >= PATIENCE and total_steps >= MIN_EVAL_STEPS:
                    print(f"\nEarly stopping triggered after {total_steps} steps!")
                    break
                elif patience_counter >= PATIENCE:
                    print(f"  (Would stop, but waiting for minimum {MIN_EVAL_STEPS} steps)")
                    patience_counter = PATIENCE - 1

            print("-" * 80 + "\n")
            model.train()

    if patience_counter >= PATIENCE and total_steps >= MIN_EVAL_STEPS:
        break

    avg_epoch_loss = epoch_loss / len(training_data_batched)
    print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

print("\n" + "=" * 80)
print("Training completed!")
print(f"Best validation loss: {best_eval_loss:.4f}")
print(f"Total steps: {total_steps}")
print("=" * 80)

plot_training_progress(training_losses, validation_losses, OUTPUT_DIR)

final_state = {
    'total_steps': total_steps,
    'epochs_completed': epoch + 1,
    'best_validation_loss': best_eval_loss,
    'final_training_loss': training_losses[-1] if training_losses else None,
    'training_config': {
        'gpus': '4x RTX A6000 (48GB)',
        'cuda_visible_devices': '2,3,4,5',
        'batch_size': BATCH_SIZE,
        'batch_size_per_gpu': BATCH_SIZE_PER_GPU,
        'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS,
        'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        'learning_rate': LEARNING_RATE,
        'max_length': MAX_LENGTH,
        'gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
    }
}

with open(os.path.join(OUTPUT_DIR, 'final_training_state.json'), 'w') as f:
    json.dump(final_state, f, indent=2)

print("\n✓ All training artifacts saved:")
print(f"  - Checkpoints: {OUTPUT_DIR}/checkpoint_step*/")
print(f"  - Training plot: {OUTPUT_DIR}/training_progress.png")
print(f"  - Training report: {OUTPUT_DIR}/training_report.txt")
print(f"  - Final state: {OUTPUT_DIR}/final_training_state.json")
print("\n" + "=" * 80)
