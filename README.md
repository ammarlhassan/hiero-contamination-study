# Data Contamination in Neural Hieroglyphic Translation

Code, data, and results for the paper:

**"Data Contamination in Neural Machine Translation of Ancient Egyptian Hieroglyphics"**
Submitted to NLP4DH 2026 (Workshop at EACL 2026)

## Key Findings

- **32% of test targets** (16/50) appear verbatim in training data after normalization
- **50% show soft leakage** (>=70% character 8-gram overlap)
- BLEU inflation: **29-47 points** on contaminated vs. clean samples
- COMET-22 inflation: **0.23-0.26 points**
- Corrected clean baselines: **30.9-39.2 BLEU** (vs. reported 61.5)

## Repository Structure

```
NLP4DH_2026_release/
├── README.md
├── scripts/
│   ├── contamination_detection/    # Core contamination analysis
│   │   ├── normalize_translations.py       # Normalization pipeline (Section 3.2)
│   │   ├── ngram_sensitivity.py            # N-gram overlap sensitivity (Table 2)
│   │   ├── intermediate_band_analysis.py   # Exact/soft/clean partition (Table 3)
│   │   ├── english_contamination_check.py  # English direction control
│   │   ├── source_side_overlap.py          # Source-side overlap analysis (Table 5)
│   │   ├── source_causal_analysis.py       # Causal attribution (Section 4.3)
│   │   ├── retrieval_baseline.py           # Oracle retrieval baseline (Table 6)
│   │   └── source_knn_baseline.py          # Source-KNN baseline (Table 6)
│   ├── evaluation/                 # Model evaluation
│   │   ├── evaluate_paper_model.py         # Released Model (mattiadc/hiero-transformer)
│   │   ├── evaluate_exact_trainpy_checkpoint.py  # Script Reproduction
│   │   ├── evaluate_hybrid_model.py        # M2M-100 Hybrid
│   │   ├── evaluate_wmt2025_model.py       # M2M-100 Conservative
│   │   ├── evaluate_mbart_model.py         # mBART-50
│   │   ├── evaluate_comet22.py             # COMET-22 evaluation (Table 4)
│   │   ├── evaluate_doc_clean_model.py     # Document-level decontamination (Table 7)
│   │   ├── evaluate_english_direction.py   # English direction evaluation
│   │   ├── bootstrap_new.py               # Bootstrap confidence intervals (Table 8)
│   │   └── recompute_bleu_settings.py     # Raw vs. normalized ablation (Table 9)
│   └── training/                   # Model training
│       ├── m2m100_training_HYBRID.py       # M2M-100 Hybrid training
│       ├── m2m100_training_WMT2025_STRATEGY.py  # M2M-100 Conservative training
│       ├── mbart50_training_a6000.py       # mBART-50 training
│       └── document_level_retrain.py       # Document-level decontamination retrain
├── results/                        # Pre-computed results
│   ├── cached_predictions.json             # All 5 models x 50 predictions
│   ├── comet22_results.json                # COMET-22 scores per model x subset
│   ├── ngram_sensitivity_results.json      # N-gram sensitivity data
│   ├── intermediate_band_results.json      # Canonical partition (16/9/25)
│   ├── source_side_overlap_results.json    # Source overlap scores
│   ├── source_knn_results.json             # Source-KNN baseline results
│   ├── retrieval_baseline_results.json     # Oracle retrieval results
│   ├── doc_clean_evaluation_results.json   # Doc-clean model results
│   ├── mixture_simulation_results.json     # Mixture simulation data
│   ├── english_contamination_results.json  # English contamination check
│   ├── bootstrap_new_results.json          # Bootstrap CIs
│   ├── enhanced_contamination_results.json # Enhanced contamination analysis
│   ├── statistical_analysis.json           # Statistical test results
│   └── test_item_catalog.csv               # Per-item catalog (all 50 items)
├── data/
│   ├── clean_test_set.json                 # 34 decontaminated test items
│   └── contaminated_test_set.json          # 16 contaminated test items
└── paper/
    ├── NLP4DH_2026_FULL.tex                # Main paper source
    ├── NLP4DH_2026_FULL.pdf                # Compiled PDF
    ├── NLP4DH_2026_FULL.bbl                # Compiled bibliography
    ├── references_full.bib                 # Bibliography
    ├── acl.sty                             # ACL style file
    ├── acl_natbib.bst                      # ACL bibliography style
    ├── comet22_table.tex                   # COMET-22 results table
    └── doc_clean_table.tex                 # Document-clean results table
```

## Quick Start

### Check contamination in the test set
```bash
cd scripts/contamination_detection
python normalize_translations.py
```

### Run n-gram sensitivity analysis
```bash
python ngram_sensitivity.py
```

### Evaluate a model
```bash
cd scripts/evaluation
python evaluate_paper_model.py  # Evaluate Released Model
```

### COMET-22 evaluation
```bash
# Requires: conda activate comet_eval (Python 3.10, unbabel-comet 2.2.7)
python evaluate_comet22.py
```

## Per-Item Test Catalog

The file `results/test_item_catalog.csv` provides a complete mapping for all 50 test items:

| Column | Description |
|--------|-------------|
| `test_index` | Index (0-49) in the test set |
| `document_id` | TLA source document ID |
| `contamination_status` | `exact`, `soft`, or `clean` |
| `target_frequency` | How many times the normalized target appears in training |
| `source_overlap_8gram` | Maximum character 8-gram overlap with any training source |
| `target_preview` | First 80 characters of the German target |

## Models Evaluated

All retrained model checkpoints are publicly available on Hugging Face:

| Model | Description | HF Checkpoint |
|-------|-------------|---------------|
| **Released Model** | Public checkpoint from original authors | [`mattiadc/hiero-transformer`](https://huggingface.co/mattiadc/hiero-transformer) |
| **Script Reproduction** | Retrained with original `train.py` (epochs=20, lr=3e-5) | [`bumblelbee/hiero-m2m100-script-reproduction`](https://huggingface.co/bumblelbee/hiero-m2m100-script-reproduction) |
| **M2M-100 Hybrid** | Fine-tuned M2M-100 (lr=3e-5, AdamW, cosine schedule) | [`bumblelbee/hiero-m2m100-hybrid`](https://huggingface.co/bumblelbee/hiero-m2m100-hybrid) |
| **M2M-100 Conservative** | Fine-tuned M2M-100 (lr=1e-5, AdamW, cosine schedule) | [`bumblelbee/hiero-m2m100-conservative`](https://huggingface.co/bumblelbee/hiero-m2m100-conservative) |
| **mBART-50** | Fine-tuned mBART-50 (611M params, cross-architecture test) | [`bumblelbee/hiero-mbart50`](https://huggingface.co/bumblelbee/hiero-mbart50) |
| **M2M-100 Doc-clean** | Document-level decontaminated variant | [`bumblelbee/hiero-m2m100-doc-clean`](https://huggingface.co/bumblelbee/hiero-m2m100-doc-clean) |

## Data

The original training and test data is from the [hiero-transformer repository](https://github.com/mattiadc/hiero-transformer).

- Training: 18,669 ea→de samples with non-empty source and target (from 61,330 total)
- Test: 50 ea→de samples (16 exact-contaminated, 9 soft-contaminated, 25 clean)
- Decontaminated test: 34 samples (no exact target overlap with training)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- sacrebleu
- unbabel-comet (for COMET-22 evaluation)

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{[authors]2026contamination,
  title={Data Contamination in Neural Machine Translation of Ancient Egyptian Hieroglyphics},
  author={[Authors]},
  booktitle={Proceedings of the Workshop on Natural Language Processing for Digital Humanities (NLP4DH 2026)},
  year={2026}
}
```

## License

This repository is released under the MIT License. The original TLA data is maintained by the Berlin-Brandenburg Academy of Sciences and Humanities and is freely available for research use.
