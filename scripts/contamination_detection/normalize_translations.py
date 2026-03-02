#!/usr/bin/env python3
"""
normalize_translations.py -- Standalone normalization tool for NMT contamination detection.

Applies the normalization pipeline used in "Data Contamination in Neural
Hieroglyphic Translation: A Reproducibility Study" (NLP4DH 2026).

Pipeline:
  1. Strip leading/trailing whitespace
  2. Lowercase
  3. Remove punctuation (keep word characters and spaces)

Usage examples:
  # As a library
  from normalize_translations import normalize, normalize_file

  normalized = normalize("Werde gekocht!")   # → "werde gekocht"

  # Command-line: normalize a text file (one sentence per line)
  python normalize_translations.py input.txt output.txt

  # Check overlap between test and training targets
  python normalize_translations.py --check-overlap test.txt train.txt
"""

import re
import sys
from pathlib import Path
from typing import Iterable, List, Set


# ── Core normalization ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Normalize a translation string for contamination detection.

    Steps:
      1. strip()        – remove leading/trailing whitespace
      2. lower()        – case fold to ASCII lowercase
      3. re.sub(...)    – remove all punctuation, keep \\w and spaces

    Args:
        text: A raw translation string (any language).

    Returns:
        Normalized string suitable for overlap comparison.

    Examples:
        >>> normalize("Werde gekocht!")
        'werde gekocht'
        >>> normalize("  Man schläft unter seinen Füßen.  ")
        'man schläft unter seinen füßen'
    """
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


# ── File utilities ────────────────────────────────────────────────────────────

def normalize_file(path: str | Path) -> List[str]:
    """Read a file (one sentence per line) and return normalized lines."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [normalize(line) for line in lines if line.strip()]


def build_norm_set(texts: Iterable[str]) -> Set[str]:
    """Return a set of normalized strings for fast membership testing."""
    return {normalize(t) for t in texts if t.strip()}


def check_overlap(test_texts: Iterable[str], train_texts: Iterable[str]) -> dict:
    """
    Check for normalized overlap between test and training targets.

    Args:
        test_texts:  Iterable of test-set target strings.
        train_texts: Iterable of training-set target strings.

    Returns:
        dict with keys:
          n_test, n_train, n_contaminated, n_clean,
          contamination_rate (float), contaminated_samples (list of str)
    """
    test_list   = [t for t in test_texts  if str(t).strip()]
    train_norm  = build_norm_set(train_texts)

    contaminated = [t for t in test_list if normalize(t) in train_norm]
    clean        = [t for t in test_list if normalize(t) not in train_norm]

    return {
        "n_test":            len(test_list),
        "n_train":           len(train_norm),
        "n_contaminated":    len(contaminated),
        "n_clean":           len(clean),
        "contamination_rate": len(contaminated) / len(test_list) if test_list else 0.0,
        "contaminated_samples": contaminated,
    }


# ── Character n-gram overlap ──────────────────────────────────────────────────

def char_ngrams(text: str, n: int) -> set:
    """
    Return the set of character n-grams from a normalized string.

    Args:
        text: Raw or pre-normalized string.
        n:    N-gram length (e.g., 8 for character 8-grams).

    Returns:
        Set of n-gram strings.
    """
    text = normalize(text)
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def ngram_max_overlap(test_text: str, train_texts: Iterable[str], n: int = 8) -> float:
    """
    Maximum character-n-gram coverage: max over training of |A∩B|/|A|
    where A = test n-grams, B = train n-grams.

    Args:
        test_text:   Test-set target string.
        train_texts: Iterable of training-set target strings.
        n:           N-gram size (default 8).

    Returns:
        Float in [0, 1]; 1.0 means full coverage (exact or near-exact match).
    """
    test_ng = char_ngrams(test_text, n)
    if not test_ng:
        return 0.0
    best = 0.0
    for train_t in train_texts:
        train_ng = char_ngrams(train_t, n)
        if not train_ng:
            continue
        overlap = len(test_ng & train_ng) / len(test_ng)
        if overlap > best:
            best = overlap
        if best >= 1.0:
            break
    return best


# ── Command-line interface ────────────────────────────────────────────────────

def _cli():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if args[0] == "--check-overlap":
        if len(args) < 3:
            print("Usage: normalize_translations.py --check-overlap test.txt train.txt")
            sys.exit(1)
        test_path  = Path(args[1])
        train_path = Path(args[2])
        test_texts  = [l for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        train_texts = [l for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        result = check_overlap(test_texts, train_texts)
        print(f"Test samples:      {result['n_test']}")
        print(f"Train targets:     {result['n_train']}")
        print(f"Contaminated:      {result['n_contaminated']} ({result['contamination_rate']*100:.1f}%)")
        print(f"Clean:             {result['n_clean']}")
        if result['contaminated_samples']:
            print("\nContaminated samples:")
            for s in result['contaminated_samples'][:20]:
                print(f"  {s!r}")
        sys.exit(0)

    # Default: normalize a file
    if len(args) < 2:
        print("Usage: normalize_translations.py input.txt output.txt")
        sys.exit(1)
    in_path  = Path(args[0])
    out_path = Path(args[1])
    normalized = normalize_file(in_path)
    out_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    print(f"Normalized {len(normalized)} lines → {out_path}")


if __name__ == "__main__":
    _cli()
