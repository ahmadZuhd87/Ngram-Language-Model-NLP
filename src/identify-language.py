#!/usr/bin/env python3
"""
identify-language.py — Character-level N-gram Language Model for Language Identification
ENCS5342 Assignment 2 — Task 3

Usage:
  python identify-language.py TRAIN   <train-dir>/<train-base> <model-dir>
  python identify-language.py PREDICT <test-dir> <model-dir>
  python identify-language.py EVALUATE <gold-file> <prediction-file>
"""

import os
import sys
import glob
import random
import re

# All 22 supported language codes
LANGUAGES = [
    "it", "lt", "et", "fr", "hu", "lv", "cs", "en", "da", "de",
    "mt", "nl", "pl", "fi", "pt", "ro", "sk", "sl", "sv", "bg", "el", "es"
]

LM_ORDER = 3   # trigram character-level LM — best balance of accuracy and coverage


def char_tokenize(text):
    """
    Convert a line of text into a space-separated sequence of characters.
    Each character (including spaces) becomes a token.
    The space character is represented as '<SP>' to avoid ambiguity.
    """
    tokens = []
    for ch in text:
        if ch == '\n':
            continue          # skip newlines; sentence boundary is handled by SRILM
        elif ch == ' ':
            tokens.append('<SP>')
        else:
            tokens.append(ch)
    return ' '.join(tokens)


def prepare_char_file(input_path, output_path):
    """Write character-tokenized version of input_path to output_path."""
    with open(input_path, encoding='utf-8', errors='replace') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            tokenized = char_tokenize(line.strip())
            if tokenized:
                fout.write(tokenized + '\n')


def prepare_char_file_single_line(input_path, output_path, line_index=None):
    """
    Write character-tokenized version using only one random line (for Task 3.2).
    If line_index is given, use that specific line; otherwise pick randomly.
    """
    with open(input_path, encoding='utf-8', errors='replace') as fin:
        lines = [l.strip() for l in fin if l.strip()]

    if not lines:
        with open(output_path, 'w') as f:
            f.write('')
        return

    if line_index is not None:
        chosen = lines[line_index % len(lines)]
    else:
        chosen = random.choice(lines)

    with open(output_path, 'w', encoding='utf-8') as fout:
        tokenized = char_tokenize(chosen)
        if tokenized:
            fout.write(tokenized + '\n')


def train(train_base, model_dir, order=LM_ORDER, single_line=False, seed=42):
    """
    TRAIN mode:
    For each language, character-tokenize the training file and build a
    character-level n-gram LM using SRILM ngram-count.
    """
    os.makedirs(model_dir, exist_ok=True)
    random.seed(seed)

    for lang in LANGUAGES:
        train_file = f"{train_base}.{lang}"
        if not os.path.exists(train_file):
            print(f"[TRAIN] WARNING: {train_file} not found, skipping.", file=sys.stderr)
            continue

        char_file = os.path.join(model_dir, f"train.{lang}.char")
        lm_file   = os.path.join(model_dir, f"train.{lang}.lm")

        # Prepare character-tokenized training data
        if single_line:
            prepare_char_file_single_line(train_file, char_file)
        else:
            prepare_char_file(train_file, char_file)

        # Build LM with Witten-Bell smoothing (best performer from Task 2)
        cmd = (
            f"ngram-count "
            f"-text {char_file} "
            f"-order {order} "
            f"-wbdiscount "
            f"-lm {lm_file} "
        )
        print(f"[TRAIN] Building LM for language: {lang}", file=sys.stderr)
        os.system(cmd)

    print(f"[TRAIN] Done. Models saved to: {model_dir}", file=sys.stderr)


def get_perplexity(lm_file, test_char_file, order=LM_ORDER):
    """
    Run ngram -ppl and extract the perplexity value from SRILM output.
    Returns float perplexity, or infinity on failure.
    """
    tmp_out = "/tmp/_ppl_result.txt"
    cmd = (
        f"ngram "
        f"-lm {lm_file} "
        f"-order {order} "
        f"-ppl {test_char_file} "
        f"> {tmp_out} 2>&1"
    )
    os.system(cmd)

    try:
        with open(tmp_out) as f:
            content = f.read()
        # Extract ppl= value from SRILM output line
        # Format: "... ppl= 123.456 ppl1= ..."
        match = re.search(r'ppl=\s*([\d.]+(?:e[+-]?\d+)?)', content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return float('inf')


def predict(test_dir, model_dir, order=LM_ORDER, single_line=False, seed=42):
    """
    PREDICT mode:
    For each test file, character-tokenize it, compute perplexity against
    every language LM, and output the language with the lowest perplexity.
    """
    random.seed(seed)

    # Find all test files (numeric extension only: dev.1, dev.2, ..., test.1, ...)
    # Explicitly exclude anything whose extension is not a pure integer (e.g. dev.gold)
    all_files = glob.glob(os.path.join(test_dir, "*"))
    numeric_files = []
    for p in all_files:
        if not os.path.isfile(p):
            continue
        parts = os.path.basename(p).rsplit('.', 1)
        if len(parts) == 2 and parts[1].isdigit():
            numeric_files.append((parts[0], int(parts[1]), p))

    test_files = [p for _, _, p in sorted(numeric_files, key=lambda x: (x[0], x[1]))]

    for test_path in test_files:
        basename = os.path.basename(test_path)
        char_file = f"/tmp/_char_{basename}.txt"

        if single_line:
            prepare_char_file_single_line(test_path, char_file)
        else:
            prepare_char_file(test_path, char_file)

        best_lang = None
        best_ppl  = float('inf')

        for lang in LANGUAGES:
            lm_file = os.path.join(model_dir, f"train.{lang}.lm")
            if not os.path.exists(lm_file):
                continue
            ppl = get_perplexity(lm_file, char_file, order=order)
            if ppl < best_ppl:
                best_ppl  = ppl
                best_lang = lang

        print(f"{basename}\t{best_lang}")


def evaluate(gold_file, pred_file):
    """
    EVALUATE mode:
    Compare predictions against gold labels and report accuracy.
    """
    with open(gold_file) as f:
        gold = {}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                gold[parts[0]] = parts[1]

    with open(pred_file) as f:
        pred = {}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pred[parts[0]] = parts[1]

    correct = 0
    total   = 0
    errors  = []

    for fname, true_lang in sorted(gold.items()):
        predicted = pred.get(fname, None)
        total += 1
        if predicted == true_lang:
            correct += 1
        else:
            errors.append((fname, true_lang, predicted))

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")

    if errors:
        print("\nErrors:")
        for fname, true_lang, predicted in errors:
            print(f"  {fname}: true={true_lang}, predicted={predicted}")


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].upper()

    if mode == "TRAIN":
        # Usage: TRAIN <train-dir>/<train-base> <model-dir>
        if len(sys.argv) < 4:
            print("Usage: python identify-language.py TRAIN <train-base> <model-dir>")
            sys.exit(1)
        train_base = sys.argv[2]
        model_dir  = sys.argv[3]
        train(train_base, model_dir)

    elif mode == "TRAIN1":
        # Task 3.2: train with 1 random line per language, order 1
        if len(sys.argv) < 4:
            print("Usage: python identify-language.py TRAIN1 <train-base> <model-dir>")
            sys.exit(1)
        train_base = sys.argv[2]
        model_dir  = sys.argv[3]
        train(train_base, model_dir, order=1, single_line=True)

    elif mode == "PREDICT":
        # Usage: PREDICT <test-dir> <model-dir>
        if len(sys.argv) < 4:
            print("Usage: python identify-language.py PREDICT <test-dir> <model-dir>")
            sys.exit(1)
        test_dir  = sys.argv[2]
        model_dir = sys.argv[3]
        predict(test_dir, model_dir)

    elif mode == "PREDICT1":
        # Task 3.2: predict using 1 random line from each test file, order 1
        if len(sys.argv) < 4:
            print("Usage: python identify-language.py PREDICT1 <test-dir> <model-dir>")
            sys.exit(1)
        test_dir  = sys.argv[2]
        model_dir = sys.argv[3]
        predict(test_dir, model_dir, order=1, single_line=True)

    elif mode == "EVALUATE":
        # Usage: EVALUATE <gold-file> <prediction-file>
        if len(sys.argv) < 4:
            print("Usage: python identify-language.py EVALUATE <gold-file> <pred-file>")
            sys.exit(1)
        evaluate(sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: TRAIN, PREDICT, EVALUATE, TRAIN1, PREDICT1")
        sys.exit(1)