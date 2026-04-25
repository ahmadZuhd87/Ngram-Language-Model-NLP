#!/bin/bash
# ENCS5342 — Assignment 2 — Task 3: Guess the Language!
# Run from the root of the assignment folder (where Europarl/ is)
# Requirements: SRILM (ngram-count, ngram), Python 3, identify-language.py in same folder

set -e
ROOT="$(pwd)"
export PATH="$ROOT/SRILM/SRILM/bin/i686-m64:$PATH"

which ngram-count
which ngram
TRAIN_BASE="Europarl/train/train"
DEV_DIR="Europarl/dev"
TEST_DIR="Europarl/test"
GOLD="Europarl/dev/dev.gold"

echo "============================================================"
echo "QUESTION 3.1 — Full model: character trigram + Witten-Bell"
echo "============================================================"

echo ""
echo "[Step 1] Training language models (all 22 languages)..."
python3 identify-language.py TRAIN $TRAIN_BASE modeldir

echo ""
echo "[Step 2] Predicting development set..."
python3 identify-language.py PREDICT $DEV_DIR modeldir > dev.predict

echo ""
echo "[Step 3] Evaluating on development set..."
python3 identify-language.py EVALUATE $GOLD dev.predict

echo ""
echo "============================================================"
echo "QUESTION 3.2 — Reduced: 1 random line, order 1"
echo "============================================================"

echo ""
echo "[Step 1] Training with 1 random line per language (order 1)..."
python3 identify-language.py TRAIN1 $TRAIN_BASE modeldir_small

echo ""
echo "[Step 2] Predicting development set (1 random line from each dev file)..."
python3 identify-language.py PREDICT1 $DEV_DIR modeldir_small > dev.predict.small

echo ""
echo "[Step 3] Evaluating reduced model..."
python3 identify-language.py EVALUATE $GOLD dev.predict.small

echo ""
echo "============================================================"
echo "QUESTION 3.3 — Blind test set predictions"
echo "============================================================"

echo ""
echo "[Step 1] Predicting blind test set using full model..."
python3 identify-language.py PREDICT $TEST_DIR modeldir > test.pred

echo ""
echo "test.pred contents:"
cat test.pred

echo ""
echo "============================================================"
echo "Task 3 complete!"
echo "Output files: dev.predict, dev.predict.small, test.pred"
echo "============================================================"