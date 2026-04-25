#!/bin/bash
# ENCS5342 — Assignment 2 — Task 2
# Run from the root of the assignment folder (where English-Mix/ and Tools/ are)

#!/bin/bash
set -e

BASE="English-Mix"
ROOT="$(pwd)"
export PATH="$ROOT/SRILM/SRILM/bin/i686-m64:$PATH"
echo "============================================================"
echo "PREPROCESSING — Create all tokenization variants"
echo "============================================================"

echo "[1/4] Tokenizing..."
cd "$ROOT/Tools"

perl tokenizer.pl -no-escape < "$ROOT/$BASE/UNCorpus.train" > "$ROOT/$BASE/UNCorpus.train.tok"
perl tokenizer.pl -no-escape < "$ROOT/$BASE/UNCorpus.test"  > "$ROOT/$BASE/UNCorpus.test.tok"
perl tokenizer.pl -no-escape < "$ROOT/$BASE/Bible.test"     > "$ROOT/$BASE/Bible.test.tok"
perl tokenizer.pl -no-escape < "$ROOT/$BASE/Fair.test"      > "$ROOT/$BASE/Fair.test.tok"

cd "$ROOT"

echo "[2/4] Lowercasing..."
tr '[:upper:]' '[:lower:]' < "$BASE/UNCorpus.train.tok" > "$BASE/UNCorpus.train.tok.lc"
tr '[:upper:]' '[:lower:]' < "$BASE/UNCorpus.test.tok"  > "$BASE/UNCorpus.test.tok.lc"
tr '[:upper:]' '[:lower:]' < "$BASE/Bible.test.tok"     > "$BASE/Bible.test.tok.lc"
tr '[:upper:]' '[:lower:]' < "$BASE/Fair.test.tok"      > "$BASE/Fair.test.tok.lc"

echo "[3/4] Stemming (Porter)..."
cd "$ROOT/Tools"

perl porter.pl < "$ROOT/$BASE/UNCorpus.train.tok.lc" > "$ROOT/$BASE/UNCorpus.train.tok.lc.port"
perl porter.pl < "$ROOT/$BASE/UNCorpus.test.tok.lc"  > "$ROOT/$BASE/UNCorpus.test.tok.lc.port"
perl porter.pl < "$ROOT/$BASE/Bible.test.tok.lc"     > "$ROOT/$BASE/Bible.test.tok.lc.port"
perl porter.pl < "$ROOT/$BASE/Fair.test.tok.lc"      > "$ROOT/$BASE/Fair.test.tok.lc.port"

cd "$ROOT"

# --- Step 4: BPE ---
echo "[4/4] BPE..."
subword-nmt learn-bpe -s 10000 < $BASE/UNCorpus.train.tok.lc > $BASE/bpe.codes
subword-nmt apply-bpe -c $BASE/bpe.codes < $BASE/UNCorpus.train.tok.lc > $BASE/UNCorpus.train.tok.lc.bpe
subword-nmt apply-bpe -c $BASE/bpe.codes < $BASE/UNCorpus.test.tok.lc  > $BASE/UNCorpus.test.tok.lc.bpe
subword-nmt apply-bpe -c $BASE/bpe.codes < $BASE/Bible.test.tok.lc     > $BASE/Bible.test.tok.lc.bpe
subword-nmt apply-bpe -c $BASE/bpe.codes < $BASE/Fair.test.tok.lc      > $BASE/Fair.test.tok.lc.bpe

echo "Preprocessing complete."
echo ""

echo "============================================================"
echo "EXPERIMENT 2.1 — Effect of Tokenization"
echo "Fixed: 100% data, trigram (3), Witten-Bell (-wbdiscount)"
echo "============================================================"

declare -a TRAIN_FILES=(
    "$BASE/UNCorpus.train"
    "$BASE/UNCorpus.train.tok"
    "$BASE/UNCorpus.train.tok.lc"
    "$BASE/UNCorpus.train.tok.lc.port"
    "$BASE/UNCorpus.train.tok.lc.bpe"
)
declare -a TEST_UN=(
    "$BASE/UNCorpus.test"
    "$BASE/UNCorpus.test.tok"
    "$BASE/UNCorpus.test.tok.lc"
    "$BASE/UNCorpus.test.tok.lc.port"
    "$BASE/UNCorpus.test.tok.lc.bpe"
)
declare -a TEST_BIBLE=(
    "$BASE/Bible.test"
    "$BASE/Bible.test.tok"
    "$BASE/Bible.test.tok.lc"
    "$BASE/Bible.test.tok.lc.port"
    "$BASE/Bible.test.tok.lc.bpe"
)
declare -a TEST_FAIR=(
    "$BASE/Fair.test"
    "$BASE/Fair.test.tok"
    "$BASE/Fair.test.tok.lc"
    "$BASE/Fair.test.tok.lc.port"
    "$BASE/Fair.test.tok.lc.bpe"
)

for i in 0 1 2 3 4; do
    LM="${TRAIN_FILES[$i]}.3.wbdiscount.lm"
    echo "--- Training: ${TRAIN_FILES[$i]} ---"
    ngram-count -text "${TRAIN_FILES[$i]}" -order 3 -wbdiscount -lm "$LM"
    ngram -lm "$LM" -order 3 -ppl "${TEST_UN[$i]}"
    ngram -lm "$LM" -order 3 -ppl "${TEST_BIBLE[$i]}"
    ngram -lm "$LM" -order 3 -ppl "${TEST_FAIR[$i]}"
    echo ""
done

echo "============================================================"
echo "EXPERIMENT 2.2 — Effect of Training Size"
echo "Fixed: tok.lc, trigram, Witten-Bell"
echo "============================================================"

for n in 70000 35000 17500 8750 4375; do
    SUBSET="/tmp/train_${n}.tok.lc"
    LM="/tmp/lm_${n}.lm"
    head -n $n $BASE/UNCorpus.train.tok.lc > "$SUBSET"
    echo "--- Training size: $n lines ---"
    ngram-count -text "$SUBSET" -order 3 -wbdiscount -lm "$LM"
    ngram -lm "$LM" -order 3 -ppl $BASE/UNCorpus.test.tok.lc
    ngram -lm "$LM" -order 3 -ppl $BASE/Bible.test.tok.lc
    ngram -lm "$LM" -order 3 -ppl $BASE/Fair.test.tok.lc
    echo ""
done

echo "============================================================"
echo "EXPERIMENT 2.3 — Effect of LM Order"
echo "Fixed: tok.lc, 100% data, Witten-Bell"
echo "============================================================"

for order in 1 2 3 4 5; do
    LM="/tmp/lm.order${order}.lm"
    echo "--- Order: $order ---"
    ngram-count -text $BASE/UNCorpus.train.tok.lc -order $order -wbdiscount -lm "$LM"
    ngram -lm "$LM" -order $order -ppl $BASE/UNCorpus.test.tok.lc
    ngram -lm "$LM" -order $order -ppl $BASE/Bible.test.tok.lc
    ngram -lm "$LM" -order $order -ppl $BASE/Fair.test.tok.lc
    echo ""
done

echo "============================================================"
echo "EXPERIMENT 2.4 — Effect of Smoothing Method"
echo "Fixed: tok.lc, 100% data, 5-gram"
echo "============================================================"

TRAIN_LC="$BASE/UNCorpus.train.tok.lc"

echo "--- Add-1 (Laplace) ---"
ngram-count -text $TRAIN_LC -order 5 -addsmooth 1 -lm /tmp/lm.add1.lm
ngram -lm /tmp/lm.add1.lm -order 5 -ppl $BASE/UNCorpus.test.tok.lc
ngram -lm /tmp/lm.add1.lm -order 5 -ppl $BASE/Bible.test.tok.lc
ngram -lm /tmp/lm.add1.lm -order 5 -ppl $BASE/Fair.test.tok.lc

echo "--- Add-0.1 ---"
ngram-count -text $TRAIN_LC -order 5 -addsmooth 0.1 -lm /tmp/lm.add01.lm
ngram -lm /tmp/lm.add01.lm -order 5 -ppl $BASE/UNCorpus.test.tok.lc
ngram -lm /tmp/lm.add01.lm -order 5 -ppl $BASE/Bible.test.tok.lc
ngram -lm /tmp/lm.add01.lm -order 5 -ppl $BASE/Fair.test.tok.lc

echo "--- Good-Turing (SRILM default) ---"
ngram-count -text $TRAIN_LC -order 5 -lm /tmp/lm.gt.lm
ngram -lm /tmp/lm.gt.lm -order 5 -ppl $BASE/UNCorpus.test.tok.lc
ngram -lm /tmp/lm.gt.lm -order 5 -ppl $BASE/Bible.test.tok.lc
ngram -lm /tmp/lm.gt.lm -order 5 -ppl $BASE/Fair.test.tok.lc

echo "--- Witten-Bell ---"
ngram-count -text $TRAIN_LC -order 5 -wbdiscount -lm /tmp/lm.wb.lm
ngram -lm /tmp/lm.wb.lm -order 5 -ppl $BASE/UNCorpus.test.tok.lc
ngram -lm /tmp/lm.wb.lm -order 5 -ppl $BASE/Bible.test.tok.lc
ngram -lm /tmp/lm.wb.lm -order 5 -ppl $BASE/Fair.test.tok.lc

echo "--- Kneser-Ney ---"
ngram-count -text $TRAIN_LC -order 5 -kndiscount -lm /tmp/lm.kn.lm
ngram -lm /tmp/lm.kn.lm -order 5 -ppl $BASE/UNCorpus.test.tok.lc
ngram -lm /tmp/lm.kn.lm -order 5 -ppl $BASE/Bible.test.tok.lc
ngram -lm /tmp/lm.kn.lm -order 5 -ppl $BASE/Fair.test.tok.lc

echo ""
echo "============================================================"
echo "Task 2 DONE. Copy the PPL/OOV numbers into your report."
echo "============================================================"