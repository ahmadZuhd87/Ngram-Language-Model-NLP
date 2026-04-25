# Ngram-Language-Model-NLP
N-gram Language Model for text generation and language identification using SRILM and Python (ENCS5342 Assignment)
# N-gram Language Model for NLP

This project implements **N-gram Language Models** for text processing and **language identification** as part of the ENCS5342 course (Information Retrieval with Applications of NLP).

---

## 📌 Project Overview

The project explores:

* Building N-gram Language Models using SRILM
* Evaluating models using **Perplexity** and **OOV rate**
* Studying effects of:

  * Tokenization
  * Training size
  * N-gram order
  * Smoothing techniques
* Implementing a **character-level language identification system**

---

## ⚙️ Technologies Used

* Python 3
* Bash scripting
* SRILM Toolkit
* subword-nmt (BPE)
* Unix/Linux (WSL recommended)

---

## 📂 Project Structure

```
.
├── src/
│   └── identify-language.py
├── scripts/
│   ├── task1.sh
│   ├── task2.sh
│   └── task3.sh
├── results/
├── Assignment2_1222332_1220829.pdf
└── README.md
```

---

## 🚀 Tasks Implemented

### 🔹 Task 1: Build Language Model

* Tokenization using Moses tokenizer
* Build trigram model with SRILM
* Evaluate perplexity on:

  * UNCorpus (in-domain)
  * Bible (out-of-domain)

---

### 🔹 Task 2: Model Experiments

Studied the effect of:

* Tokenization techniques (raw, tok, lc, stemming, BPE)
* Training data size
* N-gram order (1 to 5)
* Smoothing methods:

  * Add-1
  * Add-k
  * Witten-Bell
  * Kneser-Ney

---

### 🔹 Task 3: Language Identification

Implemented a **character-level trigram model** to classify text language.

Features:

* Supports 22 languages
* Uses perplexity to select best language
* Achieved high accuracy on development data

---

## ▶️ How to Run

### 1. Train models

```
python identify-language.py TRAIN Europarl/train/train modeldir
```

### 2. Predict languages

```
python identify-language.py PREDICT Europarl/test modeldir > test.pred
```

### 3. Evaluate

```
python identify-language.py EVALUATE Europarl/dev/dev.gold dev.predict
```

---

## 📊 Results

* Best model: Character-level trigram + Witten-Bell smoothing
* Achieved high accuracy on development dataset
* Successfully generated predictions for unseen test data

---

## 🧠 Key Insights

* Tokenization reduces OOV and improves model coverage
* Larger training data improves in-domain performance
* Higher n-gram order reduces perplexity (in-domain)
* Out-of-domain data leads to high perplexity and OOV

---

## 👨‍💻 Authors

* Ahmad Zuhd (1222332)
* Bara Mohsen (1220829)

---

## 📎 Course

ENCS5342 — Information Retrieval with Applications of NLP
Birzeit University
