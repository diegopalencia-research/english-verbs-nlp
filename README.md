# English Verb Phonetics — NLP & Data Science

---

A full-stack AI application combining linguistics, machine learning, and data science to analyze English verb morphology and phonetic behavior.

Built around a structured dataset of **298 English verbs** (160 regular, 138 irregular), enriched with IPA transcriptions, phonetic encodings, morphological features, and pronunciation rules.

---

## What It Does

- **Lemmatizer** — resolves any conjugated form to its base verb (`fought` → `fight`, `went` → `go`, `broken` → `break`)
- **Phonetic Rule Engine** — predicts -ed pronunciation (`/t/`, `/d/`, `/ɪd/`) from phonetic features
- **IPA Display** — full transcription for base form, simple past, and past participle
- **Audio Pronunciation** — browser-native speech synthesis for all three verb forms
- **ML Classifier** — classifies unknown verbs as regular or irregular with confidence score
- **Model Dashboard** — live accuracy, precision, recall, F1, confusion matrix, and feature importance

---

## Key Findings

- **47.5%** of regular verbs follow the `/d/` ending — the most common past tense sound
- **28.1%** follow `/t/` (voiceless consonants: walk → walkt, cook → cookt)
- **24.4%** follow `/ɪd/` — verbs ending in /t/ or /d/ add an extra syllable (start → startid)
- The most frequent irregular pattern is **"no vowel change"** (cut/cut, put/put, hit/hit) — 20 verbs
- Second most frequent: **iː → ɛ** (feel/felt, keep/kept, sleep/slept) — 19 verbs
- **Irregular verbs are shorter on average** — 4.4 letters vs 5.5 for regular verbs, reflecting Old English monosyllabic roots
- The **Random Forest classifier reaches 75.0% accuracy** distinguishing regular from irregular verbs using spelling features alone — outperforming the 54% random baseline by 21 percentage points
- Top predictive features: second-to-last character, verb length, consonant count, vowel count, last letter

---

## Dataset

| Property | Value |
|---|---|
| Total verbs | 298 |
| Regular verbs | 160 |
| Irregular verbs | 138 |
| Features per verb | IPA (3), Phonetic encoding (3), Last sound, -ed ending, Vowel change pattern |
| Source | Custom structured dataset — manually curated |

---

## ML Model

```
Algorithm:     Random Forest Classifier
Estimators:    200 decision trees
Max depth:     8
Validation:    5-fold stratified cross-validation
Train/Test:    80 / 20 split (stratified)

Accuracy:      75.0%  (Random Forest)
Comparison:    66.7%  (Logistic Regression)
Baseline:      54.0%  (majority class — random)
Improvement:   +21 percentage points over baseline
```

**25 engineered features:** length · vowel_count · consonant_count · syllable_count · phonetic_category (voiced/voiceless/stop/vowel) · is_voiceless · is_voiced · is_stop · is_vowel_end · 20 binary suffix patterns · bigram · trigram · last_letter · second_last

---

## Project Structure

```
english-verbs-nlp/
├── app.py                         Main Streamlit application (5 pages)
├── services/
│   ├── lemmatizer.py              Multi-form verb search (base/past/participle)
│   ├── preprocessing.py           Feature engineering pipeline — 25 features
│   └── phonetics.py               Rule-based -ed pronunciation prediction
├── scripts/
│   ├── train_model.py             Reproducible training — outputs metrics.json
│   └── evaluate_model.py          Standalone evaluation report
├── models/
│   ├── classifier.pkl             Trained RandomForest (serialized)
│   └── metrics.json               Evaluation metrics (machine-readable)
├── data/
│   └── english_verbs.xlsx         298-verb dataset (2 sheets)
├── notebooks/
│   └── analysis.ipynb             Full EDA + ML notebook (16 cells)
├── README.md
└── requirements.txt
```

---

## App Pages

| Page | Description |
|---|---|
| **Verb Lookup** | Search any form — base, past, or participle. Returns IPA, audio, phonetic rule, sibling verbs |
| **Phonetic Explorer** | Filter regular verbs by -ed ending. Browse irregular patterns with vowel change grid |
| **Charts & Analysis** | Distribution, -ed endings, irregular patterns, verb length — 4 chart tabs |
| **Model Performance** | Live accuracy, precision, recall, F1, confusion matrix, CV scores, feature importance |
| **Verb Reference** | Searchable full table — regular and irregular with pattern filter |

---

## Setup

```bash
git clone https://github.com/diegopalencia-research/english-verbs-nlp.git
cd english-verbs-nlp
pip install -r requirements.txt
streamlit run app.py
```

Retrain the model:
```bash
python scripts/train_model.py
```

Evaluate:
```bash
python scripts/evaluate_model.py
```

---

## The Phonetic Rule

When a regular verb takes **-ed**, pronunciation is determined by the **final sound** of the base form — not the final letter:

| Final Sound | -ed Sound | Examples |
|---|---|---|
| Voiceless (p, k, f, s, sh, ch) | **/t/** | walk → walkt · cook → cookt |
| Voiced (vowels, b, g, v, z, m, n, l, r) | **/d/** | call → calld · love → loved |
| /t/ or /d/ | **/ɪd/** | start → startid · need → needid |

Rule formalized in Chomsky & Halle (1968) — validated here with supervised ML.

---

## Research Foundation

> *Can phonetic features predict morphological class in English verbs?*

- **Chomsky & Halle (1968)** — *The Sound Pattern of English* — phonological voicing rules for -ed allomorphy
- **Rumelhart & McClelland (1986)** — past tense acquisition using connectionist networks — benchmark for this task
- **Berko (1958)** — Wug test — productive rule application in English morphology

Preprint in preparation — OSF Preprints.

---

## Requirements

```
pandas>=2.0
openpyxl>=3.1
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
streamlit>=1.28
scipy>=1.10
numpy>=1.24
```

---

## What I Learned

This project taught me the **complete data science workflow**:

1. **Data collection and structuring** — building a dataset from domain knowledge
2. **Exploratory data analysis** — finding patterns in linguistic data
3. **Feature engineering** — turning text/phonetics into ML-ready features
4. **Model training and evaluation** — classification with scikit-learn
5. **Deployment** — shipping a real interactive application

It also showed that **data science is a tool for understanding any domain** — not just finance or tech. Language is data too.

---

**Live App:** https://english-verbs-nlp.streamlit.app/
&nbsp;&nbsp;·&nbsp;&nbsp;
**GitHub:** github.com/diegopalencia-research/english-verbs-nlp

## Author

**Diego José Palencia Robles**
*Data Science & NLP Projects — Applied Linguistics + Machine Learning*

- GitHub; @diegopalencia-research: https://github.com/diegopalencia-research
- LinkedIn: https://www.linkedin.com/in/diego-jose-palencia-robles/

---

## License

MIT License — free to use, share, and modify with attribution.
