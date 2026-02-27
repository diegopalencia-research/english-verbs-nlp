# English Verb Phonetics — NLP & Data Science Analysis

> A structured linguistic dataset of 300+ English verbs with IPA transcriptions, phonetic encodings, and past-tense pronunciation rules — analyzed through the lens of data science and NLP.

---

## Project Overview

This project combines English linguistics with data science to explore a fundamental question:

**Can we predict how the past tense of an English verb is pronounced, just from its phonetic properties?**

Regular verbs in English follow a hidden phonetic rule most native speakers don't consciously know. Irregular verbs break that rule entirely, forcing learners to memorize exceptions. This project:

- Builds a clean, structured dataset of 298 English verbs with full IPA notation
- Analyzes phonetic patterns using Python (pandas, matplotlib, seaborn)
- Trains a machine learning classifier to predict verb type (regular vs. irregular)
- Deploys an interactive app for real-time verb lookup and prediction

---

## The Core Linguistic Rule

When a regular English verb takes the -ed suffix, it is **not** always pronounced the same way. The pronunciation depends on the **last sound** of the base form:

| Last Sound of Base Verb | -ed Pronunciation | Example |
|---|---|---|
| Voiceless consonant (p, k, f, s, sh, ch) | /t/ | walk**ed** → /wɔːkt/ |
| Voiced sound (vowel, b, g, m, n, l, r, v, z) | /d/ | call**ed** → /kɔːld/ |
| The sounds /t/ or /d/ | /ɪd/ (extra syllable) | start**ed** → /ˈstɑːrtɪd/ |

This rule is automatic for native speakers — but invisible to learners. Data science lets us visualize and verify it at scale.

---

## Dataset

| File | Description |
|---|---|
| `data/english_verbs.xlsx` | Main dataset — 4 sheets |
| `data/regular_verbs.csv` | Regular verbs only (exported) |
| `data/irregular_verbs.csv` | Irregular verbs only (exported) |

**Columns:**

| Column | Description |
|---|---|
| `Base` | Base form of the verb |
| `Simple_Past` | Simple past form |
| `Past_Participle` | Past participle form |
| `IPA_Base` | IPA transcription — base form |
| `IPA_Past` | IPA transcription — past form |
| `IPA_Past_Participle` | IPA transcription — past participle |
| `Phonetic_Base` | Simplified phonetic spelling |
| `Phonetic_Past` | Simplified phonetic spelling — past |
| `Phonetic_Part` | Simplified phonetic spelling — participle |
| `Last_Sound` (regular) | Phonetic description of the base form's final sound |
| `Ending` (regular) | Resulting -ed pronunciation: /t/, /d/, or /ɪd/ |
| `Vowel_Change` (irregular) | Pattern of vowel change (e.g., iː → ɛ) |

---

## Project Structure

```
english-verbs-nlp/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── english_verbs.xlsx        # Main Excel dataset
│   ├── regular_verbs.csv         # Exported regular verbs
│   └── irregular_verbs.csv       # Exported irregular verbs
│
├── notebooks/
│   └── analysis.ipynb            # Full EDA + ML notebook
│
└── app/
    └── app.py                    # Streamlit interactive app
```

---

## Key Findings

*(From the notebook — update this after running your analysis)*

- **X%** of regular verbs follow the /t/ ending, **X%** the /d/ ending, **X%** the /ɪd/ ending
- The most common irregular pattern is `iː → ɛ` (feel/felt, keep/kept, sleep/slept...)
- The ML classifier reaches **~X% accuracy** distinguishing regular from irregular verbs based purely on spelling features
- Verbs ending in a vowel sound are almost always regular

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/english-verbs-nlp.git
cd english-verbs-nlp
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

### 5. Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Data manipulation |
| matplotlib / seaborn | Visualization |
| scikit-learn | Machine learning |
| Streamlit | Interactive web app |
| openpyxl | Excel file handling |
| Jupyter Notebook | Analysis and documentation |

---

## requirements.txt

```
pandas>=2.0
openpyxl>=3.1
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
streamlit>=1.28
jupyter>=1.0
notebook>=7.0
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

## Author

**[Your Name]**
Aspiring Data Scientist | English C2 Learner

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## License

MIT License — free to use, share, and modify with attribution.
