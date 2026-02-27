# app.py
# Verb Phonetics — diegopalencia
# Single-file Streamlit app with:
# - robust search by any form (base / simple past / past participle)
# - IPA / readable phonetic spellings columns
# - inline browser TTS play buttons (no external libs)
# - Intro tab, Phonetic Explorer, Charts, Verb Reference
# - improved contrast, safe font fallbacks, accessible UI
# - ML classifier fallback (RandomForest) for unknown verbs
# - sample-data fallback if 'data/english_verbs.xlsx' is missing
# - extensive comments and helpers for maintainability
#
# Notes:
# - Requires: streamlit, pandas, matplotlib, scikit-learn
# - Run: `streamlit run app.py` from project root (ensure data folder present)
# - If audio silent: click anywhere on page (user gesture) and try again, or test in Chrome/Firefox.

import os
import json
import sys
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------
# Basic configuration & logging
# -------------------------
st.set_page_config(
    page_title="Verb Phonetics — diegopalencia",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure simple logger to help debug in deployed environments
logger = logging.getLogger("verb_phonetics")
if not logger.handlers:
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -------------------------
# Styling: safe font stacks and color tokens
# Use system fonts / safe fallbacks to avoid Google fonts being blocked.
# -------------------------
st.markdown(
    """
<style>
:root {
  --bg: #080C14;
  --panel: #0D1220;
  --muted: #7A8796;
  --text: #E8EAF0;
  --hint: #9FB6C6;
  --accent1: #7EB8D4;
  --accent2: #D48A90;
  --accent3: #D4B87A;
  --accent4: #8ECFB0;
  --border: #1E2535;
  --card-radius: 8px;
}

/* layout background */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Syne", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* monospace stack for technical text */
.mono, code, pre, .mono-text { font-family: "DM Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", "Courier New", monospace !important; }

/* titles and subtitles */
.page-title { font-weight: 800; font-size: 2.6rem; color:#F4F7FF; margin-bottom:0.1rem; line-height:1.02; }
.page-subtitle { font-family: DM Mono, monospace; color: var(--hint); font-size:0.88rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1.2rem; }

/* input placeholder visibility */
[data-testid="stTextInput"] input::placeholder { color: var(--hint) !important; opacity:1 !important; }

/* dataframes */
[data-testid="stDataFrame"] * { font-family:DM Mono,monospace !important; font-size:0.78rem !important; }

/* tag / badges */
.verb-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--card-radius); padding: 12px; margin-bottom: 12px; }
.ipa-text { font-family:DM Mono,monospace; color: var(--accent1); background: rgba(126,184,212,0.07); padding:6px 8px; border-radius:6px; display:inline-block; }
.rule-badge { padding:6px 10px; border-radius:6px; font-family:DM Mono,monospace; }

/* pattern cards */
.pattern-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap: 12px; margin-top:12px; }
.pattern-card { background: var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px; transition:all 0.12s; }
.pattern-card:hover { transform: translateY(-6px); border-color: var(--accent1); }

/* small text helpers */
.small-muted { font-family:DM Mono,monospace; color: var(--hint); font-size:0.82rem; }
.try-line { color: #C8CDD8; font-family:DM Mono,monospace; font-size:0.86rem; }

.audio-note { font-family:DM Mono,monospace; color:#7A8796; font-size:0.78rem; margin-top:6px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Constants: colors + small helpers
# -------------------------
CHART_BG = "#080C14"
CHART_TEXT = "#6B7794"
ACCENT1, ACCENT2, ACCENT3, ACCENT4 = "#7EB8D4", "#D48A90", "#D4B87A", "#8ECFB0"

# -------------------------
# Inline audio approach
#
# Strategy:
# - Each play button renders HTML with an inline onclick handler that constructs a
#   SpeechSynthesisUtterance and calls window.speechSynthesis.speak().
# - This ensures the TTS call runs as a direct user gesture from button click.
# - We avoid relying on a globally injected function (some platforms strip that).
# -------------------------
def _safe_js_str(s: Optional[str]) -> str:
    """Return a JSON-escaped JS string literal or 'null'."""
    if s is None:
        return "null"
    # ensure it's a str
    return json.dumps(str(s))

def render_play_buttons(base: Optional[str], past: Optional[str] = None, part: Optional[str] = None) -> None:
    """
    Render a compact inline play button block for base/past/part using inline JS handlers.
    """
    base_js = _safe_js_str(base)
    past_js = _safe_js_str(past)
    part_js = _safe_js_str(part)

    # Inline JS function (self-invoking) on each button click to ensure user gesture.
    html = f"""
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px;">
      <button onclick="(function(t){{ if(!t) return; try{{ const u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=0.86; u.pitch=1.0; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); }}catch(e){{console.warn(e);}} }})({base_js})"
        style="background:#0D1220;border:1px solid #1E2535;border-radius:8px;color:{ACCENT1};padding:7px 12px;font-family:DM Mono,monospace;cursor:pointer;">▶︎ Base</button>
      {"<button onclick=\"(function(t){ if(!t) return; try{ const u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=0.86; u.pitch=1.0; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);}catch(e){console.warn(e);} })(" + past_js + ")\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:8px;color:" + ACCENT2 + ";padding:7px 12px;font-family:DM Mono,monospace;cursor:pointer;\">▶︎ Past</button>" if past else ""}
      {"<button onclick=\"(function(t){ if(!t) return; try{ const u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=0.86; u.pitch=1.0; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);}catch(e){console.warn(e);} })(" + part_js + ")\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:8px;color:" + ACCENT3 + ";padding:7px 12px;font-family:DM Mono,monospace;cursor:pointer;\">▶︎ Part</button>" if part else ""}
      <div class="audio-note" style="margin-left:8px;">Uses browser speechSynthesis — click once to enable audio if needed.</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_play_list(rows: List[Dict[str, Any]], title: Optional[str] = None, max_items: int = 8) -> None:
    """
    Render a vertical list of small inline play controls for a list of verbs (rows).
    Each line: verb name + small ▶︎ base / ▶︎ past / ▶︎ part buttons.
    """
    items_html = []
    count = 0
    for r in rows:
        if count >= max_items:
            break
        b = str(r.get("Base", ""))
        p = str(r.get("Simple_Past", "")) or None
        pp = str(r.get("Past_Participle", "")) or None
        b_js = _safe_js_str(b)
        p_js = _safe_js_str(p)
        pp_js = _safe_js_str(pp)
        item = f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <div style="width:140px;font-family:DM Mono,monospace;color:#C8CDD8;">{b}</div>
          <button onclick="(function(t){{ if(!t) return; try{{const u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=0.86; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);}}catch(e){{console.warn(e);}}}})({b_js})"
            style="background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:{ACCENT1};padding:5px 8px;font-family:DM Mono,monospace;cursor:pointer;">▶︎ base</button>
          {"<button onclick=\"(function(t){ if(!t) return; try{const u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=0.86; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);}catch(e){console.warn(e);} })(" + p_js + ")\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:" + ACCENT2 + ";padding:5px 8px;font-family:DM Mono,monospace;cursor:pointer;\">▶︎ past</button>" if p else ""}
          {"<button onclick=\"(function(t){ if(!t) return; try{const u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=0.86; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);}catch(e){console.warn(e);} })(" + pp_js + ")\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:" + ACCENT3 + ";padding:5px 8px;font-family:DM Mono,monospace;cursor:pointer;\">▶︎ part</button>" if pp else ""}
        </div>
        """
        items_html.append(item)
        count += 1
    html = f"<div style='font-family:DM Mono,monospace;color:#C8CDD8;'>{f'<div style=\"font-weight:700;margin-bottom:8px;\">{title}</div>' if title else ''}{''.join(items_html)}</div>"
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Data loading: try to load excel dataset; if missing, fallback to small sample dataset
# The Excel file is expected to have sheets: 'Regular Verbs' and 'Irregular Verbs' with header row at index 2.
# -------------------------
@st.cache_data
def load_dataset(excel_path: str = "data/english_verbs.xlsx") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset, normalize columns and return (df_reg, df_irreg).
    If the file is missing, produce a small sample dataset so app doesn't crash.
    """
    path = Path(excel_path)
    if not path.exists():
        logger.warning(f"Dataset not found at {excel_path}. Loading fallback sample dataset.")
        # small fallback dataset
        df_reg = pd.DataFrame([
            {"Base": "walk", "Simple_Past": "walked", "Past_Participle": "walked",
             "IPA_Base": "wɔːk", "IPA_Past": "wɔːkt", "IPA_PP": "wɔːkt",
             "Phonetic_Base": "wawk", "Phonetic_Past": "wawkt", "Phonetic_PP": "wawkt",
             "Last_Sound": "k", "Ending": "/t/"},
            {"Base": "play", "Simple_Past": "played", "Past_Participle": "played",
             "IPA_Base": "pleɪ", "IPA_Past": "pleɪd", "IPA_PP": "pleɪd",
             "Phonetic_Base": "play", "Phonetic_Past": "playd", "Phonetic_PP": "playd",
             "Last_Sound": "y", "Ending": "/d/"},
        ])
        df_irreg = pd.DataFrame([
            {"Base": "write", "Simple_Past": "wrote", "Past_Participle": "written",
             "IPA_Base": "raɪt", "IPA_Past": "roʊt", "IPA_PP": "ˈrɪtən",
             "Phonetic_Base": "ryt", "Phonetic_Past": "roht", "Phonetic_PP": "rittən",
             "Vowel_Change": "aɪ → oʊ"},
            {"Base": "go", "Simple_Past": "went", "Past_Participle": "gone",
             "IPA_Base": "ɡoʊ", "IPA_Past": "wɛnt", "IPA_PP": "ɡɔn",
             "Phonetic_Base": "goh", "Phonetic_Past": "went", "Phonetic_PP": "gon",
             "Vowel_Change": "oʊ → ɛ"}
        ])
    else:
        # try reading excel with expected structure
        try:
            logger.info(f"Loading dataset from {excel_path}")
            df_reg = pd.read_excel(excel_path, sheet_name="Regular Verbs", header=2)
            df_irreg = pd.read_excel(excel_path, sheet_name="Irregular Verbs", header=2)
        except Exception as e:
            logger.exception("Failed to read dataset; using fallback sample.")
            # revert to fallback
            return load_dataset(excel_path=None)

    # standardize column names (if needed)
    reg_cols = ['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'IPA_PP',
                'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Last_Sound', 'Ending']
    irreg_cols = ['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'IPA_PP',
                  'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Vowel_Change']

    # assign columns defensively if shapes match
    def safe_assign_columns(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
        # if df already has all expected columns, keep as is
        cols = list(df.columns)
        if all(c in cols for c in expected):
            return df.copy()
        # else try to set first len(expected) columns to expected names
        df = df.copy()
        try:
            df.columns = expected[:len(df.columns)]
        except Exception as e:
            # fallback: create minimal dataframe
            logger.warning("Column assignment fallback: creating minimal normalized dataframe.")
            new = pd.DataFrame(columns=expected)
            for col in expected:
                new[col] = df.get(col, "")
            return new
        # ensure all expected exist
        for c in expected:
            if c not in df.columns:
                df[c] = ""
        return df[expected]

    df_reg = safe_assign_columns(df_reg, reg_cols)
    df_irreg = safe_assign_columns(df_irreg, irreg_cols)

    # drop empty Base rows, strip strings, fillna, create lowercase helpers
    for df in (df_reg, df_irreg):
        if 'Base' in df.columns:
            df['Base'] = df['Base'].astype(str).str.strip()
            df = df[df['Base'].notna() & (df['Base'] != '')]
    df_reg = df_reg.dropna(subset=['Base']).reset_index(drop=True)
    df_irreg = df_irreg.dropna(subset=['Base']).reset_index(drop=True)

    # ensure phonetic fields exist
    for c in ['Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP']:
        if c not in df_reg.columns:
            df_reg[c] = df_reg.get('IPA_Base', "")
        if c not in df_irreg.columns:
            df_irreg[c] = df_irreg.get('IPA_Base', "")

    # normalize columns to strings & create lowercase helpers for robust matching
    for df in (df_reg, df_irreg):
        for col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
        # helper lower-case columns for search speed and consistent behavior
        df['Base_l'] = df['Base'].str.lower().str.strip()
        df['Simple_Past_l'] = df['Simple_Past'].str.lower().str.strip()
        df['Past_Participle_l'] = df['Past_Participle'].str.lower().str.strip()
    df_reg['Type'] = 'Regular'
    df_irreg['Type'] = 'Irregular'

    # final return
    return df_reg.reset_index(drop=True), df_irreg.reset_index(drop=True)


# -------------------------
# Feature extraction + small model to predict irregularity when verb not in dataset.
# Keep this light-weight; the goal is to give a useful fallback message, not production accuracy.
# -------------------------
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract simple string-based features for model training/prediction."""
    f = pd.DataFrame()
    f['length'] = df['Base'].str.len().fillna(0)
    f['vowel_count'] = df['Base'].str.count('[aeiou]').fillna(0)
    f['consonant_count'] = df['Base'].str.count('[bcdfghjklmnpqrstvwxyz]').fillna(0)
    endings = ['e','n','d','t','l','r','k','g','w','y','ng','nd','ld','nt','in','ow','aw']
    for s in endings:
        # ends_{s} - binary
        f[f'ends_{s}'] = df['Base'].str.endswith(s).astype(int)
    le = LabelEncoder()
    # handle last_letter and second_last safely
    last = df['Base'].str[-1].fillna('_')
    sec_last = df['Base'].str[-2].fillna('_')
    try:
        f['last_letter'] = le.fit_transform(last)
        f['second_last'] = le.fit_transform(sec_last)
    except Exception:
        # fallback: simple ord values
        f['last_letter'] = last.apply(lambda x: ord(x) if x else 0)
        f['second_last'] = sec_last.apply(lambda x: ord(x) if x else 0)
    return f.fillna(0)


@st.cache_resource
def train_model(df_reg: pd.DataFrame, df_irreg: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a small RandomForest classifier to predict irregular vs regular.
    Cached as resource to avoid retraining every run.
    """
    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)
    X = extract_features(df_all)
    y = (df_all['Type'] == 'Irregular').astype(int)
    # small train-test split for reliability
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        # fallback split if stratify fails
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=7)
    rf.fit(X_tr, y_tr)
    # log out-of-sample accuracy if possible (not critical)
    try:
        acc = rf.score(X_te, y_te)
        logger.info(f"Trained RandomForest irregularity predictor — test accuracy: {acc:.3f}")
    except Exception:
        logger.info("Trained RandomForest irregularity predictor.")
    return rf


# -------------------------
# Load dataset & model now (cached)
# -------------------------
df_reg, df_irreg = load_dataset()
model = train_model(df_reg, df_irreg)

# small phonetic pattern examples (for explorer)
PATTERN_EXAMPLES = {
    "iː → ɛ":        "feel/felt · keep/kept · sleep/slept · meet/met",
    "aɪ → oʊ":       "write/wrote · ride/rode · rise/rose · drive/drove",
    "ɪ → æ → ʌ":     "sing/sang/sung · drink/drank/drunk · ring/rang/rung",
    "no vowel change":"cut/cut · put/put · hit/hit · set/set",
    "eɪ → oʊ":       "break/broke · wake/woke · speak/spoke",
    "ɪ → ʌ":         "dig/dug · stick/stuck · win/won",
    "aɪ → ɪ":        "bite/bit · hide/hid · light/lit",
    "ʌ → eɪ → ʌ":   "come/came/come · become/became/become",
    "iː → ɔː":       "see/saw · seek/sought · teach/taught",
    "aɪ → aʊ":       "find/found · bind/bound · wind/wound",
    "aɪ → ɔː":       "fight/fought · buy/bought · catch/caught",
    "oʊ → uː":       "know/knew · grow/grew · throw/threw",
}


# -------------------------
# Sidebar UI & navigation
# -------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:10px 0 12px 0;">
          <div class="mono" style="font-size:0.68rem;color:#4A5568;text-transform:uppercase;">Project</div>
          <div style="font-weight:700;font-size:1.05rem;color:#E8EAF0;line-height:1.1;">English Verb<br>Phonetics</div>
          <div class="small-muted" style="margin-top:8px;">Search by any form (base / simple past / past participle). Use the ▶︎ buttons to listen via your browser's speech engine.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Navigation pages
PAGES = ["/ Intro", "/ Verb Lookup", "/ Phonetic Explorer", "/ Charts & Analysis", "/ Verb Reference"]
default_idx = 1  # open on Verb Lookup by default
page = st.sidebar.radio("Navigate", PAGES, index=default_idx)


# small dataset summary in sidebar
with st.sidebar:
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-family:DM Mono,monospace;color:#4A5568;">
          <div style="font-size:0.62rem;text-transform:uppercase;color:#4A5568;margin-bottom:6px;">Dataset</div>
          <div><span style="color:{ACCENT4}; font-weight:700;">{len(df_reg)}</span> regular</div>
          <div style="margin-top:4px;"><span style="color:{ACCENT2}; font-weight:700;">{len(df_irreg)}</span> irregular</div>
          <div style="margin-top:6px;"><span style='color:{ACCENT1};'>{len(df_reg)+len(df_irreg)}</span> total</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# Utility helpers: phonetic_summary and robust search
# -------------------------
def phonetic_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Return top counts for phonetic columns (Phonetic_Base, Phonetic_Past, Phonetic_PP).
    Each item is a small dataframe with columns: phonetic, count.
    """
    out = {}
    for col in ['Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP']:
        if col in df.columns:
            counts = df[col].value_counts().reset_index().rename(columns={'index': 'phonetic', col: 'count'})
            out[col] = counts.head(10)
        else:
            out[col] = pd.DataFrame({'phonetic': [], 'count': []})
    return out


def find_verb_any_form(token: str) -> Tuple[Optional[str], Optional[pd.Series]]:
    """
    Robust search: exact base / exact past / exact participle / contains (partial).
    Returns (label, row_series) or (None, None) if not found.
    """
    q = str(token).strip().lower()
    if not q:
        return None, None

    # look in regular base
    m = df_reg[df_reg['Base_l'] == q]
    if not m.empty:
        return 'Regular', m.iloc[0]
    # irregular base
    m = df_irreg[df_irreg['Base_l'] == q]
    if not m.empty:
        return 'Irregular', m.iloc[0]
    # simple past exact
    m = df_reg[df_reg['Simple_Past_l'] == q]
    if not m.empty:
        return 'Regular', m.iloc[0]
    m = df_irreg[df_irreg['Simple_Past_l'] == q]
    if not m.empty:
        return 'Irregular', m.iloc[0]
    # past participle exact
    m = df_reg[df_reg['Past_Participle_l'] == q]
    if not m.empty:
        return 'Regular', m.iloc[0]
    m = df_irreg[df_irreg['Past_Participle_l'] == q]
    if not m.empty:
        return 'Irregular', m.iloc[0]
    # partial contains search across the three forms (case-insensitive)
    try:
        m = df_reg[df_reg[['Base', 'Simple_Past', 'Past_Participle']].apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)]
        if not m.empty:
            return 'Regular', m.iloc[0]
        m = df_irreg[df_irreg[['Base', 'Simple_Past', 'Past_Participle']].apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)]
        if not m.empty:
            return 'Irregular', m.iloc[0]
    except Exception as e:
        logger.exception("Partial contains search failed", exc_info=e)

    return None, None


# -------------------------
# Page: Intro
# -------------------------
if page == "/ Intro":
    st.markdown('<div class="page-title">Welcome — Verb Phonetics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">What this tool is · How it works · How to use it</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-family:DM Mono,monospace;color:#C8CDD8;line-height:1.8;">
          <b>What it is</b>
          <p>This app combines a small verb dataset (regular + irregular) with IPA, human-friendly phonetic spellings,
          browser TTS pronunciation, and a simple ML helper to guess irregularity for unknown words. It is meant as
          a research/learning tool — not a full dictionary.</p>

          <b>How it works</b>
          <ol>
            <li>Search by any form: base (write), simple past (wrote) or past participle (written).</li>
            <li>If the verb exists in the dataset, the app shows base / past / participle, IPA, phonetic spellings, audio and rule explanation.</li>
            <li>If not found, the ML model gives a regular/irregular probability and a suggested rule (fallback).</li>
          </ol>

          <b>How to use it</b>
          <ul>
            <li>Type a verb in the lookup box and press Enter (or click away to run search).</li>
            <li>Click the ▶︎ buttons to listen — each click triggers browser TTS via a user gesture.</li>
            <li>Use the Phonetic Explorer to study -ed endings and irregular vowel patterns grouped by counts.</li>
            <li>Use Charts to get a quick sample and audio examples right below each chart.</li>
          </ul>

          <b>Troubleshooting audio</b>
          <ul>
            <li>If nothing plays, click somewhere in the page first (user gesture). Some browsers require a prior interaction before allowing speechSynthesis.</li>
            <li>Try Chrome or Firefox to rule out browser-specific restrictions.</li>
            <li>If you want recorded, high-quality audio rather than TTS, I can add mp3 assets and play them instead.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# Page: Verb Lookup
# -------------------------
elif page == "/ Verb Lookup":
    st.markdown('<div class="page-title">Verb Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">IPA · PHONETICS · AUDIO · RULE EXPLANATION</div>', unsafe_allow_html=True)

    # Clear label above input so users see function immediately
    st.markdown('<div class="small-muted">Search or type a verb (base / simple past / past participle) and press Enter.</div>', unsafe_allow_html=True)
    verb = st.text_input("", placeholder="Search or type a verb — e.g., walk, think, break (press Enter)").strip()

    if verb == "" or verb is None:
        st.markdown('<div class="try-line">Try: <b>walk</b> · <b>think</b> · <b>break</b> · <b>feel</b> · <b>start</b> · <b>google</b> · <b>zoom</b></div>', unsafe_allow_html=True)
    else:
        # normalize and find
        label, row = find_verb_any_form(verb)
        if row is None:
            # prediction fallback
            st.markdown('<div class="section-label">Not in dataset — ML Prediction</div>', unsafe_allow_html=True)
            # features for the single-word
            candidate_df = pd.DataFrame([{'Base': verb}])
            features = extract_features(candidate_df)
            try:
                prob = model.predict_proba(features)[0]
                conf = max(prob) * 100.0
                pred_label = 'Irregular' if prob[1] > 0.5 else 'Regular'
            except Exception:
                pred_label = 'Regular'
                conf = 60.0
            st.markdown(
                f"""
                <div class="verb-card">
                  <div style="font-family:Syne,sans-serif;font-weight:700;color:{ACCENT4};font-size:1.05rem;">Prediction: {pred_label} — {conf:.1f}%</div>
                  <div class="small-muted" style="margin-top:8px;">Model prediction — consult an authoritative dictionary for definitive conjugations.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # playable for the input token (single button)
            render_play_buttons(verb, None, None)
        else:
            # Display the canonical row
            st.markdown(f'<div class="section-label">{label} Verb</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form", row.get("Base", "—"))
            c2.metric("Simple Past", row.get("Simple_Past", "—"))
            c3.metric("Past Participle", row.get("Past_Participle", "—"))

            st.markdown('<div class="section-label">Audio Pronunciation</div>', unsafe_allow_html=True)
            render_play_buttons(row.get("Base"), row.get("Simple_Past"), row.get("Past_Participle"))

            # IPA & phonetic
            st.markdown('<div class="section-label">IPA · Phonetic</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px;">
                  <div class="verb-card" style="flex:1;min-width:180px;">
                    <div class="small-muted">Base</div>
                    <div style="margin-top:8px;"><span class="ipa-text mono">{row.get('IPA_Base','')}</span></div>
                    <div style="margin-top:8px;" class="mono-text">{row.get('Phonetic_Base','')}</div>
                  </div>
                  <div class="verb-card" style="flex:1;min-width:180px;">
                    <div class="small-muted">Simple Past</div>
                    <div style="margin-top:8px;"><span class="ipa-text mono">{row.get('IPA_Past','')}</span></div>
                    <div style="margin-top:8px;" class="mono-text">{row.get('Phonetic_Past','')}</div>
                  </div>
                  <div class="verb-card" style="flex:1;min-width:180px;">
                    <div class="small-muted">Past Participle</div>
                    <div style="margin-top:8px;"><span class="ipa-text mono">{row.get('IPA_PP','')}</span></div>
                    <div style="margin-top:8px;" class="mono-text">{row.get('Phonetic_PP','')}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Rule explanation area
            if label == 'Regular':
                ending = row.get('Ending', '')
                last_sound = row.get('Last_Sound', '—')
                if ending == '/t/':
                    badge_html = f'<span class="rule-badge" style="background:rgba(78,160,120,0.08);border:1px solid rgba(78,160,120,0.2);color:{ACCENT4};">/t/</span>'
                    rule_txt = "Voiceless consonant → -ed pronounced /t/ (no extra syllable)."
                elif ending == '/d/':
                    badge_html = f'<span class="rule-badge" style="background:rgba(126,184,212,0.08);border:1px solid rgba(126,184,212,0.2);color:{ACCENT1};">/d/</span>'
                    rule_txt = "Voiced sound → -ed pronounced /d/."
                else:
                    badge_html = f'<span class="rule-badge" style="background:rgba(200,160,80,0.08);border:1px solid rgba(200,160,80,0.2);color:{ACCENT3};">/ɪd/</span>'
                    rule_txt = "Ends in /t/ or /d/ → -ed pronounced /ɪd/ (extra syllable)."
                st.markdown(f'<div class="verb-card">{badge_html} <span class="small-muted" style="margin-left:10px;">last sound: {last_sound}</span><div style="margin-top:8px;color:#9FB6C6;">{rule_txt}</div></div>', unsafe_allow_html=True)
            else:
                vc = row.get('Vowel_Change', '—')
                siblings = df_irreg[(df_irreg['Vowel_Change'] == vc) & (df_irreg['Base'] != row['Base'])]['Base'].tolist()[:8]
                siblings_str = ", ".join(siblings) if siblings else "—"
                ex = PATTERN_EXAMPLES.get(vc, "")
                st.markdown(f'<div class="verb-card"><div style="font-weight:700;color:{ACCENT2};">{vc}</div><div style="margin-top:8px;color:#9FB6C6;">{ex}</div><div style="margin-top:8px;color:#7A8796;">Same pattern: {siblings_str}</div></div>', unsafe_allow_html=True)

            # Example sentences (small templated contextual examples)
            st.markdown('<div class="section-label">Example Sentences</div>', unsafe_allow_html=True)
            base = row.get('Base', '')
            past = row.get('Simple_Past', '')
            part = row.get('Past_Participle', '')
            st.markdown(
                f"""
                <div style="font-family:DM Mono,monospace;color:#C8CDD8;line-height:1.8;">
                  • Present simple: I often <b>{base}</b> in the morning.<br>
                  • Simple past: Yesterday I <b>{past}</b>.<br>
                  • Present perfect: I have <b>{part}</b> many times.
                </div>
                """,
                unsafe_allow_html=True,
            )

# -------------------------
# Page: Phonetic Explorer
# -------------------------
elif page == "/ Phonetic Explorer":
    st.markdown('<div class="page-title">Phonetic Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Regular -ed rules & irregular vowel patterns</div>', unsafe_allow_html=True)

    tab_reg, tab_irreg = st.tabs(["Regular — -ed Rule", "Irregular — Vowel Patterns"])

    with tab_reg:
        endings = df_reg['Ending'].value_counts()
        st.markdown(
            f"""
            <div style="display:flex;gap:12px;margin-bottom:12px;">
              <div class="verb-card" style="flex:1;">
                <div style="font-weight:700;color:{ACCENT4};font-size:1.3rem;">{endings.get('/t/', 0)}</div>
                <div class="small-muted" style="margin-top:6px;">/t/ (voiceless)</div>
              </div>
              <div class="verb-card" style="flex:1;">
                <div style="font-weight:700;color:{ACCENT1};font-size:1.3rem;">{endings.get('/d/', 0)}</div>
                <div class="small-muted" style="margin-top:6px;">/d/ (voiced)</div>
              </div>
              <div class="verb-card" style="flex:1;">
                <div style="font-weight:700;color:{ACCENT3};font-size:1.3rem;">{endings.get('/ɪd/', 0)}</div>
                <div class="small-muted" style="margin-top:6px;">/ɪd/ (extra syllable)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Filter</div>', unsafe_allow_html=True)
        ending_filter = st.selectbox("", ["/t/ — voiceless", "/d/ — voiced", "/ɪd/ — extra syllable"], label_visibility="collapsed")
        ending_code = ending_filter.split(" ")[0]
        filtered = df_reg[df_reg['Ending'] == ending_code][['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Last_Sound', 'Ending']].reset_index(drop=True)
        st.markdown(f'<div class="section-label">{len(filtered)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(filtered, use_container_width=True, height=420)

    with tab_irreg:
        st.markdown('<div class="small-muted" style="margin-bottom:8px;">Irregular verbs form past tenses through internal vowel changes rather than adding -ed.</div>', unsafe_allow_html=True)
        vc_counts = df_irreg['Vowel_Change'].value_counts()
        # pattern cards grid (non-clickable yet; selection via dropdown)
        cards_html = '<div class="pattern-grid">'
        for pattern, count in vc_counts.items():
            ex = PATTERN_EXAMPLES.get(pattern, "")
            short_ex = ex.split("·")[0].strip() if ex else ""
            cards_html += f"""
            <div class="pattern-card">
              <div style="font-family:DM Mono,monospace;color:{ACCENT3};">{pattern}</div>
              <div style="font-weight:700;color:#C8CDD8;font-size:1.1rem;margin-top:6px;">{count}</div>
              <div class="small-muted" style="margin-top:6px">{short_ex}</div>
            </div>
            """
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:10px;">Filter by Pattern</div>', unsafe_allow_html=True)
        pattern_options = ["All patterns"] + vc_counts.index.tolist()
        selected = st.selectbox("", pattern_options, label_visibility="collapsed", key="explorer_irreg")
        if selected == "All patterns":
            df_show = df_irreg[['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Vowel_Change']].copy()
        else:
            df_show = df_irreg[df_irreg['Vowel_Change'] == selected][['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Vowel_Change']].copy()
            if selected in PATTERN_EXAMPLES:
                st.markdown(f"<div class='small-muted' style='margin-bottom:8px'>{PATTERN_EXAMPLES[selected]}</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{len(df_show)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=420)

# -------------------------
# Page: Charts & Analysis
# -------------------------
elif page == "/ Charts & Analysis":
    st.markdown('<div class="page-title">Charts & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Patterns in English verbs (phonetic columns included)</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "-ed Endings", "Irregular Patterns", "Verb Length"])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        style_ax = lambda a, f: (f.patch.set_facecolor(CHART_BG), a.set_facecolor(CHART_BG), a.tick_params(colors=CHART_TEXT, labelsize=9), a.xaxis.label.set_color(CHART_TEXT), a.yaxis.label.set_color(CHART_TEXT), a.title.set_color('#C8CDD8'), [sp.set_color('#1A2035') for sp in a.spines.values()])
        style_ax(ax, fig)
        bars = ax.bar(['Regular', 'Irregular'], [len(df_reg), len(df_irreg)], color=[ACCENT1, ACCENT2], width=0.42, edgecolor=CHART_BG, linewidth=2)
        for bar, val in zip(bars, [len(df_reg), len(df_irreg)]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha='center', fontweight='bold', fontsize=12, color='#C8CDD8')
        ax.set_title('Dataset Composition', fontsize=10, pad=12)
        ax.set_ylim(0, max(len(df_reg), len(df_irreg)) * 1.18)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(bottom=False)
        plt.tight_layout()
        st.pyplot(fig)

        # small audio lists under chart
        st.markdown('<div style="display:flex;gap:14px;margin-top:8px;">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div style='font-family:DM Mono,monospace;color:#C8CDD8;font-weight:700;margin-bottom:6px;'>Sample Regular</div>", unsafe_allow_html=True)
            render_play_list(df_reg.head(6).to_dict('records'))
        with c2:
            st.markdown("<div style='font-family:DM Mono,monospace;color:#C8CDD8;font-weight:700;margin-bottom:6px;'>Sample Irregular</div>", unsafe_allow_html=True)
            render_play_list(df_irreg.head(6).to_dict('records'))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        ec = df_reg['Ending'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for a in axes: style_ax(a, fig)
        axes[0].bar(ec.index, ec.values, color=[ACCENT4, ACCENT1, ACCENT3], edgecolor=CHART_BG, linewidth=2, width=0.4)
        for i, (idx, val) in enumerate(ec.items()):
            axes[0].text(i, val + 1, str(val), ha='center', fontweight='bold', fontsize=11, color='#C8CDD8')
        axes[0].set_title('-ed Ending Count', fontsize=10, pad=10)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)
        axes[1].pie(ec.values, labels=[f'{i} ({v})' for i, v in ec.items()], colors=[ACCENT4, ACCENT1, ACCENT3], autopct='%1.0f%%', startangle=90, wedgeprops={'edgecolor':CHART_BG, 'linewidth':3}, textprops={'color':CHART_TEXT, 'fontsize':9, 'fontfamily':'monospace'})
        axes[1].set_title('-ed Share', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div style="margin-top:8px;"><b style="color:#C8CDD8">Top phonetic spellings (regular)</b></div>', unsafe_allow_html=True)
        try:
            top_phon = phonetic_summary(df_reg)['Phonetic_Base'].rename(columns={'phonetic': 'phonetic', 'Phonetic_Base': 'count'}).head(6)
            st.dataframe(top_phon, use_container_width=True, height=150)
        except Exception as e:
            logger.exception("Failed to render phonetic summary table", exc_info=e)

        st.markdown('<div style="margin-top:10px;"><b style="color:#C8CDD8">Play examples from -ed group</b></div>', unsafe_allow_html=True)
        render_play_list(df_reg.sample(min(6, len(df_reg))).to_dict('records'))

    with tab3:
        vc = df_irreg['Vowel_Change'].value_counts().head(13)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style_ax(ax, fig)
        colors = [ACCENT2 if i == 0 else ACCENT1 if i < 4 else ACCENT3 for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1], edgecolor=CHART_BG, linewidth=2, height=0.6)
        for bar, val in zip(ax.patches, vc.values[::-1]):
            ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, str(val), va='center', fontsize=9, fontweight='bold', color='#C8CDD8')
        ax.set_title('Most Common Irregular Patterns', fontsize=10, pad=12)
        ax.set_xlim(0, vc.max() + 5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticklabels(vc.index[::-1], fontfamily='monospace', fontsize=8.5, color=CHART_TEXT)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div style="margin-top:6px;"><b style="color:#C8CDD8">Phonetic examples for irregulars</b></div>', unsafe_allow_html=True)
        st.dataframe(phonetic_summary(df_irreg)['Phonetic_Base'].rename(columns={'phonetic': 'phonetic', 'Phonetic_Base': 'count'}).head(8), use_container_width=True, height=150)

        # audio list for top irregular pattern examples
        examples = []
        for pat in vc.index[:6]:
            match = df_irreg[df_irreg['Vowel_Change'] == pat].head(1)
            if not match.empty:
                examples.append(match.iloc[0][['Base', 'Simple_Past', 'Past_Participle']].to_dict())
        if examples:
            st.markdown('<div style="margin-top:8px;"><b style="color:#C8CDD8">Play sample verbs for top irregular patterns</b></div>', unsafe_allow_html=True)
            render_play_list(examples)

    with tab4:
        df_reg['length'] = df_reg['Base'].str.len()
        df_irreg['length'] = df_irreg['Base'].str.len()
        fig, ax = plt.subplots(figsize=(10, 4))
        style_ax(ax, fig)
        ax.hist(df_reg['length'], bins=range(2, 15), alpha=0.75, color=ACCENT1, label='Regular', edgecolor=CHART_BG, linewidth=1.5)
        ax.hist(df_irreg['length'], bins=range(2, 15), alpha=0.75, color=ACCENT2, label='Irregular', edgecolor=CHART_BG, linewidth=1.5)
        ax.axvline(df_reg['length'].mean(), color=ACCENT1, linestyle='--', linewidth=1.5, label=f"Reg avg: {df_reg['length'].mean():.1f}")
        ax.axvline(df_irreg['length'].mean(), color=ACCENT2, linestyle='--', linewidth=1.5, label=f"Irreg avg: {df_irreg['length'].mean():.1f}")
        ax.set_title('Verb Length Distribution', fontsize=10, pad=12)
        ax.set_xlabel('Characters', fontsize=9)
        ax.legend(fontsize=8, framealpha=0, labelcolor=CHART_TEXT)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div class="small-muted" style="margin-top:8px;">Irregular verbs tend to be shorter — Old English legacy. New coinages are usually regular (google, tweet, zoom).</div>', unsafe_allow_html=True)

# -------------------------
# Page: Verb Reference
# -------------------------
elif page == "/ Verb Reference":
    st.markdown('<div class="page-title">Verb Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Complete searchable table — phonetic columns included</div>', unsafe_allow_html=True)

    tab_r, tab_i = st.tabs(["Regular Verbs", "Irregular Verbs"])

    with tab_r:
        search = st.text_input("", placeholder="Search regular verbs (base / phonetic / past) — press Enter", key="s_reg")
        df_show = df_reg.copy()
        if search:
            q = search.lower().strip()
            mask = df_show[['Base', 'Phonetic_Base', 'Phonetic_Past', 'Simple_Past', 'Past_Participle']].apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)
            df_show = df_show[mask]
        st.markdown(f'<div class="section-label">{len(df_show)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show[['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Last_Sound', 'Ending']].reset_index(drop=True), use_container_width=True, height=520)

    with tab_i:
        cs, cf = st.columns([2, 1])
        with cs:
            search2 = st.text_input("", placeholder="Search irregular verbs (base / pattern / phonetic / past)", key="s_irreg")
        with cf:
            vc_opts = ["All patterns"] + df_irreg['Vowel_Change'].value_counts().index.tolist()
            pat_filter = st.selectbox("", vc_opts, key="ref_irreg_filter", label_visibility="collapsed")
        df_show2 = df_irreg.copy()
        if search2:
            q = search2.lower().strip()
            mask = df_show2[['Base', 'Phonetic_Base', 'Phonetic_Past', 'Simple_Past', 'Past_Participle']].apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)
            df_show2 = df_show2[mask]
        if pat_filter != "All patterns":
            df_show2 = df_show2[df_show2['Vowel_Change'] == pat_filter]
        st.markdown(f'<div class="section-label">{len(df_show2)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show2[['Base', 'Simple_Past', 'Past_Participle', 'IPA_Base', 'IPA_Past', 'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP', 'Vowel_Change']].reset_index(drop=True), use_container_width=True, height=520)

# -------------------------
# End of app
# -------------------------
