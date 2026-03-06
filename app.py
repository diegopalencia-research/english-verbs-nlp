import os
import sys
import json
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import streamlit.components.v1 as components

# ── Local services ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.lemmatizer    import find_verb, suggest_verbs
from services.preprocessing import extract_features, count_syllables, get_phonetic_category
from services.phonetics     import (predict_ending, get_rule_explanation,
                                    get_semantic_class_info, adjective_test)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verb Phonetics — diegopalencia",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #060A10;
    color: #DCE0EA;
}
.block-container { padding: 2.2rem 3rem 5rem; max-width: 1180px; }

[data-testid="stSidebar"] {
    background: #080D18;
    border-right: 1px solid #131D2E;
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #3A4D6A !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 0;
    transition: color 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #5B9EC9 !important; }

.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.04em;
    color: #F0F2F8;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.page-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #2E4060;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 2.8rem;
}
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2E4060;
    margin: 1.8rem 0 0.7rem;
    display: block;
}
.card {
    background: #0B1120;
    border: 1px solid #131D2E;
    border-radius: 3px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.9rem;
}
.card-sm {
    background: #0B1120;
    border: 1px solid #131D2E;
    border-radius: 3px;
    padding: 0.9rem 1.1rem;
}
.ipa {
    font-family: 'DM Mono', monospace;
    font-size: 0.95rem;
    color: #5B9EC9;
    background: rgba(91,158,201,0.08);
    border: 1px solid rgba(91,158,201,0.18);
    border-radius: 2px;
    padding: 0.18rem 0.55rem;
    display: inline-block;
    letter-spacing: 0.03em;
}
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.22rem 0.65rem;
    border-radius: 2px;
    margin-right: 0.4rem;
}
.b-t   { background:rgba(82,180,130,0.08);  border:1px solid rgba(82,180,130,0.2);  color:#7DCBA8; }
.b-d   { background:rgba(91,158,201,0.08);  border:1px solid rgba(91,158,201,0.2);  color:#5B9EC9; }
.b-id  { background:rgba(210,170,80,0.08);  border:1px solid rgba(210,170,80,0.2);  color:#C9A84C; }
.b-irr { background:rgba(190,90,100,0.08);  border:1px solid rgba(190,90,100,0.2);  color:#C97080; }
.b-reg { background:rgba(82,180,130,0.08);  border:1px solid rgba(82,180,130,0.2);  color:#7DCBA8; }
.b-inf { background:rgba(120,100,200,0.08); border:1px solid rgba(120,100,200,0.2); color:#A090D8; }
.b-ok  { background:rgba(82,180,130,0.08);  border:1px solid rgba(82,180,130,0.2);  color:#7DCBA8; }
.b-part{ background:rgba(160,144,216,0.08); border:1px solid rgba(160,144,216,0.2); color:#A090D8; }
.b-emo { background:rgba(190,90,100,0.08);  border:1px solid rgba(190,90,100,0.2);  color:#C97080; }
.b-phy { background:rgba(91,158,201,0.08);  border:1px solid rgba(91,158,201,0.2);  color:#5B9EC9; }
.b-pro { background:rgba(82,180,130,0.08);  border:1px solid rgba(82,180,130,0.2);  color:#7DCBA8; }
.b-amb { background:rgba(210,170,80,0.08);  border:1px solid rgba(210,170,80,0.2);  color:#C9A84C; }

[data-testid="metric-container"] {
    background: #0B1120;
    border: 1px solid #131D2E;
    border-radius: 3px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2E4060 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #DCE0EA !important;
}
[data-testid="stTextInput"] input {
    background: #0B1120 !important;
    border: 1px solid #131D2E !important;
    border-radius: 2px !important;
    color: #DCE0EA !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
    letter-spacing: 0.04em;
    transition: border-color 0.15s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #5B9EC9 !important;
    box-shadow: 0 0 0 1px rgba(91,158,201,0.12) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 1px solid #131D2E;
    background: transparent;
    gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2E4060 !important;
    padding: 0.55rem 1.3rem !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #5B9EC9 !important;
    border-bottom: 1px solid #5B9EC9 !important;
}
[data-testid="stDataFrame"] { border: 1px solid #131D2E !important; border-radius: 3px; }
[data-testid="stDataFrame"] * { font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: #0B1120 !important;
    border: 1px solid #131D2E !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #DCE0EA !important;
}
.stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin-bottom:1.5rem; }
.stat-grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin-bottom:1.5rem; }
.stat-block { background:#0B1120; border:1px solid #131D2E; border-radius:3px; padding:1rem 1.2rem; }
.stat-num  { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; }
.stat-lbl  { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase; color:#2E4060; margin-top:0.1rem; }
.stat-sub  { font-family:'DM Mono',monospace; font-size:0.65rem; color:#1E3050; margin-top:0.4rem; }
.p-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(165px,1fr)); gap:0.7rem; margin:0.8rem 0 1.4rem; }
.p-card { background:#0B1120; border:1px solid #131D2E; border-radius:3px; padding:0.85rem 1rem; }
.p-sym  { font-family:'DM Mono',monospace; font-size:0.75rem; color:#C9A84C; margin-bottom:0.25rem; }
.p-cnt  { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#DCE0EA; }
.p-lbl  { font-family:'DM Mono',monospace; font-size:0.6rem; color:#2E4060; text-transform:uppercase; letter-spacing:0.08em; }
.p-ex   { font-family:'DM Mono',monospace; font-size:0.62rem; color:#1A2E46; margin-top:0.35rem; }
.metric-row { display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin:1rem 0; }
.m-block { background:#0B1120; border:1px solid #131D2E; border-radius:3px; padding:1.1rem; text-align:center; }
.m-val  { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; }
.m-lbl  { font-family:'DM Mono',monospace; font-size:0.6rem; color:#2E4060; text-transform:uppercase; letter-spacing:0.12em; margin-top:0.2rem; }
.found-in {
    font-family:'DM Mono',monospace;
    font-size:0.7rem;
    color:#2E4060;
    letter-spacing:0.1em;
    text-transform:uppercase;
    margin-bottom:0.3rem;
}
.hr { border:none; border-top:1px solid #0F1A28; margin:1.5rem 0; }
.cv-bar-wrap { margin: 0.4rem 0; }
.cv-bar-lbl  { font-family:'DM Mono',monospace; font-size:0.65rem; color:#2E4060; margin-bottom:0.15rem; }
.cv-bar-bg   { background:#0B1120; border:1px solid #131D2E; border-radius:2px; height:6px; width:100%; }
.cv-bar-fill { height:6px; border-radius:2px; background:#5B9EC9; }
</style>
""", unsafe_allow_html=True)


# ── Audio helper ───────────────────────────────────────────────────────────────
def speak_button(word: str, label: str = "", key: str = ""):
    safe = str(word).replace("'", "\\'").replace('"', '')
    html = f"""
    <button onclick="window.speechSynthesis.cancel();
      window.speechSynthesis.speak(Object.assign(
        new SpeechSynthesisUtterance('{safe}'),
        {{lang:'en-US', rate:0.82, pitch:1.0}}
      ))"
      style="background:#0B1120;border:1px solid #131D2E;border-radius:2px;
             color:#5B9EC9;font-family:'DM Mono',monospace;font-size:0.68rem;
             letter-spacing:0.08em;padding:0.28rem 0.85rem;cursor:pointer;
             transition:all 0.15s;white-space:nowrap;"
      onmouseover="this.style.borderColor='#5B9EC9';this.style.color='#8EC4E0'"
      onmouseout="this.style.borderColor='#131D2E';this.style.color='#5B9EC9'">
      &#9654;&nbsp;{label or word}
    </button>"""
    components.html(html, height=40)


# ── Morphological pattern classifier for irregular verbs ──────────────────────
MORPH_PATTERNS = {
    'Invariant':       ('b-t',   '#7DCBA8', 'base = past = PP',           'cut · put · hit · let · set · hurt · cost · burst'),
    'Collapsed':       ('b-d',   '#5B9EC9', 'past = PP ≠ base',           'bring · buy · think · teach · catch · sell · tell · make'),
    'Tripartite':      ('b-irr', '#C97080', 'all three forms different',  'sing · drink · begin · swim · drive · write · choose'),
    '-en Participle':  ('b-inf', '#A090D8', 'PP ends in -en / -n',        'break · fall · give · take · see · know · grow · wear'),
    'Dental Suffix':   ('b-id',  '#C9A84C', 'past/PP ends in -ght / dental','buy/bought · fight · seek · think · catch · teach'),
}

def classify_morph_pattern(row):
    base = str(row.get('Base', '')).lower().strip()
    past = str(row.get('Simple_Past', '')).lower().strip()
    pp   = str(row.get('Past_Participle', '')).lower().strip()

    # 1. Invariant — all three identical
    if base == past == pp:
        return 'Invariant'

    # 2. Dental suffix — -ght, -pt, -lt, -ld, -nt endings on past or PP
    dental_endings = ('ght', 'pt', 'lt', 'ld', 'nt')
    if past.endswith(dental_endings) or pp.endswith(dental_endings):
        return 'Dental Suffix'

    # 3. -en Participle — PP ends in -en or bare -n while past does not
    if (pp.endswith('en') or (pp.endswith('n') and not past.endswith('n'))) and past != pp:
        return '-en Participle'

    # 4. Collapsed — past == PP but base is different
    if past == pp and base != past:
        return 'Collapsed'

    # 5. Tripartite — all three genuinely different
    return 'Tripartite'


# ── Data & model ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    FILE = 'data/english_verbs.xlsx'
    df_reg   = pd.read_excel(FILE, sheet_name='Regular Verbs',          header=2)
    df_irreg = pd.read_excel(FILE, sheet_name='Irregular Verbs',        header=2)
    df_part  = pd.read_excel(FILE, sheet_name='Participial Adjectives', header=2)

    reg_cols = ['Base','Simple_Past','Past_Participle',
                'IPA_Base','IPA_Past','IPA_PP',
                'Phonetic_Base','Phonetic_Past','Phonetic_PP',
                'Last_Sound','Ending']
    irreg_cols = ['Base','Simple_Past','Past_Participle',
                  'IPA_Base','IPA_Past','IPA_PP',
                  'Phonetic_Base','Phonetic_Past','Phonetic_PP',
                  'Vowel_Change']
    part_cols = ['Base_Verb','Participial_Form','IPA_Base','IPA_Adj',
                 'Phonetic_Adj','Semantic_Class','Example_Phrase','Notes']

    df_reg.columns   = reg_cols
    df_irreg.columns = irreg_cols
    df_part.columns  = part_cols

    df_reg   = df_reg.dropna(subset=['Base']).reset_index(drop=True)
    df_irreg = df_irreg.dropna(subset=['Base']).reset_index(drop=True)
    df_part  = df_part.dropna(subset=['Base_Verb']).reset_index(drop=True)

    df_reg['Type']   = 'Regular'
    df_irreg['Type'] = 'Irregular'

    # ── PATCH: classify each irregular verb by morphological pattern ──────
    df_irreg['Morph_Pattern'] = df_irreg.apply(classify_morph_pattern, axis=1)

    return df_reg, df_irreg, df_part


@st.cache_resource
def train_model_cached(df_reg, df_irreg):
    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)
    X = extract_features(df_all)
    y = (df_all['Type'] == 'Irregular').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_leaf=2, random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cv     = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'recall':    round(recall_score(y_test, y_pred, average='weighted'), 4),
        'f1':        round(f1_score(y_test, y_pred, average='weighted'), 4),
        'cv_mean':   round(cv.mean(), 4),
        'cv_std':    round(cv.std(), 4),
        'cv_scores': [round(float(s), 4) for s in cv],
        'cm':        confusion_matrix(y_test, y_pred).tolist(),
        'report':    classification_report(y_test, y_pred,
                         target_names=['Regular','Irregular'],
                         output_dict=True),
        'importances': dict(zip(
            X.columns.tolist(),
            [round(float(v), 6) for v in model.feature_importances_]
        )),
        'train_size': len(X_train),
        'test_size':  len(X_test),
    }
    return model, metrics, X.columns.tolist()


df_reg, df_irreg, df_part = load_data()
model, METRICS, FEATURE_NAMES = train_model_cached(df_reg, df_irreg)

# ── Chart style ────────────────────────────────────────────────────────────────
BG   = '#060A10'
C1, C2, C3, C4, C5 = '#5B9EC9', '#C97080', '#C9A84C', '#7DCBA8', '#A090D8'
CT   = '#2E4060'

def sax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor('#0B1120')
    ax.tick_params(colors=CT, labelsize=8)
    ax.xaxis.label.set_color(CT)
    ax.yaxis.label.set_color(CT)
    ax.title.set_color('#8096B8')
    for sp in ax.spines.values():
        sp.set_color('#131D2E')

PATTERN_EXAMPLES = {
    "iː → ɛ":         "feel/felt · keep/kept · sleep/slept · meet/met",
    "aɪ → oʊ":        "write/wrote · ride/rode · rise/rose · drive/drove",
    "ɪ → æ → ʌ":      "sing/sang/sung · drink/drank/drunk · ring/rang/rung",
    "no vowel change": "cut/cut · put/put · hit/hit · set/set",
    "eɪ → oʊ":        "break/broke · wake/woke · speak/spoke",
    "ɪ → ʌ":          "dig/dug · stick/stuck · win/won · hang/hung",
    "aɪ → ɪ":         "bite/bit · hide/hid · slide/slid · light/lit",
    "ʌ → eɪ → ʌ":    "come/came/come · become/became/become",
    "iː → ɔː":        "see/saw · seek/sought · teach/taught",
    "aɪ → aʊ":        "find/found · bind/bound · wind/wound",
    "aɪ → ɔː":        "fight/fought · buy/bought · catch/caught",
    "oʊ → uː":        "know/knew · grow/grew · throw/threw",
}

SC_BADGE = {
    'Emotional state': 'b-emo',
    'Physical state':  'b-phy',
    'Process result':  'b-pro',
    'Ambiguous':       'b-amb',
}
SC_COLOR = {
    'Emotional state': C2,
    'Physical state':  C1,
    'Process result':  C4,
    'Ambiguous':       C3,
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1.8rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.22em;
                  text-transform:uppercase;color:#1E3050;margin-bottom:0.35rem;">Research</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
                  color:#DCE0EA;line-height:1.25;letter-spacing:-0.02em;">
        English Verb<br>Phonetics
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "/ Lookup",
        "/ Phonetic Explorer",
        "/ Participial Adjectives",
        "/ Charts",
        "/ Model Performance",
        "/ Reference"
    ], label_visibility="collapsed")

    total_verbs = len(df_reg) + len(df_irreg)
    st.markdown(f"""
    <div style="margin-top:2.5rem;font-family:'DM Mono',monospace;
                font-size:0.65rem;line-height:2.3;">
      <div style="color:#1E3050;letter-spacing:0.12em;text-transform:uppercase;
                  font-size:0.58rem;margin-bottom:0.4rem;">Dataset</div>
      <span style="color:{C4};">{len(df_reg)}</span>&nbsp;&nbsp;regular<br>
      <span style="color:{C2};">{len(df_irreg)}</span>&nbsp;&nbsp;irregular<br>
      <span style="color:{C5};">{len(df_part)}</span>&nbsp;&nbsp;participial adj<br>
      <span style="color:{C1};">{total_verbs}</span>&nbsp;&nbsp;total verbs<br>
      <span style="color:#7DCBA8;">{METRICS['accuracy']:.1%}</span>&nbsp;&nbsp;model accuracy
    </div>
    <div style="margin-top:2.5rem;font-family:'DM Mono',monospace;
                font-size:0.58rem;color:#1A2E46;line-height:1.8;">
      diegopalencia-research<br>
      <span style="color:#111D30;">github.com/diegopalencia</span>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LOOKUP
# ═════════════════════════════════════════════════════════════════════════════
if page == "/ Lookup":
    st.markdown('<div class="page-title">Verb Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Any form — base · past · participle · participial adj · audio · IPA · rule</div>', unsafe_allow_html=True)

    verb_input = st.text_input("", placeholder="fought   went   broken   excited   walk   organized...").lower().strip()

    if not verb_input:
        st.markdown("""
        <div style="margin-top:3rem;font-family:'DM Mono',monospace;font-size:0.72rem;
                    color:#1A2E46;line-height:2.6;letter-spacing:0.04em;">
          Try any form — base, past, participle, or participial adjective:<br>
          fought &nbsp;&middot;&nbsp; went &nbsp;&middot;&nbsp; bought &nbsp;&middot;&nbsp;
          broken &nbsp;&middot;&nbsp; excited &nbsp;&middot;&nbsp; exhausted &nbsp;&middot;&nbsp;
          walked &nbsp;&middot;&nbsp; organized &nbsp;&middot;&nbsp; google
        </div>
        """, unsafe_allow_html=True)
    else:
        row, verb_type, matched_form = find_verb(verb_input, df_reg, df_irreg, df_part)

        if row is not None and verb_type == 'Participial Adjective':
            # ── Participial Adjective result ──────────────────────────────
            sc   = row['Semantic_Class']
            bc   = SC_BADGE.get(sc, 'b-part')
            info = get_semantic_class_info(sc)
            tests = adjective_test(row['Participial_Form'])

            st.markdown(f"""
            <div class="found-in">
              Found as <span style="color:#A090D8;">{matched_form}</span>
              &nbsp;·&nbsp;
              <span class="badge b-part">Participial Adjective</span>
              <span class="badge {bc}">{sc}</span>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Base Verb",        row['Base_Verb'])
            c2.metric("Participial Form", row['Participial_Form'])

            st.markdown('<span class="sec-label">Audio</span>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca: speak_button(str(row['Base_Verb']),        "Base verb",        f"bv_{verb_input}")
            with cb: speak_button(str(row['Participial_Form']), "Participial form",  f"pf_{verb_input}")

            st.markdown('<span class="sec-label">IPA Transcription</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem;">
              <div class="card-sm" style="flex:1;min-width:140px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;
                            text-transform:uppercase;color:#1E3050;margin-bottom:0.35rem;">Base Verb</div>
                <span class="ipa">{row['IPA_Base']}</span>
              </div>
              <div class="card-sm" style="flex:1;min-width:140px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;
                            text-transform:uppercase;color:#1E3050;margin-bottom:0.35rem;">Participial Adjective</div>
                <span class="ipa">{row['IPA_Adj']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            color:#2E4060;margin-top:0.35rem;">{row['Phonetic_Adj']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<span class="sec-label">Usage & Classification</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
              <span class="badge {bc}">{sc}</span>
              <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                          color:#4A6280;margin-top:0.75rem;line-height:1.9;">
                {info.get('description', '')}
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                          color:#5B9EC9;margin-top:0.8rem;">
                Example: <span style="color:#DCE0EA;">{row['Example_Phrase']}</span>
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                          color:#1E3050;margin-top:0.7rem;line-height:1.9;border-top:1px solid #131D2E;padding-top:0.7rem;">
                {row['Notes']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<span class="sec-label">Adjective Tests</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                          color:#1E3050;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
                Is it an adjective? Apply these tests:
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.76rem;color:#4A6280;line-height:2.2;">
                <span style="color:#2E4060;">Very test:&nbsp;</span>
                <span style="color:#DCE0EA;">"very {row['Participial_Form']}"</span>
                &nbsp;→ if natural, it's an adjective<br>
                <span style="color:#2E4060;">Seem test:&nbsp;</span>
                <span style="color:#DCE0EA;">"seem {row['Participial_Form']}"</span>
                &nbsp;→ predicative adjective test<br>
                <span style="color:#2E4060;">Attributive:&nbsp;</span>
                <span style="color:#DCE0EA;">a {row['Participial_Form']} [noun]</span>
                &nbsp;→ adjective before noun<br>
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif row is not None:
            # ── Regular or Irregular result ───────────────────────────────
            badge_cls = 'b-reg' if verb_type == 'Regular' else 'b-irr'
            st.markdown(f"""
            <div class="found-in">
              Found as <span style="color:#5B9EC9;">{matched_form}</span>
              &nbsp;·&nbsp;
              <span class="badge {badge_cls}">{verb_type}</span>
              {f'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#1A2E46;">'
               f'— base form is <b style="color:#DCE0EA;">{row["Base"]}</b></span>'
               if matched_form != 'base form' else ''}
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form",       row['Base'])
            c2.metric("Simple Past",     row['Simple_Past'])
            c3.metric("Past Participle", row['Past_Participle'])

            st.markdown('<span class="sec-label">Audio</span>', unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca: speak_button(str(row['Base']),            "Base form",       f"b_{verb_input}")
            with cb: speak_button(str(row['Simple_Past']),     "Simple past",     f"p_{verb_input}")
            with cc: speak_button(str(row['Past_Participle']), "Past participle", f"pp_{verb_input}")

            st.markdown('<span class="sec-label">IPA Transcription</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem;">
              <div class="card-sm" style="flex:1;min-width:140px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;
                            text-transform:uppercase;color:#1E3050;margin-bottom:0.35rem;">Base</div>
                <span class="ipa">{row['IPA_Base']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            color:#2E4060;margin-top:0.35rem;">{row['Phonetic_Base']}</div>
              </div>
              <div class="card-sm" style="flex:1;min-width:140px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;
                            text-transform:uppercase;color:#1E3050;margin-bottom:0.35rem;">Simple Past</div>
                <span class="ipa">{row['IPA_Past']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            color:#2E4060;margin-top:0.35rem;">{row['Phonetic_Past']}</div>
              </div>
              <div class="card-sm" style="flex:1;min-width:140px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;
                            text-transform:uppercase;color:#1E3050;margin-bottom:0.35rem;">Past Participle</div>
                <span class="ipa">{row['IPA_PP']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            color:#2E4060;margin-top:0.35rem;">{row['Phonetic_PP']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<span class="sec-label">Phonetic Rule</span>', unsafe_allow_html=True)
            if verb_type == 'Regular':
                ending = row['Ending']
                sound  = row['Last_Sound']
                rules = {
                    '/t/':  "Last sound is a <b>voiceless consonant</b> (p, k, f, s, sh, ch) — -ed is pronounced as a sharp <b>/t/</b>. No extra syllable.",
                    '/d/':  "Last sound is <b>voiced</b> (vowel, b, g, v, z, m, n, l, r) — -ed is pronounced as a soft <b>/d/</b>.",
                    '/ɪd/': "Last sound is <b>/t/ or /d/</b> — an extra syllable <b>/ɪd/</b> is inserted to separate identical sounds."
                }
                badge_map = {'/t/': 'b-t', '/d/': 'b-d', '/ɪd/': 'b-id'}
                bc = badge_map.get(ending, 'b-reg')
                st.markdown(f"""
                <div class="card">
                  <span class="badge {bc}">{ending}</span>
                  <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#1E3050;">
                    last sound: {sound}
                  </span>
                  <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                              color:#4A6280;margin-top:0.75rem;line-height:1.85;">
                    {rules.get(ending, '')}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                vc  = row['Vowel_Change']
                mp  = row.get('Morph_Pattern', '')
                mp_info = MORPH_PATTERNS.get(mp, ('b-irr', C2, '', ''))
                mp_bc, mp_color, mp_desc, mp_ex = mp_info

                siblings = df_irreg[
                    (df_irreg['Vowel_Change'] == vc) & (df_irreg['Base'] != row['Base'])
                ]['Base'].tolist()[:7]
                sib_str = " &nbsp;&middot;&nbsp; ".join(siblings) if siblings else "—"
                ex = PATTERN_EXAMPLES.get(vc, "")

                # Morphological pattern siblings
                mp_siblings = df_irreg[
                    (df_irreg['Morph_Pattern'] == mp) & (df_irreg['Base'] != row['Base'])
                ]['Base'].tolist()[:6]
                mp_sib_str = " &nbsp;&middot;&nbsp; ".join(mp_siblings) if mp_siblings else "—"

                st.markdown(f"""
                <div class="card">
                  <div style="margin-bottom:0.6rem;">
                    <span class="badge b-irr">{vc}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#1E3050;">
                      vowel change pattern
                    </span>
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                              color:#4A6280;line-height:1.85;">
                    This verb does <b>not</b> follow the -ed rule.<br>
                    It changes its internal vowel to form the past tense.
                    {f'<br><span style="color:#1E3050;">Examples: {ex}</span>' if ex else ''}
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                              color:#1A2E46;margin-top:0.8rem;">
                    Same vowel pattern: {sib_str}
                  </div>
                  <div style="border-top:1px solid #131D2E;margin-top:0.9rem;padding-top:0.9rem;">
                    <span class="badge {mp_bc}">{mp}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#1E3050;">
                      morphological pattern &nbsp;·&nbsp; {mp_desc}
                    </span>
                    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                color:#1A2E46;margin-top:0.55rem;">
                      Same structure: {mp_sib_str}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Check if this verb also has a participial adjective entry
            part_match = df_part[df_part['Base_Verb'].str.lower() == row['Base'].lower()]
            if not part_match.empty:
                pm = part_match.iloc[0]
                bc2 = SC_BADGE.get(pm['Semantic_Class'], 'b-part')
                st.markdown(f"""
                <div class="card" style="border-color:#1A2E46;">
                  <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                              letter-spacing:0.14em;text-transform:uppercase;
                              color:#1E3050;margin-bottom:0.5rem;">Also a Participial Adjective</div>
                  <span class="badge b-part">adj</span>
                  <span class="badge {bc2}">{pm['Semantic_Class']}</span>
                  <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#5B9EC9;">
                    &nbsp;{pm['Participial_Form']}
                  </span>
                  <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                              color:#1E3050;margin-top:0.5rem;">
                    Example: <span style="color:#2E4060;">{pm['Example_Phrase']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # ── Not in dataset — ML prediction ────────────────────────────
            st.markdown('<span class="sec-label">Not in dataset — ML Prediction</span>',
                        unsafe_allow_html=True)

            row_df   = pd.DataFrame([{'Base': verb_input}])
            features = extract_features(row_df)
            prob     = model.predict_proba(features)[0]
            label    = 'Irregular' if prob[1] > 0.5 else 'Regular'
            conf     = max(prob) * 100

            phon_cat = get_phonetic_category(verb_input)
            sylls    = count_syllables(verb_input)

            speak_button(verb_input, f"Hear '{verb_input}'", f"pred_{verb_input}")

            if label == 'Regular':
                predicted_ending = predict_ending(verb_input)
                rule_text = get_rule_explanation(predicted_ending)
                badge_map = {'/t/': 'b-t', '/d/': 'b-d', '/ɪd/': 'b-id'}
                bc = badge_map.get(predicted_ending, 'b-reg')
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.7rem;">
                    <span class="badge b-reg">Regular</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                 color:#1E3050;">{conf:.1f}% confidence</span>
                  </div>
                  <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                              font-weight:700;color:{C4};">
                    {verb_input} &rarr; {verb_input}ed
                  </div>
                  <div style="margin-top:0.6rem;">
                    <span class="badge {bc}">{predicted_ending}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;
                                 color:#3A5070;">{rule_text}</span>
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.63rem;
                              color:#1A2E46;margin-top:0.8rem;line-height:1.9;">
                    Phonetic category: {phon_cat} &nbsp;&middot;&nbsp;
                    Syllables: {sylls} &nbsp;&middot;&nbsp;
                    Last letter: {verb_input[-1]}
                  </div>
                </div>
                """, unsafe_allow_html=True)
                speak_button(f"{verb_input}ed", f"Hear '{verb_input}ed'",
                             f"pred_past_{verb_input}")
            else:
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.7rem;">
                    <span class="badge b-irr">Likely Irregular</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                 color:#1E3050;">{conf:.1f}% confidence</span>
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                              color:#4A6280;line-height:1.85;">
                    This verb likely has an unpredictable past form.<br>
                    Check a dictionary for the correct conjugation.
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.63rem;
                              color:#1A2E46;margin-top:0.8rem;">
                    Phonetic category: {phon_cat} &nbsp;&middot;&nbsp; Syllables: {sylls}
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHONETIC EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "/ Phonetic Explorer":
    st.markdown('<div class="page-title">Phonetic Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Regular -ed rules · irregular vowel patterns · morphological groups</div>', unsafe_allow_html=True)

    tab_reg, tab_irr = st.tabs(["Regular — -ed Rule", "Irregular — Vowel & Structure Patterns"])

    with tab_reg:
        endings = df_reg['Ending'].value_counts()
        t = endings.get('/t/', 0)
        d = endings.get('/d/', 0)
        i = endings.get('/ɪd/', 0)

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-block">
            <div class="stat-num" style="color:{C4};">{t}</div>
            <div style="margin:0.3rem 0;"><span class="badge b-t">/t/</span></div>
            <div class="stat-lbl">Voiceless consonant</div>
            <div class="stat-sub">p · k · f · s · sh · ch · x</div>
          </div>
          <div class="stat-block">
            <div class="stat-num" style="color:{C1};">{d}</div>
            <div style="margin:0.3rem 0;"><span class="badge b-d">/d/</span></div>
            <div class="stat-lbl">Voiced sound</div>
            <div class="stat-sub">vowels · b · g · v · z · m · n · l · r</div>
          </div>
          <div class="stat-block">
            <div class="stat-num" style="color:{C3};">{i}</div>
            <div style="margin:0.3rem 0;"><span class="badge b-id">/ɪd/</span></div>
            <div class="stat-lbl">T or D ending</div>
            <div class="stat-sub">/t/ or /d/ → needs extra syllable</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="sec-label">Filter</span>', unsafe_allow_html=True)
        ef = st.selectbox("", ["/t/ — voiceless", "/d/ — voiced", "/ɪd/ — extra syllable"],
                          label_visibility="collapsed", key="ef")
        ec = ef.split(" ")[0]
        filtered = df_reg[df_reg['Ending'] == ec][
            ['Base','Simple_Past','IPA_Base','IPA_Past','Last_Sound','Ending']
        ].reset_index(drop=True)
        st.markdown(f'<span class="sec-label">{len(filtered)} verbs with {ec}</span>',
                    unsafe_allow_html=True)
        st.dataframe(filtered, use_container_width=True, height=380)

    with tab_irr:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.76rem;color:#2E4060;
                    line-height:2;margin-bottom:0.5rem;">
          Irregular verbs form past tense through internal vowel change — never by adding -ed.
          Two classification lenses: <b style="color:#DCE0EA;">vowel phonetics</b> (what sound changes)
          and <b style="color:#DCE0EA;">morphological pattern</b> (how the three forms relate structurally).
        </div>
        """, unsafe_allow_html=True)

        sub_vowel, sub_morph = st.tabs(["By Vowel Change", "By Morphological Pattern"])

        # ── Vowel change sub-tab (original content) ────────────────────────
        with sub_vowel:
            vc_counts = df_irreg['Vowel_Change'].value_counts()
            cards = '<div class="p-grid">'
            for pat, cnt in vc_counts.items():
                ex = PATTERN_EXAMPLES.get(pat, "")
                short = ex.split("·")[0].strip() if ex else ""
                cards += f"""
                <div class="p-card">
                  <div class="p-sym">{pat}</div>
                  <div class="p-cnt">{cnt}</div>
                  <div class="p-lbl">verbs</div>
                  <div class="p-ex">{short}</div>
                </div>"""
            cards += '</div>'
            st.markdown(cards, unsafe_allow_html=True)

            st.markdown('<span class="sec-label">Filter by Vowel Pattern</span>', unsafe_allow_html=True)
            opts = ["All patterns"] + vc_counts.index.tolist()
            sel  = st.selectbox("", opts, label_visibility="collapsed", key="irr_sel")

            df_sh = df_irreg if sel == "All patterns" else df_irreg[df_irreg['Vowel_Change'] == sel]
            if sel != "All patterns" and sel in PATTERN_EXAMPLES:
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                            color:#1E3050;margin-bottom:0.6rem;">{PATTERN_EXAMPLES[sel]}</div>
                """, unsafe_allow_html=True)

            st.markdown(f'<span class="sec-label">{len(df_sh)} verbs</span>', unsafe_allow_html=True)
            st.dataframe(
                df_sh[['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Vowel_Change']
                      ].reset_index(drop=True),
                use_container_width=True, height=360
            )

        # ── NEW: Morphological pattern sub-tab ────────────────────────────
        with sub_morph:
            mp_counts = df_irreg['Morph_Pattern'].value_counts()

            # Summary cards
            cards2 = '<div class="p-grid">'
            for pat, cnt in mp_counts.items():
                bc, color, desc, exs = MORPH_PATTERNS.get(pat, ('b-reg', C4, '', ''))
                short_ex = exs.split("·")[0].strip() if exs else ""
                cards2 += f"""
                <div class="p-card">
                  <div style="margin-bottom:0.4rem;">
                    <span class="badge {bc}" style="font-size:0.6rem;">{pat}</span>
                  </div>
                  <div class="p-cnt" style="color:{color};">{cnt}</div>
                  <div class="p-lbl">verbs</div>
                  <div class="p-ex" style="margin-top:0.35rem;color:#2E4060;">{desc}</div>
                  <div class="p-ex">{short_ex}</div>
                </div>"""
            cards2 += '</div>'
            st.markdown(cards2, unsafe_allow_html=True)

            # Expanded explanation for each pattern
            st.markdown('<span class="sec-label">Pattern Definitions</span>', unsafe_allow_html=True)
            for pat, (bc, color, desc, exs) in MORPH_PATTERNS.items():
                count = mp_counts.get(pat, 0)
                if count == 0:
                    continue
                st.markdown(f"""
                <div class="card" style="margin-bottom:0.6rem;border-left:2px solid {color}33;">
                  <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.45rem;">
                    <span class="badge {bc}">{pat}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{color};">
                      {count} verbs
                    </span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#1E3050;">
                      {desc}
                    </span>
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                              color:#2E4060;line-height:2.1;">
                    {exs}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Filter + table
            st.markdown('<span class="sec-label">Filter by Morphological Pattern</span>',
                        unsafe_allow_html=True)
            mp_opts = ["All patterns"] + mp_counts.index.tolist()
            mp_sel  = st.selectbox("", mp_opts, label_visibility="collapsed", key="morph_sel")

            df_mp = df_irreg if mp_sel == "All patterns" \
                             else df_irreg[df_irreg['Morph_Pattern'] == mp_sel]

            st.markdown(f'<span class="sec-label">{len(df_mp)} verbs</span>', unsafe_allow_html=True)
            st.dataframe(
                df_mp[['Base','Simple_Past','Past_Participle',
                       'IPA_Base','IPA_Past','Vowel_Change','Morph_Pattern']
                      ].reset_index(drop=True),
                use_container_width=True, height=380
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PARTICIPIAL ADJECTIVES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "/ Participial Adjectives":
    st.markdown('<div class="page-title">Participial Adjectives</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Past participles used as adjectives · 4 semantic classes · 76 entries</div>', unsafe_allow_html=True)

    sc_counts = df_part['Semantic_Class'].value_counts()

    st.markdown(f"""
    <div class="stat-grid-4">
      <div class="stat-block">
        <div class="stat-num" style="color:{C2};">{sc_counts.get('Emotional state', 0)}</div>
        <div style="margin:0.3rem 0;"><span class="badge b-emo">Emotional</span></div>
        <div class="stat-lbl">Experiencer-oriented</div>
        <div class="stat-sub">bored · excited · tired</div>
      </div>
      <div class="stat-block">
        <div class="stat-num" style="color:{C1};">{sc_counts.get('Physical state', 0)}</div>
        <div style="margin:0.3rem 0;"><span class="badge b-phy">Physical</span></div>
        <div class="stat-lbl">Visible condition</div>
        <div class="stat-sub">broken · frozen · torn</div>
      </div>
      <div class="stat-block">
        <div class="stat-num" style="color:{C4};">{sc_counts.get('Process result', 0)}</div>
        <div style="margin:0.3rem 0;"><span class="badge b-pro">Process</span></div>
        <div class="stat-lbl">Completed action</div>
        <div class="stat-sub">cooked · printed · trained</div>
      </div>
      <div class="stat-block">
        <div class="stat-num" style="color:{C3};">{sc_counts.get('Ambiguous', 0)}</div>
        <div style="margin:0.3rem 0;"><span class="badge b-amb">Ambiguous</span></div>
        <div class="stat-lbl">Verb or adjective</div>
        <div class="stat-sub">experienced · limited · mixed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-bottom:1.5rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                  text-transform:uppercase;color:#1E3050;margin-bottom:0.6rem;">What is a Participial Adjective?</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.76rem;color:#4A6280;line-height:2;">
        A past participle used as an <b style="color:#A090D8;">adjective</b> rather than a verb form.
        The same word can be both:
        <span style="color:#C97080;">"She <u>exhausted</u> her savings"</span> (verb, past tense)
        vs
        <span style="color:#7DCBA8;">"an <u>exhausted</u> runner"</span> (adjective, attributive).
        <br>
        Key test: if <b>very</b> can precede it naturally, it's functioning as an adjective.
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "All", "Emotional", "Physical", "Process Result", "Ambiguous"
    ])

    def part_table(df_filtered):
        st.markdown(f'<span class="sec-label">{len(df_filtered)} entries</span>',
                    unsafe_allow_html=True)
        st.dataframe(
            df_filtered[['Base_Verb','Participial_Form','IPA_Adj','Phonetic_Adj',
                         'Semantic_Class','Example_Phrase','Notes']
                        ].reset_index(drop=True),
            use_container_width=True, height=460
        )

    with tab1:
        s = st.text_input("", placeholder="Search participial adjectives...", key="s_part")
        df_show = df_part.copy()
        if s:
            mask = (df_show['Base_Verb'].str.contains(s.lower(), na=False) |
                    df_show['Participial_Form'].str.contains(s.lower(), na=False))
            df_show = df_show[mask]
        part_table(df_show)

    for tab_widget, sc_name in zip([tab2, tab3, tab4, tab5],
                                   ['Emotional state','Physical state',
                                    'Process result','Ambiguous']):
        with tab_widget:
            info = get_semantic_class_info(sc_name)
            bc   = SC_BADGE.get(sc_name, 'b-part')
            st.markdown(f"""
            <div class="card" style="margin-bottom:1rem;">
              <span class="badge {bc}">{sc_name}</span>
              <div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                          color:#4A6280;margin-top:0.6rem;line-height:1.9;">
                {info.get('description', '')}
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                          color:#1E3050;margin-top:0.5rem;">
                Common examples: {info.get('examples', '')}
              </div>
            </div>
            """, unsafe_allow_html=True)
            part_table(df_part[df_part['Semantic_Class'] == sc_name])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHARTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "/ Charts":
    st.markdown('<div class="page-title">Charts & Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Phonetic patterns across {len(df_reg)+len(df_irreg)} verbs + {len(df_part)} participial adjectives</div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Overview", "-ed Endings", "Irregular Patterns",
        "Morphological Groups", "Verb Length", "Participial Adjectives"
    ])

    with t1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax in axes: sax(ax, fig)

        cats   = ['Regular', 'Irregular', 'Participial\nAdj']
        vals   = [len(df_reg), len(df_irreg), len(df_part)]
        colors = [C1, C2, C5]

        axes[0].bar(cats, vals, color=colors, width=0.42, edgecolor=BG, linewidth=2.5)
        for i, (cat, val) in enumerate(zip(cats, vals)):
            axes[0].text(i, val + 1.5, str(val), ha='center',
                         fontweight='bold', fontsize=12, color='#8096B8')
        axes[0].set_title('Dataset Composition', fontsize=10, pad=10)
        axes[0].set_ylim(0, max(vals) * 1.2)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)

        axes[1].pie(vals,
                    labels=[f'{c.replace(chr(10), " ")}  ({v})' for c, v in zip(cats, vals)],
                    colors=colors, autopct='%1.0f%%', startangle=90,
                    wedgeprops={'edgecolor': BG, 'linewidth': 3},
                    textprops={'color': CT, 'fontsize': 8.5, 'fontfamily': 'monospace'})
        axes[1].set_title('Dataset Share', fontsize=10, pad=10)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#1E3050;margin-top:0.4rem;">The 10 most frequent English verbs are all irregular. Of {len(df_part)} participial adjectives, {len(df_part[df_part["Semantic_Class"]=="Emotional state"])} describe emotional states — the largest class.</div>', unsafe_allow_html=True)

    with t2:
        ec = df_reg['Ending'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax in axes: sax(ax, fig)
        axes[0].bar(ec.index, ec.values, color=[C4, C1, C3],
                    edgecolor=BG, linewidth=2.5, width=0.38)
        for i, (idx, val) in enumerate(ec.items()):
            axes[0].text(i, val + 1, str(val), ha='center',
                         fontweight='bold', fontsize=11, color='#8096B8')
        axes[0].set_title('-ed Ending Count', fontsize=10, pad=10)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)
        axes[1].pie(ec.values,
                    labels=[f'{i}  ({v})' for i, v in ec.items()],
                    colors=[C4, C1, C3], autopct='%1.0f%%', startangle=90,
                    wedgeprops={'edgecolor': BG, 'linewidth': 3},
                    textprops={'color': CT, 'fontsize': 8.5, 'fontfamily': 'monospace'})
        axes[1].set_title('-ed Share', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    with t3:
        vc = df_irreg['Vowel_Change'].value_counts().head(14)
        fig, ax = plt.subplots(figsize=(10, 5.8))
        sax(ax, fig)
        cols = [C2 if i == 0 else C1 if i < 4 else C3 for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1],
                color=cols[::-1], edgecolor=BG, linewidth=2, height=0.58)
        for bar, val in zip(ax.patches, vc.values[::-1]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', fontsize=9, fontweight='bold', color='#8096B8')
        ax.set_title('Irregular Vowel Change Patterns', fontsize=10, pad=10)
        ax.set_xlim(0, vc.max() + 5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticklabels(vc.index[::-1], fontfamily='monospace', fontsize=8.5, color=CT)
        plt.tight_layout()
        st.pyplot(fig)

    # ── NEW: Morphological groups chart ───────────────────────────────────
    with t4:
        mp_counts = df_irreg['Morph_Pattern'].value_counts()
        mp_colors = [MORPH_PATTERNS.get(k, ('b-reg', C1, '', ''))[1] for k in mp_counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax in axes: sax(ax, fig)

        axes[0].bar(range(len(mp_counts)), mp_counts.values,
                    color=mp_colors, edgecolor=BG, linewidth=2, width=0.45)
        for i, val in enumerate(mp_counts.values):
            axes[0].text(i, val + 0.3, str(val), ha='center',
                         fontweight='bold', fontsize=11, color='#8096B8')
        axes[0].set_xticks(range(len(mp_counts)))
        axes[0].set_xticklabels(
            [k.replace(' ', '\n') for k in mp_counts.index],
            fontfamily='monospace', fontsize=7.5, color=CT
        )
        axes[0].set_title('Irregular Verbs by Morphological Pattern', fontsize=10, pad=10)
        axes[0].set_ylim(0, mp_counts.max() * 1.25)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)

        wedges, texts, autotexts = axes[1].pie(
            mp_counts.values,
            labels=[f'{k}  ({v})' for k, v in mp_counts.items()],
            colors=mp_colors, autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor': BG, 'linewidth': 3},
            textprops={'color': CT, 'fontsize': 8, 'fontfamily': 'monospace'},
            pctdistance=0.78
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc='#0B1120')
        axes[1].add_patch(centre_circle)
        axes[1].text(0, 0, f'{len(df_irreg)}\nirregular', ha='center', va='center',
                     fontsize=10, fontweight='bold', color='#DCE0EA', fontfamily='monospace')
        axes[1].set_title('Morphological Pattern Distribution', fontsize=10, pad=10)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<span class="sec-label">Cross-tabulation: Pattern × Vowel Change</span>',
                    unsafe_allow_html=True)
        cross = pd.crosstab(df_irreg['Morph_Pattern'], df_irreg['Vowel_Change'])
        st.dataframe(cross, use_container_width=True, height=260)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#1E3050;margin-top:0.4rem;line-height:2;">
          Each cell = number of verbs sharing both a morphological pattern and a vowel change pattern.<br>
          Tripartite verbs span the most vowel patterns — they are the most phonetically diverse group.
        </div>
        """, unsafe_allow_html=True)

    with t5:
        df_reg['len']   = df_reg['Base'].str.len()
        df_irreg['len'] = df_irreg['Base'].str.len()
        fig, ax = plt.subplots(figsize=(10, 4))
        sax(ax, fig)
        ax.hist(df_reg['len'],   bins=range(2, 16), alpha=0.75, color=C1,
                label='Regular',   edgecolor=BG, linewidth=1.5)
        ax.hist(df_irreg['len'], bins=range(2, 16), alpha=0.75, color=C2,
                label='Irregular', edgecolor=BG, linewidth=1.5)
        ax.axvline(df_reg['len'].mean(),   color=C1, linestyle='--', linewidth=1.5,
                   label=f"Reg avg {df_reg['len'].mean():.1f}")
        ax.axvline(df_irreg['len'].mean(), color=C2, linestyle='--', linewidth=1.5,
                   label=f"Irreg avg {df_irreg['len'].mean():.1f}")
        ax.set_title('Verb Length Distribution', fontsize=10, pad=10)
        ax.set_xlabel('Characters in base form', fontsize=8)
        ax.legend(fontsize=8, framealpha=0, labelcolor=CT)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#1E3050;margin-top:0.4rem;">Irregular verbs are shorter on average — they derive from Old English. Modern coined verbs (google, zoom, tweet) are almost always regular.</div>', unsafe_allow_html=True)

    with t6:
        sc_counts = df_part['Semantic_Class'].value_counts()
        sc_colors = [SC_COLOR.get(k, C5) for k in sc_counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax in axes: sax(ax, fig)

        axes[0].bar(range(len(sc_counts)), sc_counts.values,
                    color=sc_colors, edgecolor=BG, linewidth=2, width=0.45)
        for i, val in enumerate(sc_counts.values):
            axes[0].text(i, val + 0.3, str(val), ha='center',
                         fontweight='bold', fontsize=11, color='#8096B8')
        axes[0].set_xticks(range(len(sc_counts)))
        axes[0].set_xticklabels(
            [k.replace(' ', '\n') for k in sc_counts.index],
            fontfamily='monospace', fontsize=7.5, color=CT
        )
        axes[0].set_title('Participial Adj by Semantic Class', fontsize=10, pad=10)
        axes[0].set_ylim(0, sc_counts.max() * 1.25)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)

        wedges, texts, autotexts = axes[1].pie(
            sc_counts.values,
            labels=[f'{k}  ({v})' for k, v in sc_counts.items()],
            colors=sc_colors, autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor': BG, 'linewidth': 3},
            textprops={'color': CT, 'fontsize': 8, 'fontfamily': 'monospace'},
            pctdistance=0.78
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc='#0B1120')
        axes[1].add_patch(centre_circle)
        axes[1].text(0, 0, f'{len(df_part)}\nadj', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='#DCE0EA', fontfamily='monospace')
        axes[1].set_title('Participial Adj Distribution', fontsize=10, pad=10)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<span class="sec-label">Verb–Adjective Overlap</span>', unsafe_allow_html=True)

        part_bases = set(df_part['Base_Verb'].str.lower())
        reg_bases  = set(df_reg['Base'].str.lower())
        irr_bases  = set(df_irreg['Base'].str.lower())

        overlap_reg  = part_bases & reg_bases
        overlap_irr  = part_bases & irr_bases
        only_part    = part_bases - reg_bases - irr_bases

        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        sax(ax2, fig2)
        bars = ax2.barh(
            ['From Regular verbs', 'From Irregular verbs', 'Unique to Part. Adj'],
            [len(overlap_reg), len(overlap_irr), len(only_part)],
            color=[C1, C2, C5], edgecolor=BG, linewidth=2, height=0.45
        )
        for bar, val in zip(bars, [len(overlap_reg), len(overlap_irr), len(only_part)]):
            ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     str(val), va='center', fontsize=10, fontweight='bold', color='#8096B8')
        ax2.set_title('Participial Adj: Verb Source Overlap', fontsize=10, pad=10)
        ax2.set_xlim(0, max(len(overlap_reg), len(overlap_irr), len(only_part)) + 6)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.tick_params(left=False)
        ax2.set_yticklabels(
            ['From Regular verbs', 'From Irregular verbs', 'Unique to Part. Adj'],
            fontfamily='monospace', fontsize=8.5, color=CT
        )
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#1E3050;margin-top:0.4rem;line-height:2;">
          <span style="color:{C1};">{len(overlap_reg)}</span> participial adjectives derive from regular verbs
          &nbsp;·&nbsp;
          <span style="color:{C2};">{len(overlap_irr)}</span> from irregular verbs (broken, frozen, worn...)
          &nbsp;·&nbsp;
          <span style="color:{C5};">{len(only_part)}</span> appear only in the participial sheet
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "/ Model Performance":
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">RandomForest · 200 estimators · 5-fold cross-validation</div>', unsafe_allow_html=True)

    st.markdown('<span class="sec-label">Evaluation Metrics — Test Set</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="m-block">
        <div class="m-val" style="color:{C4};">{METRICS['accuracy']:.1%}</div>
        <div class="m-lbl">Accuracy</div>
      </div>
      <div class="m-block">
        <div class="m-val" style="color:{C1};">{METRICS['precision']:.1%}</div>
        <div class="m-lbl">Precision</div>
      </div>
      <div class="m-block">
        <div class="m-val" style="color:{C3};">{METRICS['recall']:.1%}</div>
        <div class="m-lbl">Recall</div>
      </div>
      <div class="m-block">
        <div class="m-val" style="color:{C2};">{METRICS['f1']:.1%}</div>
        <div class="m-lbl">F1 Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sec-label">5-Fold Cross-Validation</span>', unsafe_allow_html=True)
    cv_cols = st.columns(5)
    for i, (col, score) in enumerate(zip(cv_cols, METRICS['cv_scores'])):
        col.markdown(f"""
        <div class="card-sm" style="text-align:center;">
          <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                      color:#1E3050;margin-bottom:0.3rem;">Fold {i+1}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                      font-weight:700;color:{C1};">{score:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    cv_mean = METRICS['cv_mean']
    cv_std  = METRICS['cv_std']
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#2E4060;margin-top:0.8rem;">
      Mean: <span style="color:{C4};font-weight:600;">{cv_mean:.1%}</span>
      &nbsp;&nbsp;±&nbsp;&nbsp;
      <span style="color:#1E3050;">{cv_std:.2%}</span> std
      &nbsp;&nbsp;·&nbsp;&nbsp; Stable model: low variance across folds.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sec-label">Confusion Matrix</span>', unsafe_allow_html=True)
    cm = METRICS['cm']
    col_cm, col_info = st.columns([1, 1])

    with col_cm:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sax(ax, fig)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    cmap='Blues', linewidths=2, linecolor=BG,
                    xticklabels=['Regular', 'Irregular'],
                    yticklabels=['Regular', 'Irregular'],
                    cbar=False,
                    annot_kws={'fontsize': 14, 'fontweight': 'bold',
                               'color': '#DCE0EA', 'fontfamily': 'monospace'})
        ax.set_xlabel('Predicted', fontsize=9, color=CT)
        ax.set_ylabel('Actual',    fontsize=9, color=CT)
        ax.tick_params(colors=CT)
        ax.set_title('Test Set Confusion Matrix', fontsize=10, color='#8096B8', pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    with col_info:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2.2;margin-top:0.5rem;">
          <div style="color:#1E3050;text-transform:uppercase;letter-spacing:0.1em;
                      font-size:0.6rem;margin-bottom:0.5rem;">Reading the matrix</div>
          <span style="color:{C4};">&#10003;</span>&nbsp;
          <span style="color:#2E4060;">True Regular:&nbsp;</span>
          <span style="color:#DCE0EA;">{tn}</span><br>
          <span style="color:{C4};">&#10003;</span>&nbsp;
          <span style="color:#2E4060;">True Irregular:&nbsp;</span>
          <span style="color:#DCE0EA;">{tp}</span><br>
          <span style="color:{C2};">&#10007;</span>&nbsp;
          <span style="color:#2E4060;">False Positives:&nbsp;</span>
          <span style="color:#DCE0EA;">{fp}</span>
          <span style="font-size:0.62rem;color:#1A2E46;"> (said Irreg, was Reg)</span><br>
          <span style="color:{C2};">&#10007;</span>&nbsp;
          <span style="color:#2E4060;">False Negatives:&nbsp;</span>
          <span style="color:#DCE0EA;">{fn}</span>
          <span style="font-size:0.62rem;color:#1A2E46;"> (said Reg, was Irreg)</span><br><br>
          <div style="color:#1E3050;text-transform:uppercase;letter-spacing:0.1em;
                      font-size:0.6rem;margin-bottom:0.5rem;">Test set split</div>
          <span style="color:#2E4060;">Train:&nbsp;</span>
          <span style="color:#DCE0EA;">{METRICS['train_size']}</span>&nbsp;verbs<br>
          <span style="color:#2E4060;">Test:&nbsp;&nbsp;</span>
          <span style="color:#DCE0EA;">{METRICS['test_size']}</span>&nbsp;verbs
          <span style="color:#1A2E46;"> (80/20 split)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<span class="sec-label">Feature Importance — Top 12</span>', unsafe_allow_html=True)
    imp_sorted = sorted(METRICS['importances'].items(), key=lambda x: x[1], reverse=True)[:12]
    imp_df = pd.DataFrame(imp_sorted, columns=['Feature', 'Importance'])

    fig, ax = plt.subplots(figsize=(10, 4))
    sax(ax, fig)
    palette = [C1 if i < 3 else C3 if i < 6 else CT for i in range(len(imp_df))]
    ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1],
            color=palette[::-1], edgecolor=BG, linewidth=1.5, height=0.55)
    for bar, val in zip(ax.patches, imp_df['Importance'][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8, color='#4A6280',
                fontfamily='monospace')
    ax.set_title('Feature Importance', fontsize=10, pad=10)
    ax.set_xlim(0, imp_df['Importance'].max() * 1.25)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(left=False)
    ax.set_yticklabels(imp_df['Feature'][::-1],
                       fontfamily='monospace', fontsize=8, color=CT)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<span class="sec-label">Model Architecture</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="font-family:'DM Mono',monospace;font-size:0.72rem;
                             color:#2E4060;line-height:2.3;">
      <span style="color:{C1};">Algorithm:</span> Random Forest Classifier<br>
      <span style="color:{C1};">Estimators:</span> 200 decision trees<br>
      <span style="color:{C1};">Max depth:</span> 8 levels<br>
      <span style="color:{C1};">Features:</span> {len(FEATURE_NAMES)} — length, vowel count, syllables,
      phonetic category (voiced/voiceless/stop), suffix patterns, n-grams, participial heuristic<br>
      <span style="color:{C1};">Class weight:</span> balanced (handles regular/irregular imbalance)<br>
      <span style="color:{C1};">Validation:</span> 5-fold cross-validation
    </div>
    """, unsafe_allow_html=True)

    # ── Misclassification Analysis ─────────────────────────────────────────
    st.markdown('<span class="sec-label">Misclassification Analysis</span>', unsafe_allow_html=True)

    df_all_for_analysis = pd.concat([df_reg, df_irreg], ignore_index=True)
    X_analysis = extract_features(df_all_for_analysis)
    y_analysis = (df_all_for_analysis['Type'] == 'Irregular').astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_analysis, y_analysis, test_size=0.20, random_state=42, stratify=y_analysis
    )

    y_pred_analysis = model.predict(X_te)
    probs_analysis  = model.predict_proba(X_te)

    df_test_mc = df_all_for_analysis.loc[X_te.index].copy()
    df_test_mc['Predicted']    = ['Irregular' if p == 1 else 'Regular' for p in y_pred_analysis]
    df_test_mc['Actual']       = ['Irregular' if a == 1 else 'Regular' for a in y_te]
    df_test_mc['Correct']      = y_pred_analysis == y_te.values
    df_test_mc['Prob_Irreg_%'] = [round(p[1]*100, 1) for p in probs_analysis]
    df_test_mc['Error_Type']   = df_test_mc.apply(
        lambda r: 'FP — said Irregular' if (r['Predicted']=='Irregular' and r['Actual']=='Regular')
                  else ('FN — said Regular' if (r['Predicted']=='Regular' and r['Actual']=='Irregular')
                        else 'Correct'), axis=1
    )

    misclassified = df_test_mc[~df_test_mc['Correct']].reset_index(drop=True)
    fp_mask = misclassified['Error_Type'].str.startswith('FP')
    fn_mask = misclassified['Error_Type'].str.startswith('FN')
    n_errors, n_fp, n_fn = len(misclassified), fp_mask.sum(), fn_mask.sum()

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-bottom:1.2rem;">
      <div class="card-sm" style="text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{C2};">{n_errors}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1E3050;text-transform:uppercase;letter-spacing:0.12em;">Total Errors</div>
      </div>
      <div class="card-sm" style="text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{C3};">{n_fp}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1E3050;text-transform:uppercase;letter-spacing:0.12em;">False Positives<br><span style="font-size:0.55rem;">Said Irregular, was Regular</span></div>
      </div>
      <div class="card-sm" style="text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{C1};">{n_fn}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1E3050;text-transform:uppercase;letter-spacing:0.12em;">False Negatives<br><span style="font-size:0.55rem;">Said Regular, was Irregular</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_mc_all, tab_mc_fp, tab_mc_fn = st.tabs([
        f"All errors ({n_errors})",
        f"False Positives ({n_fp})",
        f"False Negatives ({n_fn})",
    ])
    display_cols = ['Base', 'Type', 'Predicted', 'Prob_Irreg_%', 'Error_Type']

    with tab_mc_all:
        if n_errors > 0:
            st.dataframe(
                misclassified[display_cols].rename(columns={'Type':'Actual','Prob_Irreg_%':'P(Irregular)%'}),
                use_container_width=True, height=min(n_errors * 38 + 40, 400)
            )
            if n_errors >= 3:
                st.markdown('<span class="sec-label">Why does the model fail?</span>', unsafe_allow_html=True)
                misclassified['last_letter'] = misclassified['Base'].str[-1]
                lc = misclassified['last_letter'].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(8, 3))
                sax(ax, fig)
                ax.bar(lc.index, lc.values, color=C2, edgecolor=BG, linewidth=2, width=0.5)
                for i, (ch, v) in enumerate(lc.items()):
                    ax.text(i, v + 0.05, str(v), ha='center', fontsize=10, fontweight='bold', color='#8096B8')
                ax.set_title('Final Letter of Misclassified Verbs', fontsize=10, pad=10)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.tick_params(bottom=False)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown(f"""
                <div class="card" style="margin-top:0.5rem;">
                  <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#4A6280;line-height:2;">
                    The model uses <b style="color:#DCE0EA;">spelling features only</b> — not etymology.<br>
                    Short verbs (cut, put, hit) look regular from spelling but are irregular.<br>
                    <span style="color:#2E4060;">Adding etymology features would likely push accuracy above 85%.</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_mc_fp:
        if n_fp > 0:
            st.dataframe(misclassified[fp_mask][display_cols].rename(columns={'Type':'Actual','Prob_Irreg_%':'P(Irregular)%'}),
                         use_container_width=True, height=min(n_fp * 38 + 40, 380))

    with tab_mc_fn:
        if n_fn > 0:
            st.dataframe(misclassified[fn_mask][display_cols].rename(columns={'Type':'Actual','Prob_Irreg_%':'P(Irregular)%'}),
                         use_container_width=True, height=min(n_fn * 38 + 40, 380))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — REFERENCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "/ Reference":
    st.markdown('<div class="page-title">Verb Reference</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Complete searchable table — {len(df_reg)+len(df_irreg)} verbs · {len(df_part)} participial adjectives</div>', unsafe_allow_html=True)

    t_r, t_i, t_p = st.tabs(["Regular Verbs", "Irregular Verbs", "Participial Adjectives"])

    with t_r:
        s = st.text_input("", placeholder="Search regular verbs...", key="s_r")
        df_sh = df_reg.copy()
        if s:
            df_sh = df_sh[df_sh['Base'].str.contains(s.lower(), na=False)]
        st.markdown(f'<span class="sec-label">{len(df_sh)} verbs</span>', unsafe_allow_html=True)
        st.dataframe(
            df_sh[['Base','Simple_Past','Past_Participle',
                   'IPA_Base','IPA_Past','Last_Sound','Ending']].reset_index(drop=True),
            use_container_width=True, height=520
        )

    with t_i:
        cs, cf1, cf2 = st.columns([2, 1, 1])
        with cs:
            s2 = st.text_input("", placeholder="Search irregular verbs...", key="s_i")
        with cf1:
            vc_opts = ["All vowel patterns"] + df_irreg['Vowel_Change'].value_counts().index.tolist()
            pf = st.selectbox("", vc_opts, key="ref_pf", label_visibility="collapsed")
        with cf2:
            mp_opts2 = ["All morph patterns"] + df_irreg['Morph_Pattern'].value_counts().index.tolist()
            mp_f = st.selectbox("", mp_opts2, key="ref_mp", label_visibility="collapsed")

        df_sh2 = df_irreg.copy()
        if s2:
            df_sh2 = df_sh2[df_sh2['Base'].str.contains(s2.lower(), na=False)]
        if pf != "All vowel patterns":
            df_sh2 = df_sh2[df_sh2['Vowel_Change'] == pf]
        if mp_f != "All morph patterns":
            df_sh2 = df_sh2[df_sh2['Morph_Pattern'] == mp_f]

        st.markdown(f'<span class="sec-label">{len(df_sh2)} verbs</span>', unsafe_allow_html=True)
        st.dataframe(
            df_sh2[['Base','Simple_Past','Past_Participle',
                    'IPA_Base','IPA_Past','Vowel_Change','Morph_Pattern']].reset_index(drop=True),
            use_container_width=True, height=520
        )

    with t_p:
        cs3, cf3 = st.columns([2, 1])
        with cs3:
            s3 = st.text_input("", placeholder="Search participial adjectives...", key="s_p")
        with cf3:
            sc_opts = ["All classes"] + df_part['Semantic_Class'].value_counts().index.tolist()
            sc_f = st.selectbox("", sc_opts, key="ref_sc", label_visibility="collapsed")

        df_sh3 = df_part.copy()
        if s3:
            mask = (df_sh3['Base_Verb'].str.contains(s3.lower(), na=False) |
                    df_sh3['Participial_Form'].str.contains(s3.lower(), na=False))
            df_sh3 = df_sh3[mask]
        if sc_f != "All classes":
            df_sh3 = df_sh3[df_sh3['Semantic_Class'] == sc_f]

        st.markdown(f'<span class="sec-label">{len(df_sh3)} entries</span>', unsafe_allow_html=True)
        st.dataframe(
            df_sh3[['Base_Verb','Participial_Form','IPA_Adj','Phonetic_Adj',
                    'Semantic_Class','Example_Phrase','Notes']].reset_index(drop=True),
            use_container_width=True, height=520
        )
