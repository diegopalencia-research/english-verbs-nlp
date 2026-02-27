import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Verb Phonetics — diegopalencia",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080C14;
    color: #E8EAF0;
}
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid #1E2535;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem;
}
[data-testid="stSidebar"] .stRadio label {
    color: #6B7794 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #7EB8D4 !important; }

.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4A5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4A5568;
    margin-bottom: 0.8rem;
    margin-top: 1.6rem;
}
[data-testid="metric-container"] {
    background: #0D1220;
    border: 1px solid #1E2535;
    border-radius: 4px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4A5568 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #E8EAF0 !important;
}
[data-testid="stTextInput"] input {
    background: #0D1220 !important;
    border: 1px solid #1E2535 !important;
    border-radius: 3px !important;
    color: #E8EAF0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
    letter-spacing: 0.04em;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7EB8D4 !important;
    box-shadow: 0 0 0 1px #7EB8D420 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1A2035;
    background: transparent;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4A5568 !important;
    padding: 0.6rem 1.4rem !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #7EB8D4 !important;
    border-bottom: 1px solid #7EB8D4 !important;
}
[data-testid="stDataFrame"] { border: 1px solid #1A2035 !important; border-radius: 4px; }
[data-testid="stDataFrame"] * { font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; }

.verb-card {
    background: #0D1220;
    border: 1px solid #1E2535;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.ipa-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #7EB8D4;
    background: rgba(126,184,212,0.07);
    border: 1px solid rgba(126,184,212,0.15);
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    display: inline-block;
}
.rule-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 2px;
    margin-right: 0.5rem;
}
.badge-t   { background:rgba(78,160,120,0.1); border:1px solid rgba(78,160,120,0.25); color:#8ECFB0; }
.badge-d   { background:rgba(126,184,212,0.1); border:1px solid rgba(126,184,212,0.25); color:#7EB8D4; }
.badge-id  { background:rgba(200,160,80,0.1); border:1px solid rgba(200,160,80,0.25); color:#D4B87A; }
.badge-irreg { background:rgba(180,80,90,0.1); border:1px solid rgba(180,80,90,0.25); color:#D48A90; }

.stat-row  { display:flex; gap:1rem; margin-bottom:1.5rem; }
.stat-block { flex:1; background:#0D1220; border:1px solid #1E2535; border-radius:4px; padding:1rem 1.2rem; }
.stat-num  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#7EB8D4; }
.stat-label { font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:#4A5568; }

.pattern-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:0.8rem; margin-top:1rem; }
.pattern-card { background:#0D1220; border:1px solid #1E2535; border-radius:4px; padding:0.9rem 1rem; }
.pattern-card-symbol { font-family:'DM Mono',monospace; font-size:0.78rem; color:#D4B87A; margin-bottom:0.3rem; }
.pattern-card-count  { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#E8EAF0; }
.pattern-card-label  { font-family:'DM Mono',monospace; font-size:0.65rem; color:#4A5568; letter-spacing:0.08em; text-transform:uppercase; }
.pattern-card-ex     { font-family:'DM Mono',monospace; font-size:0.65rem; color:#2A3348; margin-top:0.4rem; }
</style>
""", unsafe_allow_html=True)


# ── Audio helper ──────────────────────────────────────────────────────────────
def speak_button(word, label="", key=""):
    safe_word = str(word).replace("'", "\\'").replace('"', '\\"')
    btn_html = f"""
    <button onclick="window.speechSynthesis.speak(Object.assign(new SpeechSynthesisUtterance('{safe_word}'),
        {{lang:'en-US', rate:0.85, pitch:1}}))"
      style="background:#0D1220; border:1px solid #1E2535; border-radius:3px;
             color:#7EB8D4; font-family:'DM Mono',monospace; font-size:0.72rem;
             letter-spacing:0.08em; padding:0.3rem 0.9rem; cursor:pointer;
             transition:all 0.2s; margin:0.2rem 0.3rem 0.2rem 0;"
      onmouseover="this.style.borderColor='#7EB8D4'"
      onmouseout="this.style.borderColor='#1E2535'">
      &#9654; {label or word}
    </button>
    """
    components.html(btn_html, height=44)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    FILE = 'data/english_verbs.xlsx'
    df_reg   = pd.read_excel(FILE, sheet_name='Regular Verbs',   header=2)
    df_irreg = pd.read_excel(FILE, sheet_name='Irregular Verbs', header=2)
    reg_cols   = ['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','IPA_PP',
                  'Phonetic_Base','Phonetic_Past','Phonetic_PP','Last_Sound','Ending']
    irreg_cols = ['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','IPA_PP',
                  'Phonetic_Base','Phonetic_Past','Phonetic_PP','Vowel_Change']
    df_reg.columns   = reg_cols
    df_irreg.columns = irreg_cols
    df_reg   = df_reg.dropna(subset=['Base']).reset_index(drop=True)
    df_irreg = df_irreg.dropna(subset=['Base']).reset_index(drop=True)
    df_reg['Type']   = 'Regular'
    df_irreg['Type'] = 'Irregular'
    return df_reg, df_irreg

def extract_features(df):
    f = pd.DataFrame()
    f['length']          = df['Base'].str.len()
    f['vowel_count']     = df['Base'].str.count('[aeiou]')
    f['consonant_count'] = df['Base'].str.count('[bcdfghjklmnpqrstvwxyz]')
    for s in ['e','n','d','t','l','r','k','g','w','y','ng','nd','ld','nt','in','ow','aw']:
        f[f'ends_{s}'] = df['Base'].str.endswith(s).astype(int)
    le = LabelEncoder()
    f['last_letter'] = le.fit_transform(df['Base'].str[-1].fillna('_'))
    f['second_last'] = le.fit_transform(df['Base'].str[-2].fillna('_'))
    return f

@st.cache_resource
def train_model(df_reg, df_irreg):
    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)
    X = extract_features(df_all)
    y = (df_all['Type'] == 'Irregular').astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    rf.fit(X_tr, y_tr)
    return rf

df_reg, df_irreg = load_data()
model            = train_model(df_reg, df_irreg)

CHART_BG   = '#080C14'
CHART_TEXT = '#6B7794'
ACCENT1, ACCENT2, ACCENT3, ACCENT4 = '#7EB8D4', '#D48A90', '#D4B87A', '#8ECFB0'

def style_ax(ax, fig):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_TEXT, labelsize=9)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.title.set_color('#C8CDD8')
    for sp in ax.spines.values(): sp.set_color('#1A2035')

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem 0;">
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;
                  text-transform:uppercase;color:#4A5568;margin-bottom:0.3rem;">Project</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                  color:#E8EAF0;line-height:1.3;">English Verb<br>Phonetics</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["/ Verb Lookup","/ Phonetic Explorer",
                         "/ Charts & Analysis","/ Verb Reference"],
                    label_visibility="collapsed")

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#2A3348;line-height:2.2;">
      <div style="color:#4A5568;letter-spacing:0.1em;text-transform:uppercase;
                  font-size:0.62rem;margin-bottom:0.5rem;">Dataset</div>
      <span style="color:#8ECFB0;">{len(df_reg)}</span> regular<br>
      <span style="color:#D48A90;">{len(df_irreg)}</span> irregular<br>
      <span style="color:#7EB8D4;">{len(df_reg)+len(df_irreg)}</span> total
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — VERB LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
if page == "/ Verb Lookup":
    st.markdown('<div class="page-title">Verb Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">IPA · phonetics · audio · rule explanation</div>', unsafe_allow_html=True)

    verb = st.text_input("", placeholder="Type a verb — walk, think, break, google...").lower().strip()

    if not verb:
        st.markdown("""
        <div style="margin-top:3rem;font-family:'DM Mono',monospace;font-size:0.75rem;
                    color:#2A3348;line-height:2.4;">
          Try: walk &nbsp;&middot;&nbsp; think &nbsp;&middot;&nbsp; break &nbsp;&middot;&nbsp;
          feel &nbsp;&middot;&nbsp; start &nbsp;&middot;&nbsp; google &nbsp;&middot;&nbsp; zoom
        </div>
        """, unsafe_allow_html=True)

    else:
        reg_match   = df_reg[df_reg['Base'].str.lower() == verb]
        irreg_match = df_irreg[df_irreg['Base'].str.lower() == verb]

        if not reg_match.empty:
            row = reg_match.iloc[0]
            st.markdown('<div class="section-label">Regular Verb</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form",       row['Base'])
            c2.metric("Simple Past",     row['Simple_Past'])
            c3.metric("Past Participle", row['Past_Participle'])

            st.markdown('<div class="section-label">Audio Pronunciation</div>', unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca: speak_button(str(row['Base']),            "Base form",       f"b{verb}")
            with cb: speak_button(str(row['Simple_Past']),     "Simple past",     f"p{verb}")
            with cc: speak_button(str(row['Past_Participle']), "Past participle", f"pp{verb}")

            st.markdown('<div class="section-label">IPA Transcription</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                            text-transform:uppercase;color:#4A5568;margin-bottom:0.4rem;">Base</div>
                <span class="ipa-text">{row['IPA_Base']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;margin-top:0.4rem;">{row['Phonetic_Base']}</div>
              </div>
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                            text-transform:uppercase;color:#4A5568;margin-bottom:0.4rem;">Simple Past</div>
                <span class="ipa-text">{row['IPA_Past']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;margin-top:0.4rem;">{row['Phonetic_Past']}</div>
              </div>
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                            text-transform:uppercase;color:#4A5568;margin-bottom:0.4rem;">Past Participle</div>
                <span class="ipa-text">{row['IPA_PP']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;margin-top:0.4rem;">{row['Phonetic_PP']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Phonetic Rule</div>', unsafe_allow_html=True)
            ending = row['Ending']
            sound  = row['Last_Sound']
            if ending == '/t/':
                badge = '<span class="rule-badge badge-t">/t/</span>'
                rule  = "Last sound is a <b>voiceless consonant</b> — -ed is pronounced as a sharp <b>/t/</b>, no extra syllable."
            elif ending == '/d/':
                badge = '<span class="rule-badge badge-d">/d/</span>'
                rule  = "Last sound is <b>voiced</b> (vowel or voiced consonant) — -ed is pronounced as a soft <b>/d/</b>."
            else:
                badge = '<span class="rule-badge badge-id">/ɪd/</span>'
                rule  = "Last sound is <b>/t/ or /d/</b> — an extra syllable <b>/ɪd/</b> is added to separate identical sounds."

            st.markdown(f"""
            <div class="verb-card">
              {badge}
              <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4A5568;">last sound: {sound}</span>
              <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#8A9BB8;margin-top:0.7rem;line-height:1.8;">{rule}</div>
            </div>
            """, unsafe_allow_html=True)

        elif not irreg_match.empty:
            row = irreg_match.iloc[0]
            st.markdown('<div class="section-label">Irregular Verb</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form",       row['Base'])
            c2.metric("Simple Past",     row['Simple_Past'])
            c3.metric("Past Participle", row['Past_Participle'])

            st.markdown('<div class="section-label">Audio Pronunciation</div>', unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca: speak_button(str(row['Base']),            "Base form",       f"b{verb}")
            with cb: speak_button(str(row['Simple_Past']),     "Simple past",     f"p{verb}")
            with cc: speak_button(str(row['Past_Participle']), "Past participle", f"pp{verb}")

            st.markdown('<div class="section-label">IPA Transcription</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                            text-transform:uppercase;color:#4A5568;margin-bottom:0.4rem;">Base</div>
                <span class="ipa-text">{row['IPA_Base']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;margin-top:0.4rem;">{row['Phonetic_Base']}</div>
              </div>
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                            text-transform:uppercase;color:#4A5568;margin-bottom:0.4rem;">Simple Past</div>
                <span class="ipa-text">{row['IPA_Past']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;margin-top:0.4rem;">{row['Phonetic_Past']}</div>
              </div>
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                            text-transform:uppercase;color:#4A5568;margin-bottom:0.4rem;">Past Participle</div>
                <span class="ipa-text">{row['IPA_PP']}</span>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;margin-top:0.4rem;">{row['Phonetic_PP']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Vowel Change Pattern</div>', unsafe_allow_html=True)
            vc = row['Vowel_Change']
            siblings = df_irreg[(df_irreg['Vowel_Change']==vc) & (df_irreg['Base']!=row['Base'])]['Base'].tolist()[:6]
            siblings_str = " &nbsp;&middot;&nbsp; ".join(siblings) if siblings else "—"
            ex = PATTERN_EXAMPLES.get(vc, "")

            st.markdown(f"""
            <div class="verb-card">
              <span class="rule-badge badge-irreg">{vc}</span>
              <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#8A9BB8;margin-top:0.7rem;line-height:1.8;">
                This verb does <b>not</b> follow the -ed rule.<br>
                It changes its internal vowel to form the past tense.
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#4A5568;margin-top:0.8rem;">
                Same pattern: {siblings_str}
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown('<div class="section-label">Not in dataset — ML Prediction</div>', unsafe_allow_html=True)
            row      = pd.DataFrame([{'Base': verb}])
            features = extract_features(row)
            prob     = model.predict_proba(features)[0]
            label    = 'Irregular' if prob[1] > 0.5 else 'Regular'
            conf     = max(prob) * 100

            speak_button(verb, "Hear pronunciation", f"pred_{verb}")

            if label == 'Regular':
                last = verb[-1]
                if last in 'td':
                    ending, rule = '/ɪd/', "Ends in /t/ or /d/ — past adds extra syllable /ɪd/"
                elif last in 'pkfscx':
                    ending, rule = '/t/', "Ends in voiceless consonant — past sounds like /t/"
                else:
                    ending, rule = '/d/', "Ends in voiced sound — past sounds like /d/"
                st.markdown(f"""
                <div class="verb-card">
                  <span class="rule-badge badge-t">Regular</span>
                  <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4A5568;">{conf:.1f}% confidence</span>
                  <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
                              color:#8ECFB0;margin-top:0.6rem;">{verb} &rarr; {verb}ed</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#8A9BB8;margin-top:0.5rem;">{rule}</div>
                </div>
                """, unsafe_allow_html=True)
                speak_button(f"{verb}ed", "Hear predicted past", f"pred_past_{verb}")
            else:
                st.markdown(f"""
                <div class="verb-card">
                  <span class="rule-badge badge-irreg">Likely Irregular</span>
                  <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4A5568;">{conf:.1f}% confidence</span>
                  <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#D48A90;margin-top:0.7rem;">
                    Past form is likely unpredictable — check a dictionary.
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PHONETIC EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "/ Phonetic Explorer":
    st.markdown('<div class="page-title">Phonetic Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Regular -ed rules &amp; irregular vowel patterns</div>', unsafe_allow_html=True)

    tab_reg, tab_irreg = st.tabs(["Regular — -ed Rule", "Irregular — Vowel Patterns"])

    with tab_reg:
        endings = df_reg['Ending'].value_counts()
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-block">
            <div class="stat-num" style="color:#8ECFB0;">{endings.get('/t/', 0)}</div>
            <div style="margin:0.3rem 0;"><span class="rule-badge badge-t">/t/</span></div>
            <div class="stat-label">voiceless consonant</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#2A3348;margin-top:0.5rem;">p &middot; k &middot; f &middot; s &middot; sh &middot; ch &middot; x</div>
          </div>
          <div class="stat-block">
            <div class="stat-num" style="color:#7EB8D4;">{endings.get('/d/', 0)}</div>
            <div style="margin:0.3rem 0;"><span class="rule-badge badge-d">/d/</span></div>
            <div class="stat-label">voiced sound</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#2A3348;margin-top:0.5rem;">vowels &middot; b &middot; g &middot; v &middot; z &middot; m &middot; n &middot; l &middot; r</div>
          </div>
          <div class="stat-block">
            <div class="stat-num" style="color:#D4B87A;">{endings.get('/ɪd/', 0)}</div>
            <div style="margin:0.3rem 0;"><span class="rule-badge badge-id">/ɪd/</span></div>
            <div class="stat-label">t or d ending</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#2A3348;margin-top:0.5rem;">/t/ or /d/ &rarr; needs extra syllable</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Filter</div>', unsafe_allow_html=True)
        ending_filter = st.selectbox("", ["/t/ — voiceless","/d/ — voiced","/ɪd/ — extra syllable"],
                                     label_visibility="collapsed")
        ending_code = ending_filter.split(" ")[0]
        filtered = df_reg[df_reg['Ending']==ending_code][
            ['Base','Simple_Past','IPA_Base','IPA_Past','Last_Sound','Ending']
        ].reset_index(drop=True)
        st.markdown(f'<div class="section-label">{len(filtered)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(filtered, use_container_width=True, height=380)

    with tab_irreg:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#6B7794;
                    line-height:2;margin-bottom:1rem;">
          Irregular verbs form their past tense through internal vowel changes,
          not by adding -ed. Click a pattern to filter the table.
        </div>
        """, unsafe_allow_html=True)

        vc_counts = df_irreg['Vowel_Change'].value_counts()
        cards_html = '<div class="pattern-grid">'
        for pattern, count in vc_counts.items():
            ex = PATTERN_EXAMPLES.get(pattern, "")
            short_ex = ex.split("·")[0].strip() if ex else ""
            cards_html += f"""
            <div class="pattern-card">
              <div class="pattern-card-symbol">{pattern}</div>
              <div class="pattern-card-count">{count}</div>
              <div class="pattern-card-label">verbs</div>
              <div class="pattern-card-ex">{short_ex}</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:2rem;">Filter by Pattern</div>',
                    unsafe_allow_html=True)
        pattern_options = ["All patterns"] + vc_counts.index.tolist()
        selected = st.selectbox("", pattern_options, label_visibility="collapsed", key="explorer_irreg")

        if selected == "All patterns":
            df_show = df_irreg[['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Vowel_Change']].copy()
        else:
            df_show = df_irreg[df_irreg['Vowel_Change']==selected][
                ['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Vowel_Change']
            ].copy()
            if selected in PATTERN_EXAMPLES:
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                            color:#4A5568;margin-bottom:0.8rem;">
                  {PATTERN_EXAMPLES[selected]}
                </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="section-label">{len(df_show)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=360)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "/ Charts & Analysis":
    st.markdown('<div class="page-title">Charts & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Patterns in 298 English verbs</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview","-ed Endings","Irregular Patterns","Verb Length"])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        style_ax(ax, fig)
        bars = ax.bar(['Regular','Irregular'], [len(df_reg),len(df_irreg)],
                      color=[ACCENT1,ACCENT2], width=0.4, edgecolor=CHART_BG, linewidth=2)
        for bar, val in zip(bars, [len(df_reg),len(df_irreg)]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(val), ha='center', fontweight='bold', fontsize=12, color='#C8CDD8')
        ax.set_title('Dataset Composition', fontsize=10, pad=12)
        ax.set_ylim(0, max(len(df_reg),len(df_irreg))*1.18)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(bottom=False)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#4A5568;margin-top:0.5rem;">The 10 most frequently used verbs in spoken English are all irregular: be, have, do, say, go, get, make, know, think, see.</div>', unsafe_allow_html=True)

    with tab2:
        ec = df_reg['Ending'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax in axes: style_ax(ax, fig)
        axes[0].bar(ec.index, ec.values, color=[ACCENT4,ACCENT1,ACCENT3],
                    edgecolor=CHART_BG, linewidth=2, width=0.4)
        for i, (idx, val) in enumerate(ec.items()):
            axes[0].text(i, val+1, str(val), ha='center', fontweight='bold', fontsize=11, color='#C8CDD8')
        axes[0].set_title('-ed Ending Count', fontsize=10, pad=10)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)
        axes[1].pie(ec.values, labels=[f'{i}  ({v})' for i,v in ec.items()],
                    colors=[ACCENT4,ACCENT1,ACCENT3], autopct='%1.0f%%', startangle=90,
                    wedgeprops={'edgecolor':CHART_BG,'linewidth':3},
                    textprops={'color':CHART_TEXT,'fontsize':9,'fontfamily':'monospace'})
        axes[1].set_title('-ed Share', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        vc = df_irreg['Vowel_Change'].value_counts().head(13)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style_ax(ax, fig)
        colors = [ACCENT2 if i==0 else ACCENT1 if i<4 else ACCENT3 for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1],
                edgecolor=CHART_BG, linewidth=2, height=0.6)
        for bar, val in zip(ax.patches, vc.values[::-1]):
            ax.text(bar.get_width()+0.08, bar.get_y()+bar.get_height()/2,
                    str(val), va='center', fontsize=9, fontweight='bold', color='#C8CDD8')
        ax.set_title('Most Common Irregular Patterns', fontsize=10, pad=12)
        ax.set_xlim(0, vc.max()+5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticklabels(vc.index[::-1], fontfamily='monospace', fontsize=8.5, color=CHART_TEXT)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        df_reg['length']   = df_reg['Base'].str.len()
        df_irreg['length'] = df_irreg['Base'].str.len()
        fig, ax = plt.subplots(figsize=(10, 4))
        style_ax(ax, fig)
        ax.hist(df_reg['length'],   bins=range(2,15), alpha=0.75, color=ACCENT1,
                label='Regular',   edgecolor=CHART_BG, linewidth=1.5)
        ax.hist(df_irreg['length'], bins=range(2,15), alpha=0.75, color=ACCENT2,
                label='Irregular', edgecolor=CHART_BG, linewidth=1.5)
        ax.axvline(df_reg['length'].mean(),   color=ACCENT1, linestyle='--', linewidth=1.5,
                   label=f"Reg avg: {df_reg['length'].mean():.1f}")
        ax.axvline(df_irreg['length'].mean(), color=ACCENT2, linestyle='--', linewidth=1.5,
                   label=f"Irreg avg: {df_irreg['length'].mean():.1f}")
        ax.set_title('Verb Length Distribution', fontsize=10, pad=12)
        ax.set_xlabel('Characters', fontsize=9)
        ax.legend(fontsize=8, framealpha=0, labelcolor=CHART_TEXT)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#4A5568;margin-top:0.5rem;">Irregular verbs tend to be shorter — they come from Old English, a compact monosyllabic language. New words invented today (google, tweet, zoom) are almost always regular.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — VERB REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "/ Verb Reference":
    st.markdown('<div class="page-title">Verb Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Complete searchable table — 298 verbs</div>', unsafe_allow_html=True)

    tab_r, tab_i = st.tabs(["Regular Verbs","Irregular Verbs"])

    with tab_r:
        search = st.text_input("", placeholder="Search regular verbs...", key="s_reg")
        df_show = df_reg.copy()
        if search:
            df_show = df_show[df_show['Base'].str.contains(search.lower(), na=False)]
        st.markdown(f'<div class="section-label">{len(df_show)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show[['Base','Simple_Past','Past_Participle',
                               'IPA_Base','IPA_Past','Last_Sound','Ending']].reset_index(drop=True),
                     use_container_width=True, height=520)

    with tab_i:
        cs, cf = st.columns([2,1])
        with cs:
            search2 = st.text_input("", placeholder="Search irregular verbs...", key="s_irreg")
        with cf:
            vc_opts = ["All patterns"] + df_irreg['Vowel_Change'].value_counts().index.tolist()
            pat_filter = st.selectbox("", vc_opts, key="ref_irreg_filter",
                                      label_visibility="collapsed")
        df_show2 = df_irreg.copy()
        if search2:
            df_show2 = df_show2[df_show2['Base'].str.contains(search2.lower(), na=False)]
        if pat_filter != "All patterns":
            df_show2 = df_show2[df_show2['Vowel_Change']==pat_filter]
        st.markdown(f'<div class="section-label">{len(df_show2)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show2[['Base','Simple_Past','Past_Participle',
                                'IPA_Base','IPA_Past','Vowel_Change']].reset_index(drop=True),
                     use_container_width=True, height=520)
