import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Verb Phonetics — diegopalencia",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── High-End Aesthetic CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080C14;
    color: #E8EAF0;
}

/* Fix for Dark Mode Input Visibility */
div[data-testid="stTextInput"] label {
    color: #7EB8D4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em;
}

input::placeholder {
    color: #4A5568 !important;
    opacity: 1;
}

[data-testid="stTextInput"] input {
    background: #0D1220 !important;
    border: 1px solid #1E2535 !important;
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
}

.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid #1E2535;
}

.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
}

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
    padding: 0.15rem 0.5rem;
    display: inline-block;
}

.rule-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: 2px;
}
.badge-t { background:rgba(78,160,120,0.1); color:#8ECFB0; border:1px solid #8ECFB044; }
.badge-d { background:rgba(126,184,212,0.1); color:#7EB8D4; border:1px solid #7EB8D444; }
.badge-id { background:rgba(200,160,80,0.1); color:#D4B87A; border:1px solid #D4B87A44; }
.badge-irreg { background:rgba(180,80,90,0.1); color:#D48A90; border:1px solid #D48A9044; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4A5568;
    margin: 1.5rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Audio Helper (Fixed for Inline Display) ───────────────────
def speak_button(word, label="", key=""):
    safe_word = str(word).replace("'", "\\'").replace('"', '\\"')
    btn_html = f"""
    <button onclick="window.speechSynthesis.speak(Object.assign(new SpeechSynthesisUtterance('{safe_word}'),
        {{lang:'en-US', rate:0.85, pitch:1}}))"
      style="background:#1A2035; border:1px solid #30364d; border-radius:3px;
             color:#7EB8D4; font-family:monospace; font-size:12px;
             padding:6px 12px; cursor:pointer; width:100%;">
      ▶ {label or word}
    </button>
    """
    components.html(btn_html, height=35)

# ── Data Core ─────────────────────────────────────────────────
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
    df_reg = df_reg.dropna(subset=['Base']).reset_index(drop=True)
    df_irreg = df_irreg.dropna(subset=['Base']).reset_index(drop=True)
    df_reg['Type'], df_irreg['Type'] = 'Regular', 'Irregular'
    return df_reg, df_irreg

def extract_features(df):
    f = pd.DataFrame()
    f['length'] = df['Base'].str.len()
    f['vowels'] = df['Base'].str.count('[aeiou]')
    for s in ['e','n','d','t','l','r','k','y','ng','ow']:
        f[f'ends_{s}'] = df['Base'].str.endswith(s).astype(int)
    le = LabelEncoder()
    f['last'] = le.fit_transform(df['Base'].str[-1].fillna('_'))
    return f

@st.cache_resource
def train_model(df_reg, df_irreg):
    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)
    X, y = extract_features(df_all), (df_all['Type'] == 'Irregular').astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    return rf

df_reg, df_irreg = load_data()
model = train_model(df_reg, df_irreg)

# ── Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:white;'>VERB LAB</h2>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["/ Verb Lookup", "/ Phonetic Explorer", "/ Charts & Analysis", "/ Verb Reference"])

# ─────────────────────────────────────────────────────────────
# PAGE 1: LOOKUP
# ─────────────────────────────────────────────────────────────
if page == "/ Verb Lookup":
    st.markdown('<div class="page-title">Verb Lookup</div>', unsafe_allow_html=True)
    verb = st.text_input("Enter base form", placeholder="Type a verb — walk, think, break...").lower().strip()

    if verb:
        reg_match   = df_reg[df_reg['Base'].str.lower() == verb]
        irreg_match = df_irreg[df_irreg['Base'].str.lower() == verb]

        if not reg_match.empty or not irreg_match.empty:
            row = reg_match.iloc[0] if not reg_match.empty else irreg_match.iloc[0]
            st.markdown(f'<div class="section-label">{"Regular" if not reg_match.empty else "Irregular"} Verb</div>', unsafe_allow_html=True)
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Base", row['Base'])
            c2.metric("Past", row['Simple_Past'])
            c3.metric("Participle", row['Past_Participle'])

            # Audio
            st.markdown('<div class="section-label">Audio</div>', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            with a1: speak_button(row['Base'], "Speak Base")
            with a2: speak_button(row['Simple_Past'], "Speak Past")
            with a3: speak_button(row['Past_Participle'], "Speak Participle")

            # Phonetics Display
            st.markdown('<div class="section-label">Phonetics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex; gap:10px;">
                <div class="verb-card" style="flex:1;"><b>{row['IPA_Base']}</b><br><small>{row['Phonetic_Base']}</small></div>
                <div class="verb-card" style="flex:1;"><b>{row['IPA_Past']}</b><br><small>{row['Phonetic_Past']}</small></div>
                <div class="verb-card" style="flex:1;"><b>{row['IPA_PP']}</b><br><small>{row['Phonetic_PP']}</small></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Verb not in dataset. Try another or check spelling.")

# ─────────────────────────────────────────────────────────────
# PAGE 2: EXPLORER (With new columns)
# ─────────────────────────────────────────────────────────────
elif page == "/ Phonetic Explorer":
    st.markdown('<div class="page-title">Phonetic Explorer</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Regular Rules", "Irregular Patterns"])
    
    with t1:
        end_filter = st.selectbox("Filter by -ed sound", ["/t/", "/d/", "/ɪd/"])
        cols = ['Base', 'Simple_Past', 'Phonetic_Base', 'Phonetic_Past', 'Ending']
        st.dataframe(df_reg[df_reg['Ending'] == end_filter][cols], use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4: REFERENCE (Full Columns)
# ─────────────────────────────────────────────────────────────
elif page == "/ Verb Reference":
    st.markdown('<div class="page-title">Full Reference</div>', unsafe_allow_html=True)
    r1, r2 = st.tabs(["Regular", "Irregular"])
    
    all_cols = ['Base', 'Simple_Past', 'Past_Participle', 'Phonetic_Base', 'Phonetic_Past', 'Phonetic_PP']
    
    with r1:
        st.dataframe(df_reg[all_cols], use_container_width=True)
    with r2:
        st.dataframe(df_irreg[all_cols + ['Vowel_Change']], use_container_width=True)
