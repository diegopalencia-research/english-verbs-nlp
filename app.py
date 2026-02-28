# app.py
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Verb Phonetics — diegopalencia",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Styling (dark navy + fonts) + improved contrast
# -------------------------
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
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.02em;
    color: #F4F7FF;
    line-height: 1.05;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    color: #8A9BB8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
}
.placeholder-examples {
    font-family: 'DM Mono', monospace;
    font-size:0.82rem;
    color:#C8CDD8;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #7A8796;
    margin-bottom: 0.8rem;
    margin-top: 1.6rem;
}
[data-testid="stTextInput"] input {
    background: #0D1220 !important;
    border: 1px solid #1E2535 !important;
    border-radius: 3px !important;
    color: #E8EAF0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.95rem !important;
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
    padding: 0.55rem 1.2rem !important;
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
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
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
    border-radius: 3px;
    margin-right: 0.5rem;
}
.badge-t   { background:rgba(78,160,120,0.1); border:1px solid rgba(78,160,120,0.25); color:#8ECFB0; }
.badge-d   { background:rgba(126,184,212,0.1); border:1px solid rgba(126,184,212,0.25); color:#7EB8D4; }
.badge-id  { background:rgba(200,160,80,0.1); border:1px solid rgba(200,160,80,0.25); color:#D4B87A; }
.badge-irreg { background:rgba(180,80,90,0.1); border:1px solid rgba(180,80,90,0.25); color:#D48A90; }

.pattern-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:0.8rem; margin-top:1rem; }
.pattern-card { background:#0D1220; border:1px solid #1E2535; border-radius:6px; padding:0.9rem 1rem; cursor:pointer; transition:transform .12s; }
.pattern-card:hover { border-color:#7EB8D4; transform:translateY(-4px); }
.pattern-card-symbol { font-family:'DM Mono',monospace; font-size:0.78rem; color:#D4B87A; margin-bottom:0.3rem; }
.pattern-card-count  { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#E8EAF0; }
.pattern-card-label  { font-family:'DM Mono',monospace; font-size:0.65rem; color:#4A5568; letter-spacing:0.08em; text-transform:uppercase; }
.pattern-card-ex     { font-family:'DM Mono',monospace; font-size:0.65rem; color:#9FB6C6; margin-top:0.4rem; }
.audio-note { font-family:'DM Mono',monospace; font-size:0.68rem; color:#7A8796; margin-top:0.4rem; }
.play-inline { display:inline-block; margin-right:0.45rem; }
.small-muted { font-family:'DM Mono',monospace;color:#9FB6C6;font-size:0.78rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Inject one in-page global audio engine (runs in main document)
# Buttons rendered with st.markdown call window.playVerb(...)
# -------------------------
st.markdown(
    """
    <script>
    window.playVerb = function(text){
        try {
            if(!text) return;
            // cancel any running utterance to avoid overlap
            if(window.speechSynthesis) {
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance(String(text));
                u.lang = 'en-US';
                u.rate = 0.85;
                u.pitch = 1.0;
                window.speechSynthesis.speak(u);
            } else {
                console.warn("speechSynthesis not available");
            }
        } catch(e) {
            console.warn("playVerb error", e);
        }
    }
    </script>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helper to render play buttons inline (uses the global window.playVerb)
# We use st.markdown with unsafe HTML so buttons live in the main page
# -------------------------
def render_play_buttons(base, past=None, part=None):
    # json.dumps to safely escape strings for JS
    base_js = json.dumps(str(base)) if base else "null"
    past_js = json.dumps(str(past)) if past else "null"
    part_js = json.dumps(str(part)) if part else "null"

    html = f"""
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
      <button onclick="window.playVerb({base_js})" aria-label="Play base"
        style="background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:#7EB8D4;padding:6px 10px;font-family:DM Mono,monospace;font-size:0.85rem;cursor:pointer;">
        ▶︎ Base
      </button>
      {"<button onclick=\"window.playVerb(" + past_js + ")\" aria-label=\"Play past\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:#D48A90;padding:6px 10px;font-family:DM Mono,monospace;font-size:0.85rem;cursor:pointer;\">▶︎ Past</button>" if past else ""}
      {"<button onclick=\"window.playVerb(" + part_js + ")\" aria-label=\"Play part\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:#D4B87A;padding:6px 10px;font-family:DM Mono,monospace;font-size:0.85rem;cursor:pointer;\">▶︎ Part</button>" if part else ""}
      <div class="audio-note" style="margin-left:10px;">Uses your browser speech engine.</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_play_list(rows, title=None, max_items=8):
    # rows: iterable of dict-like with Base, Simple_Past, Past_Participle
    items = []
    count = 0
    for r in rows:
        if count >= max_items:
            break
        b = r.get('Base', '')
        p = r.get('Simple_Past', '') or None
        pp = r.get('Past_Participle', '') or None
        # use small inline buttons per row, reuse same HTML approach
        b_js = json.dumps(str(b))
        p_js = json.dumps(str(p)) if p else "null"
        pp_js = json.dumps(str(pp)) if pp else "null"
        items.append(f"""
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:140px;font-family:DM Mono,monospace;color:#C8CDD8;">{b}</div>
            <button onclick="window.playVerb({b_js})" style="background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:#7EB8D4;padding:5px 8px;font-family:DM Mono,monospace;font-size:0.78rem;cursor:pointer;">▶︎ base</button>
            {f"<button onclick=\"window.playVerb({p_js})\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:#D48A90;padding:5px 8px;font-family:DM Mono,monospace;font-size:0.78rem;cursor:pointer;\">▶︎ past</button>" if p else ""}
            {f"<button onclick=\"window.playVerb({pp_js})\" style=\"background:#0D1220;border:1px solid #1E2535;border-radius:6px;color:#D4B87A;padding:5px 8px;font-family:DM Mono,monospace;font-size:0.78rem;cursor:pointer;\">▶︎ part</button>" if pp else ""}
          </div>
        """)
        count += 1

    html = f"""
    <div style="font-family:DM Mono,monospace;color:#C8CDD8;">
      {f'<div style="font-weight:700;margin-bottom:6px;color:#C8CDD8;">{title}</div>' if title else ''}
      {''.join(items)}
    </div>
    """
    # approximate height
    height = 36 * (count + 1)
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Data loading + feature extraction + model
# -------------------------
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
    # ensure phonetic columns exist (fallback to IPA if missing)
    for c in ['Phonetic_Base','Phonetic_Past','Phonetic_PP']:
        if c not in df_reg.columns:
            df_reg[c] = df_reg.get('IPA_Base', '')
        if c not in df_irreg.columns:
            df_irreg[c] = df_irreg.get('IPA_Base', '')
    # normalize to strings
    for df in (df_reg, df_irreg):
        for col in ['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','IPA_PP',
                    'Phonetic_Base','Phonetic_Past','Phonetic_PP','Last_Sound','Ending','Vowel_Change']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
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

# Chart colors
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

# -------------------------
# Sidebar & navigation
# -------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.2rem 0;">
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;
                  text-transform:uppercase;color:#4A5568;margin-bottom:0.3rem;">Project</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
                  color:#E8EAF0;line-height:1.2;">English Verb<br>Phonetics</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.74rem;color:#C8CDD8;margin-top:0.6rem;">
        Search by any form (base / past / participle). Play audio with ▶︎ buttons.
      </div>
    </div>
    """, unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["/ Intro","/ Verb Lookup","/ Phonetic Explorer","/ Charts & Analysis","/ Verb Reference"], index=1)

st.sidebar.markdown("<div style='margin-top:1.0rem;'></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A5568;line-height:1.8;">
  <div style="color:#4A5568;letter-spacing:0.1em;text-transform:uppercase;font-size:0.62rem;margin-bottom:0.4rem;">Dataset</div>
  <span style="color:#8ECFB0;">{len(df_reg)}</span> regular<br>
  <span style="color:#D48A90;">{len(df_irreg)}</span> irregular<br>
  <span style="color:#7EB8D4;">{len(df_reg)+len(df_irreg)}</span> total
</div>
""", unsafe_allow_html=True)

# -------------------------
# Helpers: phonetic summary + robust search
# -------------------------
def phonetic_summary(df):
    out = {}
    for col in ['Phonetic_Base','Phonetic_Past','Phonetic_PP']:
        if col in df.columns:
            counts = df[col].value_counts().reset_index().rename(columns={'index':'phonetic', col:'count'})
            out[col] = counts.head(8)
        else:
            out[col] = pd.DataFrame({'phonetic':[], 'count':[]})
    return out

def find_verb_any_form(query):
    q = str(query).strip().lower()
    # exact base
    m_reg = df_reg[df_reg['Base'].str.lower() == q]
    if not m_reg.empty:
        return 'Regular', m_reg.iloc[0]
    m_ir = df_irreg[df_irreg['Base'].str.lower() == q]
    if not m_ir.empty:
        return 'Irregular', m_ir.iloc[0]
    # exact past / participle
    m_reg2 = df_reg[(df_reg['Simple_Past'].str.lower() == q) | (df_reg['Past_Participle'].str.lower() == q)]
    if not m_reg2.empty:
        return 'Regular', m_reg2.iloc[0]
    m_ir2 = df_irreg[(df_irreg['Simple_Past'].str.lower() == q) | (df_irreg['Past_Participle'].str.lower() == q)]
    if not m_ir2.empty:
        return 'Irregular', m_ir2.iloc[0]
    # contains (partial) search across the three forms
    m_reg3 = df_reg[df_reg[['Base','Simple_Past','Past_Participle']].apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)]
    if not m_reg3.empty:
        return 'Regular', m_reg3.iloc[0]
    m_ir3 = df_irreg[df_irreg[['Base','Simple_Past','Past_Participle']].apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)]
    if not m_ir3.empty:
        return 'Irregular', m_ir3.iloc[0]
    return None, None

# -------------------------
# PAGE: Intro
# -------------------------
if page == "/ Intro":
    st.markdown('<div class="page-title">Welcome — Verb Phonetics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">What this tool is · How it works · How to use it</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:DM Mono,monospace;color:#C8CDD8;line-height:1.8;">
      <b>What it is</b>
      <p>This web app combines a compact verb dataset (regular + irregular) with IPA, readable phonetic spellings,
      quick in-browser audio (no extra libs) and simple ML to predict irregularity for unknown items.</p>

      <b>How it works</b>
      <ol>
        <li>Search by any verb form: base (write), simple past (wrote), or past participle (written).</li>
        <li>The app finds the canonical base and shows IPA, phonetic spellings, audio and rules.</li>
        <li>The Phonetic Explorer groups verbs by -ed endings and vowel-change patterns.</li>
      </ol>

      <b>How to use strategically</b>
      <ul>
        <li>Study verbs by vowel-change pattern — it's easier to generalize (e.g., write/wrote).</li>
        <li>Use the example sentences to practise in context, then click ▶︎ to listen and repeat.</li>
        <li>Use the Charts page to sample representative verbs (tiny ▶︎ lists below the charts).</li>
      </ul>

      <b>Troubleshooting audio</b>
      <ul>
        <li>If audio is silent: ensure tab is not muted and click somewhere on the page (user gesture).</li>
        <li>Try Chrome or Firefox if your browser blocks speechSynthesis.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# PAGE: Verb Lookup (search by any form)
# -------------------------
elif page == "/ Verb Lookup":
    st.markdown('<div class="page-title">Verb Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">IPA · PHONETICS · AUDIO · RULE EXPLANATION</div>', unsafe_allow_html=True)

    verb = st.text_input("", placeholder="Type a verb — e.g., walk, think, break (press Enter)").lower().strip()

    if not verb:
        st.markdown('<div class="placeholder-examples">Try: <b>walk</b> &nbsp; · &nbsp; <b>think</b> &nbsp; · &nbsp; <b>break</b> &nbsp; · &nbsp; <b>feel</b> &nbsp; · &nbsp; <b>start</b> &nbsp; · &nbsp; <b>google</b> &nbsp; · &nbsp; <b>zoom</b></div>', unsafe_allow_html=True)
    else:
        label, row = find_verb_any_form(verb)
        if row is None:
            # ML fallback
            st.markdown('<div class="section-label">Not in dataset — ML Prediction</div>', unsafe_allow_html=True)
            row_df = pd.DataFrame([{'Base': verb}])
            features = extract_features(row_df)
            prob = model.predict_proba(features)[0]
            label = 'Irregular' if prob[1] > 0.5 else 'Regular'
            conf = max(prob) * 100
            st.markdown(f"""
            <div class="verb-card">
              <div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#8ECFB0;">Prediction: {label} — {conf:.1f}%</div>
              <div style="font-family:DM Mono,monospace;color:#9FB6C6;margin-top:8px;">This is a model prediction — check a dictionary for exact forms.</div>
            </div>
            """, unsafe_allow_html=True)
            # one-button play for the input token
            render_play_buttons(verb, None, None)
        else:
            st.markdown(f'<div class="section-label">{label} Verb</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form",       row['Base'])
            c2.metric("Simple Past",     row['Simple_Past'])
            c3.metric("Past Participle", row['Past_Participle'])

            st.markdown('<div class="section-label">Audio Pronunciation</div>', unsafe_allow_html=True)
            render_play_buttons(row['Base'], row['Simple_Past'], row['Past_Participle'])

            st.markdown('<div class="section-label">IPA · Phonetic</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#7A8796;margin-bottom:0.4rem;">Base</div>
                <div><span class="ipa-text">{row.get('IPA_Base','')}</span></div>
                <div style="font-family:DM Mono,monospace;font-size:0.82rem;color:#C8CDD8;margin-top:0.6rem;">{row.get('Phonetic_Base','')}</div>
              </div>
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#7A8796;margin-bottom:0.4rem;">Simple Past</div>
                <div><span class="ipa-text">{row.get('IPA_Past','')}</span></div>
                <div style="font-family:DM Mono,monospace;font-size:0.82rem;color:#C8CDD8;margin-top:0.6rem;">{row.get('Phonetic_Past','')}</div>
              </div>
              <div class="verb-card" style="flex:1;min-width:150px;">
                <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#7A8796;margin-bottom:0.4rem;">Past Participle</div>
                <div><span class="ipa-text">{row.get('IPA_PP','')}</span></div>
                <div style="font-family:DM Mono,monospace;font-size:0.82rem;color:#C8CDD8;margin-top:0.6rem;">{row.get('Phonetic_PP','')}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Rule explanation
            if label == 'Regular':
                ending = row.get('Ending','')
                sound  = row.get('Last_Sound','—')
                if ending == '/t/':
                    badge = '<span class="rule-badge badge-t">/t/</span>'
                    rule  = "Voiceless consonant → -ed pronounced /t/ (no extra syllable)."
                elif ending == '/d/':
                    badge = '<span class="rule-badge badge-d">/d/</span>'
                    rule  = "Voiced sound → -ed pronounced /d/."
                else:
                    badge = '<span class="rule-badge badge-id">/ɪd/</span>'
                    rule  = "Ends in /t/ or /d/ → -ed pronounced as extra syllable /ɪd/."
                st.markdown(f"<div class='verb-card'>{badge} <span style='color:#7A8796;margin-left:8px'>last sound: {sound}</span><div style='color:#9FB6C6;margin-top:8px'>{rule}</div></div>", unsafe_allow_html=True)
            else:
                vc = row.get('Vowel_Change','—')
                siblings = df_irreg[(df_irreg['Vowel_Change']==vc) & (df_irreg['Base']!=row['Base'])]['Base'].tolist()[:8]
                siblings_str = ", ".join(siblings) if siblings else "—"
                ex = PATTERN_EXAMPLES.get(vc, "")
                st.markdown(f"<div class='verb-card'><span class='rule-badge badge-irreg'>{vc}</span><div style='color:#9FB6C6;margin-top:8px'>{ex}</div><div style='color:#7A8796;margin-top:8px'>Same pattern: {siblings_str}</div></div>", unsafe_allow_html=True)

            # Example sentences
            st.markdown('<div class="section-label">Example Sentences</div>', unsafe_allow_html=True)
            base = row.get('Base','')
            past = row.get('Simple_Past','')
            part = row.get('Past_Participle','')
            ex_html = f"""
              <div style="font-family:DM Mono,monospace;color:#C8CDD8;line-height:1.8;">
                <div>• Present simple: I often <b>{base}</b> in the morning.</div>
                <div>• Simple past: Yesterday I <b>{past}</b>.</div>
                <div>• Present perfect: I have <b>{part}</b> many times.</div>
              </div>
            """
            st.markdown(ex_html, unsafe_allow_html=True)

# -------------------------
# PAGE: Phonetic Explorer
# -------------------------
elif page == "/ Phonetic Explorer":
    st.markdown('<div class="page-title">Phonetic Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Regular -ed rules & irregular vowel patterns</div>', unsafe_allow_html=True)

    tab_reg, tab_irreg = st.tabs(["Regular — -ed Rule", "Irregular — Vowel Patterns"])

    with tab_reg:
        endings = df_reg['Ending'].value_counts()
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-block">
            <div class="stat-num" style="color:#8ECFB0;">{endings.get('/t/', 0)}</div>
            <div style="margin:0.3rem 0;"><span class="rule-badge badge-t">/t/</span></div>
            <div class="stat-label">voiceless consonant</div>
          </div>
          <div class="stat-block">
            <div class="stat-num" style="color:#7EB8D4;">{endings.get('/d/', 0)}</div>
            <div style="margin:0.3rem 0;"><span class="rule-badge badge-d">/d/</span></div>
            <div class="stat-label">voiced sound</div>
          </div>
          <div class="stat-block">
            <div class="stat-num" style="color:#D4B87A;">{endings.get('/ɪd/', 0)}</div>
            <div style="margin:0.3rem 0;"><span class="rule-badge badge-id">/ɪd/</span></div>
            <div class="stat-label">extra syllable</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Filter</div>', unsafe_allow_html=True)
        ending_filter = st.selectbox("", ["/t/ — voiceless","/d/ — voiced","/ɪd/ — extra syllable"], label_visibility="collapsed")
        ending_code = ending_filter.split(" ")[0]
        filtered = df_reg[df_reg['Ending']==ending_code][
            ['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Phonetic_Base','Phonetic_Past','Phonetic_PP','Last_Sound','Ending']
        ].reset_index(drop=True)
        st.markdown(f'<div class="section-label">{len(filtered)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(filtered, use_container_width=True, height=420)

    with tab_irreg:
        st.markdown("""
        <div style="font-family:DM Mono,monospace;font-size:0.82rem;color:#9FB6C6;line-height:1.7;margin-bottom:8px;">
          Irregular verbs form their past tense through internal vowel changes, not by adding -ed.
        </div>
        """, unsafe_allow_html=True)

        vc_counts = df_irreg['Vowel_Change'].value_counts()
        cards_html = '<div class="pattern-grid">'
        for pattern, count in vc_counts.items():
            ex = PATTERN_EXAMPLES.get(pattern, "")
            short_ex = ex.split("·")[0].strip() if ex else ""
            cards_html += f"""
            <div class="pattern-card" title="Filter by this pattern">
              <div class="pattern-card-symbol">{pattern}</div>
              <div class="pattern-card-count">{count}</div>
              <div class="pattern-card-label">verbs</div>
              <div class="pattern-card-ex">{short_ex}</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.0rem;">Filter by Pattern</div>', unsafe_allow_html=True)
        pattern_options = ["All patterns"] + vc_counts.index.tolist()
        selected = st.selectbox("", pattern_options, label_visibility="collapsed", key="explorer_irreg")

        if selected == "All patterns":
            df_show = df_irreg[['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Phonetic_Base','Phonetic_Past','Phonetic_PP','Vowel_Change']].copy()
        else:
            df_show = df_irreg[df_irreg['Vowel_Change']==selected][
                ['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Phonetic_Base','Phonetic_Past','Phonetic_PP','Vowel_Change']
            ].copy()
            if selected in PATTERN_EXAMPLES:
                st.markdown(f"<div style='font-family:DM Mono,monospace;color:#9FB6C6;margin-bottom:8px'>{PATTERN_EXAMPLES[selected]}</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="section-label">{len(df_show)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=420)

# -------------------------
# PAGE: Charts & Analysis (with tiny play lists)
# -------------------------
elif page == "/ Charts & Analysis":
    st.markdown('<div class="page-title">Charts & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Patterns in English verbs (phonetic columns included)</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview","-ed Endings","Irregular Patterns","Verb Length"])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        style_ax(ax, fig)
        bars = ax.bar(['Regular','Irregular'], [len(df_reg),len(df_irreg)],
                      color=[ACCENT1,ACCENT2], width=0.4, edgecolor=CHART_BG, linewidth=2)
        for bar, val in zip(bars, [len(df_reg),len(df_irreg)]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(val), ha='center', fontweight='bold', fontsize=12, color='#C8CDD8')
        ax.set_title('Dataset Composition', fontsize=10, pad=12)
        ax.set_ylim(0, max(len(df_reg),len(df_irreg))*1.18)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(bottom=False)
        plt.tight_layout()
        st.pyplot(fig)

        # small audio lists
        sample_irreg = df_irreg.head(6).to_dict('records')
        sample_reg   = df_reg.head(6).to_dict('records')
        st.markdown('<div style="display:flex;gap:2rem;margin-top:8px;">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='font-family:DM Mono,monospace;color:#C8CDD8;font-weight:700;margin-bottom:6px;'>Sample regular</div>", unsafe_allow_html=True)
            render_play_list(sample_reg)
        with col2:
            st.markdown("<div style='font-family:DM Mono,monospace;color:#C8CDD8;font-weight:700;margin-bottom:6px;'>Sample irregular</div>", unsafe_allow_html=True)
            render_play_list(sample_irreg)

    with tab2:
        ec = df_reg['Ending'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax in axes: style_ax(ax, fig)
        axes[0].bar(ec.index, ec.values, color=[ACCENT4,ACCENT1,ACCENT3], edgecolor=CHART_BG, linewidth=2, width=0.4)
        for i, (idx, val) in enumerate(ec.items()):
            axes[0].text(i, val+1, str(val), ha='center', fontweight='bold', fontsize=11, color='#C8CDD8')
        axes[0].set_title('-ed Ending Count', fontsize=10, pad=10)
        for sp in axes[0].spines.values(): sp.set_visible(False)
        axes[0].tick_params(bottom=False)
        axes[1].pie(ec.values, labels=[f'{i}  ({v})' for i,v in ec.items()], colors=[ACCENT4,ACCENT1,ACCENT3], autopct='%1.0f%%', startangle=90, wedgeprops={'edgecolor':CHART_BG,'linewidth':3}, textprops={'color':CHART_TEXT,'fontsize':9,'fontfamily':'monospace'})
        axes[1].set_title('-ed Share', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div class="section-label" style="margin-top:0.6rem;">Top phonetic spellings (regular)</div>', unsafe_allow_html=True)
        st.dataframe(phonetic_summary(df_reg)['Phonetic_Base'].rename(columns={'phonetic':'phonetic','Phonetic_Base':'count'}).head(6), use_container_width=True, height=150)

        st.markdown('<div style="margin-top:8px;"><b style="color:#C8CDD8">Play examples from -ed group</b></div>', unsafe_allow_html=True)
        # random sample of regular verbs for playing
        top_reg = df_reg.sample(min(6, len(df_reg))).to_dict('records')
        render_play_list(top_reg)

    with tab3:
        vc = df_irreg['Vowel_Change'].value_counts().head(13)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style_ax(ax, fig)
        colors = [ACCENT2 if i==0 else ACCENT1 if i<4 else ACCENT3 for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1], edgecolor=CHART_BG, linewidth=2, height=0.6)
        for bar, val in zip(ax.patches, vc.values[::-1]):
            ax.text(bar.get_width()+0.08, bar.get_y()+bar.get_height()/2, str(val), va='center', fontsize=9, fontweight='bold', color='#C8CDD8')
        ax.set_title('Most Common Irregular Patterns', fontsize=10, pad=12)
        ax.set_xlim(0, vc.max()+5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticklabels(vc.index[::-1], fontfamily='monospace', fontsize=8.5, color=CHART_TEXT)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div class="section-label" style="margin-top:0.6rem;">Phonetic examples for irregulars</div>', unsafe_allow_html=True)
        st.dataframe(phonetic_summary(df_irreg)['Phonetic_Base'].rename(columns={'phonetic':'phonetic','Phonetic_Base':'count'}).head(8), use_container_width=True, height=150)

        # audio list for top irregular patterns (one example verb per pattern)
        examples = []
        for pat in vc.index[:6]:
            match = df_irreg[df_irreg['Vowel_Change']==pat].head(1)
            if not match.empty:
                examples.append(match.iloc[0][['Base','Simple_Past','Past_Participle']].to_dict())
        if examples:
            st.markdown('<div style="margin-top:8px;"><b style="color:#C8CDD8">Play sample verbs for top irregular patterns</b></div>', unsafe_allow_html=True)
            render_play_list(examples)

    with tab4:
        df_reg['length']   = df_reg['Base'].str.len()
        df_irreg['length'] = df_irreg['Base'].str.len()
        fig, ax = plt.subplots(figsize=(10, 4))
        style_ax(ax, fig)
        ax.hist(df_reg['length'],   bins=range(2,15), alpha=0.75, color=ACCENT1, label='Regular',   edgecolor=CHART_BG, linewidth=1.5)
        ax.hist(df_irreg['length'], bins=range(2,15), alpha=0.75, color=ACCENT2, label='Irregular', edgecolor=CHART_BG, linewidth=1.5)
        ax.axvline(df_reg['length'].mean(),   color=ACCENT1, linestyle='--', linewidth=1.5, label=f"Reg avg: {df_reg['length'].mean():.1f}")
        ax.axvline(df_irreg['length'].mean(), color=ACCENT2, linestyle='--', linewidth=1.5, label=f"Irreg avg: {df_irreg['length'].mean():.1f}")
        ax.set_title('Verb Length Distribution', fontsize=10, pad=12)
        ax.set_xlabel('Characters', fontsize=9)
        ax.legend(fontsize=8, framealpha=0, labelcolor=CHART_TEXT)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#4A5568;margin-top:0.5rem;">Irregular verbs tend to be shorter — they come from Old English. New coinages are usually regular.</div>', unsafe_allow_html=True)

# -------------------------
# PAGE: Verb Reference
# -------------------------
elif page == "/ Verb Reference":
    st.markdown('<div class="page-title">Verb Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Complete searchable table — phonetic columns included</div>', unsafe_allow_html=True)

    tab_r, tab_i = st.tabs(["Regular Verbs","Irregular Verbs"])

    with tab_r:
        search = st.text_input("", placeholder="Search regular verbs (base / phonetic / past) — press Enter", key="s_reg")
        df_show = df_reg.copy()
        if search:
            mask = df_show[['Base','Phonetic_Base','Phonetic_Past','Simple_Past','Past_Participle']].apply(lambda row: row.astype(str).str.lower().str.contains(search.lower()).any(), axis=1)
            df_show = df_show[mask]
        st.markdown(f'<div class="section-label">{len(df_show)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show[['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Phonetic_Base','Phonetic_Past','Phonetic_PP','Last_Sound','Ending']].reset_index(drop=True), use_container_width=True, height=520)

    with tab_i:
        cs, cf = st.columns([2,1])
        with cs:
            search2 = st.text_input("", placeholder="Search irregular verbs (base / pattern / phonetic / past)", key="s_irreg")
        with cf:
            vc_opts = ["All patterns"] + df_irreg['Vowel_Change'].value_counts().index.tolist()
            pat_filter = st.selectbox("", vc_opts, key="ref_irreg_filter", label_visibility="collapsed")
        df_show2 = df_irreg.copy()
        if search2:
            mask = df_show2[['Base','Phonetic_Base','Phonetic_Past','Simple_Past','Past_Participle']].apply(lambda row: row.astype(str).str.lower().str.contains(search2.lower()).any(), axis=1)
            df_show2 = df_show2[mask]
        if pat_filter != "All patterns":
            df_show2 = df_show2[df_show2['Vowel_Change']==pat_filter]
        st.markdown(f'<div class="section-label">{len(df_show2)} verbs</div>', unsafe_allow_html=True)
        st.dataframe(df_show2[['Base','Simple_Past','Past_Participle','IPA_Base','IPA_Past','Phonetic_Base','Phonetic_Past','Phonetic_PP','Vowel_Change']].reset_index(drop=True), use_container_width=True, height=520)

# End of file
