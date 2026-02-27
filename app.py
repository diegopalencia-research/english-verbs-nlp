import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="English Verb Phonetics",
    page_icon="📖",
    layout="wide"
)

# ── Load and prepare data ─────────────────────────────────────
@st.cache_data
def load_data():
    FILE = 'data/english_verbs.xlsx'

    df_reg   = pd.read_excel(FILE, sheet_name='Regular Verbs',   header=2)
    df_irreg = pd.read_excel(FILE, sheet_name='Irregular Verbs', header=2)

    reg_cols = ['Base','Simple_Past','Past_Participle',
                'IPA_Base','IPA_Past','IPA_PP',
                'Phonetic_Base','Phonetic_Past','Phonetic_PP',
                'Last_Sound','Ending']
    irreg_cols = ['Base','Simple_Past','Past_Participle',
                  'IPA_Base','IPA_Past','IPA_PP',
                  'Phonetic_Base','Phonetic_Past','Phonetic_PP',
                  'Vowel_Change']

    df_reg.columns   = reg_cols
    df_irreg.columns = irreg_cols

    df_reg   = df_reg.dropna(subset=['Base']).reset_index(drop=True)
    df_irreg = df_irreg.dropna(subset=['Base']).reset_index(drop=True)

    df_reg['Type']   = 'Regular'
    df_irreg['Type'] = 'Irregular'

    return df_reg, df_irreg

# ── Feature engineering ───────────────────────────────────────
def extract_features(df):
    features = pd.DataFrame()
    features['length']          = df['Base'].str.len()
    features['vowel_count']     = df['Base'].str.count('[aeiou]')
    features['consonant_count'] = df['Base'].str.count('[bcdfghjklmnpqrstvwxyz]')
    for suffix in ['e','n','d','t','l','r','k','g','w','y',
                   'ng','nd','ld','nt','in','ow','aw']:
        features[f'ends_{suffix}'] = df['Base'].str.endswith(suffix).astype(int)
    le = LabelEncoder()
    features['last_letter'] = le.fit_transform(df['Base'].str[-1].fillna('_'))
    features['second_last'] = le.fit_transform(df['Base'].str[-2].fillna('_'))
    return features

# ── Train model ───────────────────────────────────────────────
@st.cache_resource
def train_model(df_reg, df_irreg):
    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)
    X = extract_features(df_all)
    y = (df_all['Type'] == 'Irregular').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    rf.fit(X_train, y_train)
    return rf

# ── Load everything ───────────────────────────────────────────
df_reg, df_irreg = load_data()
model            = train_model(df_reg, df_irreg)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Verb Lookup",
    "Phonetic Rule Explorer",
    "Charts & Analysis",
    "Full Verb Table"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Regular verbs:** {len(df_reg)}")
st.sidebar.markdown(f"**Irregular verbs:** {len(df_irreg)}")
st.sidebar.markdown(f"**Total:** {len(df_reg)+len(df_irreg)}")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [diegopalencia-research](https://github.com/diegopalencia-research)")

# ─────────────────────────────────────────────────────────────
# PAGE 1 — VERB LOOKUP
# ─────────────────────────────────────────────────────────────
if page == "Verb Lookup":
    st.title("English Verb Phonetics")
    st.markdown("Type any English verb to see its full phonetic analysis and past tense prediction.")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        verb = st.text_input("Enter a verb (base form):",
                             placeholder="e.g. walk, think, google...").lower().strip()

    if verb:
        # Check if it's in our dataset
        reg_match   = df_reg[df_reg['Base'].str.lower() == verb]
        irreg_match = df_irreg[df_irreg['Base'].str.lower() == verb]

        if not reg_match.empty:
            row = reg_match.iloc[0]
            st.success(f"Found in dataset — **Regular Verb**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form",       row['Base'])
            c2.metric("Simple Past",     row['Simple_Past'])
            c3.metric("Past Participle", row['Past_Participle'])

            st.markdown("#### Pronunciation")
            c4, c5, c6 = st.columns(3)
            c4.markdown(f"**IPA Base:** `{row['IPA_Base']}`")
            c5.markdown(f"**IPA Past:** `{row['IPA_Past']}`")
            c6.markdown(f"**IPA PP:**   `{row['IPA_PP']}`")

            c7, c8, c9 = st.columns(3)
            c7.markdown(f"**Sound:** {row['Phonetic_Base']}")
            c8.markdown(f"**Sound:** {row['Phonetic_Past']}")
            c9.markdown(f"**Sound:** {row['Phonetic_PP']}")

            st.markdown("#### Phonetic Rule")
            ending = row['Ending']
            sound  = row['Last_Sound']

            if ending == '/t/':
                st.info(f"**Last sound:** {sound}  \n"
                        f"**Rule:** Voiceless consonant → -ed is pronounced **/t/**  \n"
                        f"**Example:** walk → walk**t**")
            elif ending == '/d/':
                st.info(f"**Last sound:** {sound}  \n"
                        f"**Rule:** Voiced sound → -ed is pronounced **/d/**  \n"
                        f"**Example:** call → call**d**")
            elif ending == '/ɪd/':
                st.warning(f"**Last sound:** {sound}  \n"
                           f"**Rule:** Ends in /t/ or /d/ → -ed adds **extra syllable /ɪd/**  \n"
                           f"**Example:** start → start**id**")

        elif not irreg_match.empty:
            row = irreg_match.iloc[0]
            st.error(f"Found in dataset — **Irregular Verb**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Base Form",       row['Base'])
            c2.metric("Simple Past",     row['Simple_Past'])
            c3.metric("Past Participle", row['Past_Participle'])

            st.markdown("#### Pronunciation")
            c4, c5, c6 = st.columns(3)
            c4.markdown(f"**IPA Base:** `{row['IPA_Base']}`")
            c5.markdown(f"**IPA Past:** `{row['IPA_Past']}`")
            c6.markdown(f"**IPA PP:**   `{row['IPA_PP']}`")

            st.markdown("#### Vowel Change Pattern")
            st.warning(f"**Pattern:** {row['Vowel_Change']}  \n"
                       f"This verb does **not** follow the -ed rule. "
                       f"It changes its internal vowel to form the past tense.")

        else:
            st.markdown("---")
            st.markdown(f"**'{verb}'** is not in the dataset — predicting with ML model...")

            row      = pd.DataFrame([{'Base': verb}])
            features = extract_features(row)
            prob     = model.predict_proba(features)[0]
            label    = 'Irregular' if prob[1] > 0.5 else 'Regular'
            conf     = max(prob) * 100

            if label == 'Regular':
                st.success(f"Prediction: **Regular** ({conf:.1f}% confidence)")
                last = verb[-1]
                if last in 'td':
                    rule = "ends in /t/ or /d/ → past tense adds extra syllable, pronounced **/ɪd/**"
                elif last in 'pkfscx':
                    rule = "ends in voiceless consonant → past tense pronounced **/t/**"
                else:
                    rule = "ends in voiced sound → past tense pronounced **/d/**"
                st.info(f"**Predicted past tense:** {verb}ed  \n**Phonetic rule:** {rule}")
            else:
                st.error(f"Prediction: **Irregular** ({conf:.1f}% confidence)")
                st.warning("This verb likely has an unpredictable past form. Check a dictionary.")

# ─────────────────────────────────────────────────────────────
# PAGE 2 — PHONETIC RULE EXPLORER
# ─────────────────────────────────────────────────────────────
elif page == "Phonetic Rule Explorer":
    st.title("The Phonetic Rule for -ed")
    st.markdown("---")

    st.markdown("""
    When a regular English verb takes **-ed**, the pronunciation depends
    on the **last sound** of the base form — not the last letter.

    | Last Sound | Pronunciation | Reason |
    |---|---|---|
    | Voiceless: /p/ /k/ /f/ /s/ /ʃ/ /tʃ/ | **/t/** | Easier to say after voiceless |
    | Voiced: vowels, /b/ /g/ /v/ /z/ /m/ /n/ /l/ /r/ | **/d/** | Easier to say after voiced |
    | /t/ or /d/ | **/ɪd/** | Need extra syllable to separate sounds |
    """)

    st.markdown("---")
    ending_filter = st.selectbox(
        "Show me all verbs with this ending:",
        ["/t/ — voiceless", "/d/ — voiced", "/ɪd/ — extra syllable"]
    )

    ending_code = ending_filter.split(" ")[0]
    filtered = df_reg[df_reg['Ending'] == ending_code][
        ['Base','Simple_Past','IPA_Base','IPA_Past','Last_Sound','Ending']
    ]

    st.markdown(f"**{len(filtered)} verbs** with `{ending_code}` ending:")
    st.dataframe(filtered, use_container_width=True, height=400)

# ─────────────────────────────────────────────────────────────
# PAGE 3 — CHARTS
# ─────────────────────────────────────────────────────────────
elif page == "Charts & Analysis":
    st.title("Charts & Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Distribution", "Ending Sounds", "Vowel Patterns"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = {'Regular': len(df_reg), 'Irregular': len(df_irreg)}
        bars = ax.bar(counts.keys(), counts.values(),
                      color=['#1D4E5A','#4A1C2A'],
                      width=0.5, edgecolor='white')
        for bar, val in zip(bars, counts.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(val), ha='center', fontweight='bold', fontsize=13)
        ax.set_title('Regular vs Irregular Verbs', fontweight='bold')
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.info("English has more regular verbs, but the most-used verbs (be, have, go, do) are all irregular.")

    with tab2:
        ending_counts = df_reg['Ending'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(ending_counts.index, ending_counts.values,
                    color=['#2E7D52','#1A3A7A','#8B6914'], edgecolor='white')
        for i, (idx, val) in enumerate(ending_counts.items()):
            axes[0].text(i, val+1, str(val), ha='center', fontweight='bold')
        axes[0].set_title('-ed Pronunciation Count', fontweight='bold')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].pie(ending_counts.values,
                    labels=[f'{i}\n({v})' for i,v in ending_counts.items()],
                    colors=['#2E7D52','#1A3A7A','#8B6914'],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor':'white','linewidth':2})
        axes[1].set_title('-ed Pronunciation Share', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        vc = df_irreg['Vowel_Change'].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette('rocket_r', len(vc))
        ax.barh(vc.index[::-1], vc.values[::-1],
                color=palette[::-1], edgecolor='white')
        for bar, val in zip(ax.patches, vc.values[::-1]):
            ax.text(bar.get_width()+0.1,
                    bar.get_y()+bar.get_height()/2,
                    str(val), va='center', fontweight='bold')
        ax.set_title('Most Common Irregular Patterns', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.info("The most common pattern is iː→ɛ: feel/felt, keep/kept, sleep/slept, meet/met")

# ─────────────────────────────────────────────────────────────
# PAGE 4 — FULL TABLE
# ─────────────────────────────────────────────────────────────
elif page == "Full Verb Table":
    st.title("Complete Verb Reference")
    st.markdown("---")

    tab_r, tab_i = st.tabs(["Regular Verbs", "Irregular Verbs"])

    with tab_r:
        search = st.text_input("Search regular verbs:", placeholder="Type any verb...")
        df_show = df_reg.copy()
        if search:
            df_show = df_show[df_show['Base'].str.contains(search.lower(), na=False)]
        st.markdown(f"Showing **{len(df_show)}** verbs")
        st.dataframe(
            df_show[['Base','Simple_Past','Past_Participle',
                     'IPA_Base','IPA_Past','Last_Sound','Ending']],
            use_container_width=True, height=500
        )

    with tab_i:
        search2 = st.text_input("Search irregular verbs:", placeholder="Type any verb...")
        df_show2 = df_irreg.copy()
        if search2:
            df_show2 = df_show2[df_show2['Base'].str.contains(search2.lower(), na=False)]
        st.markdown(f"Showing **{len(df_show2)}** verbs")
        st.dataframe(
            df_show2[['Base','Simple_Past','Past_Participle',
                      'IPA_Base','IPA_Past','Vowel_Change']],
            use_container_width=True, height=500
        )
```

Click **Commit new file** at the bottom.

---

### STEP 2 — Add requirements.txt

Make sure your `requirements.txt` in the repo contains exactly this:
```
pandas
openpyxl
matplotlib
seaborn
scikit-learn
streamlit
