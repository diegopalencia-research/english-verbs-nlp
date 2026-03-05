"""
lemmatizer.py
Searches for a verb in any conjugated form across all datasets.
If a user types 'fought', 'went', or 'bought' — this finds the base entry.
Now also searches Participial Adjectives sheet.
"""


def find_verb(query: str, df_reg, df_irreg, df_part=None):
    """
    Search for any verb form across base, simple past, and past participle.
    Also checks participial adjectives sheet if provided.

    Returns:
        tuple: (row, verb_type, matched_form) or (None, None, None)
        verb_type: 'Regular' | 'Irregular' | 'Participial Adjective'
        matched_form: 'base form' | 'simple past' | 'past participle' |
                      'participial form'
    """
    q = query.lower().strip()

    # ── Regular verbs ─────────────────────────────────────────────────────
    col_map = {
        'Base':            'base form',
        'Simple_Past':     'simple past',
        'Past_Participle': 'past participle',
    }
    for col, label in col_map.items():
        match = df_reg[df_reg[col].str.lower() == q]
        if not match.empty:
            return match.iloc[0], 'Regular', label

    # ── Irregular verbs ───────────────────────────────────────────────────
    for col, label in col_map.items():
        match = df_irreg[df_irreg[col].str.lower() == q]
        if not match.empty:
            return match.iloc[0], 'Irregular', label

    # ── Participial Adjectives ─────────────────────────────────────────────
    if df_part is not None:
        # Search by participial form first (e.g. "broken", "excited")
        match = df_part[df_part['Participial_Form'].str.lower() == q]
        if not match.empty:
            return match.iloc[0], 'Participial Adjective', 'participial form'
        # Then by base verb (e.g. "break", "excite")
        match = df_part[df_part['Base_Verb'].str.lower() == q]
        if not match.empty:
            return match.iloc[0], 'Participial Adjective', 'base verb'

    return None, None, None


def suggest_verbs(query: str, df_reg, df_irreg, n: int = 5, df_part=None):
    """
    Return up to n verb base forms that start with the query string.
    Used for search suggestions. Now includes participial adjectives.
    """
    q = query.lower().strip()
    seen = set()
    results = []

    for col in ['Base', 'Simple_Past', 'Past_Participle']:
        for df in [df_reg, df_irreg]:
            for base in df[df[col].str.lower().str.startswith(q)]['Base'].tolist():
                if base not in seen:
                    seen.add(base)
                    results.append(base)
                if len(results) >= n:
                    return results

    if df_part is not None:
        for col in ['Base_Verb', 'Participial_Form']:
            for val in df_part[df_part[col].str.lower().str.startswith(q)][col].tolist():
                if val not in seen:
                    seen.add(val)
                    results.append(val)
                if len(results) >= n:
                    return results

    return results
    
