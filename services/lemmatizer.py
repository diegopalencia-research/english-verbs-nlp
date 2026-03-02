"""
lemmatizer.py
Searches for a verb in any conjugated form across both datasets.
If a user types 'fought', 'went', or 'bought' — this finds the base entry.
"""


def find_verb(query: str, df_reg, df_irreg):
    """
    Search for any verb form across base, simple past, and past participle.

    Returns:
        tuple: (row, verb_type, matched_form) or (None, None, None)
    """
    q = query.lower().strip()

    col_map = {
        'Base':             'base form',
        'Simple_Past':      'simple past',
        'Past_Participle':  'past participle',
    }

    for col, label in col_map.items():
        match = df_reg[df_reg[col].str.lower() == q]
        if not match.empty:
            return match.iloc[0], 'Regular', label

    for col, label in col_map.items():
        match = df_irreg[df_irreg[col].str.lower() == q]
        if not match.empty:
            return match.iloc[0], 'Irregular', label

    return None, None, None


def suggest_verbs(query: str, df_reg, df_irreg, n: int = 5):
    """
    Return up to n verb base forms that start with the query string.
    Used for search suggestions.
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

    return results
