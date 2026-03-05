"""
auto_lookup.py
Auto-lookup service for missing verbs.

Flow:
  1. User searches a verb not in the local Excel dataset
  2. This service calls the Free Dictionary API (no API key needed)
  3. Applies phonetic rules from phonetics.py to classify the verb
  4. Saves the result to Supabase (PostgreSQL) automatically
  5. Next search for the same verb is served from Supabase — instant

Setup (one time, ~10 minutes):
  1. Create free account at https://supabase.com
  2. Create new project
  3. Go to SQL Editor → paste and run the schema below
  4. Go to Project Settings → API → copy URL and anon key
  5. Add to Streamlit secrets:
       SUPABASE_URL = "https://xxxxx.supabase.co"
       SUPABASE_KEY = "your-anon-key"
  6. pip install supabase

Supabase SQL schema (run once in Supabase SQL Editor):
─────────────────────────────────────────────────────────
CREATE TABLE auto_verbs (
    id              SERIAL PRIMARY KEY,
    base_verb       TEXT NOT NULL UNIQUE,
    simple_past     TEXT,
    past_participle TEXT,
    ipa_base        TEXT,
    ipa_past        TEXT,
    ipa_pp          TEXT,
    phonetic_base   TEXT,
    phonetic_past   TEXT,
    phonetic_pp     TEXT,
    last_sound      TEXT,
    ending          TEXT,
    verb_type       TEXT DEFAULT 'Regular',
    vowel_change    TEXT,
    confidence      FLOAT,
    source          TEXT DEFAULT 'auto',
    status          TEXT DEFAULT 'auto',
    searched_count  INT DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_searched   TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookup
CREATE INDEX idx_auto_verbs_base ON auto_verbs (LOWER(base_verb));
CREATE INDEX idx_auto_verbs_past ON auto_verbs (LOWER(simple_past));
CREATE INDEX idx_auto_verbs_pp   ON auto_verbs (LOWER(past_participle));

-- Allow public read/write (adjust RLS policy as needed)
ALTER TABLE auto_verbs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "allow_all" ON auto_verbs FOR ALL USING (true) WITH CHECK (true);
─────────────────────────────────────────────────────────
"""

import re
import streamlit as st

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Local imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.phonetics import predict_ending, get_rule_explanation


# ── Supabase client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> "Client | None":
    """Return Supabase client or None if not configured."""
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


# ── Dictionary API ─────────────────────────────────────────────────────────────
DICT_API = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

def fetch_from_dictionary(word: str) -> dict | None:
    """
    Call the Free Dictionary API for a word.
    Returns a simplified dict with IPA and phonetic info, or None if not found.
    No API key required. Rate limit: generous for personal use.
    """
    if not REQUESTS_AVAILABLE:
        return None

    try:
        resp = requests.get(DICT_API.format(word=word), timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data or not isinstance(data, list):
            return None

        entry = data[0]

        # Extract IPA from phonetics list
        ipa_base = ""
        for ph in entry.get("phonetics", []):
            if ph.get("text"):
                ipa_base = ph["text"]
                break

        # Check if it's actually a verb
        is_verb = False
        for meaning in entry.get("meanings", []):
            if meaning.get("partOfSpeech") == "verb":
                is_verb = True
                break

        return {
            "word":     word,
            "ipa_base": ipa_base if ipa_base else f"/{word}/",
            "is_verb":  is_verb,
            "found":    True,
        }

    except Exception:
        return None


# ── Past form generation ───────────────────────────────────────────────────────
def generate_past_forms(base: str) -> tuple[str, str]:
    """
    Generate simple past and past participle for a regular verb.
    Handles the main English spelling rules.
    Returns: (simple_past, past_participle)
    """
    w = base.lower().strip()

    # Already ends in -e: just add -d
    if w.endswith('e'):
        past = w + 'd'

    # Ends in consonant + y: change y → ied
    elif (len(w) >= 2 and w[-1] == 'y'
          and w[-2] not in 'aeiou'):
        past = w[:-1] + 'ied'

    # Single-syllable CVC pattern: double the final consonant
    elif (len(w) >= 3
          and w[-1] not in 'aeiouywx'
          and w[-2] in 'aeiou'
          and w[-3] not in 'aeiou'
          and _count_syllables(w) == 1):
        past = w + w[-1] + 'ed'

    # Multi-syllable ending in stressed CVC (e.g. "admit", "occur"):
    # simplified — just add -ed (handles most common cases)
    else:
        past = w + 'ed'

    return past, past  # regular verbs: past = past participle


def _count_syllables(word: str) -> int:
    word = word.lower()
    count = 0
    vowels = 'aeiou'
    prev = False
    for ch in word:
        v = ch in vowels
        if v and not prev:
            count += 1
        prev = v
    if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
        count = max(1, count - 1)
    return max(1, count)


def build_phonetic(word: str) -> str:
    """
    Generate a simple stress-marked phonetic spelling.
    Approximation — not IPA. Matches existing dataset format.
    Example: "validate" → "VAL-i-dayt"
    """
    # Very simplified — uppercase first syllable
    w = word.upper()
    if len(w) <= 4:
        return w
    # Insert hyphens at vowel boundaries (rough approximation)
    parts = re.sub(r'([AEIOU]{2,})', r'-\1-', w).strip('-')
    return parts


def build_ipa_past(ipa_base: str, ending: str) -> str:
    """Approximate IPA for past form by appending the ending sound."""
    clean = ipa_base.rstrip('/')
    suffix_map = {'/t/': 't/', '/d/': 'd/', '/ɪd/': 'ɪd/'}
    return clean + suffix_map.get(ending, 'd/')


# ── Supabase read ──────────────────────────────────────────────────────────────
def lookup_in_supabase(query: str) -> dict | None:
    """
    Search Supabase for a verb in base, simple_past, or past_participle columns.
    Returns row dict or None.
    """
    sb = get_supabase()
    if sb is None:
        return None

    q = query.lower().strip()
    try:
        # Search all three columns
        for col in ['base_verb', 'simple_past', 'past_participle']:
            resp = (sb.table('auto_verbs')
                      .select('*')
                      .ilike(col, q)
                      .limit(1)
                      .execute())
            if resp.data:
                row = resp.data[0]
                # Increment search counter
                sb.table('auto_verbs').update({
                    'searched_count': row['searched_count'] + 1,
                    'last_searched':  'now()'
                }).eq('id', row['id']).execute()
                return row
    except Exception:
        pass
    return None


# ── Supabase write ─────────────────────────────────────────────────────────────
def save_to_supabase(verb_data: dict) -> bool:
    """Save a new auto-looked-up verb to Supabase. Returns True on success."""
    sb = get_supabase()
    if sb is None:
        return False
    try:
        sb.table('auto_verbs').upsert(verb_data, on_conflict='base_verb').execute()
        return True
    except Exception:
        return False


# ── Main public function ───────────────────────────────────────────────────────
def auto_lookup_verb(query: str, ml_model=None, feature_extractor=None) -> dict | None:
    """
    Complete auto-lookup pipeline for a missing verb.

    Steps:
        1. Check Supabase cache (instant)
        2. Call Free Dictionary API
        3. Apply phonetic rules
        4. Apply ML classifier (optional, for regular/irregular prediction)
        5. Save to Supabase
        6. Return structured result

    Returns a dict with keys matching the app's expected structure,
    or None if the word is not found as a verb in the dictionary.
    """

    # ── Step 1: Check Supabase cache ─────────────────────────────────────────
    cached = lookup_in_supabase(query)
    if cached:
        return {
            'source':       'supabase_cache',
            'base':         cached['base_verb'],
            'simple_past':  cached['simple_past'],
            'past_participle': cached['past_participle'],
            'ipa_base':     cached['ipa_base'],
            'ipa_past':     cached['ipa_past'],
            'ipa_pp':       cached['ipa_pp'],
            'phonetic_base': cached['phonetic_base'],
            'verb_type':    cached['verb_type'],
            'ending':       cached['ending'],
            'last_sound':   cached['last_sound'],
            'confidence':   cached.get('confidence', 0),
            'searched_count': cached['searched_count'],
            'status':       cached['status'],
        }

    # ── Step 2: Call Dictionary API ───────────────────────────────────────────
    api_result = fetch_from_dictionary(query)
    if api_result is None or not api_result.get('is_verb'):
        return None  # Not found or not a verb

    base     = query.lower().strip()
    ipa_base = api_result.get('ipa_base', f'/{base}/')

    # ── Step 3: Phonetic rule engine ──────────────────────────────────────────
    ending    = predict_ending(base)
    past, pp  = generate_past_forms(base)
    ipa_past  = build_ipa_past(ipa_base, ending)
    phonetic  = build_phonetic(base)

    last_sound_map = {
        '/t/':  f'voiceless /{base[-1]}/',
        '/d/':  f'voiced /{base[-1]}/',
        '/ɪd/': f'voiceless /t/' if base[-1] == 't' else f'voiced /d/',
    }
    last_sound = last_sound_map.get(ending, f'/{base[-1]}/')

    # ── Step 4: ML classifier (optional) ─────────────────────────────────────
    verb_type  = 'Regular'
    confidence = 0.0

    if ml_model is not None and feature_extractor is not None:
        try:
            import pandas as pd
            row_df   = pd.DataFrame([{'Base': base}])
            features = feature_extractor(row_df)
            prob     = ml_model.predict_proba(features)[0]
            verb_type  = 'Irregular' if prob[1] > 0.5 else 'Regular'
            confidence = round(float(max(prob)) * 100, 1)
        except Exception:
            verb_type  = 'Regular'
            confidence = 70.0

    # ── Step 5: Save to Supabase ──────────────────────────────────────────────
    verb_data = {
        'base_verb':       base,
        'simple_past':     past,
        'past_participle': pp,
        'ipa_base':        ipa_base,
        'ipa_past':        ipa_past,
        'ipa_pp':          ipa_past,
        'phonetic_base':   phonetic,
        'phonetic_past':   phonetic.upper() + 'D' if ending == '/d/' else phonetic + 'T',
        'phonetic_pp':     phonetic.upper() + 'D' if ending == '/d/' else phonetic + 'T',
        'last_sound':      last_sound,
        'ending':          ending,
        'verb_type':       verb_type,
        'vowel_change':    None,
        'confidence':      confidence,
        'source':          'dictionary_api',
        'status':          'auto',
        'searched_count':  1,
    }

    saved = save_to_supabase(verb_data)

    return {
        'source':          'dictionary_api' if saved else 'local_only',
        'base':            base,
        'simple_past':     past,
        'past_participle': pp,
        'ipa_base':        ipa_base,
        'ipa_past':        ipa_past,
        'ipa_pp':          ipa_past,
        'phonetic_base':   phonetic,
        'verb_type':       verb_type,
        'ending':          ending,
        'last_sound':      last_sound,
        'confidence':      confidence,
        'searched_count':  1,
        'status':          'auto',
        'saved':           saved,
    }


# ── Admin helpers ──────────────────────────────────────────────────────────────
def get_pending_review(limit: int = 50) -> list[dict]:
    """Return verbs auto-added but not yet manually confirmed."""
    sb = get_supabase()
    if sb is None:
        return []
    try:
        resp = (sb.table('auto_verbs')
                  .select('*')
                  .eq('status', 'auto')
                  .order('searched_count', desc=True)
                  .limit(limit)
                  .execute())
        return resp.data or []
    except Exception:
        return []


def get_supabase_stats() -> dict:
    """Return counts for the sidebar display."""
    sb = get_supabase()
    if sb is None:
        return {'total': 0, 'auto': 0, 'confirmed': 0, 'available': False}
    try:
        resp = sb.table('auto_verbs').select('status').execute()
        rows = resp.data or []
        total     = len(rows)
        auto      = sum(1 for r in rows if r['status'] == 'auto')
        confirmed = sum(1 for r in rows if r['status'] == 'confirmed')
        return {'total': total, 'auto': auto, 'confirmed': confirmed, 'available': True}
    except Exception:
        return {'total': 0, 'auto': 0, 'confirmed': 0, 'available': False}
