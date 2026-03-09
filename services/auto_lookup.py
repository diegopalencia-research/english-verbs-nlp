"""
auto_lookup.py
──────────────
Supabase-backed verb lookup service.

Flow for any verb NOT in the local Excel dataset:
  1. Check Supabase cache (search_logs + pending_verbs)
  2. If not cached → call dictionaryapi.dev (free, no key needed)
  3. Apply phonetic rule engine  → predict -ed ending
  4. Run ML classifier           → predict Regular / Irregular
  5. Save result to Supabase     → next search is instant
  6. Log every search            → analytics / Community Trends

Environment:
  Streamlit secrets  (.streamlit/secrets.toml  locally,
                      Streamlit Cloud → App Settings → Secrets)

  [supabase]
  url = "https://xxxxxxxxxxxx.supabase.co"
  key = "your-anon-public-key"
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ── Supabase client (cached so it's created once per session) ─────────────────

@st.cache_resource
def get_supabase() -> Optional[Client]:
    """
    Returns a Supabase client or None if credentials are missing.
    Failing silently means the app still works without Supabase —
    it just falls back to the ML prediction path.
    """
    try:
        url: str = st.secrets["supabase"]["url"]
        key: str = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        logger.warning("Supabase credentials not found — running without database.")
        return None


# ── Search logging ────────────────────────────────────────────────────────────

def log_search(
    verb: str,
    found: bool,
    verb_type: Optional[str] = None,
    matched_as: Optional[str] = None,
) -> None:
    """
    Insert one row into search_logs for every lookup.
    Non-blocking: exceptions are caught and logged, never raised.
    """
    client = get_supabase()
    if client is None:
        return
    try:
        client.table("search_logs").insert({
            "verb":       verb.lower().strip(),
            "found":      found,
            "verb_type":  verb_type,
            "matched_as": matched_as,
        }).execute()
    except Exception as e:
        logger.warning("search_logs insert failed: %s", e)


# ── Pending verb cache ────────────────────────────────────────────────────────

def get_cached_verb(verb: str) -> Optional[dict]:
    """
    Check if we already looked up this verb before.
    Returns the row dict or None.
    """
    client = get_supabase()
    if client is None:
        return None
    try:
        result = (
            client.table("pending_verbs")
            .select("*")
            .eq("verb", verb.lower().strip())
            .limit(1)
            .execute()
        )
        if result.data:
            # Bump search_count so we can track popularity
            row = result.data[0]
            client.table("pending_verbs").update({
                "search_count": row["search_count"] + 1
            }).eq("id", row["id"]).execute()
            return row
    except Exception as e:
        logger.warning("pending_verbs select failed: %s", e)
    return None


def save_pending_verb(
    verb: str,
    ml_label: str,
    ml_conf: float,
    predicted_ending: Optional[str] = None,
) -> None:
    """
    Save a newly discovered verb to pending_verbs.
    Uses upsert so duplicate searches don't create duplicate rows.
    """
    client = get_supabase()
    if client is None:
        return
    try:
        client.table("pending_verbs").upsert({
            "verb":              verb.lower().strip(),
            "search_count":      1,
            "ml_label":          ml_label,
            "ml_conf":           round(float(ml_conf), 2),
            "predicted_ending":  predicted_ending,
        }, on_conflict="verb").execute()
    except Exception as e:
        logger.warning("pending_verbs upsert failed: %s", e)


# ── Dictionary API ────────────────────────────────────────────────────────────

def fetch_from_dictionary_api(verb: str) -> Optional[dict]:
    """
    Call dictionaryapi.dev — free, no API key required.
    Returns a dict with keys: phonetic, audio_url, definition
    or None if the verb is not found or the request fails.
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{verb.lower().strip()}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        if not isinstance(data, list) or not data:
            return None

        entry    = data[0]
        phonetic = entry.get("phonetic", "")

        # Extract audio URL from phonetics array
        audio_url = ""
        for p in entry.get("phonetics", []):
            if p.get("audio"):
                audio_url = p["audio"]
                break

        # Extract first definition
        definition = ""
        meanings = entry.get("meanings", [])
        if meanings:
            defs = meanings[0].get("definitions", [])
            if defs:
                definition = defs[0].get("definition", "")

        return {
            "phonetic":   phonetic,
            "audio_url":  audio_url,
            "definition": definition,
        }

    except requests.exceptions.Timeout:
        logger.warning("Dictionary API timeout for '%s'", verb)
        return None
    except Exception as e:
        logger.warning("Dictionary API error for '%s': %s", verb, e)
        return None


# ── Main public function ──────────────────────────────────────────────────────

def auto_lookup(
    verb: str,
    model,
    extract_features_fn,
    predict_ending_fn,
    get_rule_explanation_fn,
    get_phonetic_category_fn,
    count_syllables_fn,
) -> dict:
    """
    Full pipeline for a verb not found in the local Excel dataset.

    Returns a result dict with keys:
        source          'cache' | 'api' | 'ml_only'
        label           'Regular' | 'Irregular'
        confidence      float 0-100
        predicted_ending str or None   (e.g. '/t/', '/d/', '/ɪd/')
        rule_text       str
        phonetic        str   (from API or empty)
        audio_url       str   (from API or empty)
        definition      str   (from API or empty)
        phonetic_cat    str
        syllables       int
    """
    verb = verb.lower().strip()

    # ── Step 1: Supabase cache ────────────────────────────────────────────
    cached = get_cached_verb(verb)
    if cached:
        log_search(verb, found=False, verb_type=f"cached:{cached['ml_label']}")
        return {
            "source":           "cache",
            "label":            cached["ml_label"],
            "confidence":       cached["ml_conf"],
            "predicted_ending": cached.get("predicted_ending"),
            "rule_text":        get_rule_explanation_fn(cached.get("predicted_ending", "")),
            "phonetic":         "",
            "audio_url":        "",
            "definition":       "",
            "phonetic_cat":     get_phonetic_category_fn(verb),
            "syllables":        count_syllables_fn(verb),
        }

    # ── Step 2: Dictionary API ────────────────────────────────────────────
    import pandas as pd
    api_data   = fetch_from_dictionary_api(verb) or {}
    row_df     = pd.DataFrame([{"Base": verb}])
    features   = extract_features_fn(row_df)
    prob       = model.predict_proba(features)[0]
    label      = "Irregular" if prob[1] > 0.5 else "Regular"
    confidence = round(float(max(prob)) * 100, 1)

    # ── Step 3 & 4: Phonetic rule + ML ───────────────────────────────────
    predicted_ending = predict_ending_fn(verb) if label == "Regular" else None
    rule_text        = get_rule_explanation_fn(predicted_ending or "")
    phonetic_cat     = get_phonetic_category_fn(verb)
    syllables        = count_syllables_fn(verb)

    # ── Step 5: Save to Supabase ──────────────────────────────────────────
    save_pending_verb(verb, label, confidence, predicted_ending)

    # ── Step 6: Log search ────────────────────────────────────────────────
    source = "api" if api_data else "ml_only"
    log_search(verb, found=False, verb_type=f"{source}:{label}")

    return {
        "source":           source,
        "label":            label,
        "confidence":       confidence,
        "predicted_ending": predicted_ending,
        "rule_text":        rule_text,
        "phonetic":         api_data.get("phonetic", ""),
        "audio_url":        api_data.get("audio_url", ""),
        "definition":       api_data.get("definition", ""),
        "phonetic_cat":     phonetic_cat,
        "syllables":        syllables,
    }


# ── Analytics helpers ─────────────────────────────────────────────────────────

def get_top_searched(limit: int = 10) -> list[dict]:
    """
    Returns the top N most-searched pending verbs.
    Used for the Community Trends tab.
    """
    client = get_supabase()
    if client is None:
        return []
    try:
        result = (
            client.table("pending_verbs")
            .select("verb, search_count, ml_label, ml_conf")
            .order("search_count", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.warning("get_top_searched failed: %s", e)
        return []


def get_search_stats() -> dict:
    """
    Returns aggregate search stats.
    Used for the sidebar or analytics dashboard.
    """
    client = get_supabase()
    if client is None:
        return {"total": 0, "found": 0, "not_found": 0, "unique_new": 0}
    try:
        logs = client.table("search_logs").select("found").execute()
        rows = logs.data or []
        total     = len(rows)
        found     = sum(1 for r in rows if r["found"])
        not_found = total - found

        pending = client.table("pending_verbs").select("id", count="exact").execute()
        unique_new = pending.count or 0

        return {
            "total":      total,
            "found":      found,
            "not_found":  not_found,
            "unique_new": unique_new,
        }
    except Exception as e:
        logger.warning("get_search_stats failed: %s", e)
        return {"total": 0, "found": 0, "not_found": 0, "unique_new": 0}
