"""
preprocessing.py
Feature engineering pipeline for the verb classifier.
Includes phonetic category, syllable count, n-grams,
voiced/voiceless encoding — real NLP feature engineering.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Phonetic constants
VOICELESS = {'p', 'k', 'f', 's', 'x', 'c'}
VOICELESS_DIGRAPHS = {'sh', 'ch', 'tch', 'ck', 'gh'}
STOP_ENDINGS = {'t', 'd'}


def count_syllables(word: str) -> int:
    """
    Approximate syllable count by counting vowel groups.
    Handles silent-e rule for more accuracy.
    """
    word = word.lower()
    count = 0
    vowels = 'aeiou'
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Silent e at end does not count as syllable
    if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
        count = max(1, count - 1)

    return max(1, count)


def get_phonetic_category(word: str) -> str:
    """
    Classify the phonetic category of the word's final sound.

    Categories:
        'voiceless' — p, k, f, s, sh, ch, x
        'stop'      — t, d  (need extra syllable when -ed is added)
        'vowel'     — ends in vowel sound
        'voiced'    — all other consonants: b, g, v, z, m, n, l, r

    Returns: one of {'voiceless', 'stop', 'vowel', 'voiced'}
    """
    w = word.lower()

    # Check digraphs first (order matters)
    if len(w) >= 3 and w[-3:] == 'tch':
        return 'voiceless'
    if len(w) >= 2:
        if w[-2:] in VOICELESS_DIGRAPHS:
            return 'voiceless'

    last = w[-1] if w else ''

    if last in 'aeiou':
        return 'vowel'
    if last in STOP_ENDINGS:
        return 'stop'
    if last in VOICELESS:
        return 'voiceless'
    return 'voiced'


def get_ngram(word: str, n: int) -> str:
    """Return the last n characters of a word, zero-padded if shorter."""
    w = str(word).lower()
    return w[-n:] if len(w) >= n else w.zfill(n)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature extraction pipeline.

    Features created:
    ─ Basic:          length, vowel_count, consonant_count
    ─ Phonetic:       syllable_count, phonetic_category (encoded),
                      is_voiceless, is_voiced, is_stop, is_vowel_end
    ─ Suffix binary:  ends_* for 20 common endings
    ─ N-grams:        bigram, trigram (last 2-3 chars, label-encoded)
    ─ Characters:     last_letter, second_last (label-encoded)
    """
    f = pd.DataFrame(index=df.index)

    base = df['Base'].astype(str).str.lower()

    # ── Basic ────────────────────────────────────────────────
    f['length']          = base.str.len()
    f['vowel_count']     = base.str.count('[aeiou]')
    f['consonant_count'] = base.str.count('[bcdfghjklmnpqrstvwxyz]')

    # ── Phonetic ─────────────────────────────────────────────
    f['syllable_count']  = base.apply(count_syllables)

    phon = base.apply(get_phonetic_category)
    phon_map = {'vowel': 0, 'voiced': 1, 'voiceless': 2, 'stop': 3}
    f['phonetic_category'] = phon.map(phon_map).fillna(1).astype(int)

    f['is_voiceless'] = (phon == 'voiceless').astype(int)
    f['is_voiced']    = (phon == 'voiced').astype(int)
    f['is_stop']      = (phon == 'stop').astype(int)
    f['is_vowel_end'] = (phon == 'vowel').astype(int)

    # ── Suffix binary features ───────────────────────────────
    suffixes = ['e','n','d','t','l','r','k','g','w','y',
                'ng','nd','ld','nt','in','ow','aw','ck','ll','se']
    for s in suffixes:
        f[f'ends_{s}'] = base.str.endswith(s).astype(int)

    # ── N-grams (label-encoded) ──────────────────────────────
    le = LabelEncoder()
    f['bigram']      = le.fit_transform(base.apply(lambda w: get_ngram(w, 2)))
    f['trigram']     = le.fit_transform(base.apply(lambda w: get_ngram(w, 3)))
    f['last_letter'] = le.fit_transform(base.str[-1:].fillna('_'))
    f['second_last'] = le.fit_transform(base.str[-2:-1].fillna('_'))

    return f
