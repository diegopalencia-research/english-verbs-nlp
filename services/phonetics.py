"""
phonetics.py
Rule-based phonetic analysis for English verb past tense pronunciation.
Implements the three-way -ed pronunciation rule:
  /t/  — voiceless consonant ending
  /d/  — voiced sound ending
  /ɪd/ — /t/ or /d/ ending (extra syllable)
"""

VOICELESS_FINALS = {'p', 'k', 'f', 'x'}
VOICELESS_DIGRAPHS = {'sh', 'ch', 'tch', 'gh'}
STOP_FINALS = {'t', 'd'}

ENDING_LABELS = {
    '/t/':  'Voiceless consonant → -ed sounds like /t/',
    '/d/':  'Voiced sound → -ed sounds like /d/',
    '/ɪd/': 'Ends in /t/ or /d/ → adds extra syllable /ɪd/',
}

ENDING_SOUNDS = {
    '/t/':  ['p', 'k', 'f', 's', 'sh', 'ch', 'x'],
    '/d/':  ['vowel', 'b', 'g', 'v', 'z', 'm', 'n', 'l', 'r'],
    '/ɪd/': ['/t/', '/d/'],
}


def predict_ending(base_form: str) -> str:
    """
    Predict the -ed pronunciation for a regular verb.

    Returns: '/t/', '/d/', or '/ɪd/'
    """
    w = base_form.lower().strip()

    # Check digraphs first
    if len(w) >= 3 and w[-3:] == 'tch':
        return '/t/'
    if len(w) >= 2 and w[-2:] in VOICELESS_DIGRAPHS:
        return '/t/'

    last = w[-1] if w else ''

    if last in STOP_FINALS:
        # Silent e: "vote" ends in /t/ but last *letter* is 'e'
        return '/ɪd/'

    if last == 'e' and len(w) >= 2:
        # Check the sound before the silent e
        second = w[-2]
        if second in STOP_FINALS:
            return '/ɪd/'
        if second in VOICELESS_FINALS:
            return '/t/'
        return '/d/'

    if last in VOICELESS_FINALS or last == 's':
        return '/t/'

    return '/d/'


def get_rule_explanation(ending: str) -> str:
    """Return the human-readable explanation for a given ending code."""
    return ENDING_LABELS.get(ending, 'Unknown rule')


def phonetic_summary(base_form: str, ending: str) -> dict:
    """Return a full phonetic summary dict for a regular verb."""
    return {
        'base':    base_form,
        'past':    f'{base_form}ed',
        'ending':  ending,
        'rule':    get_rule_explanation(ending),
        'example': f'{base_form} → {base_form}ed ({ending})',
    }
