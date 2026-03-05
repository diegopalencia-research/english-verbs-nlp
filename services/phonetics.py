"""
phonetics.py
Rule-based phonetic analysis for English verb past tense pronunciation.

Implements the three-way -ed pronunciation rule:
  /t/  — voiceless consonant ending
  /d/  — voiced sound ending
  /ɪd/ — /t/ or /d/ ending (extra syllable)

Also provides helpers for participial adjective phonetic display.
"""

VOICELESS_FINALS  = {'p', 'k', 'f', 'x'}
VOICELESS_DIGRAPHS = {'sh', 'ch', 'tch', 'gh'}
STOP_FINALS       = {'t', 'd'}

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

# Semantic class descriptions for participial adjectives
SEMANTIC_CLASS_INFO = {
    'Emotional state': {
        'color_key': 'b-irr',
        'description': (
            'Describes the <b>experiencer</b> — the person who feels the emotion. '
            'Distinct from present participle adjectives (boring, exciting) which '
            'describe the <b>stimulus</b>. Example: "a bored student" (experiencer) '
            'vs "a boring lecture" (stimulus).'
        ),
        'examples': 'bored · excited · tired · worried · frustrated · thrilled',
    },
    'Physical state': {
        'color_key': 'b-d',
        'description': (
            'Describes a <b>visible physical condition</b> resulting from a past action. '
            'Typically describes objects or body parts. Many come from irregular verbs '
            '(broken, frozen, torn) and are fully lexicalized as adjectives.'
        ),
        'examples': 'broken · frozen · worn · torn · cracked · swollen',
    },
    'Process result': {
        'color_key': 'b-t',
        'description': (
            'Describes the <b>outcome of a completed process</b>. Common in culinary, '
            'technical, and business contexts. The action has been applied to produce '
            'a new state. Often interchangeable with the past tense but clearly '
            'adjectival in attributive position.'
        ),
        'examples': 'cooked · printed · trained · updated · organized · polished',
    },
    'Ambiguous': {
        'color_key': 'b-id',
        'description': (
            'Can function as <b>either past tense verb or adjective</b> depending on '
            'context. "She experienced hardship" (verb) vs "an experienced nurse" '
            '(adjective). Context, syntax position, and stress determine interpretation. '
            'Key test: can you substitute "very"? "a very experienced nurse" ✓ → adjective.'
        ),
        'examples': 'experienced · advanced · determined · mixed · limited · involved',
    },
}


def predict_ending(base_form: str) -> str:
    """
    Predict the -ed pronunciation for a regular verb.
    Returns: '/t/', '/d/', or '/ɪd/'
    """
    w = base_form.lower().strip()

    # Check digraphs first (order matters)
    if len(w) >= 3 and w[-3:] == 'tch':
        return '/t/'
    if len(w) >= 2 and w[-2:] in VOICELESS_DIGRAPHS:
        return '/t/'

    last = w[-1] if w else ''

    if last in STOP_FINALS:
        return '/ɪd/'

    if last == 'e' and len(w) >= 2:
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


def get_semantic_class_info(semantic_class: str) -> dict:
    """Return display info for a participial adjective semantic class."""
    return SEMANTIC_CLASS_INFO.get(semantic_class, {
        'color_key': 'b-reg',
        'description': '',
        'examples': '',
    })


def adjective_test(participial_form: str) -> dict:
    """
    Return the standard linguistic tests for whether a participial form
    is functioning as an adjective (for display in the UI).
    """
    return {
        'very_test':    f'very {participial_form}',
        'seem_test':    f'seem {participial_form}',
        'attributive':  f'a {participial_form} [noun]',
        'predicative':  f'[noun] is {participial_form}',
        'note': (
            'If "very" can precede the form and it still sounds natural, '
            'it is functioning as an adjective, not a verb.'
        ),
    }


def phonetic_summary(base_form: str, ending: str) -> dict:
    """Return a full phonetic summary dict for a regular verb."""
    return {
        'base':    base_form,
        'past':    f'{base_form}ed',
        'ending':  ending,
        'rule':    get_rule_explanation(ending),
        'example': f'{base_form} → {base_form}ed ({ending})',
    }
