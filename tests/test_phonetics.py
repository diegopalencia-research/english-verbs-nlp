"""
tests/test_services.py
Unit tests for lemmatizer, phonetics, and preprocessing services.

Run from project root:
    python -m pytest tests/ -v

Or without pytest:
    python tests/test_services.py
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.phonetics     import predict_ending, get_rule_explanation, get_semantic_class_info
from services.preprocessing import (count_syllables, get_phonetic_category,
                                     extract_features, get_ngram)
from services.lemmatizer    import find_verb, suggest_verbs


# ── Fixtures ───────────────────────────────────────────────────────────────────
def make_reg_df():
    return pd.DataFrame({
        'Base':            ['walk',  'start', 'love', 'push', 'call'],
        'Simple_Past':     ['walked','started','loved','pushed','called'],
        'Past_Participle': ['walked','started','loved','pushed','called'],
        'IPA_Base':  ['/wɔːk/', '/stɑːrt/', '/lʌv/', '/pʊʃ/', '/kɔːl/'],
        'IPA_Past':  ['/wɔːkt/','/stɑːrtɪd/','/lʌvd/','/pʊʃt/','/kɔːld/'],
        'IPA_PP':    ['/wɔːkt/','/stɑːrtɪd/','/lʌvd/','/pʊʃt/','/kɔːld/'],
        'Phonetic_Base': ['WAWK', 'START', 'LUV', 'PUSH', 'KAWL'],
        'Phonetic_Past': ['WAWKT','STARTID','LUVD','PUSHT','KAWLD'],
        'Phonetic_PP':   ['WAWKT','STARTID','LUVD','PUSHT','KAWLD'],
        'Last_Sound': ['voiceless /k/', 'voiced /t/', 'voiced /v/',
                       'voiceless /ʃ/', 'voiced /l/'],
        'Ending': ['/t/', '/ɪd/', '/d/', '/t/', '/d/'],
        'Type':   ['Regular'] * 5,
    })


def make_irreg_df():
    return pd.DataFrame({
        'Base':            ['go',   'fight',  'buy',   'break', 'know'],
        'Simple_Past':     ['went', 'fought', 'bought','broke', 'knew'],
        'Past_Participle': ['gone', 'fought', 'bought','broken','known'],
        'IPA_Base':  ['/ɡoʊ/', '/faɪt/', '/baɪ/', '/breɪk/', '/noʊ/'],
        'IPA_Past':  ['/wɛnt/','/fɔːt/', '/bɔːt/', '/broʊk/', '/njuː/'],
        'IPA_PP':    ['/ɡɒn/', '/fɔːt/', '/bɔːt/', '/ˈbroʊkən/','known'],
        'Phonetic_Base': ['GOH',  'FITE',   'BY',    'BRAYK',  'NOH'],
        'Phonetic_Past': ['WENT', 'FAWT',   'BAWT',  'BROHK',  'NYOO'],
        'Phonetic_PP':   ['GAWN', 'FAWT',   'BAWT',  'BROHKEN','NOHN'],
        'Vowel_Change': ['special form', 'aɪ → ɔː', 'aɪ → ɔː',
                         'eɪ → oʊ', 'oʊ → uː'],
        'Type': ['Irregular'] * 5,
    })


def make_part_df():
    return pd.DataFrame({
        'Base_Verb':       ['bore',   'excite',   'break',  'experience'],
        'Participial_Form':['bored',  'excited',  'broken', 'experienced'],
        'IPA_Base':  ['/bɔːr/', '/ɪkˈsaɪt/', '/breɪk/', '/ɪkˈspɪərɪəns/'],
        'IPA_Adj':   ['/bɔːrd/','/ɪkˈsaɪtɪd/','/ˈbroʊkən/','/ɪkˈspɪərɪənst/'],
        'Phonetic_Adj':    ['BORD','ek-SITE-id','BROH-ken','ek-SPEER-eenst'],
        'Semantic_Class':  ['Emotional state','Emotional state',
                            'Physical state', 'Ambiguous'],
        'Example_Phrase':  ['a bored student','an excited child',
                            'a broken window','an experienced nurse'],
        'Notes': ['', '', '', ''],
    })


# ═════════════════════════════════════════════════════════════════════════════
# PHONETICS TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestPredictEnding(unittest.TestCase):

    def test_voiceless_k(self):
        self.assertEqual(predict_ending('walk'),  '/t/')

    def test_voiceless_p(self):
        self.assertEqual(predict_ending('jump'),  '/t/')

    def test_voiceless_f(self):
        self.assertEqual(predict_ending('laugh'), '/t/')

    def test_voiceless_s(self):
        self.assertEqual(predict_ending('miss'),  '/t/')

    def test_voiceless_sh_digraph(self):
        self.assertEqual(predict_ending('push'),  '/t/')

    def test_voiceless_ch_digraph(self):
        self.assertEqual(predict_ending('teach'), '/t/')

    def test_voiceless_tch_trigraph(self):
        self.assertEqual(predict_ending('watch'), '/t/')

    def test_voiced_vowel(self):
        self.assertEqual(predict_ending('love'),  '/d/')

    def test_voiced_l(self):
        self.assertEqual(predict_ending('call'),  '/d/')

    def test_voiced_n(self):
        self.assertEqual(predict_ending('clean'), '/d/')

    def test_voiced_r(self):
        self.assertEqual(predict_ending('answer'),'/d/')

    def test_voiced_b(self):
        self.assertEqual(predict_ending('rob'),   '/d/')

    def test_stop_t(self):
        self.assertEqual(predict_ending('start'), '/ɪd/')

    def test_stop_d(self):
        self.assertEqual(predict_ending('need'),  '/ɪd/')

    def test_silent_e_voiceless(self):
        # "bake" ends in silent e, second-to-last is voiceless /k/
        self.assertEqual(predict_ending('bake'),  '/t/')

    def test_silent_e_voiced(self):
        # "love" ends in silent e, voiced
        self.assertEqual(predict_ending('love'),  '/d/')

    def test_rule_explanation_t(self):
        result = get_rule_explanation('/t/')
        self.assertIn('/t/', result)
        self.assertIsInstance(result, str)

    def test_rule_explanation_d(self):
        result = get_rule_explanation('/d/')
        self.assertIn('/d/', result)

    def test_rule_explanation_id(self):
        result = get_rule_explanation('/ɪd/')
        self.assertIn('/ɪd/', result)

    def test_rule_explanation_unknown(self):
        result = get_rule_explanation('/x/')
        self.assertIsInstance(result, str)


class TestSemanticClassInfo(unittest.TestCase):

    def test_emotional_state_returns_dict(self):
        info = get_semantic_class_info('Emotional state')
        self.assertIsInstance(info, dict)
        self.assertIn('description', info)
        self.assertIn('examples', info)

    def test_all_classes_return_dict(self):
        for cls in ['Emotional state', 'Physical state', 'Process result', 'Ambiguous']:
            info = get_semantic_class_info(cls)
            self.assertIsInstance(info, dict)

    def test_unknown_class_returns_dict(self):
        info = get_semantic_class_info('NonexistentClass')
        self.assertIsInstance(info, dict)


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestCountSyllables(unittest.TestCase):

    def test_one_syllable(self):
        self.assertEqual(count_syllables('walk'), 1)

    def test_two_syllables(self):
        # 'open' has two clear vowel groups: o-e and e-n
        self.assertEqual(count_syllables('open'), 2)

    def test_three_syllables(self):
        self.assertEqual(count_syllables('validate'), 3)

    def test_silent_e(self):
        self.assertEqual(count_syllables('love'), 1)

    def test_minimum_one(self):
        self.assertGreaterEqual(count_syllables('the'), 1)


class TestGetPhoneticCategory(unittest.TestCase):

    def test_voiceless_k(self):
        self.assertEqual(get_phonetic_category('walk'),   'voiceless')

    def test_voiceless_sh(self):
        self.assertEqual(get_phonetic_category('push'),   'voiceless')

    def test_voiceless_ch(self):
        self.assertEqual(get_phonetic_category('teach'),  'voiceless')

    def test_stop_t(self):
        self.assertEqual(get_phonetic_category('start'),  'stop')

    def test_stop_d(self):
        self.assertEqual(get_phonetic_category('need'),   'stop')

    def test_vowel_end(self):
        self.assertEqual(get_phonetic_category('go'),     'vowel')

    def test_voiced_l(self):
        self.assertEqual(get_phonetic_category('call'),   'voiced')

    def test_voiced_n(self):
        self.assertEqual(get_phonetic_category('clean'),  'voiced')


class TestExtractFeatures(unittest.TestCase):

    def setUp(self):
        df = pd.concat([make_reg_df(), make_irreg_df()], ignore_index=True)
        self.features = extract_features(df)

    def test_returns_dataframe(self):
        self.assertIsInstance(self.features, pd.DataFrame)

    def test_no_nulls(self):
        self.assertFalse(self.features.isnull().any().any(),
                         "Feature matrix contains NaN values")

    def test_expected_columns_present(self):
        for col in ['length', 'vowel_count', 'consonant_count',
                    'syllable_count', 'phonetic_category',
                    'is_voiceless', 'is_voiced', 'is_stop', 'is_vowel_end',
                    'bigram', 'trigram', 'last_letter']:
            self.assertIn(col, self.features.columns, f"Missing column: {col}")

    def test_participial_feature_present(self):
        self.assertIn('is_likely_participial', self.features.columns)

    def test_row_count_matches(self):
        df = pd.concat([make_reg_df(), make_irreg_df()], ignore_index=True)
        self.assertEqual(len(self.features), len(df))

    def test_length_values_correct(self):
        df = pd.concat([make_reg_df(), make_irreg_df()], ignore_index=True)
        feats = extract_features(df)
        # "walk" has length 4
        walk_idx = df[df['Base'] == 'walk'].index[0]
        self.assertEqual(feats.loc[walk_idx, 'length'], 4)


class TestGetNgram(unittest.TestCase):

    def test_bigram(self):
        self.assertEqual(get_ngram('walk', 2), 'lk')

    def test_trigram(self):
        self.assertEqual(get_ngram('walk', 3), 'alk')

    def test_short_word(self):
        result = get_ngram('go', 3)
        self.assertEqual(len(result), 3)  # zero-padded


# ═════════════════════════════════════════════════════════════════════════════
# LEMMATIZER TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestFindVerb(unittest.TestCase):

    def setUp(self):
        self.reg   = make_reg_df()
        self.irreg = make_irreg_df()
        self.part  = make_part_df()

    # Regular verbs
    def test_find_regular_base(self):
        row, vtype, form = find_verb('walk', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Regular')
        self.assertEqual(form, 'base form')

    def test_find_regular_past(self):
        row, vtype, form = find_verb('walked', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Regular')
        self.assertEqual(form, 'simple past')

    def test_find_regular_pp(self):
        # 'loved' == simple_past in fixture; use 'gone' (irregular pp distinct from past)
        row, vtype, form = find_verb('gone', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Irregular')
        self.assertEqual(form, 'past participle')

    # Irregular verbs
    def test_find_irregular_base(self):
        row, vtype, form = find_verb('go', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Irregular')

    def test_find_irregular_past(self):
        row, vtype, form = find_verb('went', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Irregular')
        self.assertEqual(form, 'simple past')

    def test_find_irregular_pp(self):
        row, vtype, form = find_verb('broken', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Irregular')
        self.assertEqual(form, 'past participle')

    # Participial adjectives
    def test_find_participial_form(self):
        row, vtype, form = find_verb('excited', self.reg, self.irreg, self.part)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Participial Adjective')
        self.assertEqual(form, 'participial form')

    def test_find_participial_base_verb(self):
        row, vtype, form = find_verb('bore', self.reg, self.irreg, self.part)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Participial Adjective')
        self.assertEqual(form, 'base verb')

    # Case insensitivity
    def test_case_insensitive(self):
        row, vtype, _ = find_verb('WALK', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Regular')

    def test_case_insensitive_irregular(self):
        row, vtype, _ = find_verb('WENT', self.reg, self.irreg)
        self.assertIsNotNone(row)
        self.assertEqual(vtype, 'Irregular')

    # Not found
    def test_not_found_returns_none_tuple(self):
        row, vtype, form = find_verb('xyzabc123', self.reg, self.irreg)
        self.assertIsNone(row)
        self.assertIsNone(vtype)
        self.assertIsNone(form)

    # Without participial df (backward compat)
    def test_find_verb_without_part_df(self):
        row, vtype, form = find_verb('walk', self.reg, self.irreg)
        self.assertIsNotNone(row)

    # Whitespace handling
    def test_whitespace_stripped(self):
        row, vtype, _ = find_verb('  walk  ', self.reg, self.irreg)
        self.assertIsNotNone(row)


class TestSuggestVerbs(unittest.TestCase):

    def setUp(self):
        self.reg   = make_reg_df()
        self.irreg = make_irreg_df()
        self.part  = make_part_df()

    def test_returns_list(self):
        result = suggest_verbs('wa', self.reg, self.irreg)
        self.assertIsInstance(result, list)

    def test_prefix_match(self):
        result = suggest_verbs('wa', self.reg, self.irreg)
        self.assertTrue(any('walk' in r for r in result))

    def test_respects_n_limit(self):
        result = suggest_verbs('a', self.reg, self.irreg, n=2)
        self.assertLessEqual(len(result), 2)

    def test_no_duplicates(self):
        result = suggest_verbs('w', self.reg, self.irreg)
        self.assertEqual(len(result), len(set(result)))

    def test_empty_query_returns_list(self):
        result = suggest_verbs('', self.reg, self.irreg)
        self.assertIsInstance(result, list)

    def test_with_participial_df(self):
        result = suggest_verbs('bor', self.reg, self.irreg, df_part=self.part)
        self.assertIsInstance(result, list)


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestIntegration(unittest.TestCase):
    """
    End-to-end tests: find verb → extract features → predict ending.
    Verifies that the pipeline components work together correctly.
    """

    def setUp(self):
        self.reg   = make_reg_df()
        self.irreg = make_irreg_df()

    def test_regular_verb_pipeline(self):
        """walk → found as Regular → features extracted → ending predicted"""
        row, vtype, _ = find_verb('walk', self.reg, self.irreg)
        self.assertEqual(vtype, 'Regular')

        df_single = pd.DataFrame([{'Base': row['Base']}])
        feats = extract_features(df_single)
        self.assertEqual(len(feats), 1)
        self.assertFalse(feats.isnull().any().any())

        ending = predict_ending(row['Base'])
        self.assertIn(ending, ['/t/', '/d/', '/ɪd/'])
        # walk ends in voiceless /k/ → must be /t/
        self.assertEqual(ending, '/t/')

    def test_start_pipeline(self):
        """start → ends in /t/ → must predict /ɪd/"""
        ending = predict_ending('start')
        self.assertEqual(ending, '/ɪd/')

    def test_love_pipeline(self):
        """love → ends in voiced → must predict /d/"""
        ending = predict_ending('love')
        self.assertEqual(ending, '/d/')

    def test_feature_count_consistent(self):
        """Feature count must be the same for all verbs (model compatibility)"""
        df1 = pd.DataFrame([{'Base': 'walk'}])
        df2 = pd.DataFrame([{'Base': 'validate'}])
        f1 = extract_features(df1)
        f2 = extract_features(df2)
        self.assertEqual(f1.shape[1], f2.shape[1],
                         "Feature count differs between verbs — model will fail")


if __name__ == '__main__':
    # Run with verbose output even without pytest
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
