"""
evaluate_model.py
Loads the saved classifier and prints a full evaluation report.

Usage:
    python scripts/evaluate_model.py
"""

import os
import sys
import json
import pickle

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.preprocessing import extract_features
from sklearn.model_selection import train_test_split

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'classifier.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'english_verbs.xlsx')


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run: python scripts/train_model.py")
        sys.exit(1)

    print("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print("Loading data...")
    df_reg   = pd.read_excel(DATA_PATH, sheet_name='Regular Verbs',   header=2)
    df_irreg = pd.read_excel(DATA_PATH, sheet_name='Irregular Verbs', header=2)

    reg_cols = ['Base','Simple_Past','Past_Participle',
                'IPA_Base','IPA_Past','IPA_PP',
                'Phonetic_Base','Phonetic_Past','Phonetic_PP',
                'Last_Sound','Ending']
    irreg_cols = reg_cols[:-1] + ['Vowel_Change']

    df_reg.columns   = reg_cols
    df_irreg.columns = irreg_cols
    df_reg   = df_reg.dropna(subset=['Base']).reset_index(drop=True)
    df_irreg = df_irreg.dropna(subset=['Base']).reset_index(drop=True)
    df_reg['Type']   = 'Regular'
    df_irreg['Type'] = 'Irregular'

    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)
    X = extract_features(df_all)
    y = (df_all['Type'] == 'Irregular').astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    y_pred = model.predict(X_test)

    print("\n── Classification Report ───────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=['Regular', 'Irregular']))

    cm = confusion_matrix(y_test, y_pred)
    print("── Confusion Matrix ────────────────────────")
    print(f"                Pred Regular   Pred Irregular")
    print(f"  Actual Regular     {cm[0][0]:4d}           {cm[0][1]:4d}")
    print(f"  Actual Irregular   {cm[1][0]:4d}           {cm[1][1]:4d}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        print(f"\n── Stored Metrics ──────────────────────────")
        print(f"  Accuracy:   {m['accuracy']:.1%}")
        print(f"  Precision:  {m['precision']:.1%}")
        print(f"  Recall:     {m['recall']:.1%}")
        print(f"  F1 Score:   {m['f1_score']:.1%}")
        print(f"  CV 5-fold:  {m['cv_mean']:.1%} ± {m['cv_std']:.1%}")


if __name__ == '__main__':
    main()
