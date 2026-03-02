"""
train_model.py
Reproducible training pipeline for the English verb classifier.

Usage:
    python scripts/train_model.py

Output:
    models/classifier.pkl     — trained RandomForest model
    models/metrics.json       — evaluation metrics
    models/feature_names.json — feature column names
"""

import os
import json
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.preprocessing import extract_features

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'english_verbs.xlsx')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')
METRICS_PATH  = os.path.join(MODEL_DIR, 'metrics.json')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.json')

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print(f"Loading data from: {DATA_PATH}")
    df_reg   = pd.read_excel(DATA_PATH, sheet_name='Regular Verbs',   header=2)
    df_irreg = pd.read_excel(DATA_PATH, sheet_name='Irregular Verbs', header=2)

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

    print(f"  Regular:   {len(df_reg)} verbs")
    print(f"  Irregular: {len(df_irreg)} verbs")
    return df_reg, df_irreg


def train():
    df_reg, df_irreg = load_data()
    df_all = pd.concat([df_reg, df_irreg], ignore_index=True)

    X = extract_features(df_all)
    y = (df_all['Type'] == 'Irregular').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining: {len(X_train)} verbs | Test: {len(X_test)} verbs")

    # ── Train ──────────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')
    cm        = confusion_matrix(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    metrics = {
        'accuracy':              round(float(acc), 4),
        'precision':             round(float(precision), 4),
        'recall':                round(float(recall), 4),
        'f1_score':              round(float(f1), 4),
        'cv_mean':               round(float(cv_scores.mean()), 4),
        'cv_std':                round(float(cv_scores.std()), 4),
        'cv_scores':             [round(float(s), 4) for s in cv_scores],
        'confusion_matrix':      cm.tolist(),
        'train_size':            int(len(X_train)),
        'test_size':             int(len(X_test)),
        'n_features':            int(X.shape[1]),
        'n_estimators':          200,
        'class_report':          classification_report(
                                     y_test, y_pred,
                                     target_names=['Regular', 'Irregular'],
                                     output_dict=True
                                 ),
        'feature_importances':   dict(zip(
                                     X.columns.tolist(),
                                     [round(float(v), 6) for v in model.feature_importances_]
                                 )),
    }

    print(f"\n── Results ────────────────────────────────")
    print(f"  Accuracy:      {acc:.1%}")
    print(f"  Precision:     {precision:.1%}")
    print(f"  Recall:        {recall:.1%}")
    print(f"  F1 Score:      {f1:.1%}")
    print(f"  CV Mean (5-fold): {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Regular', 'Irregular']))

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(FEATURES_PATH, 'w') as f:
        json.dump(X.columns.tolist(), f)

    print(f"\nSaved:")
    print(f"  {MODEL_PATH}")
    print(f"  {METRICS_PATH}")
    print(f"  {FEATURES_PATH}")
    print(f"\nDone.")

    return model, metrics


if __name__ == '__main__':
    train()
