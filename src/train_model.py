"""Telco churn: leakage-safe model comparison and evaluation."""
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def scores(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return dict(threshold=round(float(threshold), 3),
                accuracy=round(accuracy_score(y, pred), 4),
                precision=round(precision_score(y, pred, zero_division=0), 4),
                recall=round(recall_score(y, pred, zero_division=0), 4),
                f1=round(f1_score(y, pred, zero_division=0), 4),
                roc_auc=round(roc_auc_score(y, prob), 4),
                tn=int(((y == 0) & (pred == 0)).sum()),
                fp=int(((y == 0) & (pred == 1)).sum()),
                fn=int(((y == 1) & (pred == 0)).sum()),
                tp=int(((y == 1) & (pred == 1)).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--output', default='reports')
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        parser.error(f'Dataset missing: {data_path}. See README.md for the Kaggle download link.')
    df = pd.read_csv(data_path)
    required = {'Churn', 'TotalCharges', 'tenure', 'MonthlyCharges'}
    if not required.issubset(df.columns):
        parser.error(f'Missing columns: {required - set(df.columns)}')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['Churn'].isna().any() or not set(df['Churn'].unique()) <= {'Yes', 'No'}:
        parser.error('Churn must contain non-missing Yes/No labels.')
    X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
    y = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=.25, stratify=y_tv, random_state=42)
    num = X.select_dtypes(include='number').columns.tolist()
    cat = X.select_dtypes(exclude='number').columns.tolist()
    def pipeline(estimator):
        pre = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler())]), num),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat)])
        return Pipeline([('preprocess', pre), ('classifier', estimator)])
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                random_state=42, n_jobs=-1)}
    rows, fitted = [], {}
    thresholds = np.round(np.arange(.10, .901, .01), 2)
    for name, estimator in candidates.items():
        model = pipeline(estimator)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        fitted[name] = model
        for t in thresholds:
            rows.append({'model': name, **scores(y_val.to_numpy(), prob, t)})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    val = pd.DataFrame(rows)
    val.to_csv(out / 'validation_thresholds.csv', index=False)
    best = val.sort_values(['f1', 'roc_auc', 'model'], ascending=[False, False, True]).iloc[0]
    name, threshold = str(best['model']), float(best['threshold'])
    model = fitted[name]
    prob_test = model.predict_proba(X_test)[:, 1]
    result = scores(y_test.to_numpy(), prob_test, threshold)
    report = {'selected_model': name, 'selection_metric': 'validation F1',
              'selected_threshold': threshold,
              'split': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)},
              'validation_metrics': {k: float(best[k]) for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
              'test_metrics': result,
              'note': 'The held-out test set is evaluated once after selection on validation.'}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    Path('models').mkdir(exist_ok=True)
    joblib.dump({'pipeline': model, 'threshold': threshold}, 'models/churn_pipeline.joblib')
    ConfusionMatrixDisplay.from_predictions(y_test, (prob_test >= threshold).astype(int),
                                            display_labels=['No churn', 'Churn'], colorbar=False)
    plt.title('Confusion matrix — held-out test'); plt.tight_layout()
    plt.savefig(out / 'confusion_matrix.png', dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    plt.plot(fpr, tpr, label=f'{name} (AUC {result["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC — held-out test'); plt.legend(); plt.tight_layout()
    plt.savefig(out / 'roc_curve.png', dpi=150); plt.close()
    selected = val[val['model'] == name]
    for metric in ['precision', 'recall', 'f1']:
        plt.plot(selected['threshold'], selected[metric], label=metric)
    plt.axvline(threshold, color='gray', linestyle='--', label=f'Selected {threshold:.2f}')
    plt.xlabel('Threshold'); plt.ylabel('Validation score'); plt.legend(); plt.tight_layout()
    plt.savefig(out / 'threshold_analysis.png', dpi=150); plt.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
