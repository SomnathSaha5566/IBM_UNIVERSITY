"""Train and evaluate failure-type classifiers.

Usage:
    python src/train.py --data data/predictive_maintenance.csv
"""
import argparse
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import load_and_clean, split_xy  # noqa: E402

SEED = 42
MIN_CLASS_SIZE = 10  # classes smaller than this can't be stratified / cross-validated


def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_and_clean(args.data)
    X, y = split_xy(df)
    print("Class distribution:\n", y.value_counts(), "\n")

    counts = y.value_counts()
    rare = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if rare:
        print("Dropping very rare classes:", rare)
        mask = ~y.isin(rare)
        X, y = X[mask], y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    candidates = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, model in candidates.items():
        s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_scores[name] = float(s.mean())
        print(f"{name:22s} CV macro-F1 = {s.mean():.3f} (+/- {s.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    best = candidates[best_name].fit(X_tr, y_tr)
    pred = best.predict(X_te)
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_te, pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best_name}")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150); plt.close()

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns)
    else:
        r = permutation_importance(best, X_te, y_te, scoring="f1_macro", n_repeats=5, random_state=SEED)
        imp = pd.Series(r.importances_mean, index=X.columns)
    plt.figure(figsize=(7, 5)); imp.sort_values().plot.barh(color="#1f6feb")
    plt.title("Feature importance"); plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150); plt.close()

    joblib.dump({"model": best, "features": list(X.columns), "classes": list(best.classes_)},
                "models/model.joblib")
    metrics = {
        "best_model": best_name,
        "cv_macro_f1": cv_scores,
        "test_macro_f1": float(f1_score(y_te, pred, average="macro")),
        "test_weighted_f1": float(f1_score(y_te, pred, average="weighted")),
        "test_accuracy": float((pred == y_te).mean()),
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/predictive_maintenance.csv")
    main(ap.parse_args())
