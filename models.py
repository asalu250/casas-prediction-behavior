"""
models.py
=========
Trains and compares the model suite for predicting behavioral deviation
days. Two families are used, matching the two things this project is
actually testing:

  SUPERVISED (trained against label_statistical from labeling.py):
    Logistic Regression, Random Forest, Gradient Boosting (XGBoost if
    available, else sklearn's GradientBoostingClassifier), SVM.
    These answer: "given the statistical-deviation label, can we PREDICT
    it from features -- including lagged/rolling features that would be
    available *before* the deviation day itself?"

  UNSUPERVISED (no label used):
    Isolation Forest, One-Class SVM, Local Outlier Factor.
    These answer: "does an anomaly detector find the same days unusual,
    without ever being told which days were flagged?"

A critical methodological point: for the "predict future deviation"
framing to be honest, supervised models must be evaluated on FUTURE days
relative to what they trained on -- a random shuffled train/test split
would let the model see data from after the test day, which isn't a
prediction task at all, it's leakage. So this module uses a chronological
(time-ordered) split and TimeSeriesSplit for cross-validation, never a
random split.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def prepare_xy(feat: pd.DataFrame, labels: pd.DataFrame, label_col: str = "label_statistical"):
    """
    Joins features + label, drops rows with missing label or feature
    warmup NaNs, and returns (X, y, dates) in chronological order.

    Deliberately EXCLUDES same-day raw z-score columns (z_*) and the
    n_features_deviant column from X, since those are literally the
    inputs used to construct the label -- including them would let a
    model "predict" the label by trivially reading it back out.
    """
    df = feat.join(labels[[label_col]], how="inner")
    df = df.dropna(subset=[label_col])
    df = df.sort_index()

    feature_cols = [c for c in feat.columns]
    X = df[feature_cols].copy()
    y = df[label_col].astype(int)

    # any remaining NaNs (rolling-window warmup at the very start) -> drop
    valid = X.dropna().index
    X, y = X.loc[valid], y.loc[valid]
    return X, y


def get_supervised_models(random_state: int = 42) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                   random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6,
                                                class_weight="balanced",
                                                random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                         random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=random_state),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state,
        )
    return models


def chronological_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    """Last `test_frac` of the (time-ordered) data is held out as the test
    set -- the model is trained only on the past and tested on the future,
    consistent with the project's predictive framing."""
    n_test = int(len(X) * test_frac)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def run_supervised_comparison(X: pd.DataFrame, y: pd.Series, cv_splits: int = 5):
    """
    Trains each supervised model on a chronological train/test split, and
    separately runs TimeSeriesSplit cross-validation (still strictly
    forward-in-time on every fold) to get a more stable estimate of
    generalization than a single holdout split can provide.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    X_train, X_test, y_train, y_test = chronological_split(X_scaled, y)

    results = {}
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for name, model in get_supervised_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, "predict_proba") else None)
        holdout_metrics = evaluate_predictions(y_test, y_pred, y_score)

        cv_f1 = []
        for train_idx, val_idx in tscv.split(X_scaled):
            m = get_supervised_models()[name]
            m.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])
            pred = m.predict(X_scaled.iloc[val_idx])
            cv_f1.append(f1_score(y.iloc[val_idx], pred, zero_division=0))

        results[name] = {
            "holdout": holdout_metrics,
            "cv_f1_mean": float(np.mean(cv_f1)),
            "cv_f1_std": float(np.std(cv_f1)),
            "model": model,
        }
    return results, (X_train, X_test, y_train, y_test), scaler


def run_unsupervised_comparison(X: pd.DataFrame, y: pd.Series, contamination: float = 0.1):
    """
    Runs Isolation Forest, One-Class SVM, and LOF WITHOUT using y for
    training, then reports how well their unsupervised anomaly flags
    align with the statistical label -- an independent check on whether
    "deviant" days are structurally distinguishable at all, not just
    something the labeling rule invented.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results = {}

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = (iso.fit_predict(Xs) == -1).astype(int)
    results["IsolationForest"] = evaluate_predictions(y, iso_pred)

    ocsvm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    ocsvm_pred = (ocsvm.fit_predict(Xs) == -1).astype(int)
    results["OneClassSVM"] = evaluate_predictions(y, ocsvm_pred)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_pred = (lof.fit_predict(Xs) == -1).astype(int)
    results["LocalOutlierFactor"] = evaluate_predictions(y, lof_pred)

    return results


if __name__ == "__main__":
    feat = pd.read_parquet("data/features_daily.parquet")
    labels = pd.read_parquet("data/labels_daily.parquet")
    X, y = prepare_xy(feat, labels)
    print(f"X: {X.shape}, y positive rate: {y.mean():.3f}")

    sup_results, split, scaler = run_supervised_comparison(X, y)
    print("\n=== Supervised (holdout) ===")
    for name, r in sup_results.items():
        h = r["holdout"]
        print(f"{name:20s} F1={h['f1']:.3f} Prec={h['precision']:.3f} "
              f"Recall={h['recall']:.3f} ROC-AUC={h.get('roc_auc', float('nan')):.3f} "
              f"| CV F1={r['cv_f1_mean']:.3f}+/-{r['cv_f1_std']:.3f}")

    unsup_results = run_unsupervised_comparison(X, y)
    print("\n=== Unsupervised (vs statistical label) ===")
    for name, r in unsup_results.items():
        print(f"{name:20s} F1={r['f1']:.3f} Prec={r['precision']:.3f} Recall={r['recall']:.3f}")
