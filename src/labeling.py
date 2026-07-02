"""
labeling.py
===========
Generates "behavioral deviation" labels for days with no ground truth.

Why this is the hardest design decision in the project
--------------------------------------------------------
We have no clinician-verified labels for "this day represented a genuine
behavioral change." Everything downstream (supervised models, precision/
recall, ROC-AUC) needs *some* label to train and evaluate against, so we
have to manufacture one -- but how we manufacture it silently determines
what the "supervised" models can ever learn. If the label-generating rule
and the "predictive" model use the same signal, the project would really
just be re-deriving its own label, which would inflate performance
metrics without proving anything.

To avoid that, we deliberately generate labels using a DIFFERENT method
family than the "prediction" models the pipeline later evaluates:

  1. STATISTICAL DEVIATION LABEL (this module's primary label):
     A day is flagged "deviant" if enough individual features fall
     outside the person's own recent rolling distribution (z-score
     beyond a threshold, using the same rolling mean/std already
     computed in features.py). This is transparent, interpretable, and
     directly tied to the "trend-based" framing of the research
     question (deviation from *this person's own recent normal*, not
     from a population norm).

  2. UNSUPERVISED ANOMALY LABEL (secondary / validation label):
     Isolation Forest and Local Outlier Factor scores on the full
     feature matrix, used as an independent check -- if the statistical
     label and an unrelated unsupervised method agree on which days are
     unusual, that's much stronger evidence the label reflects a real
     pattern rather than an artifact of the labeling rule itself.

Both labels are kept in the output so the notebook can report their
agreement (e.g. Cohen's kappa) as a sanity check before trusting either
one enough to train supervised models against it -- and that agreement
analysis should go directly into the report's methodology section as
justification for the labeling approach.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# Feature columns used for statistical z-score deviation (base features,
# not their own rolling stats, to avoid the label leaking directly into
# features a model might also use).
BASE_FEATURES_FOR_ZSCORE = [
    "total_events", "motion_events", "item_events", "door_events",
    "morning_frac", "afternoon_frac", "evening_frac", "night_frac",
    "bed_night_events", "night_motion_events", "n_long_inactivity_gaps",
    "max_inactivity_gap_min", "room_transitions", "routine_similarity_cos",
]


def statistical_deviation_labels(
    feat: pd.DataFrame,
    z_threshold: float = 2.0,
    min_features_deviant: int = 3,
    window: int = 14,
) -> pd.DataFrame:
    """
    For each base feature, compute a rolling z-score of that day's value
    against the trailing `window` days (excluding the current day, so the
    day being scored can't inflate its own baseline). A day is labeled
    'deviant' (1) if at least `min_features_deviant` features have
    |z| >= z_threshold simultaneously -- requiring multiple co-occurring
    deviations is a deliberate choice to reduce false positives from a
    single noisy feature (e.g. sensor glitch), which matters a lot given
    we're using this label to train and evaluate models.
    """
    out = pd.DataFrame(index=feat.index)
    z_cols = []
    for col in BASE_FEATURES_FOR_ZSCORE:
        roll_mean = feat[col].shift(1).rolling(window, min_periods=5).mean()
        roll_std = feat[col].shift(1).rolling(window, min_periods=5).std()
        z = (feat[col] - roll_mean) / roll_std.replace(0, np.nan)
        out[f"z_{col}"] = z
        z_cols.append(f"z_{col}")

    out["n_features_deviant"] = (out[z_cols].abs() >= z_threshold).sum(axis=1)
    out["label_statistical"] = (out["n_features_deviant"] >= min_features_deviant).astype(int)
    out.loc[out[z_cols].isna().all(axis=1), "label_statistical"] = np.nan
    return out


def unsupervised_anomaly_scores(
    feat: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Independent anomaly scoring using Isolation Forest and Local Outlier
    Factor on standardized base features. `contamination` is an assumed
    prior fraction of anomalous days -- 0.1 is a conventional default
    when there's no basis to estimate the true rate, and should be
    reported as an assumption, not a finding. These scores exist mainly
    to CHECK the statistical label (see module docstring), not as the
    primary label -- but they're saved here in case the notebook wants
    to explore them as alternative labels too.
    """
    if feature_cols is None:
        feature_cols = BASE_FEATURES_FOR_ZSCORE

    X = feat[feature_cols].copy()
    valid = X.dropna().index
    Xv = X.loc[valid]
    Xs = StandardScaler().fit_transform(Xv)

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso_pred = iso.fit_predict(Xs)          # -1 = anomaly, 1 = normal
    iso_score = iso.decision_function(Xs)    # lower = more anomalous

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_pred = lof.fit_predict(Xs)
    lof_score = lof.negative_outlier_factor_

    out = pd.DataFrame(index=feat.index)
    out.loc[valid, "label_isoforest"] = (iso_pred == -1).astype(int)
    out.loc[valid, "score_isoforest"] = iso_score
    out.loc[valid, "label_lof"] = (lof_pred == -1).astype(int)
    out.loc[valid, "score_lof"] = lof_score
    return out


def combine_labels(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full label set and reports agreement between the
    statistical label and the two unsupervised methods (Cohen's kappa),
    which the notebook should surface explicitly as evidence for (or
    against) trusting the statistical label for supervised training.
    """
    from sklearn.metrics import cohen_kappa_score

    stat = statistical_deviation_labels(feat)
    unsup = unsupervised_anomaly_scores(feat)
    labels = stat.join(unsup)

    agreements = {}
    valid = labels.dropna(subset=["label_statistical", "label_isoforest", "label_lof"])
    if len(valid) > 5:
        agreements["stat_vs_isoforest_kappa"] = cohen_kappa_score(
            valid["label_statistical"], valid["label_isoforest"])
        agreements["stat_vs_lof_kappa"] = cohen_kappa_score(
            valid["label_statistical"], valid["label_lof"])

    return labels, agreements


if __name__ == "__main__":
    feat = pd.read_parquet("data/features_daily.parquet")
    labels, agreements = combine_labels(feat)
    labels.to_parquet("data/labels_daily.parquet")
    print("Label prevalence (statistical):", labels["label_statistical"].mean())
    print("Label prevalence (isoforest):", labels["label_isoforest"].mean())
    print("Label prevalence (lof):", labels["label_lof"].mean())
    print("Agreement:", agreements)
