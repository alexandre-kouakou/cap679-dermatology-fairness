import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def choose_threshold(val_df: pd.DataFrame, target_sensitivity: float = 0.90) -> float:
    """Highest threshold achieving >= target_sensitivity on val_df (frozen for test/DDI)."""
    fpr, tpr, thresholds = roc_curve(val_df["target"], val_df["prob"])
    # roc_curve returns thresholds in decreasing order; argmax on a bool array
    # returns the first True, i.e. the highest threshold that still hits the target.
    return float(thresholds[np.argmax(tpr >= target_sensitivity)])


def _metrics_at(df: pd.DataFrame, threshold: float) -> dict:
    """AUC, sensitivity, specificity for a single (sub)frame."""
    y = df["target"].values
    p = df["prob"].values
    pred = p >= threshold

    n = len(df)
    n_mal = int((y == 1).sum())
    n_ben = int((y == 0).sum())

    auc = roc_auc_score(y, p) if 0 < n_mal < n else float("nan")

    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {"n": n, "n_malignant": n_mal, "auc": auc,
            "sensitivity": sens, "specificity": spec}


def group_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per skin group, plus an 'all' row computed on the whole frame."""
    rows = []
    for group, d in df.groupby("skin_group", sort=True):
        rows.append({"skin_group": group, **_metrics_at(d, threshold)})
    rows.append({"skin_group": "all", **_metrics_at(df, threshold)})
    return pd.DataFrame(rows).set_index("skin_group").reindex(["light", "medium", "dark", "all"])


def fairness_gaps(df: pd.DataFrame, threshold: float) -> dict:
    """AUC gap (light−dark), sensitivity gap (light−dark), equalized odds difference."""
    m = group_metrics(df, threshold)

    auc_gap = m.loc["light", "auc"] - m.loc["dark", "auc"]
    sens_gap = m.loc["light", "sensitivity"] - m.loc["dark", "sensitivity"]

    # Equalized odds difference = max of |sensitivity gap| and |FPR gap| across groups.
    # Fairlearn's convention is largest minus smallest across all groups; match that.
    fpr = 1.0 - m.drop(index="all")["specificity"]
    eod = max(
        m.drop(index="all")["sensitivity"].max() - m.drop(index="all")["sensitivity"].min(),
        fpr.max() - fpr.min(),
    )

    return {
        "auc_gap_light_minus_dark": float(auc_gap),
        "sensitivity_gap_light_minus_dark": float(sens_gap),
        "equalized_odds_difference": float(eod),
    }