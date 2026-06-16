import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    best = {"threshold": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for th in np.unique(score):
        pred = (score >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(th),
                "f1": float(f1),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }
    return best


def classification_metrics(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    best = best_f1_threshold(y_true, score)
    pred = (score >= best["threshold"]).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, pred, zero_division=0)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "best_threshold": best["threshold"],
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_acc": float(0.5 * (sensitivity + specificity)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_prediction_scores(df):
    score_cols = [
        c for c in df.columns
        if c.startswith("score_") and pd.api.types.is_numeric_dtype(df[c])
    ]

    fold_rows = []
    for score_col in score_cols:
        for (part, fold), g in df.groupby(["base_part", "fold"]):
            if g["y_true"].nunique() < 2:
                continue
            metrics = classification_metrics(g["y_true"].values, g[score_col].values)
            fold_rows.append({
                "score_col": score_col,
                "base_part": part,
                "fold": int(fold),
                "n": int(len(g)),
                "n_normal": int((g["y_true"] == 0).sum()),
                "n_foreign": int((g["y_true"] == 1).sum()),
                **metrics,
            })

    fold_df = pd.DataFrame(fold_rows)

    part_df = (
        fold_df.groupby(["score_col", "base_part"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            acc_mean=("acc", "mean"),
            f1_mean=("f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            specificity_mean=("specificity", "mean"),
            balanced_acc_mean=("balanced_acc", "mean"),
        )
        .reset_index()
    )

    macro_df = (
        part_df.groupby("score_col")
        .agg(
            macro_auc=("auc_mean", "mean"),
            macro_auc_std_over_parts=("auc_mean", "std"),
            macro_pr_auc=("pr_auc_mean", "mean"),
            macro_pr_auc_std_over_parts=("pr_auc_mean", "std"),
            macro_f1=("f1_mean", "mean"),
            macro_precision=("precision_mean", "mean"),
            macro_recall=("recall_mean", "mean"),
            macro_specificity=("specificity_mean", "mean"),
            macro_balanced_acc=("balanced_acc_mean", "mean"),
        )
        .reset_index()
        .sort_values(["macro_auc", "macro_pr_auc", "macro_f1"], ascending=False)
    )

    return fold_df, part_df, macro_df

