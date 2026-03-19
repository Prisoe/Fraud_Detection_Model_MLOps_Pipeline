"""
Training — Fraud Detection (v2)
================================
Upgrades over v1:
  1. Threshold sweep over PR curve → optimal threshold stored with model
  2. Cross-validation metrics for tighter confidence intervals
  3. SHAP values computed and saved for explainability
  4. Business cost function for threshold selection
     (configurable FN cost / FP cost ratio)
  5. Model bundle = {model.joblib, threshold.json, shap_summary.json}
     all packed into model.tar.gz so evaluate + inference use same threshold
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_CHANNEL_VAL   = os.environ.get("SM_CHANNEL_VAL",   "/opt/ml/input/data/val")
SM_MODEL_DIR     = os.environ.get("SM_MODEL_DIR",      "/opt/ml/model")
SM_OUTPUT_DIR    = os.environ.get("SM_OUTPUT_DATA_DIR","/opt/ml/output/data")


def load_split(directory: str, fname: str) -> pd.DataFrame:
    path = os.path.join(directory, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {fname} at: {path}")
    return pd.read_csv(path)


def find_optimal_threshold(y_true, y_prob, fn_cost: float, fp_cost: float) -> dict:
    """
    Sweep PR curve and find threshold that minimises:
        cost = fn_cost * FN + fp_cost * FP
    This makes the threshold selection explicit about business tradeoffs.
    Default: fn_cost=10, fp_cost=1 (missing fraud 10× worse than false alarm)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    best_cost = float("inf")
    best_threshold = 0.5
    best_metrics = {}

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p == 0 and r == 0:
            continue
        # Recover TP, FP, FN from precision/recall
        tp = r * n_pos
        fn = n_pos - tp
        fp = tp / p - tp if p > 0 else 0
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
            best_metrics = {
                "threshold":  float(t),
                "precision":  float(p),
                "recall":     float(r),
                "estimated_fn": float(fn),
                "estimated_fp": float(fp),
                "business_cost": float(cost),
                "fn_cost_weight": fn_cost,
                "fp_cost_weight": fp_cost,
            }

    return best_metrics


def cross_validate(X, y, model_params: dict, n_splits: int = 5) -> dict:
    """Stratified k-fold CV to get confidence intervals on AUPRC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auprcs, roc_aucs = [], []

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        clf = XGBClassifier(**model_params, scale_pos_weight=spw,
                            use_label_encoder=False, eval_metric="aucpr",
                            verbosity=0)
        clf.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        probs = clf.predict_proba(X_vl)[:, 1]
        auprcs.append(average_precision_score(y_vl, probs))
        roc_aucs.append(roc_auc_score(y_vl, probs))
        print(f"  Fold {fold+1}/{n_splits}: AUPRC={auprcs[-1]:.4f}  ROC-AUC={roc_aucs[-1]:.4f}")

    return {
        "cv_auprc_mean":   float(np.mean(auprcs)),
        "cv_auprc_std":    float(np.std(auprcs)),
        "cv_auprc_min":    float(np.min(auprcs)),
        "cv_auprc_max":    float(np.max(auprcs)),
        "cv_roc_auc_mean": float(np.mean(roc_aucs)),
        "cv_roc_auc_std":  float(np.std(roc_aucs)),
        "cv_folds":        n_splits,
    }


def compute_shap_summary(model, X: np.ndarray, feature_names: list, max_rows: int = 2000) -> dict:
    """
    Compute SHAP values and return a summary suitable for JSON storage.
    Uses a sample for speed — full SHAP on 200k rows is slow.
    """
    try:
        import shap
        sample = X[:max_rows]
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # For binary classification XGBoost, shap_values shape = (n, features)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        importance  = dict(sorted(
            zip(feature_names, mean_abs.tolist()),
            key=lambda x: x[1], reverse=True
        ))
        return {
            "method":           "TreeExplainer",
            "n_samples":        len(sample),
            "feature_importance_mean_abs_shap": importance,
            "top_5_features":   list(importance.keys())[:5],
        }
    except ImportError:
        print("[train] shap not installed — skipping SHAP computation")
        return {"method": "unavailable", "reason": "shap package not installed"}
    except Exception as e:
        print(f"[train] SHAP failed: {e}")
        return {"method": "failed", "reason": str(e)}


def main():
    ap = argparse.ArgumentParser()
    # Hyperparameters
    ap.add_argument("--n-estimators",     type=int,   default=300)
    ap.add_argument("--max-depth",        type=int,   default=6)
    ap.add_argument("--learning-rate",    type=float, default=0.05)
    ap.add_argument("--subsample",        type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--random-state",     type=int,   default=42)
    # Threshold optimisation
    ap.add_argument("--fn-cost",          type=float, default=10.0,
                    help="Business cost of a False Negative (missed fraud)")
    ap.add_argument("--fp-cost",          type=float, default=1.0,
                    help="Business cost of a False Positive (false alarm)")
    # CV
    ap.add_argument("--cv-folds",         type=int,   default=5)
    ap.add_argument("--skip-cv",          action="store_true",
                    help="Skip cross-validation (faster, less rigorous)")
    # Channel overrides
    ap.add_argument("--train-dir",        default=SM_CHANNEL_TRAIN)
    ap.add_argument("--val-dir",          default=SM_CHANNEL_VAL)
    args = ap.parse_args()

    train_df = load_split(args.train_dir, "train.csv")
    val_df   = load_split(args.val_dir,   "val.csv")

    X_train       = train_df.drop(columns=["label"]).values
    y_train       = train_df["label"].values
    X_val         = val_df.drop(columns=["label"]).values
    y_val         = val_df["label"].values
    feature_names = train_df.drop(columns=["label"]).columns.tolist()

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    print(f"[train] Train: {len(y_train):,} rows | Fraud: {n_pos:,} ({n_pos/len(y_train):.4%})")
    print(f"[train] scale_pos_weight: {spw:.1f}")
    print(f"[train] Cost weights: FN={args.fn_cost}× FP={args.fp_cost}×")

    model_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    # ── Cross-validation (on combined train+val for CV purposes)
    cv_metrics = {}
    if not args.skip_cv:
        print(f"\n[train] Running {args.cv_folds}-fold stratified CV...")
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        cv_metrics = cross_validate(X_all, y_all, model_params, n_splits=args.cv_folds)
        print(f"[train] CV AUPRC: {cv_metrics['cv_auprc_mean']:.4f} ± {cv_metrics['cv_auprc_std']:.4f}")

    # ── Final model trained on full train split
    print("\n[train] Training final model on train split...")
    model = XGBClassifier(
        **model_params,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="aucpr",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # ── Validation metrics
    y_prob = model.predict_proba(X_val)[:, 1]
    avg_precision = float(average_precision_score(y_val, y_prob))
    roc_auc       = float(roc_auc_score(y_val, y_prob))

    # ── Optimal threshold via cost function
    print("\n[train] Sweeping PR curve for optimal threshold...")
    threshold_info = find_optimal_threshold(y_val, y_prob, args.fn_cost, args.fp_cost)
    opt_threshold  = threshold_info["threshold"]
    y_pred         = (y_prob >= opt_threshold).astype(int)

    f1        = float(f1_score(y_val, y_pred, zero_division=0))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall    = float(recall_score(y_val, y_pred, zero_division=0))

    print(f"\n[train] Validation metrics:")
    print(f"  AUPRC (avg_precision) : {avg_precision:.4f}  ← quality gate")
    print(f"  ROC-AUC               : {roc_auc:.4f}")
    print(f"  Optimal threshold     : {opt_threshold:.4f}  (cost: FN={args.fn_cost}× FP={args.fp_cost}×)")
    print(f"  F1  @ opt threshold   : {f1:.4f}")
    print(f"  Precision             : {precision:.4f}")
    print(f"  Recall                : {recall:.4f}")

    # ── SHAP
    print("\n[train] Computing SHAP feature importance...")
    shap_summary = compute_shap_summary(model, X_val, feature_names)
    if "top_5_features" in shap_summary:
        print(f"  Top features: {shap_summary['top_5_features']}")

    # ── Save model bundle
    os.makedirs(SM_MODEL_DIR, exist_ok=True)

    # 1. Model
    joblib.dump(model, os.path.join(SM_MODEL_DIR, "model.joblib"))

    # 2. Threshold — loaded by evaluate.py and inference.py
    threshold_bundle = {
        "threshold":      opt_threshold,
        "fn_cost":        args.fn_cost,
        "fp_cost":        args.fp_cost,
        "selection_method": "cost_minimisation",
        **threshold_info,
    }
    with open(os.path.join(SM_MODEL_DIR, "threshold.json"), "w") as fh:
        json.dump(threshold_bundle, fh, indent=2)

    # 3. SHAP summary
    with open(os.path.join(SM_MODEL_DIR, "shap_summary.json"), "w") as fh:
        json.dump(shap_summary, fh, indent=2)

    print(f"\n[train] Model bundle saved to {SM_MODEL_DIR}/")
    print(f"  model.joblib | threshold.json | shap_summary.json")

    # ── Metrics artifact
    os.makedirs(SM_OUTPUT_DIR, exist_ok=True)
    metrics = {
        "avg_precision":   avg_precision,
        "roc_auc":         roc_auc,
        "f1":              f1,
        "precision":       precision,
        "recall":          recall,
        "optimal_threshold": opt_threshold,
        "n_train":         int(len(y_train)),
        "n_fraud_train":   n_pos,
        "scale_pos_weight": float(spw),
        **cv_metrics,
        "threshold_selection": threshold_info,
    }
    with open(os.path.join(SM_OUTPUT_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("✅ Training complete")


if __name__ == "__main__":
    main()
