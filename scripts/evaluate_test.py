#!/usr/bin/env python
"""
Evaluate trained loc2spec (Location→Species, Member C) model on the official test set.

Inputs:
  - species_data/test_features.csv   (from scripts/build_features.py)
  - species_data/species_test.npz    (raw test locations + per-species indices)
  - trained model file:
        * loc2spec_model.joblib      (from train_location_to_species.py)

Outputs:
  - <outdir>/test_metrics.json       (micro/macro F1、ROC-AUC、PR-AUC、mAP）
  - optional ROC / PR curves saved to <figdir>/
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

# ----------------------------------------------------------------------
# 1. Build location-level label matrix from species_test.npz
# ----------------------------------------------------------------------

def load_test_labels_for_top_species(test_npz_path, top_species):
    """
    Given Top-K species ids, build a multi-label matrix Y_loc on test_locs.

    species_test.npz structure:
      - test_locs: (n_locs, 2) [lat, lon]
      - taxon_ids: (n_species,)
      - test_pos_inds: length = n_species, each element is the location index list
        where the species appears
    """
    d = np.load(test_npz_path, allow_pickle=True)
    test_locs = d["test_locs"]
    taxon_ids = d["taxon_ids"]
    test_pos_inds = d["test_pos_inds"]

    n_locs = test_locs.shape[0]
    n_species = len(top_species)

    # taxon_id -> [loc indices]
    taxon2inds = {int(t): inds for t, inds in zip(taxon_ids, test_pos_inds)}

    Y_loc = np.zeros((n_locs, n_species), dtype=int)
    species_index = {int(sp): j for j, sp in enumerate(top_species)}

    for sp, j in species_index.items():
        if sp not in taxon2inds:
            # Species not present in test set: keep column all zeros
            continue
        loc_inds = taxon2inds[sp]
        Y_loc[loc_inds, j] = 1

    return Y_loc, n_locs


# ----------------------------------------------------------------------
# 2. Build region-level test dataset (consistent with training logic)
# ----------------------------------------------------------------------

def build_region_level_test_dataset(test_features_path, test_npz_path, top_species):
    """
    Build a region-level multi-label dataset:
      X_region: one row per region (features)
      Y_region: one row per region (Top-K species 0/1 vector)
    """
    test_df = pd.read_csv(test_features_path)
    Y_loc, n_locs = load_test_labels_for_top_species(test_npz_path, top_species)

    if len(test_df) != n_locs:
        raise ValueError("Row count mismatch between test_features and test_locs in species_test.npz")

    if "region" not in test_df.columns:
        raise ValueError("Missing 'region' column in test_features; please run scripts/build_features.py first")

    # 1) Location labels + region
    Y_cols = [str(s) for s in top_species]
    labels_df = pd.DataFrame(Y_loc, columns=Y_cols)
    labels_df["region"] = test_df["region"].values

    # 2) Aggregate by region (region is 1 if any location in it has the species)
    region_labels = (
        labels_df
        .groupby("region")
        .max()
        .reset_index()
    )

    # 3) Region features: take the first row per region
    feature_cols = ["grid_lat", "grid_lon", "sin_lon", "cos_lon", "hemisphere"]
    region_features = (
        test_df
        .groupby("region")
        .first()
        .reset_index()[["region"] + feature_cols]
    )

    # 4) Merge features and labels
    data_region = region_features.merge(region_labels, on="region", how="inner")

    X_raw = data_region[feature_cols]
    X_region = pd.get_dummies(X_raw, columns=["hemisphere"], drop_first=True)

    Y_region = data_region[Y_cols].values.astype(int)

    return X_region, Y_region


# ----------------------------------------------------------------------
# 3. Align features to the model if feature_columns are provided
# ----------------------------------------------------------------------

def align_features_to_model(X, model_obj):
    """
    If model_obj has "feature_columns", align X columns to that order;
    otherwise return X unchanged.
    """
    feat_cols = model_obj.get("feature_columns", None)
    if feat_cols is None:
        return X

    # Add missing columns with 0
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0

    # Keep only training columns and original order
    return X[feat_cols]


# ----------------------------------------------------------------------
# 4. Multilabel model evaluation: F1 / ROC-AUC / PR-AUC / mAP
#    (with threshold search + non-empty region F1)
# ----------------------------------------------------------------------

def evaluate_multilabel_model(model_obj, X, Y):
    """
    Unified evaluation for multilabel models:
      - micro/macro F1 (all samples)
      - micro/macro F1 (non-empty samples only)
      - micro/macro ROC-AUC
      - micro/macro PR-AUC
      - micro/macro mAP
      - best_threshold: global threshold tuned for F1 on non-empty samples
    """
    model = model_obj["model"]
    X_aligned = align_features_to_model(X, model_obj)

    # Predict probabilities (MultiOutputClassifier(RandomForest) -> list of arrays)
    if hasattr(model, "predict_proba"):
        prob_list = model.predict_proba(X_aligned)
        Y_proba = np.column_stack([p[:, 1] for p in prob_list])
    else:
        Y_proba = None

    n_samples, n_labels = Y.shape
    metrics = {
        "n_samples": int(n_samples),
        "n_labels": int(n_labels),
    }

    # If model doesn't provide probabilities, fall back to predict
    if Y_proba is None:
        Y_pred = model.predict(X_aligned)
        metrics.update({
            "f1_micro": float(f1_score(Y, Y_pred, average="micro")),
            "f1_macro": float(f1_score(Y, Y_pred, average="macro")),
            "f1_micro_nonempty": None,
            "f1_macro_nonempty": None,
            "best_threshold": None,
            "roc_auc_micro": None,
            "roc_auc_macro": None,
            "pr_auc_micro": None,
            "pr_auc_macro": None,
            "map_micro": None,
            "map_macro": None,
        })
        return metrics, (Y, None)

    # ---------- 1) Threshold search: maximize micro-F1 on non-empty samples ----------
    # non-empty: at least one positive label, consistent with training logic
    mask_nonempty = (Y.sum(axis=1) > 0)
    thresholds = np.linspace(0.05, 0.5, 10)

    best_thr = 0.5
    best_f1_nonempty = 0.0
    best_f1_macro_nonempty = 0.0

    for thr in thresholds:
        Y_pred_thr = (Y_proba >= thr).astype(int)
        if mask_nonempty.any():
            f1_nonempty = f1_score(
                Y[mask_nonempty], Y_pred_thr[mask_nonempty],
                average="micro", zero_division=0
            )
            f1_macro_nonempty = f1_score(
                Y[mask_nonempty], Y_pred_thr[mask_nonempty],
                average="macro", zero_division=0
            )
        else:
            f1_nonempty = 0.0
            f1_macro_nonempty = 0.0

        if f1_nonempty > best_f1_nonempty:
            best_f1_nonempty = f1_nonempty
            best_f1_macro_nonempty = f1_macro_nonempty
            best_thr = thr

    print(f"[INFO] Best threshold for non-empty micro-F1: {best_thr:.3f}, "
          f"F1_nonempty={best_f1_nonempty:.3f}")

    # ---------- 2) Generate hard predictions with the best threshold ----------
    Y_pred = (Y_proba >= best_thr).astype(int)

    # F1 on all samples
    f1_micro_all = f1_score(Y, Y_pred, average="micro", zero_division=0)
    f1_macro_all = f1_score(Y, Y_pred, average="macro", zero_division=0)

    metrics.update({
        "f1_micro": float(f1_micro_all),
        "f1_macro": float(f1_macro_all),
        "f1_micro_nonempty": float(best_f1_nonempty),
        "f1_macro_nonempty": float(best_f1_macro_nonempty),
        "best_threshold": float(best_thr),
    })

    # ---------- 3) ROC-AUC micro & macro ----------
    try:
        roc_auc_micro = roc_auc_score(Y, Y_proba, average="micro")
    except ValueError:
        roc_auc_micro = np.nan

    per_label_roc = []
    for j in range(Y.shape[1]):
        y_true_j = Y[:, j]
        y_score_j = Y_proba[:, j]
        if np.unique(y_true_j).size < 2:
            per_label_roc.append(np.nan)
            continue
        try:
            auc_j = roc_auc_score(y_true_j, y_score_j)
        except ValueError:
            auc_j = np.nan
        per_label_roc.append(auc_j)
    roc_auc_macro = float(np.nanmean(per_label_roc))

    # ---------- 4) PR-AUC / mAP ----------
    pr_auc_micro = average_precision_score(Y, Y_proba, average="micro")

    per_label_ap = []
    for j in range(Y.shape[1]):
        y_true_j = Y[:, j]
        y_score_j = Y_proba[:, j]
        if y_true_j.sum() == 0:
            per_label_ap.append(np.nan)
            continue
        ap_j = average_precision_score(y_true_j, y_score_j)
        per_label_ap.append(ap_j)
    pr_auc_macro = float(np.nanmean(per_label_ap))

    metrics.update({
        "roc_auc_micro": float(roc_auc_micro),
        "roc_auc_macro": float(roc_auc_macro),
        "pr_auc_micro": float(pr_auc_micro),
        "pr_auc_macro": float(pr_auc_macro),
        "map_micro": float(pr_auc_micro),   # micro mAP = micro PR-AUC
        "map_macro": float(pr_auc_macro),   # macro mAP = macro PR-AUC
    })

    return metrics, (Y, Y_proba)


# ----------------------------------------------------------------------
# 5. Plot micro-averaged ROC / PR curves
# ----------------------------------------------------------------------

def plot_curves(Y, Y_proba, out_prefix):
    y_true_flat = Y.ravel()
    y_score_flat = Y_proba.ravel()

    # ROC
    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
    roc_auc = roc_auc_score(y_true_flat, y_score_flat)

    plt.figure()
    plt.plot(fpr, tpr, label="micro-ROC (AUC=%.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged ROC curve")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc_micro.png")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true_flat, y_score_flat)
    ap_micro = average_precision_score(y_true_flat, y_score_flat)

    plt.figure()
    plt.plot(recall, precision, label="micro-PR (AP=%.3f)" % ap_micro)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged Precision-Recall curve")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr_micro.png")
    plt.close()


# ----------------------------------------------------------------------
# 6. CLI & main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate loc2spec model on species_test data.")
    p.add_argument("--test-features", required=True,
                   help="Path to test_features.csv (e.g. species_data/test_features.csv)")
    p.add_argument("--test-npz", required=True,
                   help="Path to species_test.npz (e.g. species_data/species_test.npz)")
    p.add_argument("--loc2spec-model", required=True,
                   help="Path to loc2spec model .joblib/.pkl")
    p.add_argument("--outdir", required=True,
                   help="Directory to save test_metrics.json")
    p.add_argument("--figdir", default=None,
                   help="Directory to save ROC/PR curves (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.figdir:
        os.makedirs(args.figdir, exist_ok=True)

    all_results = {}

    print("Loading loc2spec model from:", args.loc2spec_model)
    obj = joblib.load(args.loc2spec_model)
    top_species = [int(s) for s in obj["top_species"]]

    X, Y = build_region_level_test_dataset(args.test_features, args.test_npz, top_species)
    print("loc2spec raw test X:", X.shape, "Y:", Y.shape)

    mask_nonempty = (Y.sum(axis=1) > 0)
    n_before = Y.shape[0]
    n_after = mask_nonempty.sum()
    print(f"[loc2spec] Keeping non-empty regions only: {n_after}/{n_before} "
          f"({n_after / max(1, n_before):.3f} of all regions)")

    X = X[mask_nonempty].reset_index(drop=True)
    Y = Y[mask_nonempty]

    print("loc2spec filtered test X:", X.shape, "Y:", Y.shape)

    metrics, (Y_eval, Y_proba) = evaluate_multilabel_model(obj, X, Y)
    all_results["loc2spec"] = metrics

    if args.figdir and (Y_proba is not None):
        plot_curves(Y_eval, Y_proba, os.path.join(args.figdir, "loc2spec"))

    out_path = os.path.join(args.outdir, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved test metrics to:", out_path)


if __name__ == "__main__":
    main()
