#!/usr/bin/env python
"""
Evaluate Species→Location (Member B) model on the test set
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_test_labels(test_npz_path, top_species):
    """Load test labels for top species"""
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
            continue
        loc_inds = taxon2inds[sp]
        Y_loc[loc_inds, j] = 1
    
    return Y_loc, n_locs

def build_region_test_dataset(test_features_path, test_npz_path, top_species):
    """Build region-level test dataset"""
    test_df = pd.read_csv(test_features_path)
    Y_loc, n_locs = load_test_labels(test_npz_path, top_species)
    
    if len(test_df) != n_locs:
        raise ValueError("test_features row count doesn't match test_locs count")
    
    # Location labels + region
    Y_cols = [str(s) for s in top_species]
    labels_df = pd.DataFrame(Y_loc, columns=Y_cols)
    labels_df["region"] = test_df["region"].values
    
    # Aggregate by region (if any location in region has species, region = 1)
    region_labels = (
        labels_df
        .groupby("region")
        .max()
        .reset_index()
    )
    
    # Region features: take first row per region
    feature_cols = ["grid_lat", "grid_lon", "sin_lon", "cos_lon", "hemisphere"]
    region_features = (
        test_df
        .groupby("region")
        .first()
        .reset_index()[["region"] + feature_cols]
    )
    
    # Merge features and labels
    data_region = region_features.merge(region_labels, on="region", how="inner")
    
    X_raw = data_region[feature_cols]
    X_region = pd.get_dummies(X_raw, columns=["hemisphere"], drop_first=True)
    
    Y_region = data_region[Y_cols].values.astype(int)
    
    return X_region, Y_region

def evaluate_spec2loc_model(model_obj, X_test, Y_test):
    """Evaluate spec2loc model (multiple individual species models)"""
    models = model_obj["models"]
    top_species = model_obj["top_species"]
    feature_columns = model_obj.get("feature_columns", X_test.columns.tolist())
    
    # Align features
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_aligned = X_test[feature_columns]
    
    # Predict probabilities for each species
    n_samples, n_species = Y_test.shape
    Y_proba = np.zeros((n_samples, n_species))
    
    for i, species_id in enumerate(top_species):
        if species_id in models:
            model = models[species_id]
            if hasattr(model, "predict_proba"):
                Y_proba[:, i] = model.predict_proba(X_aligned)[:, 1]
            else:
                try:
                    Y_proba[:, i] = model.decision_function(X_aligned)
                except:
                    Y_proba[:, i] = model.predict(X_aligned)
        else:
            Y_proba[:, i] = 0.0
    
    # Calculate metrics
    metrics = {
        "n_samples": int(n_samples),
        "n_species": int(n_species),
        "n_species_with_models": len(models)
    }
    
    # Micro-averaged metrics
    y_true_flat = Y_test.ravel()
    y_score_flat = Y_proba.ravel()
    
    try:
        micro_roc = roc_auc_score(y_true_flat, y_score_flat)
    except:
        micro_roc = np.nan
    
    try:
        micro_pr = average_precision_score(y_true_flat, y_score_flat)
    except:
        micro_pr = np.nan
    
    # Find optimal threshold for F1
    best_thr = 0.5
    best_f1 = 0.0
    
    thresholds = np.linspace(0.05, 0.5, 20)
    for thr in thresholds:
        y_pred = (Y_proba >= thr).astype(int)
        f1 = f1_score(Y_test, y_pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    # Final predictions with best threshold
    Y_pred_best = (Y_proba >= best_thr).astype(int)
    
    # All metrics
    f1_micro = f1_score(Y_test, Y_pred_best, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred_best, average="macro", zero_division=0)
    
    metrics.update({
        "micro_roc_auc": float(micro_roc),
        "micro_pr_auc": float(micro_pr),
        "micro_f1": float(f1_micro),
        "macro_f1": float(f1_macro),
        "best_threshold": float(best_thr)
    })
    
    # Per-species metrics
    per_species = []
    for i, species_id in enumerate(top_species):
        y_true_species = Y_test[:, i]
        y_score_species = Y_proba[:, i]
        
        if y_true_species.sum() > 0:  # Only if species has positive samples
            try:
                roc_species = roc_auc_score(y_true_species, y_score_species)
            except:
                roc_species = np.nan
            
            try:
                pr_species = average_precision_score(y_true_species, y_score_species)
            except:
                pr_species = np.nan
            
            y_pred_species = Y_pred_best[:, i]
            f1_species = f1_score(y_true_species, y_pred_species, zero_division=0)
            
            per_species.append({
                "species_id": int(species_id),
                "n_positive_regions": int(y_true_species.sum()),
                "n_total_regions": int(len(y_true_species)),
                "prevalence": float(y_true_species.sum() / len(y_true_species)),
                "roc_auc": float(roc_species),
                "pr_auc": float(pr_species),
                "f1": float(f1_species)
            })
    
    return metrics, per_species, (Y_test, Y_proba)

def plot_curves_spec2loc(Y_test, Y_proba, out_prefix):
    """Plot ROC and PR curves for spec2loc"""
    y_true_flat = Y_test.ravel()
    y_score_flat = Y_proba.ravel()
    
    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
        roc_auc = roc_auc_score(y_true_flat, y_score_flat)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f"micro-ROC (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Species→Location: Micro-averaged ROC curve")
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_prefix + "_roc_micro.png", dpi=150)
        plt.close()
    except:
        print("Could not plot ROC curve")
    
    # PR curve
    try:
        precision, recall, _ = precision_recall_curve(y_true_flat, y_score_flat)
        ap_micro = average_precision_score(y_true_flat, y_score_flat)
        
        plt.figure()
        plt.plot(recall, precision, label=f"micro-PR (AP={ap_micro:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Species→Location: Micro-averaged Precision-Recall curve")
        plt.grid(True)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(out_prefix + "_pr_micro.png", dpi=150)
        plt.close()
    except:
        print("Could not plot PR curve")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Species→Location model on test set")
    parser.add_argument("--test-features", required=True, help="Test features CSV")
    parser.add_argument("--test-npz", required=True, help="Test NPZ file")
    parser.add_argument("--model", required=True, help="Spec2loc model file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--figdir", help="Output directory for figures (optional)")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    if args.figdir:
        os.makedirs(args.figdir, exist_ok=True)
    
    print("=" * 60)
    print("Species→Location (Member B) Test Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"Loading model from: {args.model}")
    model_obj = joblib.load(args.model)
    top_species = [int(s) for s in model_obj["top_species"]]
    print(f"Model contains {len(top_species)} species")
    
    # Build test dataset
    print("Building test dataset...")
    X_test, Y_test = build_region_test_dataset(args.test_features, args.test_npz, top_species)
    print(f"Test set: X={X_test.shape}, Y={Y_test.shape}")
    
    # Evaluate
    print("Evaluating model...")
    metrics, per_species, (Y_eval, Y_proba) = evaluate_spec2loc_model(model_obj, X_test, Y_test)
    
    # Print results
    print("\n" + "=" * 40)
    print("OVERALL RESULTS")
    print("=" * 40)
    print(f"Micro F1:     {metrics['micro_f1']:.4f}")
    print(f"Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"Micro ROC-AUC: {metrics['micro_roc_auc']:.4f}")
    print(f"Micro PR-AUC: {metrics['micro_pr_auc']:.4f}")
    print(f"Best threshold: {metrics['best_threshold']:.3f}")
    print(f"Species with models: {metrics['n_species_with_models']}/{metrics['n_species']}")
    
    # Save results
    results = {
        "model_type": "spec2loc",
        "overall_metrics": metrics,
        "per_species_results": per_species,
        "test_set_info": {
            "n_samples": int(Y_test.shape[0]),
            "n_species": int(Y_test.shape[1]),
            "n_positive_labels": int(Y_test.sum())
        }
    }
    
    out_file = os.path.join(args.outdir, "spec2loc_test_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")
    
    # Plot curves if requested
    if args.figdir:
        print("Generating plots...")
        plot_curves_spec2loc(Y_eval, Y_proba, os.path.join(args.figdir, "spec2loc"))
    
    print("\n" + "=" * 60)
    print("Evaluation completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
