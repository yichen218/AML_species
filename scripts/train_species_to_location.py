#!/usr/bin/env python
"""
Species → Location Prediction Model (Member B)

Task Definition:
Given a species, predict the geographic regions where it may occur.

Input: Species features (taxon_id, basic ecological characteristics)
Output: Geographic region presence probability (grid-based binary classification)

Implementation Strategy:
1. Train independent binary classification models for each species separately
2. For each species, build positive samples (regions where the species occurs) and negative samples (other regions)
3. Use spatial K-fold cross-validation to prevent geographic leakage
4. Evaluate prediction performance for each species
"""

import argparse
import os
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib


def build_species_location_dataset(train_path: str, top_k: int = 50):
    """
    Build species→location prediction dataset
    
    Logic:
    1. Select Top-K frequent species
    2. For each species, build binary classification task:
       - Positive samples: regions where the species occurs
       - Negative samples: other regions (random sampling, maintaining balance)
    3. Features: regional geographic features (grid_lat, grid_lon, sin_lon, cos_lon, hemisphere)
    4. Labels: whether the species occurs in the region
    """
    train = pd.read_csv(train_path)
    
    # 1. Count species frequency, select Top-K species
    species_counts = Counter(train['taxon_id'])
    top_species = [species for species, _ in species_counts.most_common(top_k)]
    
    # 2. Build regional features
    feature_cols = ["grid_lat", "grid_lon", "sin_lon", "cos_lon", "hemisphere"]
    region_features = (
        train.groupby("region")
        .first()
        .reset_index()[["region"] + feature_cols]
    )
    
    # One-hot encode hemisphere
    region_features_encoded = pd.get_dummies(region_features, columns=["hemisphere"], drop_first=True)
    
    # 3. Build label matrix for each species
    # species_region_matrix: rows=regions, columns=species, values=whether species occurs in region
    species_region_matrix = np.zeros((len(region_features_encoded), top_k), dtype=int)
    
    # Build species to index mapping
    species_to_idx = {species: idx for idx, species in enumerate(top_species)}
    
    # Fill matrix
    for _, row in train.iterrows():
        region = row['region']
        species = row['taxon_id']
        
        if species in species_to_idx:
            region_idx = region_features_encoded[region_features_encoded['region'] == region].index[0]
            species_idx = species_to_idx[species]
            species_region_matrix[region_idx, species_idx] = 1
    
    # 4. Build GroupKFold groups (by grid location)
    groups = (region_features_encoded["grid_lat"].astype(int) * 1000 + 
              region_features_encoded["grid_lon"].astype(int))
    
    # 5. Prepare feature matrix X (remove region column)
    X = region_features_encoded.drop(columns=['region'])
    
    return X, species_region_matrix, groups, top_species


def train_species_models(X, Y, groups, top_species, n_splits: int = 5, model_type: str = "rf", seed: int = 42):
    """
    Train independent binary classification models for each species
    
    Returns:
    - models: dict, models for each species
    - cv_results: cross-validation results
    """
    gkf = GroupKFold(n_splits=n_splits)
    n_species = Y.shape[1]
    
    # Store models and results for each species
    models = {}
    cv_results = []
    per_species_metrics = {"species_id": [], "roc_auc": [], "pr_auc": [], "f1": []}
    
    print(f"Starting to train models for {n_species} species...")
    
    for species_idx in range(n_species):
        species_id = top_species[species_idx]
        y_species = Y[:, species_idx]
        
        # Skip species with no positive samples
        if y_species.sum() == 0:
            print(f"Species {species_id} has no positive samples, skipping")
            continue
            
        print(f"\nTraining species {species_idx+1}/{n_species}: {species_id}")
        
        # Cross-validation metrics per fold
        fold_metrics = {"roc_auc": [], "pr_auc": [], "f1": []}
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_species, groups=groups), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_species[train_idx], y_species[val_idx]
            
            # Train model
            if model_type == "lr":
                model = LogisticRegression(
                    max_iter=200, 
                    class_weight="balanced", 
                    solver="liblinear", 
                    random_state=seed + fold
                )
            elif model_type == "rf":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=seed + fold,
                    class_weight="balanced"
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model.fit(X_train, y_train)
            
            # Predict probabilities
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            
            # Calculate metrics
            if len(np.unique(y_val)) > 1:  # Need both positive and negative samples
                try:
                    roc_auc = roc_auc_score(y_val, y_proba)
                    pr_auc = average_precision_score(y_val, y_proba)
                    f1 = f1_score(y_val, y_pred)
                    
                    fold_metrics["roc_auc"].append(roc_auc)
                    fold_metrics["pr_auc"].append(pr_auc)
                    fold_metrics["f1"].append(f1)
                except:
                    pass
        
        # Calculate average performance for this species
        if fold_metrics["roc_auc"]:
            avg_roc = np.mean(fold_metrics["roc_auc"])
            avg_pr = np.mean(fold_metrics["pr_auc"])
            avg_f1 = np.mean(fold_metrics["f1"])
            
            # Train final model (using all data)
            if model_type == "lr":
                final_model = LogisticRegression(
                    max_iter=200, 
                    class_weight="balanced", 
                    solver="liblinear", 
                    random_state=seed
                )
            else:
                final_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=seed,
                    class_weight="balanced"
                )
            
            final_model.fit(X, y_species)
            models[species_id] = final_model
            
            # Record results
            per_species_metrics["species_id"].append(species_id)
            per_species_metrics["roc_auc"].append(avg_roc)
            per_species_metrics["pr_auc"].append(avg_pr)
            per_species_metrics["f1"].append(avg_f1)
            
            cv_results.append({
                "species_id": species_id,
                "n_positive_regions": int(y_species.sum()),
                "n_total_regions": int(len(y_species)),
                "roc_auc": float(avg_roc),
                "pr_auc": float(avg_pr),
                "f1": float(avg_f1)
            })
            
            print(f"  ROC-AUC: {avg_roc:.3f}, PR-AUC: {avg_pr:.3f}, F1: {avg_f1:.3f}")
        else:
            print(f"  Cannot calculate metrics (insufficient data)")
    
    return models, cv_results, per_species_metrics


def calculate_baselines(cv_results):
    """
    Calculate baseline performance (frequency baseline and majority class baseline)
    """
    baselines = []
    
    for result in cv_results:
        species_id = result["species_id"]
        n_positive = result["n_positive_regions"]
        n_total = result["n_total_regions"]
        prevalence = n_positive / n_total
        
        # Frequency baseline: performance of random prediction
        random_roc = 0.5
        random_pr = prevalence
        
        # Majority class baseline: performance of always predicting majority class
        if prevalence > 0.5:
            majority_f1 = 2 * prevalence / (1 + prevalence)  # precision=prevalence, recall=1
        else:
            majority_f1 = 0  # always predicting negative class yields F1=0
        
        baselines.append({
            "species_id": species_id,
            "prevalence": prevalence,
            "random_baseline_roc": random_roc,
            "random_baseline_pr": random_pr,
            "majority_baseline_f1": majority_f1
        })
    
    return baselines


def parse_args():
    p = argparse.ArgumentParser(description="Species→Location Prediction Model")
    p.add_argument("--train-features", type=str, required=True,
                  help="Training features file path")
    p.add_argument("--topk", type=int, default=50,
                  help="Select Top-K frequent species")
    p.add_argument("--cv-splits", type=int, default=5,
                  help="Cross-validation folds")
    p.add_argument("--model", type=str, choices=["lr", "rf"], default="rf",
                  help="Model type: lr=Logistic Regression, rf=Random Forest")
    p.add_argument("--outdir", type=str, required=True,
                  help="Output directory")
    p.add_argument("--modeldir", type=str, required=True,
                  help="Model save directory")
    p.add_argument("--seed", type=int, default=42,
                  help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)
    
    print("=" * 60)
    print("Species→Location Prediction Model")
    print("=" * 60)
    
    # Build dataset
    print(f"Building dataset (Top-{args.topk} species)...")
    X, Y, groups, top_species = build_species_location_dataset(
        args.train_features, args.topk
    )
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Number of regions: {X.shape[0]}")
    print(f"Number of species: {len(top_species)}")
    
    # Train models
    print(f"\nStarting model training ({args.model})...")
    models, cv_results, per_species_metrics = train_species_models(
        X, Y, groups, top_species, args.cv_splits, args.model, args.seed
    )
    
    # Calculate baselines
    baselines = calculate_baselines(cv_results)
    
    # Save cross-validation results
    cv_df = pd.DataFrame(cv_results)
    cv_path = os.path.join(args.outdir, "spec2loc_cv_metrics.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"\nSaving CV results: {cv_path}")
    
    # Save baseline results
    baseline_df = pd.DataFrame(baselines)
    baseline_path = os.path.join(args.outdir, "spec2loc_baselines.csv")
    baseline_df.to_csv(baseline_path, index=False)
    
    # Compute overall performance (macro averages across species)
    if per_species_metrics["roc_auc"]:
        overall_roc = np.mean(per_species_metrics["roc_auc"])
        overall_pr = np.mean(per_species_metrics["pr_auc"])
        overall_f1 = np.mean(per_species_metrics["f1"])
        
        print(f"\n=== Overall Performance ===")
        print(f"Average ROC-AUC: {overall_roc:.4f}")
        print(f"Average PR-AUC: {overall_pr:.4f}")
        print(f"Average F1: {overall_f1:.4f}")
        
        # Compare with baselines
        avg_random_roc = 0.5
        avg_random_pr = np.mean([b["random_baseline_pr"] for b in baselines])
        
        print(f"\n=== Comparison with Baselines ===")
        print(f"ROC-AUC improvement: {overall_roc - avg_random_roc:.4f} (vs random: {avg_random_roc})")
        print(f"PR-AUC improvement: {overall_pr - avg_random_pr:.4f} (vs random: {avg_random_pr:.4f})")
        
        # Save overall results row
        summary_results = [{
            "species_id": "macro_avg",
            "n_positive_regions": "-",
            "n_total_regions": "-",
            "roc_auc": float(overall_roc),
            "pr_auc": float(overall_pr),
            "f1": float(overall_f1)
        }]
        
        summary_df = pd.DataFrame(summary_results)
        final_df = pd.concat([cv_df, summary_df], ignore_index=True)
        final_df.to_csv(cv_path, index=False)
    
    # Save training logs
    logs_obj = {
        "params": {
            "topk": args.topk,
            "cv_splits": args.cv_splits,
            "model": args.model,
            "seed": args.seed,
            "n_species_trained": len(models),
            "n_total_regions": X.shape[0]
        },
        "cv_results": cv_results,
        "feature_columns": X.columns.tolist()
    }
    
    logs_path = os.path.join(args.outdir, "spec2loc_cv_logs.json")
    with open(logs_path, "w") as f:
        json.dump(logs_obj, f, indent=2)
    print(f"Saving training logs: {logs_path}")
    
    # Save models
    model_obj = {
        "models": models,  # Models for each species
        "top_species": top_species,
        "feature_columns": X.columns.tolist(),
        "model_type": args.model
    }
    
    model_name = f"spec2loc_{args.model}.pkl"
    model_path = os.path.join(args.modeldir, model_name)
    joblib.dump(model_obj, model_path)
    print(f"Saving model: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
