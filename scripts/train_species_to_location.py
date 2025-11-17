import argparse
import os
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import joblib


def build_species_to_location_dataset(train_path: str, top_k: int = 50):
    train = pd.read_csv(train_path)
    region_species = (
        train.groupby("region")["taxon_id"].apply(lambda x: sorted(set(x))).reset_index(name="species_list")
    )
    feature_cols = ["grid_lat", "grid_lon", "sin_lon", "cos_lon", "hemisphere"]
    region_features = (
        train.groupby("region").first().reset_index()[["region"] + feature_cols]
    )
    data = region_features.merge(region_species, on="region")
    all_species = [s for lst in data["species_list"] for s in lst]
    counter = Counter(all_species)
    top_species = [s for s, _ in counter.most_common(top_k)]
    data["species_topk"] = data["species_list"].apply(lambda lst: [s for s in lst if s in top_species])
    data_ml = data.copy()
    X_raw = data_ml[feature_cols]
    X = pd.get_dummies(X_raw, columns=["hemisphere"], drop_first=True)
    Y_cols = top_species
    Y = np.zeros((data_ml.shape[0], len(Y_cols)), dtype=int)
    species_set_per_region = data_ml["species_topk"].tolist()
    for j, sp in enumerate(Y_cols):
        Y[:, j] = [1 if sp in lst else 0 for lst in species_set_per_region]
    groups = data_ml["grid_lat"].astype(int) * 1000 + data_ml["grid_lon"].astype(int)
    return X, Y, groups, top_species


def get_base_estimator(model_type: str, seed: int):
    if model_type == "lr":
        return LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear", random_state=seed)
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)
    raise ValueError("Unsupported model type")


def cross_validation(X, Y, groups, n_splits: int, model_type: str, seed: int):
    gkf = GroupKFold(n_splits=n_splits)
    base = get_base_estimator(model_type, seed)
    clf = MultiOutputClassifier(base)
    fold_logs = []
    per_species_metrics = {"roc_auc": [], "pr_auc": [], "f1": []}
    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        clf.fit(X_train, Y_train)
        if hasattr(clf, "predict_proba"):
            Y_proba = np.column_stack([p[:, 1] for p in clf.predict_proba(X_val)])
        else:
            Y_proba = clf.predict(X_val)
        Y_pred = clf.predict(X_val)
        fold_roc_micro = roc_auc_score(Y_val, Y_proba, average="micro")
        fold_pr_micro = average_precision_score(Y_val, Y_proba, average="micro")
        fold_f1_micro = f1_score(Y_val, Y_pred, average="micro")
        fold_roc_macro = roc_auc_score(Y_val, Y_proba, average="macro")
        fold_pr_macro = average_precision_score(Y_val, Y_proba, average="macro")
        fold_f1_macro = f1_score(Y_val, Y_pred, average="macro")
        fold_logs.append({
            "fold": fold_id,
            "roc_auc_micro": float(fold_roc_micro),
            "pr_auc_micro": float(fold_pr_micro),
            "f1_micro": float(fold_f1_micro),
            "roc_auc_macro": float(fold_roc_macro),
            "pr_auc_macro": float(fold_pr_macro),
            "f1_macro": float(fold_f1_macro),
        })
        for j in range(Y.shape[1]):
            try:
                ps_roc = roc_auc_score(Y_val[:, j], Y_proba[:, j])
            except ValueError:
                ps_roc = np.nan
            ps_pr = average_precision_score(Y_val[:, j], Y_proba[:, j]) if (Y_val[:, j].sum() > 0) else np.nan
            ps_f1 = f1_score(Y_val[:, j], Y_pred[:, j]) if (Y_val[:, j].sum() > 0 and (1 - Y_val[:, j]).sum() > 0) else np.nan
            per_species_metrics["roc_auc"].append(ps_roc)
            per_species_metrics["pr_auc"].append(ps_pr)
            per_species_metrics["f1"].append(ps_f1)
    micro_means = {
        "roc_auc_micro": float(np.nanmean([f["roc_auc_micro"] for f in fold_logs])),
        "pr_auc_micro": float(np.nanmean([f["pr_auc_micro"] for f in fold_logs])),
        "f1_micro": float(np.nanmean([f["f1_micro"] for f in fold_logs])),
    }
    macro_means = {
        "roc_auc_macro": float(np.nanmean([f["roc_auc_macro"] for f in fold_logs])),
        "pr_auc_macro": float(np.nanmean([f["pr_auc_macro"] for f in fold_logs])),
        "f1_macro": float(np.nanmean([f["f1_macro"] for f in fold_logs])),
    }
    return fold_logs, per_species_metrics, micro_means, macro_means, clf


def aggregate_per_species(top_species, per_species_metrics, n_splits):
    m = len(top_species)
    agg = []
    for i in range(m):
        idxs = list(range(i, m * n_splits, m))
        vals_roc = [per_species_metrics["roc_auc"][k] for k in idxs]
        vals_pr = [per_species_metrics["pr_auc"][k] for k in idxs]
        vals_f1 = [per_species_metrics["f1"][k] for k in idxs]
        agg.append({
            "taxon_id": int(top_species[i]),
            "roc_auc": float(np.nanmean(vals_roc)) if len(vals_roc) > 0 else np.nan,
            "pr_auc": float(np.nanmean(vals_pr)) if len(vals_pr) > 0 else np.nan,
            "f1": float(np.nanmean(vals_f1)) if len(vals_f1) > 0 else np.nan,
        })
    return pd.DataFrame(agg)


def parse_args():
    p = argparse.ArgumentParser(description="Species→Location spatial CV")
    p.add_argument("--train-features", type=str, required=True)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--model", type=str, choices=["lr", "rf"], default="lr")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--modeldir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)
    X, Y, groups, top_species = build_species_to_location_dataset(args.train_features, args.topk)
    fold_logs, per_species_metrics, micro_means, macro_means, clf = cross_validation(
        X, Y, groups, args.cv_splits, args.model, args.seed
    )
    per_species_df = aggregate_per_species(top_species, per_species_metrics, args.cv_splits)
    summary_rows = [
        {"taxon_id": "micro", "roc_auc": micro_means["roc_auc_micro"], "pr_auc": micro_means["pr_auc_micro"], "f1": micro_means["f1_micro"]},
        {"taxon_id": "macro", "roc_auc": macro_means["roc_auc_macro"], "pr_auc": macro_means["pr_auc_macro"], "f1": macro_means["f1_macro"]},
    ]
    metrics_df = pd.concat([per_species_df, pd.DataFrame(summary_rows)], ignore_index=True)
    metrics_path = os.path.join(args.outdir, "spec2loc_cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logs_obj = {
        "params": {
            "topk": args.topk,
            "cv_splits": args.cv_splits,
            "model": args.model,
            "seed": args.seed,
        },
        "folds": fold_logs,
    }
    logs_path = os.path.join(args.outdir, "spec2loc_cv_logs.json")
    with open(logs_path, "w") as f:
        json.dump(logs_obj, f)
    base = get_base_estimator(args.model, args.seed)
    final_clf = MultiOutputClassifier(base)
    final_clf.fit(X, Y)
    model_obj = {
        "model": final_clf,
        "top_species": top_species,
        "feature_columns": X.columns.tolist(),
    }
    model_name = f"spec2loc_{args.model}.pkl"
    model_path = os.path.join(args.modeldir, model_name)
    joblib.dump(model_obj, model_path)


if __name__ == "__main__":
    main()

