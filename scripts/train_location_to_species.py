import argparse
import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import joblib


def build_location_to_species_dataset(train_path: str, top_k: int = 50):
    """
    Turn train_features.csv into a region-level multi-label dataset

    Returns:
        X: location feature matrix
        Y: multi-label 0/1 matrix for the Top-K species
        groups: group labels for GroupKFold
        top_species: list of Top-K taxon_id
    """
    print(f"Reading train features from: {train_path}")
    train = pd.read_csv(train_path)
    print("Train shape:", train.shape)
    print("Columns:", train.columns.tolist())

    # 1. region -> species_list
    # Each region corresponds to a list of unique species ids
    region_species = (
        train
        .groupby("region")["taxon_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="species_list")
    )
    print("region_species shape:", region_species.shape)

    # 2. region -> location features
    feature_cols = ["grid_lat", "grid_lon", "sin_lon", "cos_lon", "hemisphere"]

    region_features = (
        train
        .groupby("region")
        .first()
        .reset_index()[["region"] + feature_cols]
    )
    print("region_features shape:", region_features.shape)

    # 3. Merge features and species
    data = region_features.merge(region_species, on="region")
    print("Merged data shape:", data.shape)

    # 4. Count species frequency and select Top-K
    all_species = [s for lst in data["species_list"] for s in lst]
    counter = Counter(all_species)
    print("Total number of distinct species:", len(counter))
    print("Top 10 species by frequency:", counter.most_common(10))

    top_species = [s for s, _ in counter.most_common(top_k)]
    print(f"Number of Top-{top_k} species:", len(top_species))
    print("First 10 in Top-K:", top_species[:10])

    # 5. Keep Top-K species
    data["species_topk"] = data["species_list"].apply(
        lambda lst: [s for s in lst if s in top_species]
    )

    # Drop regions with no Top-K species
    data_ml = data[data["species_topk"].apply(len) > 0].reset_index(drop=True)
    print("Number of regions used for multi-label modelling:", data_ml.shape[0])

    # 6. Build feature matrix X
    X_raw = data_ml[feature_cols]

    # One-hot encode hemisphere
    X = pd.get_dummies(X_raw, columns=["hemisphere"], drop_first=True)
    print("X shape:", X.shape)

    # 7. Build multi-label matrix Y
    mlb = MultiLabelBinarizer(classes=top_species)
    Y = mlb.fit_transform(data_ml["species_topk"])
    print("Y shape:", Y.shape)

    # 8. Build GroupKFold groups
    groups = (data_ml["grid_lat"].astype(int) * 1000 + data_ml["grid_lon"].astype(int))

    return X, Y, groups, top_species


def cross_validation(X, Y, groups, n_splits: int = 5, random_state: int = 42):
    """
    GroupKFold CV with RandomForest and MultiOutputClassifier

    Returns:
        fold_results: list of dicts with per-fold micro/macro F1
        micro_mean: mean micro-F1 across folds
        macro_mean: mean macro-F1 across folds
    """
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    fold_id = 0

    for train_idx, val_idx in gkf.split(X, Y, groups=groups):
        fold_id += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        base_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )
        clf = MultiOutputClassifier(base_clf)

        print(f"\n==== Training Fold {fold_id} ====")
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_val)

        micro_f1 = f1_score(Y_val, Y_pred, average="micro")
        macro_f1 = f1_score(Y_val, Y_pred, average="macro")

        print(f"Fold {fold_id}: micro F1 = {micro_f1:.4f}, macro F1 = {macro_f1:.4f}")

        fold_results.append(
            {
                "fold": fold_id,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
            }
        )

    micro_mean = np.mean([r["micro_f1"] for r in fold_results])
    macro_mean = np.mean([r["macro_f1"] for r in fold_results])

    print("\n==== CV mean scores ====")
    print(f"micro F1 mean: {micro_mean:.4f}")
    print(f"macro F1 mean: {macro_mean:.4f}")

    return fold_results, micro_mean, macro_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a location-to-species multi-label RandomForest"
    )

    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to train_features.csv",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of most frequent species as labels",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for metrics and model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Build region-level multi-label dataset
    X, Y, groups, top_species = build_location_to_species_dataset(
        train_path=args.train_features,
        top_k=args.topk,
    )

    # 2. Cross validation
    fold_results, micro_mean, macro_mean = cross_validation(
        X, Y, groups,
        n_splits=args.cv_splits,
        random_state=args.seed,
    )

    # 3. Save CV results
    results_df = pd.DataFrame(fold_results)
    results_df.loc[len(results_df)] = {
        "fold": "mean",
        "micro_f1": micro_mean,
        "macro_f1": macro_mean,
    }

    metrics_path = os.path.join(args.outdir, "cv_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print("Saved CV metrics to:", metrics_path)

    # 4. Train model
    base_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=args.seed,
    )
    final_clf = MultiOutputClassifier(base_clf)
    final_clf.fit(X, Y)

    model_obj = {
        "model": final_clf,
        "top_species": top_species,
    }

    model_path = os.path.join(args.outdir, "loc2spec_model.joblib")
    joblib.dump(model_obj, model_path)
    print("Model saved to:", model_path)

if __name__ == "__main__":
    main()

