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
    从 train_features.csv 构造地点→物种的多标签数据集：
    - X: 地点特征
    - Y: Top-K 物种的多标签 0/1 矩阵
    - groups: 用于 GroupKFold 的 group（按 grid_lat, grid_lon 粗略分块）
    - top_species: Top-K 物种的 taxon_id 列表
    """
    print(f"读取训练特征: {train_path}")
    train = pd.read_csv(train_path)
    print("train 形状:", train.shape)
    print("列名:", train.columns.tolist())

    # ========= 1. region -> species_list =========
    # 每个 region 对应一个去重后的物种列表
    region_species = (
        train
        .groupby("region")["taxon_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="species_list")
    )
    print("region_species 形状:", region_species.shape)

    # ========= 2. region -> 地点特征 =========
    feature_cols = ["grid_lat", "grid_lon", "sin_lon", "cos_lon", "hemisphere"]

    region_features = (
        train
        .groupby("region")
        .first()
        .reset_index()[["region"] + feature_cols]
    )
    print("region_features 形状:", region_features.shape)

    # ========= 3. 合并 =========
    data = region_features.merge(region_species, on="region")
    print("合并后的 data 形状:", data.shape)

    # ========= 4. 统计物种频次，选 Top-K =========
    all_species = [s for lst in data["species_list"] for s in lst]
    counter = Counter(all_species)
    print("不同物种总数:", len(counter))
    print("出现次数最多的前 10 个物种:", counter.most_common(10))

    top_species = [s for s, _ in counter.most_common(top_k)]
    print(f"Top-{top_k} 物种数:", len(top_species))
    print("Top-K 前 10 个:", top_species[:10])

    # ========= 5. 只保留 Top-K 物种 =========
    data["species_topk"] = data["species_list"].apply(
        lambda lst: [s for s in lst if s in top_species]
    )

    # 过滤掉不包含任何 Top-K 物种的 region
    data_ml = data[data["species_topk"].apply(len) > 0].reset_index(drop=True)
    print("用于多标签建模的 region 数:", data_ml.shape[0])

    # ========= 6. 构造特征矩阵 X =========
    X_raw = data_ml[feature_cols]
    # hemisphere 是字符串类别，做 one-hot
    X = pd.get_dummies(X_raw, columns=["hemisphere"], drop_first=True)
    print("X 形状:", X.shape)

    # ========= 7. 构造多标签矩阵 Y =========
    mlb = MultiLabelBinarizer(classes=top_species)
    Y = mlb.fit_transform(data_ml["species_topk"])
    print("Y 形状:", Y.shape)

    # ========= 8. 构造 GroupKFold 的 groups =========
    # 简单做法：用 (grid_lat, grid_lon) 合成一个 group_id
    groups = (
        data_ml["grid_lat"].astype(int) * 1000
        + data_ml["grid_lon"].astype(int)
    )

    return X, Y, groups, top_species


def cross_val_evaluate_rf(X, Y, groups, n_splits: int = 5, random_state: int = 42):
    """
    使用 GroupKFold + RandomForest + MultiOutputClassifier 做交叉验证，
    返回每折的 micro/macro F1 以及整体均值。
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

        print(f"\n==== 训练 Fold {fold_id} ====")
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

    print("\n==== CV 平均结果 ====")
    print(f"micro F1 mean: {micro_mean:.4f}")
    print(f"macro F1 mean: {macro_mean:.4f}")

    return fold_results, micro_mean, macro_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train location-to-species multi-label model with RandomForest"
    )

    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="路径：train_features.csv",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="使用频次最高的前 K 个物种 (default: 50)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="GroupKFold 的折数 (default: 5)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="结果输出目录（会写入 metrics 和 模型）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 构造多标签数据集
    X, Y, groups, top_species = build_location_to_species_dataset(
        train_path=args.train_features,
        top_k=args.topk,
    )

    # 2. 交叉验证评估
    fold_results, micro_mean, macro_mean = cross_val_evaluate_rf(
        X, Y, groups,
        n_splits=args.cv_splits,
        random_state=args.seed,
    )

    # 3. 保存 CV 结果到 csv
    results_df = pd.DataFrame(fold_results)
    results_df.loc[len(results_df)] = {
        "fold": "mean",
        "micro_f1": micro_mean,
        "macro_f1": macro_mean,
    }

    metrics_path = os.path.join(args.outdir, "multilabel_cv_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print("已保存 CV 结果到:", metrics_path)

    # 4. 用全部数据训练最终模型并保存
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

    model_path = os.path.join(args.outdir, "loc2spec_rf_model.joblib")
    joblib.dump(model_obj, model_path)
    print("已保存最终模型到:", model_path)


if __name__ == "__main__":
    main()
