import argparse
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def build_region_species(train_path: str):
    """
    从 train_features.csv 构造：
    - region_species: DataFrame，每行一个 region，对应一个物种列表
    - species_name_map: dict, taxon_id -> taxon_name（方便结果里带名字）
    """
    print(f"读取训练特征: {train_path}")
    train = pd.read_csv(train_path)
    print("train 形状:", train.shape)
    print("列名:", train.columns.tolist())

    # region -> 物种列表（去重）
    region_species = (
        train
        .groupby("region")["taxon_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="species_list")
    )
    print("region_species 形状:", region_species.shape)

    # taxon_id -> taxon_name 映射（有的物种可能有多个名字版本，这里随便取一个）
    species_name_map = (
        train[["taxon_id", "taxon_name"]]
        .drop_duplicates()
        .set_index("taxon_id")["taxon_name"]
        .to_dict()
    )

    return region_species, species_name_map


def mine_cooccurrence(region_species: pd.DataFrame,
                      species_name_map: dict,
                      min_species_support: int = 10,
                      min_pair_count: int = 20):
    """
    基于 region_species 挖掘物种共现关系。

    参数：
    - min_species_support: 一个物种至少出现在多少个 region 中，才参与后续分析
    - min_pair_count: 两个物种至少在多少个 region 中共同出现，结果才保留
    """
    # N = region 个数
    num_regions = region_species.shape[0]
    print("region 总数:", num_regions)

    # 给 region_species 增加一个行索引，作为 region_id（0..N-1）
    region_species = region_species.reset_index(drop=True)
    region_species["region_id"] = region_species.index

    # ========= 1. 建立 物种 -> 出现的 region_id 集合 =========
    species_to_regions = defaultdict(set)

    for _, row in region_species.iterrows():
        rid = row["region_id"]
        for s in row["species_list"]:
            species_to_regions[s].add(rid)

    # ========= 2. 过滤掉支持度太低的物种 =========
    species_support_counts = {s: len(rids) for s, rids in species_to_regions.items()}

    print("物种总数:", len(species_support_counts))

    frequent_species = [
        s for s, cnt in species_support_counts.items()
        if cnt >= min_species_support
    ]
    print(
        f"支持度 >= {min_species_support} 的物种数:",
        len(frequent_species),
    )

    # 为了后面方便，排序一下（频率从高到低）
    frequent_species.sort(key=lambda s: species_support_counts[s], reverse=True)

    # ========= 3. 枚举物种对，计算共现指标 =========
    results = []

    S = len(frequent_species)
    print("开始枚举物种对，总对数大约:", S * (S - 1) // 2)

    for i in range(S):
        a = frequent_species[i]
        regions_a = species_to_regions[a]
        n_a = len(regions_a)

        for j in range(i + 1, S):
            b = frequent_species[j]
            regions_b = species_to_regions[b]
            n_b = len(regions_b)

            # 交集 = a 和 b 同时出现的 region
            inter = regions_a & regions_b
            n_ab = len(inter)
            if n_ab < min_pair_count:
                continue  # 共现次数太少，跳过

            # 计算各种指标
            union_size = n_a + n_b - n_ab
            if union_size == 0:
                continue

            support_a = n_a / num_regions
            support_b = n_b / num_regions
            support_ab = n_ab / num_regions

            jaccard = n_ab / union_size
            p_a_given_b = n_ab / n_b
            lift = support_ab / (support_a * support_b) if support_a * support_b > 0 else np.nan

            results.append(
                {
                    "species_a": a,
                    "species_b": b,
                    "species_a_name": species_name_map.get(a, ""),
                    "species_b_name": species_name_map.get(b, ""),
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_ab": n_ab,
                    "support_a": support_a,
                    "support_b": support_b,
                    "support_ab": support_ab,
                    "jaccard": jaccard,
                    "p_a_given_b": p_a_given_b,
                    "lift": lift,
                }
            )

    results_df = pd.DataFrame(results)
    print("共现对数:", results_df.shape[0])

    # 可以按 lift 或 jaccard 排序，方便查看最强的共现关系
    results_df = results_df.sort_values(by="lift", ascending=False)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine species co-occurrence from train_features.csv"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="路径：train_features.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="输出目录，将保存 species_cooccurrence.csv",
    )
    parser.add_argument(
        "--min-species-support",
        type=int,
        default=10,
        help="单个物种至少出现在多少个 region 中才纳入分析 (default: 10)",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=20,
        help="两个物种至少在多少个 region 中共同出现，才保留这对 (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. 构造 region -> species_list
    region_species, species_name_map = build_region_species(args.train_features)

    # 2. 共现挖掘
    results_df = mine_cooccurrence(
        region_species,
        species_name_map,
        min_species_support=args.min_species_support,
        min_pair_count=args.min_pair_count,
    )

    # 3. 保存结果
    out_path = os.path.join(args.outdir, "species_cooccurrence.csv")
    results_df.to_csv(out_path, index=False)
    print("已保存共现结果到:", out_path)


if __name__ == "__main__":
    main()
