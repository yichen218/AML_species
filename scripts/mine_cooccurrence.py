import argparse
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def build_region_species(train_path: str):
    """
    Load train_features.csv and build:

    - region_species: region with a list of species
    - species_name_map: taxon_id -> taxon_name
    """
    print(f"Read train features: {train_path}")
    train = pd.read_csv(train_path)
    print("Train set shape:", train.shape)
    print("Columns:", train.columns.tolist())

    # region -> list of unique species ids
    region_species = (
        train
        .groupby("region")["taxon_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="species_list")
    )
    print("region_species shape:", region_species.shape)

    # taxon_id -> taxon_name
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
    Find species pairs that often show up in the same regions

    Args:
        min_species_support: species must appear in at least this number of times to be considered
        min_pair_count: keep only pairs that co-occur in at least this number of times
    """
    num_regions = region_species.shape[0]
    print("Total regions:", num_regions)

    # Add region_id
    region_species = region_species.reset_index(drop=True)
    region_species["region_id"] = region_species.index

    # Build species -> set of region_ids where it appears
    species_to_regions = defaultdict(set)

    for _, row in region_species.iterrows():
        rid = row["region_id"]
        for s in row["species_list"]:
            species_to_regions[s].add(rid)

    # Count in how many regions each species appears
    species_support_counts = {s: len(rids) for s, rids in species_to_regions.items()}
    print("Total number of species:", len(species_support_counts))

    # Keep species with enough support
    frequent_species = [
        s for s, cnt in species_support_counts.items()
        if cnt >= min_species_support
    ]
    print(
        f"Number of species with support >= {min_species_support}:",
        len(frequent_species),
    )

    frequent_species.sort(key=lambda s: species_support_counts[s], reverse=True)

    # Enumerate species pairs and compute co-occurrence measures
    results = []

    S = len(frequent_species)

    for i in range(S):
        a = frequent_species[i]
        regions_a = species_to_regions[a]
        n_a = len(regions_a)

        for j in range(i + 1, S):
            b = frequent_species[j]
            regions_b = species_to_regions[b]
            n_b = len(regions_b)

            # Intersection = regions where both a and b appear
            inter = regions_a & regions_b
            n_ab = len(inter)
            if n_ab < min_pair_count:
                continue

            union_size = n_a + n_b - n_ab
            if union_size == 0:
                continue

            support_a = n_a / num_regions
            support_b = n_b / num_regions
            support_ab = n_ab / num_regions

            jaccard = n_ab / union_size
            p_a_given_b = n_ab / n_b
            lift = (
                support_ab / (support_a * support_b)
                if support_a * support_b > 0
                else np.nan
            )

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
    print("Number of co-occurring pairs:", results_df.shape[0])

    # Sort by lift or jaccard so that the strongest pairs at the top
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
        help="Path to train_features.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory; will contain species_cooccurrence.csv",
    )
    parser.add_argument(
        "--min-species-support",
        type=int,
        default=10,
        help="Minimum number of regions a species must appear in to be included",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=20,
        help="Minimum number of regions a species pair must co-occur in to be kept",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Build region -> species_list
    region_species, species_name_map = build_region_species(args.train_features)

    # 2. Mine co-occurrence patterns
    results_df = mine_cooccurrence(
        region_species,
        species_name_map,
        min_species_support=args.min_species_support,
        min_pair_count=args.min_pair_count,
    )

    # 3. Save results
    out_path = os.path.join(args.outdir, "species_cooccurrence.csv")
    results_df.to_csv(out_path, index=False)
    print("Results saved to ", out_path)


if __name__ == "__main__":
    main()
