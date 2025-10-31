from pathlib import Path
import numpy as np
import pandas as pd

def load_train_npz(path: Path) -> dict:
    """Load train data from npz file"""
    d = np.load(path, allow_pickle=True)
    return {
        "train_locs": d["train_locs"],     # [lat, lon]
        "train_ids": d["train_ids"],       # taxon_id per row
        "taxon_ids": d["taxon_ids"],       # all species ids
        "taxon_names": d.get("taxon_names", None)
    }


def load_test_npz(path: Path) -> dict:
    """Load test data from npz file"""
    d = np.load(path, allow_pickle=True)
    return {"test_locs": d["test_locs"]}   # [lat, lon]


def transform_train_df(df_dict: dict) -> pd.DataFrame:
    """Transform train data into pandas dataframe"""
    df = pd.DataFrame(df_dict["train_locs"], columns=["lat", "lon"])
    df["taxon_id"] = df_dict["train_ids"].astype(int)
    if df_dict["taxon_names"] is not None:
        id2name = dict(zip(df_dict["taxon_ids"].astype(int), df_dict["taxon_names"]))
        df["taxon_name"] = df["taxon_id"].map(id2name)
    else:
        df["taxon_name"] = np.nan
    return df


def transform_test_df(df_dict: dict) -> pd.DataFrame:
    """Transform test data into pandas dataframe"""
    return pd.DataFrame(df_dict["test_locs"], columns=["lat", "lon"])


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometry features"""
    df_new = df.copy()
    df_new["grid_lat"] = np.floor(df_new["lat"]).astype(int)
    df_new["grid_lon"] = np.floor(df_new["lon"]).astype(int)
    df_new["grid_group"] = df_new["grid_lat"].astype(str) + "_" + df_new["grid_lon"].astype(str)
    df_new["lat_band"] = pd.cut(df_new["lat"], bins=[-90, -60, -30, 0, 30, 60, 90], include_lowest=True)
    df_new["hemisphere"] = np.where(df_new["lat"] >= 0, "N", "S")

    rad = np.radians
    df_new["sin_lon"] = np.sin(rad(df_new["lon"]))
    df_new["cos_lon"] = np.cos(rad(df_new["lon"]))
    return df_new


def concat_df(train_df: pd.DataFrame, train_extra_df: pd.DataFrame | None) -> pd.DataFrame:
    """Concat train and extra train dataframes"""
    merged = pd.concat([train_df, train_extra_df], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["lat", "lon", "taxon_id"]).reset_index(drop=True)
    return merged


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "species_data"

    # Load raw train data from npz file
    train_npz = load_train_npz(data_dir / "species_train.npz")
    train_extra_npz = load_train_npz(data_dir / "species_train_extra.npz")
    test_npz  = load_test_npz(data_dir / "species_test.npz")

    # Build dataframes from npz
    train_df = transform_train_df(train_npz)
    train_extra_df = transform_train_df(train_extra_npz)
    test_df  = transform_test_df(test_npz)

    train_df = concat_df(train_df, train_extra_df)

    # Feature engineering
    train_features = add_geometry_features(train_df)
    test_features  = add_geometry_features(test_df)

    # Save
    train_features.to_csv(data_dir / "train_features.csv", index=False)
    test_features.to_csv(data_dir / "test_features.csv", index=False)
    print("Features built")


if __name__ == "__main__":
    main()