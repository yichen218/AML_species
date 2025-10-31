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


def build_train_df(df_dict: dict) -> pd.DataFrame:
    """Transform train data into pandas dataframe"""
    df = pd.DataFrame(df_dict["train_locs"], columns=["lat", "lon"])
    df["taxon_id"] = df_dict["train_ids"].astype(int)
    if df_dict["taxon_names"] is not None:
        id2name = dict(zip(df_dict["taxon_ids"].astype(int), df_dict["taxon_names"]))
        df["taxon_name"] = df["taxon_id"].map(id2name)
    else:
        df["taxon_name"] = np.nan
    return df


def build_test_df(df_dict: dict) -> pd.DataFrame:
    """Transform test data into pandas dataframe"""
    return pd.DataFrame(df_dict["test_locs"], columns=["lat", "lon"])


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometry features"""
    df_0 = df.copy()
    df_0["grid_lat"] = np.floor(df_0["lat"]).astype(int)
    df_0["grid_lon"] = np.floor(df_0["lon"]).astype(int)
    df_0["grid_group"] = df_0["grid_lat"].astype(str) + "_" + df_0["grid_lon"].astype(str)    #1x1 region
    df_0["lat_band"] = pd.cut(df_0["lat"], bins=[-90, -60, -30, 0, 30, 60, 90], include_lowest=True)
    df_0["hemisphere"] = np.where(df_0["lat"] >= 0, "N", "S")

    rad = np.radians
    df_0["sin_lon"] = np.sin(rad(df_0["lon"]))
    df_0["cos_lon"] = np.cos(rad(df_0["lon"]))
    return df_0


def define_geo_categories():
    """Define stable categories for lat_band and hemisphere"""
    lat_bins = [-90, -60, -30, 0, 30, 60, 90]
    lat_cats = pd.IntervalIndex.from_breaks(lat_bins, closed="right")  # (-90,-60], ..., (60,90]
    hemi_cats = ["S", "N"]
    return lat_cats, hemi_cats


def normalize_geo_categories(df: pd.DataFrame, lat_cats, hemi_cats) -> pd.DataFrame:
    """Cast lat_band and hemisphere to fixed categorical domains"""
    df_1 = df.copy()
    df_1["hemisphere"] = df_1["hemisphere"].astype(str).str.strip().str.upper()
    df_1["hemisphere"] = pd.Categorical(df_1["hemisphere"], categories=hemi_cats, ordered=True)
    df_1["lat_band"] = pd.Categorical(df_1["lat_band"], categories=lat_cats, ordered=True)
    return df_1


def onehot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append OHE columns for lat_band and hemisphere"""
    df_2 = df.copy()
    lat_ohe = pd.get_dummies(df_2["lat_band"], prefix="lat_band", dtype=int)
    hemi_ohe = pd.get_dummies(df_2["hemisphere"], prefix="hemi", dtype=int)
    df_2 = pd.concat([df_2, lat_ohe, hemi_ohe], axis=1)
    return df_2, list(lat_ohe.columns) + list(hemi_ohe.columns)


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
    train_df = build_train_df(train_npz)
    train_extra_df = build_train_df(train_extra_npz)
    test_df  = build_test_df(test_npz)

    # Feature engineering
    train_features = add_geo_features(concat_df(train_df, train_extra_df))
    test_features  = add_geo_features(test_df)

    # Normalize categories
    lat_cats, hemi_cats = define_geo_categories()
    train_features = normalize_geo_categories(train_features, lat_cats, hemi_cats)
    test_features = normalize_geo_categories(test_features, lat_cats, hemi_cats)

    # OneHot encoding
    train_features, ohe_cols = onehot(train_features)
    test_features, _ = onehot(test_features)


    train_features.to_csv(data_dir / "train_features.csv", index=False)
    test_features.to_csv(data_dir / "test_features.csv", index=False)
    print("Features built")


if __name__ == "__main__":
    main()