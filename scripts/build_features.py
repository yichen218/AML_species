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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometry features"""
    df0 = df.copy()
    df0["grid_lat"] = np.floor(df0["lat"]).astype(int)
    df0["grid_lon"] = np.floor(df0["lon"]).astype(int)
    df0["region"] = df0["grid_lat"].astype(str) + "°" + df0["grid_lon"].astype(str)  + "°"  #1x1 region
    df0["hemisphere"] = np.where(df0["lat"] >= 0, "N", "S")

    rad = np.radians
    df0["sin_lon"] = np.sin(rad(df0["lon"]))
    df0["cos_lon"] = np.cos(rad(df0["lon"]))
    return df0



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
    train_features = add_features(concat_df(train_df, train_extra_df))
    test_features  = add_features(test_df)


    train_features.to_csv(data_dir / "train_features.csv", index=False)
    test_features.to_csv(data_dir / "test_features.csv", index=False)
    print("Features built")


if __name__ == "__main__":
    main()