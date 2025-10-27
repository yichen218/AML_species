from pathlib import Path
import numpy as np
import pandas as pd

def load_train_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "train_locs": d["train_locs"],     # [lat, lon]
        "train_ids": d["train_ids"],       # taxon_id per row
        "taxon_ids": d["taxon_ids"],       # all species ids
        "taxon_names": d.get("taxon_names", None)
    }


def load_test_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {"test_locs": d["test_locs"]}   # [lat, lon]


def transform_train_df(df_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(df_dict["train_locs"], columns=["lat", "lon"])
    df["taxon_id"] = df_dict["train_ids"].astype(int)
    if df_dict["taxon_names"] is not None:
        id2name = dict(zip(df_dict["taxon_ids"].astype(int), df_dict["taxon_names"]))
        df["taxon_name"] = df["taxon_id"].map(id2name)
    else:
        df["taxon_name"] = np.nan
    return df


def transform_test_df(df_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(df_dict["test_locs"], columns=["lat", "lon"])


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["grid_lat"] = np.floor(out["lat"]).astype(int)
    out["grid_lon"] = np.floor(out["lon"]).astype(int)
    out["grid_group"] = out["grid_lat"].astype(str) + "_" + out["grid_lon"].astype(str)
    out["lat_band"] = pd.cut(out["lat"], bins=[-90, -60, -30, 0, 30, 60, 90], include_lowest=True)
    out["hemisphere"] = np.where(out["lat"] >= 0, "N", "S")

    rad = np.radians
    out["sin_lat"] = np.sin(rad(out["lat"]))
    out["cos_lat"] = np.cos(rad(out["lat"]))
    out["sin_lon"] = np.sin(rad(out["lon"]))
    out["cos_lon"] = np.cos(rad(out["lon"]))
    return out


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "species_data"

    # Load raw npz file
    train_npz = load_train_npz(data_dir / "species_train.npz")
    test_npz  = load_test_npz(data_dir / "species_test.npz")

    # Build dataframes from npz
    train_df = transform_train_df(train_npz)
    test_df  = transform_test_df(test_npz)

    # Feature engineering
    train_features = add_geometry_features(train_df)
    test_features  = add_geometry_features(test_df)

    # Save
    train_features.to_csv(data_dir / "train_features.csv", index=False)
    test_features.to_csv(data_dir / "test_features.csv", index=False)
    print("Features built")


if __name__ == "__main__":
    main()