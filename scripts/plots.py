from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_global_distribution(train: pd.DataFrame, out_path: Path) -> None:
    """Show where observations are globally"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5), dpi=120)
    plt.scatter(train["lon"], train["lat"], s=1, alpha=0.35)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Observation global distribution")
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_spatial_hotspots(train: pd.DataFrame, out_path: Path) -> None:
    """Focus on spatial hotspots with density"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5), dpi=120)
    hb = plt.hexbin(train["lon"], train["lat"], gridsize=80, mincnt=1)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Observation spatial hotspots")
    cb = plt.colorbar(hb); cb.set_label("Count")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_latitude_distribution(train: pd.DataFrame, out_path: Path) -> None:
    """Show latitude sampling bias"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(train["lat"], bins=60)
    plt.xlabel("Latitude"); plt.ylabel("Count"); plt.title("Latitude distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_longitude_distribution(train: pd.DataFrame, out_path: Path) -> None:
    """Show longitude sampling bias"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(train["lon"], bins=60)
    plt.xlabel("Longitude"); plt.ylabel("Count"); plt.title("Longitude distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_latitude_band_distribution(train: pd.DataFrame, out_path: Path) -> None:
    """Show observation count in each latitude band"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    band = train.groupby("lat_band").size().reset_index(name="n")
    plt.figure(figsize=(9, 4), dpi=120)
    plt.bar(band["lat_band"].astype(str), band["n"])
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Count")
    plt.title("Latitude bands")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_hemisphere_distribution(train: pd.DataFrame, out_path: Path) -> None:
    """Show observation count in N/S hemisphere"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    distribution = train.groupby("hemisphere").size().reset_index(name="n")
    plt.figure(figsize=(6, 4), dpi=120)
    plt.bar(distribution["hemisphere"], distribution["n"])
    plt.xlabel("Hemisphere"); plt.ylabel("Count"); plt.title("Hemisphere distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()



# -------------------------------- main --------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "species_data"
    plot_dir = base_dir / "plots"

    train_path = data_dir / "train_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    train = pd.read_csv(train_path)


    plot_global_distribution(train, plot_dir / "global_distribution.png")

    plot_spatial_hotspots(train, plot_dir / "spatial_hotspots_density.png")

    plot_latitude_distribution(train, plot_dir / "lat_distribution.png")

    plot_longitude_distribution(train, plot_dir / "lon_distribution.png")

    plot_latitude_band_distribution(train, plot_dir / "lat_band_distribution.png")

    plot_hemisphere_distribution(train, plot_dir / "hemisphere_distribution.png")

    print("Plots build")

if __name__ == "__main__":
    main()