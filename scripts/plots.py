from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_global_scatter(train: pd.DataFrame, out_path: Path) -> None:
    """Show where observations occur globally."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5), dpi=120)
    plt.scatter(train["lon"], train["lat"], s=1, alpha=0.35)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Training observations — global scatter")
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_hexbin_density(train: pd.DataFrame, out_path: Path) -> None:
    """Highlight spatial hotspots with density."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5), dpi=120)
    hb = plt.hexbin(train["lon"], train["lat"], gridsize=80, mincnt=1)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Training observations — density (hexbin)")
    cb = plt.colorbar(hb); cb.set_label("Count")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def export_top_species_table_and_bar(train: pd.DataFrame, table_path: Path, fig_path: Path, topk: int = 20) -> None:
    """Export top-k species table and bar chart"""
    top = (train[["taxon_id", "taxon_name"]]
           .dropna(subset=["taxon_id"])
           .groupby(["taxon_id", "taxon_name"], dropna=False)
           .size().reset_index(name="count")
           .sort_values("count", ascending=False).head(topk))
    table_path.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(table_path, index=False)

    labels = top.apply(
        lambda r: str(int(r["taxon_id"])) if pd.isna(r["taxon_name"]) or str(r["taxon_name"]).strip() == ""
        else str(r["taxon_name"]), axis=1)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.bar(range(len(top)), top["count"])
    plt.xticks(range(len(top)), labels, rotation=75, ha="right")
    plt.ylabel("Observations"); plt.title("Top species by observations")
    plt.tight_layout(); plt.savefig(fig_path, bbox_inches="tight"); plt.close()


def plot_latitude_hist(train: pd.DataFrame, out_path: Path) -> None:
    """Show latitude sampling bias."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(train["lat"], bins=60)
    plt.xlabel("Latitude"); plt.ylabel("Count"); plt.title("Latitude distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_longitude_hist(train: pd.DataFrame, out_path: Path) -> None:
    """Show longitude sampling bias."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(train["lon"], bins=60)
    plt.xlabel("Longitude"); plt.ylabel("Count"); plt.title("Longitude distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_latitude_band_counts(train: pd.DataFrame, out_path: Path) -> None:
    """Summarize coarse latitudinal preferences via predefined latitude bands."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    band = train.groupby("lat_band").size().reset_index(name="n")
    plt.figure(figsize=(9, 4), dpi=120)
    plt.bar(band["lat_band"].astype(str), band["n"])
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Count")
    plt.title("Latitude bands")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_hemisphere_distribution(train: pd.DataFrame, out_path: Path) -> None:
    """Show north vs. south hemisphere share."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    share = train.groupby("hemisphere").size().reset_index(name="n")
    plt.figure(figsize=(6, 4), dpi=120)
    plt.bar(share["hemisphere"], share["n"])
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


    plot_global_scatter(train, plot_dir / "train_scatter_global.png")
    plot_hexbin_density(train, plot_dir / "train_hexbin_density.png")
    export_top_species_table_and_bar(
        train,
        plot_dir / "top_species_table.csv",
        plot_dir / "top_species_counts.png",
        topk=20,
    )
    plot_latitude_hist(train, plot_dir / "lat_hist.png")
    plot_longitude_hist(train, plot_dir / "lon_hist.png")
    plot_latitude_band_counts(train, plot_dir / "lat_band_counts.png")
    plot_hemisphere_distribution(train, plot_dir / "hemisphere_distribution.png")

    print("Plots build")

if __name__ == "__main__":
    main()