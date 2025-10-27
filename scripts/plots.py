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


def plot_top20_coverage(train: pd.DataFrame, table_path: Path, fig_path: Path, topk: int = 20) -> None:
    """Export top-k species by spatial coverage (unique grid cells) + bar chart."""
    # id→name mapping for display (prefer first non-null name)
    id2name = (train.dropna(subset=["taxon_name"])
                    .drop_duplicates("taxon_id")
                    .set_index("taxon_id")["taxon_name"]
                    .to_dict())

    cov = (train.groupby("taxon_id")["grid_group"]
           .nunique()
           .reset_index(name="grid_coverage")
           .sort_values("grid_coverage", ascending=False)
           .head(topk))

    cov["taxon_name"] = cov["taxon_id"].map(id2name).fillna("")

    table_path.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(table_path, index=False)

    labels = cov.apply(
        lambda r: str(int(r["taxon_id"])) if not r["taxon_name"]
        else (r["taxon_name"] if len(str(r["taxon_name"])) <= 25 else str(r["taxon_name"])[:25] + "…"),
        axis=1
    )

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.bar(range(len(cov)), cov["grid_coverage"])
    plt.xticks(range(len(cov)), labels, rotation=75, ha="right")
    plt.ylabel("Unique grid cells (1°)");
    plt.title(f"Top species by spatial coverage (topk={topk})")
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


    plot_global_scatter(train, plot_dir / "train_scatter_global.png")
    plot_hexbin_density(train, plot_dir / "train_hexbin_density.png")
    plot_top20_coverage(
        train,
        plot_dir / "top_species_coverage_table.csv",
        plot_dir / "top_species_coverage.png",
        topk=20,
    )
    plot_latitude_hist(train, plot_dir / "lat_hist.png")
    plot_longitude_hist(train, plot_dir / "lon_hist.png")
    plot_latitude_band_counts(train, plot_dir / "lat_band_counts.png")
    plot_hemisphere_distribution(train, plot_dir / "hemisphere_distribution.png")

    print("Plots build")

if __name__ == "__main__":
    main()