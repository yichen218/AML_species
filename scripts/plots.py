from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_scatter_global(train: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,5), dpi=120)
    plt.scatter(train["lon"], train["lat"], s=1, alpha=0.35)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Training observations — global scatter")
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_train_hexbin_density(train: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,5), dpi=120)
    hb = plt.hexbin(train["lon"], train["lat"], gridsize=80, mincnt=1)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Training observations — density (hexbin)")
    cb = plt.colorbar(hb); cb.set_label("Count")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_top_species_counts(top: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = top.apply(
        lambda r: str(int(r["taxon_id"])) if pd.isna(r["taxon_name"]) or str(r["taxon_name"]).strip()==""
        else str(r["taxon_name"]), axis=1)
    plt.figure(figsize=(10,6), dpi=120)
    plt.bar(range(len(top)), top["count"])
    plt.xticks(range(len(top)), labels, rotation=75, ha="right")
    plt.ylabel("Observations"); plt.title("Top species by observations")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_lat_hist(train: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4), dpi=120)
    plt.hist(train["lat"], bins=60)
    plt.xlabel("Latitude"); plt.ylabel("Count"); plt.title("Latitude distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_lon_hist(train: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4), dpi=120)
    plt.hist(train["lon"], bins=60)
    plt.xlabel("Longitude"); plt.ylabel("Count"); plt.title("Longitude distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_lat_band_counts(train: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    band = train.groupby("lat_band").size().reset_index(name="n")
    plt.figure(figsize=(9,4), dpi=120)
    plt.bar(band["lat_band"].astype(str), band["n"])
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Count")
    plt.title("Latitude bands")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_hemisphere_distribution(train: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    share = train.groupby("hemisphere").size().reset_index(name="n")
    plt.figure(figsize=(6,4), dpi=120)
    plt.bar(share["hemisphere"], share["n"])
    plt.xlabel("Hemisphere"); plt.ylabel("Count"); plt.title("Hemisphere distribution")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_grid_coverage(train: pd.DataFrame, out_path: Path, top_dense_csv: Path, topk: int = 20) -> None:
    # Grid-level counts for quick density-by-cell; also export top-dense cells table.
    grid_counts = (train.groupby(["grid_lat","grid_lon"])
                   .size().reset_index(name="n_points")
                   .sort_values("n_points", ascending=False))
    top_dense_csv.parent.mkdir(parents=True, exist_ok=True)
    grid_counts.head(topk).to_csv(top_dense_csv, index=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,5), dpi=120)
    sizes = 10 + 40 * (grid_counts["n_points"] / grid_counts["n_points"].max())
    plt.scatter(grid_counts["grid_lon"], grid_counts["grid_lat"], s=sizes)
    plt.xlabel("Grid lon"); plt.ylabel("Grid lat")
    plt.title("Grid coverage (1° cells)")
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_train_test_grid_overlap(train: pd.DataFrame, test: pd.DataFrame | None,
                                 fig_path: Path, json_path: Path) -> None:
    if test is None:
        return
    g_train = set(train["grid_id"].astype(str).unique())
    g_test  = set(test["grid_id"].astype(str).unique())
    overlap = g_train & g_test
    stats = {
        "train_grid_n": len(g_train),
        "test_grid_n": len(g_test),
        "overlap_n": len(overlap),
        "overlap_ratio": (len(overlap)/len(g_test)) if g_test else 0.0
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, indent=2))

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4), dpi=120)
    keys = ["train_grid_n","test_grid_n","overlap_n"]
    vals = [stats[k] for k in keys]
    plt.bar(keys, vals)
    plt.title("Train/Test grid overlap"); plt.ylabel("Count")
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()

def plot_species_bbox_vs_count(train: pd.DataFrame, out_img: Path, out_csv: Path, topk: int = 50) -> None:
    # Simple spread proxy: (max_lat-min_lat)*(max_lon-min_lon) for top-k species.
    top = (train.groupby(["taxon_id","taxon_name"]).size()
           .reset_index(name="count").sort_values("count", ascending=False).head(topk))
    rows = []
    for _, r in top.iterrows():
        sid = r["taxon_id"]; sub = train[train["taxon_id"] == sid]
        bbox = max(0.0, sub["lat"].max()-sub["lat"].min()) * max(0.0, sub["lon"].max()-sub["lon"].min())
        rows.append({"taxon_id": sid, "taxon_name": r["taxon_name"], "count": int(r["count"]), "bbox_area_deg2": float(bbox)})
    df = pd.DataFrame(rows).sort_values(["bbox_area_deg2","count"], ascending=[False, False])
    out_csv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)

    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5), dpi=120)
    plt.scatter(df["bbox_area_deg2"], df["count"], s=20, alpha=0.7)
    plt.xlabel("BBox area (deg²)"); plt.ylabel("Observations")
    plt.title("Top species: bbox area vs. observations")
    plt.tight_layout(); plt.savefig(out_img, bbox_inches="tight"); plt.close()

def plot_species_long_tail(train: pd.DataFrame, out_img: Path, out_csv: Path) -> None:
    freq = (train.groupby(["taxon_id","taxon_name"]).size()
            .reset_index(name="count").sort_values("count", ascending=False))
    freq["cum_count"] = freq["count"].cumsum()
    freq["cum_ratio"] = freq["cum_count"] / freq["count"].sum()
    out_csv.parent.mkdir(parents=True, exist_ok=True); freq.to_csv(out_csv, index=False)

    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,5), dpi=120)
    plt.plot(np.arange(1, len(freq)+1), freq["count"].values)
    plt.xlabel("Species rank (by frequency)"); plt.ylabel("Observations")
    plt.title("Species frequency long tail")
    plt.tight_layout(); plt.savefig(out_img, bbox_inches="tight"); plt.close()


def main() -> None:
    # Project-relative defaults (robust to CWD)
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "species_data"
    fig_dir  = base_dir / "plots"
    out_dir  = base_dir / "results"

    ap = argparse.ArgumentParser(description="Make basic EDA plots and summaries.")
    ap.add_argument("--train-features", default=str(data_dir / "train_features.csv"))
    ap.add_argument("--test-features",  default=str(data_dir / "test_features.csv"))
    ap.add_argument("--figdir",         default=str(fig_dir))
    ap.add_argument("--outdir",         default=str(out_dir))
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    train_path = Path(args.train_features)
    test_path  = Path(args.test_features)
    figdir = Path(args.figdir)
    outdir = Path(args.outdir)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path.exists() else None

    # Plots
    plot_train_scatter_global(train, figdir / "train_scatter_global.png")
    plot_train_hexbin_density(train, figdir / "train_hexbin_density.png")

    # Top species table + bar
    top = (train[["taxon_id","taxon_name"]]
           .dropna(subset=["taxon_id"])
           .groupby(["taxon_id","taxon_name"], dropna=False)
           .size().reset_index(name="count")
           .sort_values("count", ascending=False).head(min(20, args.topk)))
    (outdir / "top_species_table.csv").parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(outdir / "top_species_table.csv", index=False)
    plot_top_species_counts(top, figdir / "top_species_counts.png")

    # Distributions
    plot_lat_hist(train, figdir / "lat_hist.png")
    plot_lon_hist(train, figdir / "lon_hist.png")
    plot_lat_band_counts(train, figdir / "lat_band_counts.png")
    plot_hemisphere_distribution(train, figdir / "hemisphere_distribution.png")

    # Grid coverage + overlap
    plot_grid_coverage(train, figdir / "grid_coverage.png", outdir / "top_dense_cells.csv",
                       topk=min(20, args.topk))
    plot_train_test_grid_overlap(train, test, figdir / "train_test_grid_overlap.png",
                                 outdir / "train_test_grid_overlap.json")

    # Spread proxy + long tail
    plot_species_bbox_vs_count(train, figdir / "species_bbox_vs_count.png",
                               outdir / "species_bbox_summary.csv",
                               topk=max(20, args.topk))
    plot_species_long_tail(train, figdir / "species_long_tail.png",
                           outdir / "species_frequency_summary.csv")

    print(f"[OK] EDA done.\n  Plots  → {figdir}\n  Tables → {outdir}")


if __name__ == "__main__":
    main()