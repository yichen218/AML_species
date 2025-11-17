1. Operate in terminal:
    pip install -r requirements.txt
    python scripts/build_features.py
    python scripts/plots.py
    python scripts/mine_cooccurrence.py --train-features species_data/train_features.csv --outdir results/cooccurrence
    python scripts/train_location_to_species_baseline.py --train-features species_data/train_features.csv --outdir results/location_to_species --topk 50 --cv-splits 5 --seed 42


2. Plots:
    global_distribution.png 
    spatial_hotspots_density.png 
    lat_distribution.png 
    lon_distribution.png 
    hemisphere_distribution.png

3. Features:
    taxon_id (int)
    taxon_name (str)
    lat (float), lon (float): original coordinate
    grid_lat (int), grid_lon (int): grid cell coordinate
    region (str):"{grid_lat}_{grid_lon}"
    hemisphere (category): N/S
    sin_lon (float), cos_lon (float): Longitude cycle


