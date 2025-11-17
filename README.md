1. Operate in terminal:
    pip install -r requirements.txt
    python scripts/build_features.py
    python scripts/plots.py
    python scripts/mine_cooccurrence.py --train-features species_data/train_features.csv --outdir results/cooccurrence
    python scripts/train_location_to_species_baseline.py --train-features species_data/train_features.csv --outdir results/location_to_species --topk 50 --cv-splits 5 --seed 42
2. Features:
    taxon_id (int)
    taxon_name (str)
    lat (float), lon (float): original coordinate
    grid_lat (int), grid_lon (int): grid cell coordinate
    region (str):"{grid_lat}_{grid_lon}"
    hemisphere (category): N/S
    sin_lon (float), cos_lon (float)


3. Build features
Input:
species_data/species_train.npz
species_data/species_train_extra.npz
species_data/species_test.npz

Output:
species_data/train_features.csv
species_data/test_features.csv

4. Build plots
Input:
species_data/train_features.csv

Output: 
plots/global_distribution.png
plots/spatial_hotspots_density.png
plots/lat_distribution.png
plots/lon_distribution.png
plots/hemisphere_distribution.png

5. Species co-occurrence
Input:
species_data/train_features.csv

Output:
results/cooccurrence/species_cooccurrence.csv

6. Location → species
Input:
species_data/train_features.csv

Output: 
results/location_to_species/multilabel_cv_metrics.csv
results/location_to_species/loc2spec_rf_model.joblib

