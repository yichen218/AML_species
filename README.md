# Biodiversity Prediction Project (Team 38)

This repository contains two complementary models for predicting biodiversity patterns:
- Species→Location (Member B): given a species, predict likely regions where it occurs
- Location→Species (Member C): given a region, predict likely species present

## Environment
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Data Layout
We use a simple layout under `species_data/` (equivalent to the example `data/raw` and `data/processed`):
```
species_data/
├── species_train.npz
├── species_train_extra.npz
└── species_test.npz
```
Features are generated to:
```
species_data/train_features.csv
species_data/test_features.csv
```

## How to Run
```bash
# 1) Build features
python3 scripts/build_features.py

# 2) Train Species→Location (Member B)
python3 scripts/train_species_to_location.py \
  --train-features species_data/train_features.csv \
  --topk 50 --cv-splits 5 --model rf \
  --outdir results --modeldir models --seed 42

# 3) Train Location→Species (Member C)
python3 scripts/train_location_to_species.py \
  --train-features species_data/train_features.csv \
  --topk 50 --cv-splits 5 \
  --outdir results --seed 42

# 4) Co-occurrence mining
python3 scripts/mine_cooccurrence.py \
  --train-features species_data/train_features.csv \
  --min-support 50 --outdir results

# 5) Test evaluation
## C (loc2spec)
python3 scripts/evaluate_test.py \
  --test-features species_data/test_features.csv \
  --test-npz species_data/species_test.npz \
  --loc2spec-model results/loc2spec_model.joblib \
  --outdir results/test_eval_loc2spec --figdir figures

## B (spec2loc)
python3 scripts/evaluate_spec2loc_test.py \
  --test-features species_data/test_features.csv \
  --test-npz species_data/species_test.npz \
  --model models/spec2loc_rf.pkl \
  --outdir results/test_eval_spec2loc --figdir results/figures
```

## Outputs
- Models: `models/*.pkl` or `results/loc2spec_model.joblib`
- CV metrics: `results/spec2loc_cv_metrics.csv`, `results/multilabel_cv_metrics.csv`
- Test metrics: `results/test_eval_*/test_metrics.json`
- Figures: `figures/*.png` (loc2spec) and `results/figures/*.png` (spec2loc)

## Important Notes
- Test labels (`species_test.npz:test_pos_inds`) are only used in final evaluation; never in training/tuning.
- Spatial GroupKFold is used to avoid geographic leakage.

### About spec2loc vs. report (38.pdf)
The spec2loc evaluation results in this repository do not match some numbers in the report. After submitting the report, we identified issues in the previous unified evaluation and fixed the spec2loc code to correctly handle per-species models and test aggregation. As a result, the current code reflects corrected, reproducible metrics which differ from the report’s hardcoded values.

## Repository Structure
```
scripts/              # Core scripts
species_data/         # Raw NPZ and generated features
models/               # Saved models
results/              # Metrics and analysis
figures/              # Evaluation curves
requirements.txt
README.md
```
