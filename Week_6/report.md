# Week 6 Programming Assignment

## Dataset Recap and Preprocessing
- Source: `../week_4/week4_classification.csv` (lon, lat, label) and `../week_4/week4_regression.csv` (lon, lat, value) generated in HW4 from `O-A0038-003.xml`.
- Features: geographic coordinates (longitude, latitude).
- Labels: binary flag for valid temperature (1) vs missing value (-999) (0).
- Regression targets: actual temperature values for valid grid points.

## Gaussian Discriminant Analysis (GDA)
- GDA treats each class as a multivariate Gaussian with mean `μ_y` and shared covariance `Σ`. Priors `P(y)` come from class frequencies.
- Decision rule: classify `x` as label 1 when `log p(x | y=1) + log P(y=1) > log p(x | y=0) + log P(y=0)`. With shared covariance this yields a linear boundary, suited to smoothly varying spatial data.
- Implementation: `week_6/gda.py` (NumPy only). Computes class means, shared covariance with regularisation, and uses log-likelihood for `predict_proba`.
- Training: stratified 80/20 split (random seed 42). Test accuracy printed by `train_week6.py` (`GDA (shared covariance) test accuracy: 0.5267`).
- Visualization: `outputs/gda_decision_boundary.png` shows the separating line overlaying all grid points.

## Regression Model
- Mirrors Week 4 pipeline: polynomial features (degree 3) + linear regression.
- 80/20 split (random seed 42). Metrics from `train_week6.py`:
  - MAE = 3.033°C
  - RMSE = 4.445°C
  - R² = 0.418

## Piecewise Function h(x)
- Classification component `C(x)`: Week 4 style `StandardScaler + LogisticRegression(balanced)` retrained on the latest Week 4 dataset (80/20 stratified split).
- Regression component `R(x)`: the polynomial regression above.
- Definition:  
  ```text
  h(x) = R(x) if C(x) = 1
       = -999 otherwise
  ```
- Verification:
  - `piecewise_sample.csv` samples predictions; rows with `C_week4 = 0` always return `-999`.
  - `piecewise_h_heatmap.png` visualises the piecewise surface over lon/lat grid.
  - `piecewise_h_gda_heatmap.png` optionally shows the same construction using the GDA classifier for comparison.

## How to Reproduce
```bash
cd week_6
pip install -r requirements.txt
python train_week6.py
```
- All metrics, plots, and sample tables are generated under `week_6/outputs/`.

## Notes
- GDA implementation is entirely custom as required.
- Decision boundary and heatmaps provide the requested visual evidence.
- Piecewise `h(x)` combines the Week 4 classifier/regressor exactly as specified and has been checked for the correct `-999` fallback.
