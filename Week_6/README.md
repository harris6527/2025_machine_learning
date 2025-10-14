Week 6 Assignment – GDA + Piecewise Model

What this contains
- `gda.py`: From‑scratch Gaussian Discriminant Analysis (binary). Supports shared covariance (LDA boundary) and per‑class covariances (QDA boundary).
- `train_week6.py`:
  - Loads Week 4 CSVs: `../week_4/week4_classification.csv` and `../week_4/week4_regression.csv`.
  - Trains GDA (shared covariance) and reports test accuracy.
  - Plots and saves the GDA decision boundary.
  - Trains a polynomial regression model (degree=3) like Week 4 and reports MAE/RMSE/R2.
  - Builds the piecewise function h(x) using the Week 4‑style classifier (StandardScaler + LogisticRegression) and the polynomial regressor, and saves sample outputs and heatmaps.

How to run
1) Install dependencies (numpy, pandas, matplotlib, scikit‑learn):
   pip install -r requirements.txt

2) Execute the training script from this folder:
   python train_week6.py

Outputs
- `outputs/gda_decision_boundary.png`: Decision boundary over all points.
- `outputs/metrics_summary.csv`: Accuracies and regression metrics.
- `outputs/piecewise_sample.csv`: Sample rows of h(x) over classification grid.
- `outputs/piecewise_h_heatmap.png`: Heatmap of h(x) using Week4 classifier.
- `outputs/piecewise_h_gda_heatmap.png`: Heatmap of h(x) using GDA classifier (optional comparison).

Notes
- The GDA implementation avoids built‑in classifiers as required.
- For the piecewise model, the classifier is the Week 4‑style LogisticRegression (balanced) and the regressor is the Week 4 polynomial (degree=3), aligning with your HW4 pipeline.
