import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gda import GDA


def load_week4_datasets(base_dir: Path):
    week4_dir = base_dir.parent / "week_4"
    cls_path = week4_dir / "week4_classification.csv"
    reg_path = week4_dir / "week4_regression.csv"
    if not cls_path.exists() or not reg_path.exists():
        raise FileNotFoundError(f"Expected week_4 CSVs at {cls_path} and {reg_path}.")
    cls = pd.read_csv(cls_path)
    reg = pd.read_csv(reg_path)
    # Standardize column names
    cls.columns = [c.strip().lower() for c in cls.columns]
    reg.columns = [c.strip().lower() for c in reg.columns]
    return cls, reg


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=False):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    if stratify:
        # simple stratified split for binary labels
        idx0 = idx[y == 0]
        idx1 = idx[y == 1]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        n0_te = int(len(idx0) * test_size)
        n1_te = int(len(idx1) * test_size)
        te_idx = np.concatenate([idx0[:n0_te], idx1[:n1_te]])
        tr_idx = np.concatenate([idx0[n0_te:], idx1[n1_te:]])
        rng.shuffle(te_idx)
        rng.shuffle(tr_idx)
    else:
        rng.shuffle(idx)
        n_te = int(n * test_size)
        te_idx = idx[:n_te]
        tr_idx = idx[n_te:]
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def evaluate_gda(X, y, shared_cov=True):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)
    gda = GDA(shared_cov=shared_cov)
    gda.fit(X_tr, y_tr)
    y_hat = gda.predict(X_te)
    acc = accuracy(y_te, y_hat)
    return gda, acc, (X_tr, X_te, y_tr, y_te)


def plot_decision_boundary(model: GDA, X: np.ndarray, y: np.ndarray, out_path: Path, title: str):
    # Prepare grid based on lon/lat range
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6), dpi=120)
    plt.contourf(xx, yy, zz, levels=[-0.1, 0.5, 1.1], colors=["#f0f0f0", "#e0f7fa"], alpha=0.8)
    # scatter points
    m0 = y == 0
    m1 = y == 1
    plt.scatter(X[m0, 0], X[m0, 1], s=6, c="#bdbdbd", label="label=0", edgecolors="none")
    plt.scatter(X[m1, 0], X[m1, 1], s=6, c="#0277bd", label="label=1", edgecolors="none")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.legend(loc="best", fontsize=9)
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_regression_model(Xr: np.ndarray, yr: np.ndarray):
    # Use scikit-learn PolynomialFeatures + LinearRegression to mirror week_4
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("lin", LinearRegression()),
    ])
    model.fit(Xr, yr)
    return model


def regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def piecewise_h(C_model, R_model, X_all: np.ndarray):
    # C_model: classifier with predict(X) -> {0,1}
    # R_model: regressor with predict(X) -> value
    cls = C_model.predict(X_all)
    vals = np.full(X_all.shape[0], -999.0, dtype=float)
    mask = (cls == 1)
    if np.any(mask):
        vals[mask] = R_model.predict(X_all[mask])
    return vals


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets from week_4
    cls_df, reg_df = load_week4_datasets(here)
    Xc = cls_df[["lon", "lat"]].to_numpy(float)
    yc = cls_df["label"].to_numpy(int)
    Xr = reg_df[["lon", "lat"]].to_numpy(float)
    yr = reg_df["value"].to_numpy(float)

    # 1) GDA classification (from scratch)
    gda, acc, (Xc_tr, Xc_te, yc_tr, yc_te) = evaluate_gda(Xc, yc, shared_cov=True)
    print(f"GDA (shared covariance) test accuracy: {acc:.4f}")

    # Save decision boundary plot over all classification points
    plot_decision_boundary(
        gda,
        Xc,
        yc,
        out_dir / "gda_decision_boundary.png",
        title=f"GDA Decision Boundary (acc={acc:.3f})",
    )

    # 2) Regression model mirroring week_4 (Polynomial degree=3)
    from sklearn.model_selection import train_test_split as skl_split
    Xr_tr, Xr_te, yr_tr, yr_te = skl_split(Xr, yr, test_size=0.2, random_state=42)
    regr = build_regression_model(Xr_tr, yr_tr)
    y_pred = regr.predict(Xr_te)
    mae, rmse, r2 = regression_metrics(yr_te, y_pred)
    print(f"Regression metrics -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # 3) Build piecewise function h(x)
    #    Per assignment wording, h combines the two models from Week 4.
    #    We'll replicate Week 4 classifier (LogisticRegression with StandardScaler) and reuse polynomial regr.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    Xc_tr2, Xc_te2, yc_tr2, yc_te2 = skl_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    clf_week4 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    clf_week4.fit(Xc_tr2, yc_tr2)
    acc_week4 = float((clf_week4.predict(Xc_te2) == yc_te2).mean())
    print(f"Week4-like classifier (LogReg) test accuracy: {acc_week4:.4f}")

    # h(x) using Week4-like classifier + polynomial regression
    hx_all = piecewise_h(clf_week4, regr, Xc)
    # Save a small sample table to inspect
    sample_idx = np.linspace(0, Xc.shape[0]-1, num=min(200, Xc.shape[0]), dtype=int)
    sample_df = pd.DataFrame({
        "lon": Xc[sample_idx, 0],
        "lat": Xc[sample_idx, 1],
        "C_week4": clf_week4.predict(Xc[sample_idx]),
        "h(x)": hx_all[sample_idx],
    })
    sample_df.to_csv(out_dir / "piecewise_sample.csv", index=False)

    # Also, produce a heatmap-like plot of h(x) over a lon/lat grid for visualization
    x_min, x_max = Xc[:, 0].min(), Xc[:, 0].max()
    y_min, y_max = Xc[:, 1].min(), Xc[:, 1].max()
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150)
    )
    grid_pts = np.c_[gx.ravel(), gy.ravel()]
    grid_vals = piecewise_h(clf_week4, regr, grid_pts).reshape(gx.shape)

    plt.figure(figsize=(7, 6), dpi=120)
    img = plt.imshow(
        grid_vals,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
        cmap="coolwarm",
        vmin=min(-999.0, np.nanmin(grid_vals[grid_vals > -999 + 1e-9]) if np.any(grid_vals > -999) else -999.0),
    )
    plt.colorbar(img, label="h(x) value (-999 for invalid)")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title("Piecewise h(x): Week4 classifier + Poly Regr")
    plt.tight_layout()
    plt.savefig(out_dir / "piecewise_h_heatmap.png")
    plt.close()

    # Optional: compare piecewise built with GDA as classifier
    hx_gda = piecewise_h(gda, regr, grid_pts).reshape(gx.shape)
    plt.figure(figsize=(7, 6), dpi=120)
    img2 = plt.imshow(
        hx_gda,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
        cmap="coolwarm",
        vmin=min(-999.0, np.nanmin(hx_gda[hx_gda > -999 + 1e-9]) if np.any(hx_gda > -999) else -999.0),
    )
    plt.colorbar(img2, label="h_gda(x) value (-999 for invalid)")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title("Piecewise h(x) using GDA classifier")
    plt.tight_layout()
    plt.savefig(out_dir / "piecewise_h_gda_heatmap.png")
    plt.close()

    # Save simple metrics summaries
    pd.DataFrame({
        "metric": ["acc_gda", "acc_week4_logreg", "mae_reg", "rmse_reg", "r2_reg"],
        "value": [acc, acc_week4, mae, rmse, r2]
    }).to_csv(out_dir / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()

