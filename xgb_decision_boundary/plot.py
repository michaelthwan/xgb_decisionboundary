import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def plot_decision_boundary(
    model,
    X,
    y,
    features=None,
    resolution=200,
    figsize=(8, 6),
    title=None,
    ax=None,
    cmap=None,
    scatter_kwargs=None,
):
    """Plot XGBoost decision boundary over the top 2 most important features.

    Parameters
    ----------
    model : fitted XGBClassifier
    X : array-like of shape (n_samples, n_features)
        Training or test data (DataFrame or ndarray).
    y : array-like of shape (n_samples,)
        True binary labels (0/1).
    features : tuple of (feat1, feat2), optional
        Two feature names (str) or column indices (int) to plot.
        If None, auto-selects the top 2 by feature importance.
    resolution : int
        Grid resolution per axis (default 200).
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Plot title. Auto-generates "XGBoost ({acc}%)" if None.
    ax : matplotlib Axes, optional
        Existing axes to draw on.
    cmap : matplotlib colormap, optional
        Colormap for contourf and scatter (default: RdBu_r).
    scatter_kwargs : dict, optional
        Extra keyword arguments passed to ax.scatter.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import pandas as pd

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = np.asarray(y)
    n_features = X_arr.shape[1]

    # --- Resolve feature indices and names ---
    if isinstance(X, pd.DataFrame):
        col_names = list(X.columns)
    else:
        col_names = [str(i) for i in range(n_features)]

    if features is not None:
        idx = []
        names = []
        for f in features:
            if isinstance(f, str):
                i = col_names.index(f)
            else:
                i = int(f)
            idx.append(i)
            names.append(col_names[i])
        feat_idx = idx
        feat_names = names
    else:
        importances = model.feature_importances_
        top2 = np.argsort(importances)[-2:][::-1]
        feat_idx = [int(top2[0]), int(top2[1])]
        feat_names = [col_names[feat_idx[0]], col_names[feat_idx[1]]]

    # --- Build meshgrid ---
    x0 = X_arr[:, feat_idx[0]]
    x1 = X_arr[:, feat_idx[1]]
    margin0 = (x0.max() - x0.min()) * 0.05
    margin1 = (x1.max() - x1.min()) * 0.05
    xx = np.linspace(x0.min() - margin0, x0.max() + margin0, resolution)
    yy = np.linspace(x1.min() - margin1, x1.max() + margin1, resolution)
    xx_grid, yy_grid = np.meshgrid(xx, yy)

    # --- Build prediction input (median-filled) ---
    grid_points = xx_grid.ravel().shape[0]
    medians = np.median(X_arr, axis=0)
    grid_data = np.tile(medians, (grid_points, 1))
    grid_data[:, feat_idx[0]] = xx_grid.ravel()
    grid_data[:, feat_idx[1]] = yy_grid.ravel()

    # --- Predict probabilities ---
    Z = model.predict_proba(grid_data)[:, 1]
    Z = Z.reshape(xx_grid.shape)

    # --- Plot ---
    if cmap is None:
        cmap = "RdBu_r"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    levels = np.linspace(0, 1, 11)
    cf = ax.contourf(xx_grid, yy_grid, Z, levels=levels, cmap=cmap, alpha=0.8)
    fig.colorbar(cf, ax=ax, label="P(class 1)")

    # --- Scatter data points ---
    s_kwargs = dict(edgecolors="k", s=10, linewidths=0.3, alpha=0.7)
    if scatter_kwargs:
        s_kwargs.update(scatter_kwargs)
    ax.scatter(x0, x1, c=y_arr, cmap=cmap, **s_kwargs)

    # --- Annotations ---
    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])

    if title is None:
        acc = accuracy_score(y_arr, model.predict(X_arr))
        title = f"XGBoost ({acc * 100:.1f}%)"
    ax.set_title(title)

    return fig, ax
