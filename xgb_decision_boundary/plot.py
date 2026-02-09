import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class DecisionBoundary:
    """Computes and plots XGBoost decision boundaries over the top 2 features.

    Usage
    -----
        db = DecisionBoundary(model, features=None, resolution=200)
        db.fit(X, y)          # compute grid predictions
        fig, ax = db.plot()   # render the figure
    """

    def __init__(self, model, features=None, resolution=200):
        self.model = model
        self.features = features
        self.resolution = resolution
        self._fitted = False

    def fit(self, X, y):
        """Compute meshgrid predictions and store results.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), binary labels (0/1)
        """
        import pandas as pd

        self.X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self.y_arr = np.asarray(y)
        n_features = self.X_arr.shape[1]

        # --- Resolve feature indices and names ---
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
        else:
            col_names = [str(i) for i in range(n_features)]

        if self.features is not None:
            idx, names = [], []
            for f in self.features:
                if isinstance(f, str):
                    i = col_names.index(f)
                else:
                    i = int(f)
                idx.append(i)
                names.append(col_names[i])
            self.feat_idx = idx
            self.feat_names = names
        else:
            importances = self.model.feature_importances_
            top2 = np.argsort(importances)[-2:][::-1]
            self.feat_idx = [int(top2[0]), int(top2[1])]
            self.feat_names = [col_names[self.feat_idx[0]], col_names[self.feat_idx[1]]]

        # --- Build meshgrid ---
        x0 = self.X_arr[:, self.feat_idx[0]]
        x1 = self.X_arr[:, self.feat_idx[1]]
        margin0 = (x0.max() - x0.min()) * 0.05
        margin1 = (x1.max() - x1.min()) * 0.05
        xx = np.linspace(x0.min() - margin0, x0.max() + margin0, self.resolution)
        yy = np.linspace(x1.min() - margin1, x1.max() + margin1, self.resolution)
        self.xx_grid, self.yy_grid = np.meshgrid(xx, yy)

        # --- Build prediction input (median-filled) ---
        grid_points = self.xx_grid.ravel().shape[0]
        medians = np.median(self.X_arr, axis=0)
        grid_data = np.tile(medians, (grid_points, 1))
        grid_data[:, self.feat_idx[0]] = self.xx_grid.ravel()
        grid_data[:, self.feat_idx[1]] = self.yy_grid.ravel()

        # --- Predict probabilities ---
        self.Z = self.model.predict_proba(grid_data)[:, 1].reshape(self.xx_grid.shape)
        self._fitted = True
        return self

    def plot(self, figsize=(8, 6), title=None, ax=None, cmap=None, scatter_kwargs=None):
        """Render the decision boundary plot.

        Parameters
        ----------
        figsize : tuple
        title : str, optional
            Auto-generates "XGBoost ({acc}%)" if None.
        ax : matplotlib Axes, optional
        cmap : colormap (default: RdBu_r)
        scatter_kwargs : dict, optional

        Returns
        -------
        fig, ax
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(X, y) before .plot()")

        if cmap is None:
            cmap = "RdBu_r"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Background contour
        levels = np.linspace(0, 1, 11)
        cf = ax.contourf(self.xx_grid, self.yy_grid, self.Z,
                         levels=levels, cmap=cmap, alpha=0.8)
        fig.colorbar(cf, ax=ax, label="P(class 1)")

        # Scatter data points
        x0 = self.X_arr[:, self.feat_idx[0]]
        x1 = self.X_arr[:, self.feat_idx[1]]
        s_kwargs = dict(edgecolors="k", s=10, linewidths=0.3, alpha=0.7)
        if scatter_kwargs:
            s_kwargs.update(scatter_kwargs)
        ax.scatter(x0, x1, c=self.y_arr, cmap=cmap, **s_kwargs)

        # Annotations
        ax.set_xlabel(self.feat_names[0])
        ax.set_ylabel(self.feat_names[1])
        if title is None:
            acc = accuracy_score(self.y_arr, self.model.predict(self.X_arr))
            title = f"XGBoost ({acc * 100:.1f}%)"
        ax.set_title(title)

        return fig, ax


def plot_decision_boundary(
    model, X, y, features=None, resolution=200,
    figsize=(8, 6), title=None, ax=None, cmap=None, scatter_kwargs=None,
):
    """Convenience function — fit and plot in one call."""
    db = DecisionBoundary(model, features=features, resolution=resolution)
    db.fit(X, y)
    return db.plot(figsize=figsize, title=title, ax=ax, cmap=cmap,
                   scatter_kwargs=scatter_kwargs)
