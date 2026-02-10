import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class DecisionBoundary:
    """Computes and plots XGBoost decision boundaries in 2D.

    Supports two methods:
      - "features": plots over the top 2 most important features (default)
      - "reduce": projects all features into 2D via dimensionality reduction

    Usage
    -----
        # Method 1: top 2 features
        db = DecisionBoundary(model)
        db.fit(X, y)
        fig, ax = db.plot()

        # Method 2: dimensionality reduction (PCA by default)
        db = DecisionBoundary(model, method="reduce")
        db.fit(X, y)
        fig, ax = db.plot()

        # Method 2 with custom reducer
        from sklearn.manifold import TSNE
        db = DecisionBoundary(model, method="reduce", reducer=TSNE(n_components=2))
        db.fit(X, y)
        fig, ax = db.plot()
    """

    def __init__(self, model, features=None, resolution=200,
                 method="features", reducer=None):
        """
        Parameters
        ----------
        model : fitted XGBClassifier
        features : tuple of (feat1, feat2), optional
            Only used when method="features". Override auto-selection.
        resolution : int
            Grid resolution per axis.
        method : str
            "features" — top 2 features with median imputation.
            "reduce"  — dimensionality reduction of full feature vector.
        reducer : sklearn transformer, optional
            Must have fit_transform(). Only used when method="reduce".
            If it has inverse_transform(), grid points are mapped back to
            original space for exact model predictions. Otherwise, a KNN
            interpolator approximates the background.
            Defaults to PCA(n_components=2).
        """
        self.model = model
        self.features = features
        self.resolution = resolution
        self.method = method
        self.reducer = reducer
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

        if self.method == "features":
            self._fit_features(X)
        elif self.method == "reduce":
            self._fit_reduce()
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'features' or 'reduce'.")

        self._fitted = True
        return self

    def _fit_features(self, X):
        """Top-2 features mode: meshgrid + median imputation."""
        import pandas as pd

        n_features = self.X_arr.shape[1]
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

        # Meshgrid
        x0 = self.X_arr[:, self.feat_idx[0]]
        x1 = self.X_arr[:, self.feat_idx[1]]
        margin0 = (x0.max() - x0.min()) * 0.05
        margin1 = (x1.max() - x1.min()) * 0.05
        xx = np.linspace(x0.min() - margin0, x0.max() + margin0, self.resolution)
        yy = np.linspace(x1.min() - margin1, x1.max() + margin1, self.resolution)
        self.xx_grid, self.yy_grid = np.meshgrid(xx, yy)

        # Median-filled prediction input
        grid_points = self.xx_grid.ravel().shape[0]
        medians = np.median(self.X_arr, axis=0)
        grid_data = np.tile(medians, (grid_points, 1))
        grid_data[:, self.feat_idx[0]] = self.xx_grid.ravel()
        grid_data[:, self.feat_idx[1]] = self.yy_grid.ravel()

        self.Z = self.model.predict_proba(grid_data)[:, 1].reshape(self.xx_grid.shape)
        # Store scatter coordinates
        self._scatter_x0 = x0
        self._scatter_x1 = x1

    def _fit_reduce(self):
        """Dimensionality reduction mode: project full X into 2D."""
        from sklearn.manifold import TSNE

        if self.reducer is None:
            self.reducer = TSNE(n_components=2, random_state=42)

        # Reduce data to 2D
        self.X_reduced = self.reducer.fit_transform(self.X_arr)
        self.feat_names = ["Component 1", "Component 2"]

        # Meshgrid over reduced space
        r0, r1 = self.X_reduced[:, 0], self.X_reduced[:, 1]
        margin0 = (r0.max() - r0.min()) * 0.05
        margin1 = (r1.max() - r1.min()) * 0.05
        xx = np.linspace(r0.min() - margin0, r0.max() + margin0, self.resolution)
        yy = np.linspace(r1.min() - margin1, r1.max() + margin1, self.resolution)
        self.xx_grid, self.yy_grid = np.meshgrid(xx, yy)

        grid_2d = np.c_[self.xx_grid.ravel(), self.yy_grid.ravel()]

        if hasattr(self.reducer, "inverse_transform"):
            # PCA etc. — map grid back to original space, predict exactly
            grid_original = self.reducer.inverse_transform(grid_2d)
            self.Z = self.model.predict_proba(grid_original)[:, 1]
        else:
            # t-SNE, UMAP etc. — approximate with KNN interpolation
            from sklearn.neighbors import KNeighborsRegressor
            proba = self.model.predict_proba(self.X_arr)[:, 1]
            knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
            knn.fit(self.X_reduced, proba)
            self.Z = knn.predict(grid_2d)

        self.Z = self.Z.reshape(self.xx_grid.shape)
        # Store scatter coordinates
        self._scatter_x0 = self.X_reduced[:, 0]
        self._scatter_x1 = self.X_reduced[:, 1]

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
        s_kwargs = dict(edgecolors="k", s=10, linewidths=0.3, alpha=0.7)
        if scatter_kwargs:
            s_kwargs.update(scatter_kwargs)
        ax.scatter(self._scatter_x0, self._scatter_x1,
                   c=self.y_arr, cmap=cmap, **s_kwargs)

        # Annotations
        ax.set_xlabel(self.feat_names[0])
        ax.set_ylabel(self.feat_names[1])
        if title is None:
            acc = accuracy_score(self.y_arr, self.model.predict(self.X_arr))
            title = f"XGBoost ({acc * 100:.1f}%)"
        ax.set_title(title)

        return fig, ax


def plot_decision_boundary(
    model, X, y, features=None, resolution=200, method="features", reducer=None,
    figsize=(8, 6), title=None, ax=None, cmap=None, scatter_kwargs=None,
):
    """Convenience function — fit and plot in one call."""
    db = DecisionBoundary(model, features=features, resolution=resolution,
                          method=method, reducer=reducer)
    db.fit(X, y)
    return db.plot(figsize=figsize, title=title, ax=ax, cmap=cmap,
                   scatter_kwargs=scatter_kwargs)
