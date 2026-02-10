"""Basic example: decision boundary — top-2 features vs dimensionality reduction."""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

from xgb_decision_boundary import DecisionBoundary

# Generate ~30k samples with 20 features
X, y = make_classification(
    n_samples=30_000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_clusters_per_class=3,
    flip_y=0.05,
    random_state=42,
)
feature_names = [f"feat_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Train model
model = XGBClassifier(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
model.fit(X_train, y_train)

# --- Mode 1: Top 2 features ---
db_feat = DecisionBoundary(model)
db_feat.fit(X_test, y_test)

# --- Mode 2: t-SNE (default reducer) ---
db_tsne = DecisionBoundary(model, method="reduce")
db_tsne.fit(X_test, y_test)

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
db_feat.plot(ax=ax1, title="Top 2 Features")
db_tsne.plot(ax=ax2, title="t-SNE Reduction")
plt.tight_layout()
plt.show()
