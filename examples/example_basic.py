"""Basic example: decision boundary on a ~30k sample synthetic dataset."""

import matplotlib.pyplot as plt
import numpy as np
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

# Compute decision boundary (separate from plotting)
db = DecisionBoundary(model)
db.fit(X_test, y_test)

# Plot
fig, ax = db.plot()
plt.tight_layout()
plt.show()
