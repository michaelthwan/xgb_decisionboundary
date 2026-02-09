"""Basic example: decision boundary on sklearn's breast cancer dataset."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

from xgb_decision_boundary import plot_decision_boundary

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
model.fit(X_train, y_train)

# Plot decision boundary on test set
fig, ax = plot_decision_boundary(model, X_test, y_test)
plt.tight_layout()
plt.show()
