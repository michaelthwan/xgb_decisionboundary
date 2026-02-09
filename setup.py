from setuptools import setup, find_packages

setup(
    name="xgb-decision-boundary",
    version="0.1.0",
    description="Visualize XGBoost decision boundaries over top features",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "xgboost",
        "matplotlib",
        "numpy",
        "scikit-learn",
        "pandas",
    ],
)
