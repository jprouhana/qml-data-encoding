"""
Dataset utilities for encoding experiments.
"""

import numpy as np
from sklearn.datasets import make_moons, load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_moons(n_samples=200, noise=0.15, seed=42):
    """Load make_moons dataset scaled to [0, pi]."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def load_iris_binary(seed=42):
    """Load Iris dataset (first 2 classes, first 4 features), scaled to [0, pi]."""
    data = load_iris()
    mask = data.target < 2
    X = data.data[mask]
    y = data.target[mask]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def load_breast_cancer_pca(n_components=4, seed=42):
    """Load breast cancer dataset with PCA reduction, scaled to [0, pi]."""
    from sklearn.decomposition import PCA
    data = load_breast_cancer()
    pca = PCA(n_components=n_components, random_state=seed)
    X = pca.fit_transform(data.data)
    y = data.target
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=seed)
