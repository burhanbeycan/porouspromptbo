from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RFEnsembleSurrogate:
    """Random forest surrogate with uncertainty via tree ensembles.

    Predictive mean: mean over trees
    Predictive std: std over trees

    This is a pragmatic baseline for mixed categorical/continuous spaces.
    """

    n_estimators: int = 200
    random_state: int = 0
    min_samples_leaf: int = 2

    def __post_init__(self) -> None:
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFEnsembleSurrogate":
        self.rf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.stack([t.predict(X) for t in self.rf.estimators_], axis=0)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0) + 1e-9
        return mu, sigma
