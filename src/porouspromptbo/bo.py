from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    imp = mu - best_y - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def propose_next(mu: np.ndarray, sigma: np.ndarray, best_y: float, n: int = 1) -> np.ndarray:
    ei = expected_improvement(mu, sigma, best_y=best_y)
    idx = np.argsort(-ei)[:n]
    return idx
