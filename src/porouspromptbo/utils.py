from __future__ import annotations

import numpy as np


def scalar_score(yield_pct: np.ndarray, surface_area: np.ndarray, crystallinity: np.ndarray) -> np.ndarray:
    """A simple scalarisation baseline.

    Encourages:
    - high yield
    - high surface area
    - crystallinity above ~0.5

    Replace this with Pareto optimisation in a full system.
    """
    y = yield_pct / 100.0
    sa = surface_area / 2500.0
    cryst_pen = np.clip(0.5 - crystallinity, 0, 1.0)

    return 1.0 * y + 1.2 * sa - 0.8 * cryst_pen
