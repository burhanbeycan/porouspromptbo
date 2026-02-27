from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .data import BuildingBlockLibrary
from .design_space import DesignSpace


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(space: DesignSpace, designs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Synthetic porous-material outcomes.

    Outputs:
    - yield_pct: 0-100
    - surface_area_m2_g: 0-2500
    - crystallinity_score: 0-1

    Underlying structure:
    - aromaticity & low flexibility help surface area/crystallinity
    - temperature and catalyst influence yield/crystallinity
    - concentration too high reduces crystallinity (fast precipitation)
    """
    rng = np.random.default_rng(seed)
    blocks = space.blocks.df.set_index("block_id")

    A = blocks.loc[designs["block_A"].values]
    B = blocks.loc[designs["block_B"].values]

    aromatic = (A["aromatic_rings"].values + B["aromatic_rings"].values)
    flex = (A["flexibility"].values + B["flexibility"].values) / 2.0
    mw = (A["mw"].values + B["mw"].values)

    temp = designs["temperature_C"].to_numpy(float)
    conc = designs["concentration_M"].to_numpy(float)
    catalyst = designs["catalyst"].values
    solvent = designs["solvent"].values

    catalyst_boost = np.where(catalyst == "acetic_acid", 1.0, np.where(catalyst == "tfa", 1.2, 0.7))

    solvent_factor = np.where(solvent == "dioxane", 1.1, np.where(solvent == "mesitylene", 1.05, 0.95))

    # crystallinity: improved by aromaticity, reduced by flexibility and too-high concentration
    cryst_raw = 0.25 * aromatic - 2.0 * flex - 3.0 * np.maximum(conc - 0.25, 0) + 0.02 * (temp - 60) + rng.normal(0, 0.25, size=len(designs))
    crystallinity = _sigmoid(cryst_raw)

    # surface area: correlates with aromaticity and crystallinity; penalised by very high MW (pore collapse)
    sa = (
        400
        + 180 * aromatic
        + 900 * crystallinity
        - 0.15 * np.maximum(mw - 500, 0)
        + 60 * solvent_factor
        + rng.normal(0, 120, size=len(designs))
    )
    sa = np.clip(sa, 0, 2500)

    # yield: depends on catalyst and temperature with an optimum; penalised by low crystallinity
    temp_opt = 75
    yield_pct = (
        55
        + 25 * catalyst_boost
        - 0.03 * (temp - temp_opt) ** 2
        + 15 * solvent_factor
        - 20 * (1 - crystallinity)
        + rng.normal(0, 8, size=len(designs))
    )
    yield_pct = np.clip(yield_pct, 0, 100)

    out = designs.copy()
    out["yield_pct"] = yield_pct
    out["surface_area_m2_g"] = sa
    out["crystallinity_score"] = crystallinity
    return out
