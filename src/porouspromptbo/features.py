from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data import BuildingBlockLibrary
from .design_space import DesignSpace


BLOCK_DESC_COLS = (
    "mw",
    "aromatic_rings",
    "hbd",
    "hba",
    "rot_bonds",
    "tpsa",
    "logp",
    "flexibility",
)


def _one_hot(values: np.ndarray, categories: Tuple[str, ...]) -> np.ndarray:
    cat_to_i = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(values), len(categories)), dtype=float)
    for r, v in enumerate(values):
        out[r, cat_to_i[str(v)]] = 1.0
    return out


def featurize(df: pd.DataFrame, space: DesignSpace) -> np.ndarray:
    """Featurise designs into numeric vectors.

    Strategy:
    - concatenate descriptors of block A and block B (precomputed)
    - one-hot solvent and catalyst
    - append continuous conditions (T, concentration)
    """
    blocks = space.blocks.df.set_index("block_id")

    A = blocks.loc[df["block_A"].values, list(BLOCK_DESC_COLS)].to_numpy(float)
    B = blocks.loc[df["block_B"].values, list(BLOCK_DESC_COLS)].to_numpy(float)

    solvent_oh = _one_hot(df["solvent"].values, space.solvents)
    catalyst_oh = _one_hot(df["catalyst"].values, space.catalysts)

    cont = df[["temperature_C", "concentration_M"]].to_numpy(float)

    X = np.concatenate([A, B, solvent_oh, catalyst_oh, cont], axis=1)
    return X
