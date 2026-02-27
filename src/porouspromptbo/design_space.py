from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import BuildingBlockLibrary


@dataclass(frozen=True)
class DesignSpace:
    blocks: BuildingBlockLibrary
    solvents: Tuple[str, ...] = ("dioxane", "mesitylene", "dmf", "ethanol", "thf")
    catalysts: Tuple[str, ...] = ("acetic_acid", "tfa", "none")

    # continuous ranges
    temperature_C: Tuple[float, float] = (25.0, 120.0)
    concentration_M: Tuple[float, float] = (0.05, 0.5)

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        A = rng.choice(self.blocks.df["block_id"].to_numpy(), size=n, replace=True)
        B = rng.choice(self.blocks.df["block_id"].to_numpy(), size=n, replace=True)

        solvent = rng.choice(self.solvents, size=n, replace=True)
        catalyst = rng.choice(self.catalysts, size=n, replace=True)
        temp = rng.uniform(self.temperature_C[0], self.temperature_C[1], size=n)
        conc = rng.uniform(self.concentration_M[0], self.concentration_M[1], size=n)

        return pd.DataFrame(
            {
                "block_A": A,
                "block_B": B,
                "solvent": solvent,
                "catalyst": catalyst,
                "temperature_C": temp,
                "concentration_M": conc,
            }
        )
