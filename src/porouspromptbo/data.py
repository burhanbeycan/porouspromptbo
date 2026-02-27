from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_BLOCKS_PATH = Path(__file__).resolve().parents[2] / "data" / "building_blocks.csv"


@dataclass(frozen=True)
class BuildingBlockLibrary:
    df: pd.DataFrame

    @staticmethod
    def load(path: Optional[Path] = None) -> "BuildingBlockLibrary":
        p = path or DEFAULT_BLOCKS_PATH
        df = pd.read_csv(p)
        return BuildingBlockLibrary(df=df)

    def get(self, block_id: str) -> pd.Series:
        row = self.df[self.df["block_id"] == block_id]
        if len(row) != 1:
            raise KeyError(f"Unknown block_id: {block_id}")
        return row.iloc[0]
