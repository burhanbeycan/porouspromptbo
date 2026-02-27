from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class PromptConfig:
    system_role: str = "You are an expert porous-materials chemist and optimisation scientist."
    max_history: int = 10


def build_llm_prompt(
    history: pd.DataFrame,
    candidate: Dict[str, str | float],
    objective_description: str,
    cfg: PromptConfig = PromptConfig(),
) -> str:
    """Build a prompt for LLM assistance (API-agnostic).

    Intended uses:
    - feasibility screening (solubility, reversibility, catalyst choice)
    - safety notes (acid strength, pressure/temperature)
    - rationalising why a candidate is promising
    """
    recent = history.tail(cfg.max_history).copy()

    lines = []
    lines.append(f"SYSTEM: {cfg.system_role}")
    lines.append("")
    lines.append("We are running an active-learning campaign for porous organic materials.")
    lines.append(f"Objective: {objective_description}")
    lines.append("")
    lines.append("Recent experiments (most recent last):")
    lines.append(recent.to_csv(index=False))
    lines.append("")
    lines.append("Proposed next candidate:")
    for k, v in candidate.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("TASK:")
    lines.append("1) Assess whether the proposed solvent/catalyst/temperature/concentration is reasonable.")
    lines.append("2) Identify potential failure modes (amorphous precipitation, low reversibility).")
    lines.append("3) Suggest one alternative condition set that could improve crystallinity.")
    return "\n".join(lines)
