from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .data import BuildingBlockLibrary
from .design_space import DesignSpace
from .features import featurize
from .models import RFEnsembleSurrogate
from .bo import propose_next
from .simulator import simulate
from .utils import scalar_score
from .llm_retrieval import SnippetCorpus

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run_al(rounds: int = 8, seed: int = 11, n0: int = 16, n_candidates: int = 2500) -> None:
    """Run an active-learning loop on a synthetic porous-material task."""
    lib = BuildingBlockLibrary.load()
    space = DesignSpace(blocks=lib)

    # initial random set
    obs = space.sample(n0, seed=seed)
    obs = simulate(space, obs, seed=seed + 100)

    history = []

    for t in range(rounds):
        X = featurize(obs, space)
        y = scalar_score(obs["yield_pct"].to_numpy(), obs["surface_area_m2_g"].to_numpy(), obs["crystallinity_score"].to_numpy())

        model = RFEnsembleSurrogate(random_state=seed + t).fit(X, y)
        best_y = float(np.max(y))

        cand = space.sample(n_candidates, seed=seed + 1000 + t)
        Xc = featurize(cand, space)
        mu, sigma = model.predict(Xc)
        idx = propose_next(mu, sigma, best_y=best_y, n=1)[0]

        next_design = cand.iloc[[idx]].copy()
        next_obs = simulate(space, next_design, seed=seed + 2000 + t)

        obs = pd.concat([obs, next_obs], ignore_index=True)

        r = next_obs.iloc[0]
        s = float(scalar_score(np.array([r.yield_pct]), np.array([r.surface_area_m2_g]), np.array([r.crystallinity_score]))[0])

        history.append(
            {
                "round": t + 1,
                "block_A": r.block_A,
                "block_B": r.block_B,
                "solvent": r.solvent,
                "catalyst": r.catalyst,
                "temperature_C": float(r.temperature_C),
                "concentration_M": float(r.concentration_M),
                "yield_pct": float(r.yield_pct),
                "surface_area_m2_g": float(r.surface_area_m2_g),
                "crystallinity_score": float(r.crystallinity_score),
                "score": s,
            }
        )

        console.print(f"Round {t+1}: score={s:.3f} | yield={r.yield_pct:.1f}% | SA={r.surface_area_m2_g:.0f} m²/g | cryst={r.crystallinity_score:.2f}")

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(outdir / "al_history.csv", index=False)
    console.print(f"[green]Saved history to {outdir/'al_history.csv'}[/green]")


@app.command()
def retrieve(query: str = "imine cage solvent and temperature", top_k: int = 3) -> None:
    """Search the local snippet corpus (RAG-style stub)."""
    corpus = SnippetCorpus.load()
    hits = corpus.search(query, top_k=top_k)
    table = Table(title="Snippet hits")
    table.add_column("id")
    table.add_column("source")
    table.add_column("ranges")
    table.add_column("text")
    for s in hits:
        ranges = corpus.extract_ranges(s.text)
        table.add_row(s.sid, s.source, str(ranges), s.text[:140] + ("..." if len(s.text) > 140 else ""))
    console.print(table)


if __name__ == "__main__":
    app()
