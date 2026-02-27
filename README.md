# PorousPromptBO — Text-informed active learning for porous organic materials

This mini-project demonstrates how to build an **active learning** loop for a porous-material
search space that includes:

- building-block choices (categorical)
- reaction conditions (continuous + categorical)
- multi-objective targets: **yield** + **surface area**, with a **crystallinity/quality** constraint

The emphasis is on *deployable structure* and *clear extension points* for real porous-materials data.

---

## Run the demo (simulated)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
porouspromptbo run-al --rounds 8 --seed 11
```

Outputs are written to `outputs/`.

---

## What is “text-informed” here?

A tiny TF‑IDF retriever loads `data/literature_snippets.jsonl` and can return
condition ranges (e.g., typical temperatures or solvents) to encode feasibility constraints.

This is a stand-in for LLM/RAG workflows you’d run against a real literature corpus.

---

## Data files

- `data/building_blocks.csv` — synthetic building-block descriptors (precomputed)
- `data/literature_snippets.jsonl` — local corpus of hints/constraints
