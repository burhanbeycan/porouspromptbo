import pandas as pd

from porouspromptbo.data import BuildingBlockLibrary
from porouspromptbo.design_space import DesignSpace
from porouspromptbo.simulator import simulate
from porouspromptbo.features import featurize
from porouspromptbo.models import RFEnsembleSurrogate


def test_block_library_loads():
    lib = BuildingBlockLibrary.load()
    assert len(lib.df) > 10
    assert "mw" in lib.df.columns


def test_simulation_outputs_columns():
    lib = BuildingBlockLibrary.load()
    space = DesignSpace(blocks=lib)
    designs = space.sample(8, seed=0)
    out = simulate(space, designs, seed=1)
    assert "yield_pct" in out.columns
    assert "surface_area_m2_g" in out.columns
    assert "crystallinity_score" in out.columns


def test_model_fit_predict():
    lib = BuildingBlockLibrary.load()
    space = DesignSpace(blocks=lib)
    designs = space.sample(30, seed=2)
    out = simulate(space, designs, seed=3)

    X = featurize(out, space)
    y = out["yield_pct"].to_numpy()

    m = RFEnsembleSurrogate(random_state=0).fit(X, y)
    mu, sigma = m.predict(X[:5])
    assert mu.shape == (5,)
    assert sigma.shape == (5,)
