import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import positions_from_proba, run_backtest
from stock_predictor.config import Config
from stock_predictor.data import synthetic_prices
from stock_predictor.features import build_dataset, build_features
from stock_predictor.pipeline import predict_latest, run_training
from stock_predictor.validation import walk_forward_splits


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(n=1500, seed=1)


def test_features_are_causal(prices):
    """Changing future prices must not change past features (no look-ahead)."""
    full = build_features(prices)
    cut = 1000
    truncated = build_features(prices.iloc[:cut])
    a = full.iloc[:cut].dropna()
    b = truncated.loc[a.index]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-7, atol=1e-9)


def test_dataset_shapes(prices):
    ds = build_dataset(prices, None, horizon=5)
    assert len(ds.X) == len(ds.y) == len(ds.future_ret)
    assert not ds.X.isna().any().any()
    assert set(ds.y.unique()) <= {0, 1}
    # last labeled row must be at least `horizon` bars before the end
    assert ds.X.index[-1] <= prices.index[-6]
    assert len(ds.X_live) > len(ds.X)


def test_walk_forward_purge():
    for tr, te in walk_forward_splits(1000, 5, 400, gap=5):
        assert tr.max() + 5 < te.min()
        assert len(np.intersect1d(tr, te)) == 0


def test_backtest_costs_and_positions():
    idx = pd.bdate_range("2020-01-01", periods=4)
    proba = pd.Series([0.6, 0.6, 0.4, 0.6], idx)
    ret = pd.Series([0.01, 0.01, 0.01, 0.01], idx)
    assert list(positions_from_proba(proba.values)) == [1, 1, 0, 1]
    bt, _ = run_backtest(proba, ret, cost_bps=10)
    # day0: enter (cost 10bp), day1: hold, day2: exit (cost), day3: enter (cost)
    expected = [0.01 - 0.001, 0.01, -0.001, 0.01 - 0.001]
    np.testing.assert_allclose(bt["strategy_ret"].values, expected)


def test_end_to_end_synthetic(tmp_path):
    cfg = Config(ticker="TEST", models=("logreg", "rf"), n_splits=3, min_train_size=500,
                 data_dir=str(tmp_path / "d"), model_dir=str(tmp_path / "m"), report_dir=str(tmp_path / "r"))
    res = run_training(cfg, synthetic=True, make_plots=False)
    assert 0.0 <= res["prediction"]["prob_up"] <= 1.0
    assert "ensemble" in res["metrics"]
    pred = predict_latest(cfg, synthetic=True)
    assert abs(pred["prob_up"] - res["prediction"]["prob_up"]) < 1e-9
