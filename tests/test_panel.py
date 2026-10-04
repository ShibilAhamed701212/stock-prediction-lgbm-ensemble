"""End-to-end test of the S&P 500 panel workflow on a small synthetic panel (no network)."""
import numpy as np
import pandas as pd
import pytest

from stock_predictor import panel
from stock_predictor.data import synthetic_prices

N_SYMBOLS = 24


def _fake_load_panel(data_dir="data", version=None, min_history=300):
    frames = []
    for i in range(N_SYMBOLS):
        df = synthetic_prices(n=2000, seed=i, start="2013-01-01")
        df["Symbol"] = f"S{i}"
        frames.append(df.reset_index())
    prices = pd.concat(frames).set_index(["Date", "Symbol"]).sort_index().astype("float32")
    companies = pd.DataFrame({
        "Symbol": [f"S{i}" for i in range(N_SYMBOLS)], "Shortname": "Synthetic",
        "Sector": ["Tech", "Energy", "Health"] * (N_SYMBOLS // 3), "Industry": "x", "Marketcap": 1.0,
    }).set_index("Symbol")
    return prices, companies


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("panel")
    (root / "data").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(panel, "load_panel", _fake_load_panel)
    cfg = panel.PanelConfig(n_trials=0, n_jobs=1, data_dir=str(root / "data"),
                            model_dir=str(root / "models"), report_dir=str(root / "reports"))
    res = panel.run_panel(cfg, make_plots=False)
    yield cfg, res
    mp.undo()


def test_run_panel_outputs(trained):
    cfg, res = trained
    hold = res["metrics"]["holdout"]
    assert -1 <= hold["ic_mean"] <= 1
    assert set(hold["portfolio"]) >= {"long_short_net", "long_only_top_net", "universe_equal_weight"}
    assert {"hit_rate_up", "hit_rate_beat_median"} <= set(res["deciles"].columns)
    assert (res["per_year"]["year"].diff().dropna() > 0).all()


def test_predict_panel_ranks_latest_date(trained, monkeypatch):
    cfg, res = trained
    monkeypatch.setattr(panel, "load_panel", _fake_load_panel)
    out = panel.predict_panel(cfg, refresh=False)
    assert len(out) == N_SYMBOLS
    assert out["score"].is_monotonic_decreasing
    assert out["as_of"].nunique() == 1
    # prob_beat_median must come from the beat-the-median rate, not the absolute-up rate
    expected = out["decile"].map(res["deciles"]["hit_rate_beat_median"])
    np.testing.assert_allclose(out["prob_beat_median"], expected)
    np.testing.assert_allclose(out["prob_up"], out["decile"].map(res["deciles"]["hit_rate_up"]))
