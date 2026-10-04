"""Regression tests for bugs fixed during the 2026-10 audit."""
import os
import subprocess
import sys

import pandas as pd
import pytest

from stock_predictor.config import Config
from stock_predictor.pipeline import predict_latest, run_training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _synthetic_last_close(hash_seed: str) -> str:
    code = (
        "from stock_predictor.config import Config;"
        "from stock_predictor.data import load_data;"
        "p, _ = load_data(Config(ticker='DEMO'), synthetic=True);"
        "print(repr(float(p['Close'].iloc[-1])))"
    )
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    return subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True).strip()


def test_synthetic_data_is_stable_across_processes():
    """`train --synthetic` and a later `predict --synthetic` run in different processes
    and must see the same series (hash() is salted per process)."""
    assert _synthetic_last_close("1") == _synthetic_last_close("2")


def test_predict_with_missing_market_features_gives_clear_error(tmp_path):
    cfg = Config(ticker="MKT", models=("logreg",), n_splits=3, min_train_size=500,
                 data_dir=str(tmp_path / "d"), model_dir=str(tmp_path / "m"), report_dir=str(tmp_path / "r"))
    run_training(cfg, synthetic=True, make_plots=False)
    cfg.market_ticker = None  # trained with market features, now predicting without them
    with pytest.raises(ValueError, match="--market"):
        predict_latest(cfg, synthetic=True)


def test_dashboard_panel_mode_runs(monkeypatch):
    """The dashboard's panel mode used `asdict` without importing it (NameError on click)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import stock_predictor.panel as panel

    seen = {}

    def fake_run_panel(cfg, make_plots=True):
        seen["cfg"] = cfg
        port = {"cagr": 0.1, "total_return": 0.2}
        return {"metrics": {"holdout": {"ic_mean": 0.01, "portfolio": {
            "long_short_net": port, "long_only_top_net": port, "universe_equal_weight": port}}}}

    monkeypatch.setattr(panel, "run_panel", fake_run_panel)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert seen["cfg"].n_trials == 3
    assert at.metric[0].value == "0.0100"


def test_panel_decile_table_reports_beat_median_rate():
    from stock_predictor.panel import decile_table

    dates = pd.to_datetime(["2020-01-01"] * 10 + ["2020-01-02"] * 10)
    oos = pd.DataFrame({
        "date": dates,
        "score": list(range(10)) * 2,
        "fwd": [-0.01] * 20,               # every stock fell ...
        "y": [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5,  # ... but the top half beat the median
    })
    t = decile_table(oos)
    assert (t["hit_rate_up"] == 0).all()
    assert t.loc[10, "hit_rate_beat_median"] == 1.0
    assert t.loc[1, "hit_rate_beat_median"] == 0.0
