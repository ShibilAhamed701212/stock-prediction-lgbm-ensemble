"""Interactive dashboard:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_predictor.config import DEFAULT_MODELS, Config
from stock_predictor.panel import PanelConfig, load_features, run_panel
from stock_predictor.pipeline import run_training

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 ML Stock Predictor")
st.caption("Educational use only — not financial advice.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["S&P 500 Panel (Kaggle)", "Single Stock"])
    if mode == "Single Stock":
        ticker = st.text_input("Ticker", "AAPL").strip().upper()
        market = st.text_input("Market context ticker", "SPY").strip()
    else:
        st.info("Cross-sectional model on ~1.7M rows of S&P 500 data. Downloads ~95MB on first run.")

    horizon = st.slider("Horizon (trading days)", 1, 20, 5)
    start = st.date_input("History start", pd.Timestamp("2010-01-01"))
    market = st.text_input("Market ticker (blank = none)", "SPY").strip()
    models = st.multiselect("Models", list(DEFAULT_MODELS), default=list(DEFAULT_MODELS))
    splits = st.slider("Walk-forward folds", 3, 12, 8)
    st.subheader("Strategy")
    long_th = st.slider("Long threshold P(up) ≥", 0.50, 0.70, 0.55, 0.01)
    allow_short = st.checkbox("Allow shorting", False)
    short_th = st.slider("Short threshold P(up) ≤", 0.30, 0.50, 0.45, 0.01)
    sizing = st.radio("Sizing", ["binary", "scaled"], horizontal=True)
    cost = st.number_input("Cost (bps per turnover)", 0.0, 50.0, 5.0, 0.5)
    if mode == "Single Stock":
        synthetic = st.checkbox("Use synthetic data (offline)", False)
    else:
        synthetic = False
    run = st.button("🚀 Train & Predict", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def _train_single(cfg_dict: dict, synthetic: bool):
    cfg_dict = dict(cfg_dict, models=tuple(cfg_dict["models"]))
    res = run_training(Config(**cfg_dict), synthetic=synthetic, make_plots=False)
    return {k: res[k] for k in ("metrics", "backtest", "folds", "importance", "bt", "oos", "prediction")}


@st.cache_data(show_spinner=False)
def _train_panel(cfg_dict: dict):
    res = run_panel(PanelConfig(**cfg_dict), make_plots=False)
    return res


if run:
    if mode == "Single Stock":
        if not models:
            st.error("Select at least one model.")
            st.stop()
        cfg = Config(ticker=ticker, start=str(start), market_ticker=market or None, horizon=horizon,
                     models=tuple(models), n_splits=splits, long_threshold=long_th, short_threshold=short_th,
                     allow_short=allow_short, sizing=sizing, cost_bps=cost)
        with st.spinner(f"Training single stock on {ticker} …"):
            try:
                st.session_state["res"] = _train_single(cfg.to_dict(), synthetic)
                st.session_state["mode"] = mode
            except Exception as exc:
                st.error(f"Failed: {exc}")
                st.stop()
    else:
        cfg = PanelConfig(horizon=horizon, cost_bps=cost, n_trials=3)  # keeping trials low for the UI demo
        with st.spinner("Training global panel model on ~1.7M rows. This will take ~2-3 minutes …"):
            try:
                st.session_state["res"] = _train_panel(asdict(cfg))
                st.session_state["mode"] = mode
            except Exception as exc:
                st.error(f"Failed: {exc}")
                st.stop()

res = st.session_state.get("res")
if not res:
    st.info("Choose settings in the sidebar and click **Train & Predict**.")
    st.stop()

if st.session_state.get("mode") == "S&P 500 Panel (Kaggle)":
    m = res["metrics"]["holdout"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Information Coefficient (IC)", f"{m['ic_mean']:.4f}")
    c2.metric("Long/Short CAGR (Holdout)", f"{m['portfolio']['long_short_net']['cagr']*100:.1f}%")
    c3.metric("Top Decile Returns (Holdout)", f"{m['portfolio']['long_only_top_net']['total_return']*100:.1f}%")
    c4.metric("S&P 500 Equal-weight", f"{m['portfolio']['universe_equal_weight']['total_return']*100:.1f}%")

    st.subheader("Model Performance")
    st.write("Cross-sectional model trained on S&P 500 panel. Evaluated on the holdout period.")
    st.dataframe(pd.DataFrame({"Holdout": m}).T)
    st.stop()


p = res["prediction"]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"P(up) next {p['horizon_days']}d", f"{p['prob_up'] * 100:.1f}%")
c2.metric("Signal", p["signal"])
c3.metric("Last close", f"{p['last_close']:.2f}", help=f"as of {p['as_of']}")
c4.metric("OOS AUC", f"{res['metrics']['ensemble']['auc']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Backtest", "Model metrics", "Features", "Predictions"])

with tab1:
    bt = res["bt"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="ML strategy"), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="Buy & hold"), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["drawdown"], name="Drawdown", fill="tozeroy",
                             line=dict(color="crimson")), 2, 1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(height=550, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(res["backtest"]).T.round(3), use_container_width=True)

with tab2:
    st.subheader("Out-of-sample metrics by model")
    st.dataframe(pd.DataFrame(res["metrics"]).T.round(4), use_container_width=True)
    st.subheader("Walk-forward folds")
    st.dataframe(res["folds"].round(4), use_container_width=True)

with tab3:
    imp = res["importance"].head(25)[::-1]
    st.plotly_chart(go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h")).update_layout(height=650),
                    use_container_width=True)

with tab4:
    st.write("Per-model probabilities for the latest bar:")
    st.json(p["per_model"])
    st.write("Out-of-sample predictions (last 250 rows):")
    st.dataframe(res["oos"].tail(250).round(4), use_container_width=True)
