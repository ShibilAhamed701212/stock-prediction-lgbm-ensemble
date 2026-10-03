"""Central configuration for the stock prediction system."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

DEFAULT_MODELS: Tuple[str, ...] = ("logreg", "rf", "et", "xgb", "lgbm")


@dataclass
class Config:
    # --- Data -------------------------------------------------------------
    ticker: str = "AAPL"
    start: str = "2010-01-01"
    end: Optional[str] = None                 # None = up to today
    market_ticker: Optional[str] = "SPY"      # market context features (None to disable)

    # --- Target -----------------------------------------------------------
    horizon: int = 5                          # predict direction of the next N trading days
    threshold: float = 0.0                    # label = 1 if future return > threshold

    # --- Models -----------------------------------------------------------
    models: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODELS)
    random_state: int = 42

    # --- Walk-forward validation -----------------------------------------
    n_splits: int = 8
    min_train_size: int = 756                 # ~3 years of trading days

    # --- Trading strategy / backtest -------------------------------------
    long_threshold: float = 0.55              # go long when P(up) >= this
    short_threshold: float = 0.45             # go short (if allowed) when P(up) <= this
    allow_short: bool = False
    sizing: str = "binary"                    # "binary" or "scaled"
    cost_bps: float = 5.0                     # cost per unit of turnover, in basis points

    # --- Paths ------------------------------------------------------------
    data_dir: str = "data"
    model_dir: str = "models"
    report_dir: str = "reports"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        return d
