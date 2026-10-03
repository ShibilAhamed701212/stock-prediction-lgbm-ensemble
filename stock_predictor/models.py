"""Model zoo and a soft-voting ensemble classifier."""
from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def make_model(name: str, random_state: int = 42):
    """Create an (unfitted) estimator. Hyper-parameters are deliberately conservative
    (shallow trees, strong regularization) because financial data is very noisy."""
    if name == "logreg":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.05, max_iter=3000))]
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=25, max_features="sqrt",
            n_jobs=-1, random_state=random_state,
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=25, max_features="sqrt",
            n_jobs=-1, random_state=random_state,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=10, reg_lambda=5.0,
            eval_metric="logloss", n_jobs=-1, random_state=random_state, verbosity=0,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300, num_leaves=15, max_depth=4, learning_rate=0.03,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_samples=40,
            reg_lambda=5.0, random_state=random_state, verbose=-1, n_jobs=-1,
        )
    raise ValueError(f"Unknown model '{name}'. Choose from: logreg, rf, et, xgb, lgbm")


def available_models(names: Iterable[str], random_state: int = 42) -> list:
    """Filter out models whose libraries are not installed / cannot load."""
    ok = []
    for n in names:
        try:
            make_model(n, random_state)
            ok.append(n)
        except ValueError:
            raise
        except Exception as exc:  # ImportError, missing libomp, etc.
            log.warning("Skipping model '%s' (%s)", n, exc)
    if not ok:
        raise RuntimeError("No models available.")
    return ok


class EnsembleClassifier:
    """Soft-voting ensemble: averages P(up) across heterogeneous models."""

    def __init__(self, model_names: Iterable[str], random_state: int = 42,
                 weights: Optional[Dict[str, float]] = None):
        self.model_names = available_models(model_names, random_state)
        self.random_state = random_state
        self.weights = weights
        self.models_: Dict[str, object] = {}
        self.feature_names_: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleClassifier":
        self.feature_names_ = list(X.columns)
        self.models_ = {}
        for name in self.model_names:
            model = make_model(name, self.random_state)
            model.fit(X, y)
            self.models_[name] = model
        return self

    def predict_proba_each(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_names_]
        out = {name: m.predict_proba(X)[:, 1] for name, m in self.models_.items()}
        return pd.DataFrame(out, index=X.index)

    def combine(self, each: pd.DataFrame) -> np.ndarray:
        w = np.array([(self.weights or {}).get(n, 1.0) for n in each.columns], dtype=float)
        return each.values @ (w / w.sum())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.combine(self.predict_proba_each(X))

    def feature_importance(self) -> pd.Series:
        """Average of per-model normalized importances (|coef| for linear models)."""
        imps = []
        for m in self.models_.values():
            est = m.steps[-1][1] if isinstance(m, Pipeline) else m
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_, dtype=float)
            elif hasattr(est, "coef_"):
                imp = np.abs(np.ravel(est.coef_))
            else:
                continue
            if imp.sum() > 0:
                imps.append(imp / imp.sum())
        if not imps:
            return pd.Series(dtype=float)
        return pd.Series(np.mean(imps, axis=0), index=self.feature_names_).sort_values(ascending=False)

    # ---- persistence ------------------------------------------------------
    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self, "metadata": metadata or {}}, path)

    @staticmethod
    def load(path: str):
        obj = joblib.load(path)
        return obj["model"], obj.get("metadata", {})
