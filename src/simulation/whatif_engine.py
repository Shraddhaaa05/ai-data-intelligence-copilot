"""
What-if simulation engine.
Lets users change individual feature values and see how the model
prediction probability changes in real time.
"""
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.config import MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")


def load_model():
    """Load the persisted best model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. Train a model first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def simulate(
    model,
    base_row: pd.DataFrame,
    overrides: Dict[str, float],
    problem_type: str = "classification",
) -> Dict:
    """
    Apply feature overrides to a single-row DataFrame and return
    the original and new predictions.

    Args:
        model:        fitted sklearn-compatible estimator
        base_row:     single-row DataFrame (preprocessed, same columns as training)
        overrides:    dict of {column_name: new_value}
        problem_type: "classification" or "regression"

    Returns:
        {
            "original_value":      float (probability or predicted value)
            "new_value":           float
            "delta":               float
            "direction":           "up" | "down" | "unchanged"
            "recommendation":      str
            "feature_deltas":      dict of feature -> old_value, new_value, change
        }
    """
    modified = base_row.copy()

    feature_deltas = {}
    for col, new_val in overrides.items():
        if col in modified.columns:
            old_val = float(modified[col].values[0])
            modified[col] = new_val
            feature_deltas[col] = {
                "old": round(old_val, 4),
                "new": round(float(new_val), 4),
                "change": round(float(new_val) - old_val, 4),
            }

    if problem_type == "classification" and hasattr(model, "predict_proba"):
        orig_val = float(model.predict_proba(base_row)[0][1])
        new_val = float(model.predict_proba(modified)[0][1])
    else:
        orig_val = float(model.predict(base_row)[0])
        new_val = float(model.predict(modified)[0])

    delta = round(new_val - orig_val, 4)
    direction = "up" if delta > 0.005 else "down" if delta < -0.005 else "unchanged"

    return {
        "original_value":  round(orig_val, 4),
        "new_value":       round(new_val, 4),
        "delta":           delta,
        "direction":       direction,
        "recommendation":  _recommend(new_val, problem_type),
        "feature_deltas":  feature_deltas,
    }


def _recommend(value: float, problem_type: str) -> str:
    """Rule-based action recommendation."""
    if problem_type == "regression":
        return f"Predicted value under the new scenario: {value:.4f}."

    if value >= 0.75:
        return (
            "High risk (≥75%): Immediate intervention recommended — "
            "consider a personalised discount, service upgrade, or loyalty reward."
        )
    elif value >= 0.50:
        return (
            "Medium-high risk (50–75%): Proactive outreach advised — "
            "send a satisfaction survey or personalised retention email."
        )
    elif value >= 0.30:
        return (
            "Medium risk (30–50%): Monitor closely — "
            "enrol in a loyalty programme and track engagement metrics."
        )
    else:
        return "Low risk (<30%): No immediate action required. Continue standard engagement."


def plot_probability_gauge(probability: float, title: str = "Prediction Probability") -> go.Figure:
    """Plotly gauge chart for prediction probability."""
    color = "#ef4444" if probability >= 0.7 else "#f97316" if probability >= 0.4 else "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30],  "color": "#d1fae5"},
                {"range": [30, 50], "color": "#fef3c7"},
                {"range": [50, 75], "color": "#fed7aa"},
                {"range": [75, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.8,
                "value": 70,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def plot_delta_waterfall(feature_deltas: dict, delta: float) -> go.Figure:
    """
    Waterfall chart showing how each changed feature contributed
    to the overall prediction shift.
    """
    if not feature_deltas:
        return go.Figure()

    features = list(feature_deltas.keys())
    changes = [v["change"] for v in feature_deltas.values()]

    measure = ["relative"] * len(features) + ["total"]
    x = features + ["Net change"]
    y = changes + [delta]
    colors = ["#10b981" if c < 0 else "#ef4444" for c in changes] + ["#1a56db"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=x,
        y=y,
        connector={"line": {"color": "#9ca3af"}},
        increasing={"marker": {"color": "#ef4444"}},
        decreasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#1a56db"}},
        text=[f"{v:+.3f}" for v in y],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Change Impact on Prediction",
        template="plotly_white",
        height=360,
        yaxis_title="Value change",
    )
    return fig
