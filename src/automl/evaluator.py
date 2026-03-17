"""
Model evaluation metrics for classification and regression.
Handles binary AND multiclass classification correctly throughout.
"""
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from utils.logger import get_logger

logger = get_logger(__name__)
TEMPLATE = "plotly_white"


# ── helpers ───────────────────────────────────────────────────────────────────

def _n_classes(y) -> int:
    return len(np.unique(y))

def _is_binary(y) -> bool:
    return _n_classes(y) == 2

def _avg(y) -> str:
    return "binary" if _is_binary(y) else "weighted"

def _roc_auc(model, X_test, y_test) -> float:
    """Safe ROC-AUC for binary and multiclass."""
    if not hasattr(model, "predict_proba"):
        return float("nan")
    proba = model.predict_proba(X_test)
    try:
        if _is_binary(y_test):
            return float(roc_auc_score(y_test, proba[:, 1]))
        else:
            return float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


# ── Core metrics dict ─────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, problem_type: str) -> Dict:
    preds = model.predict(X_test)

    if problem_type == "classification":
        avg = _avg(y_test)
        auc = _roc_auc(model, X_test, y_test)
        result = {
            "accuracy":  round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, average=avg, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, preds, average=avg, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test, preds, average=avg, zero_division=0)), 4),
            "roc_auc":   round(auc, 4) if not np.isnan(auc) else 0.0,
        }
        return result
    else:
        mse = mean_squared_error(y_test, preds)
        return {
            "mae":  round(float(mean_absolute_error(y_test, preds)), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2":   round(float(r2_score(y_test, preds)), 4),
        }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, class_names=None) -> go.Figure:
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    labels = class_names or [str(c) for c in sorted(set(y_test))]

    # Guard: if label count doesn't match cm size, use numeric labels
    if len(labels) != cm.shape[0]:
        labels = [str(i) for i in range(cm.shape[0])]

    # Avoid div-by-zero on empty rows
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm.astype(float) / row_sums * 100

    text = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)"
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=True,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        template=TEMPLATE, height=max(400, 80 * len(labels)),
    )
    return fig


# ── ROC curve — binary + multiclass ──────────────────────────────────────────

def plot_roc_curve(model, X_test, y_test) -> go.Figure:
    """
    Binary: single ROC curve.
    Multiclass: one-vs-rest ROC curve per class (OvR macro).
    Falls back gracefully if predict_proba not available.
    """
    if not hasattr(model, "predict_proba"):
        fig = go.Figure()
        fig.add_annotation(text="Model does not support probability estimates",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    proba = model.predict_proba(X_test)
    classes = model.classes_ if hasattr(model, "classes_") else np.unique(y_test)
    n_cls = len(classes)

    fig = go.Figure()

    if n_cls == 2:
        # ── Binary ────────────────────────────────────────────────────────────
        try:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            auc = roc_auc_score(y_test, proba[:, 1])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#1a56db", width=2),
                name=f"ROC (AUC = {auc:.3f})",
            ))
        except Exception as e:
            fig.add_annotation(text=f"ROC error: {e}",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    else:
        # ── Multiclass OvR ────────────────────────────────────────────────────
        y_bin = label_binarize(y_test, classes=classes)   # (n_samples, n_classes)
        colors = px.colors.qualitative.Set2
        auc_scores = []

        for i, cls in enumerate(classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                auc = roc_auc_score(y_bin[:, i], proba[:, i])
                auc_scores.append(auc)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    line=dict(color=colors[i % len(colors)], width=1.8),
                    name=f"Class {cls} (AUC={auc:.3f})",
                ))
            except Exception:
                continue

        macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0
        fig.update_layout(title=f"ROC Curve — One-vs-Rest (Macro AUC = {macro_auc:.3f})")

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#9ca3af", dash="dash", width=1),
        name="Random", showlegend=True,
    ))

    title = fig.layout.title.text or f"ROC Curve (AUC = {roc_auc_score(y_test, proba[:, 1]):.3f})" \
        if n_cls == 2 else fig.layout.title.text
    fig.update_layout(
        title=title or "ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=TEMPLATE,
        height=420,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )
    return fig


# ── Regression charts ─────────────────────────────────────────────────────────

def plot_predicted_vs_actual(model, X_test, y_test) -> go.Figure:
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    lo = min(float(min(y_test)), float(min(preds)))
    hi = max(float(max(y_test)), float(max(preds)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(y_test), y=list(preds), mode="markers",
        marker=dict(color="#1a56db", opacity=0.6, size=6),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#ef4444", dash="dash"),
        name="Perfect fit",
    ))
    fig.update_layout(
        title=f"Predicted vs Actual (R² = {r2:.3f})",
        xaxis_title="Actual", yaxis_title="Predicted",
        template=TEMPLATE, height=400,
    )
    return fig


def plot_residuals(model, X_test, y_test) -> go.Figure:
    preds = model.predict(X_test)
    residuals = np.array(y_test) - preds
    fig = px.histogram(
        x=residuals, nbins=50,
        title="Residual Distribution",
        labels={"x": "Residual", "y": "Count"},
        template=TEMPLATE,
        color_discrete_sequence=["#1a56db"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.update_layout(height=350)
    return fig


# ── Model leaderboard chart ───────────────────────────────────────────────────

def plot_leaderboard(leaderboard: list, problem_type: str) -> go.Figure:
    metric_cols = (
        ["accuracy", "f1", "roc_auc"] if problem_type == "classification"
        else ["r2", "mae", "rmse"]
    )
    models = [m["model"] for m in leaderboard]
    colors = ["#1a56db", "#10b981", "#f59e0b", "#ef4444"]

    fig = go.Figure()
    for i, metric in enumerate(metric_cols):
        vals = [m.get(metric, 0) for m in leaderboard]
        fig.add_trace(go.Bar(
            name=metric.upper(), x=models, y=vals,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))
    fig.update_layout(
        title="Model Leaderboard — Metric Comparison",
        barmode="group", template=TEMPLATE, height=420,
        yaxis_title="Score", legend_title="Metric",
        yaxis=dict(range=[0, 1.15] if problem_type == "classification" else None),
    )
    return fig
