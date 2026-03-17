"""
Dataset profiling module.
Produces Plotly charts for distributions, correlations, and missing values.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.logger import get_logger

logger = get_logger(__name__)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Set2
PRIMARY_COLOR = "#1a56db"


def profile_dataset(df: pd.DataFrame, schema) -> Dict[str, Any]:
    """
    Generate a full suite of profiling charts.

    Returns a dict with keys:
      - 'correlation'      : Plotly heatmap figure
      - 'missing'          : Plotly bar chart figure
      - 'distributions'    : dict of {col_name: Plotly figure}
      - 'target_breakdown' : dict of {target_col: Plotly figure}
      - 'summary_stats'    : pd.DataFrame of descriptive stats
    """
    results: Dict[str, Any] = {}

    results["summary_stats"] = _summary_stats(df, schema)
    results["correlation"] = _correlation_heatmap(df, schema)
    results["missing"] = _missing_value_chart(df)
    results["distributions"] = _distribution_charts(df, schema)
    results["target_breakdown"] = _target_breakdown(df, schema)

    logger.info("Profiling complete for dataset with %d rows × %d cols", *df.shape)
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _summary_stats(df: pd.DataFrame, schema) -> pd.DataFrame:
    """Extended descriptive statistics for numeric columns."""
    if not schema.numeric_cols:
        return pd.DataFrame()
    desc = df[schema.numeric_cols].describe().T
    desc["skewness"] = df[schema.numeric_cols].skew()
    desc["kurtosis"] = df[schema.numeric_cols].kurtosis()
    desc["missing_pct"] = (df[schema.numeric_cols].isna().sum() / len(df) * 100).round(2)
    return desc.round(4)


def _correlation_heatmap(df: pd.DataFrame, schema) -> go.Figure:
    """Pearson correlation heatmap for numeric columns."""
    num_df = df[schema.numeric_cols] if schema.numeric_cols else df.select_dtypes("number")

    if num_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation analysis",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Limit to 20 columns for readability
    num_df = num_df.iloc[:, :20]
    corr = num_df.corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Feature Correlation Heatmap",
        template=PLOTLY_TEMPLATE,
        text_auto=".2f",
    )
    fig.update_layout(
        title_font_size=16,
        coloraxis_colorbar_title="Pearson r",
        height=max(400, 30 * len(corr.columns)),
    )
    return fig


def _missing_value_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing missing-value percentage per column."""
    missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=True)
    missing = missing[missing > 0]

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values found",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16))
        fig.update_layout(title="Missing Value Analysis", template=PLOTLY_TEMPLATE)
        return fig

    colors_bar = [
        "#ef4444" if v > 40 else "#f97316" if v > 20 else "#facc15" if v > 5 else PRIMARY_COLOR
        for v in missing.values
    ]

    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in missing.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Missing Value Analysis (% per column)",
        xaxis_title="Missing %",
        yaxis_title="Column",
        template=PLOTLY_TEMPLATE,
        height=max(300, 30 * len(missing)),
        xaxis=dict(range=[0, min(missing.max() * 1.25, 100)]),
    )
    return fig


def _distribution_charts(df: pd.DataFrame, schema) -> Dict[str, go.Figure]:
    """Histogram + box plot for each numeric column; bar chart for categorical."""
    figs = {}

    for col in schema.numeric_cols[:15]:  # limit to 15 for performance
        try:
            clean = df[col].dropna()
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])

            fig.add_trace(
                go.Histogram(x=clean, nbinsx=40, marker_color=PRIMARY_COLOR,
                             name="Distribution", showlegend=False),
                row=1, col=1,
            )
            fig.add_trace(
                go.Box(y=clean, marker_color=PRIMARY_COLOR,
                       name="Box Plot", showlegend=False),
                row=1, col=2,
            )
            fig.update_layout(
                title=f"{col} — Distribution",
                template=PLOTLY_TEMPLATE,
                height=350,
            )
            figs[col] = fig
        except Exception as exc:
            logger.warning("Could not plot distribution for '%s': %s", col, exc)

    for col in schema.categorical_cols[:10]:
        if col in schema.high_cardinality_cols:
            continue
        try:
            vc = df[col].value_counts().head(20)
            fig = px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": col, "y": "Count"},
                title=f"{col} — Value Counts",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig.update_layout(height=350)
            figs[col] = fig
        except Exception as exc:
            logger.warning("Could not plot counts for '%s': %s", col, exc)

    return figs


def _target_breakdown(df: pd.DataFrame, schema) -> Dict[str, go.Figure]:
    """Value-count pie/bar charts for each suggested target column."""
    figs = {}
    for target in schema.suggested_targets[:3]:
        try:
            vc = df[target].value_counts()
            fig = px.pie(
                names=vc.index.astype(str),
                values=vc.values,
                title=f"Target: {target} — Class Distribution",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            fig.update_traces(textinfo="percent+label")
            figs[target] = fig
        except Exception as exc:
            logger.warning("Could not plot target breakdown for '%s': %s", target, exc)
    return figs
