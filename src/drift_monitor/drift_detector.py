"""
📉 Data Drift Detection
Compares training distribution vs new data using statistical tests.
Uses KS-test for numeric, chi-squared for categorical.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from utils.logger import get_logger

logger = get_logger(__name__)

DRIFT_THRESHOLD = 0.05   # p-value threshold for drift detection


class DriftDetector:
    def __init__(self, reference_df: pd.DataFrame, schema=None):
        self.reference = reference_df.copy()
        self.schema = schema
        self.results: Dict = {}

    def detect(self, new_df: pd.DataFrame) -> Dict:
        """
        Compare new_df against reference_df column by column.
        Returns a dict of {column: drift_result}.
        """
        self.results = {}
        all_cols = [c for c in self.reference.columns if c in new_df.columns]

        for col in all_cols:
            ref_col = self.reference[col].dropna()
            new_col = new_df[col].dropna()
            if len(ref_col) < 5 or len(new_col) < 5:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                self.results[col] = self._ks_test(col, ref_col, new_col)
            else:
                self.results[col] = self._chi2_test(col, ref_col, new_col)

        return self.results

    def _ks_test(self, col: str, ref: pd.Series, new: pd.Series) -> Dict:
        try:
            stat, pval = stats.ks_2samp(ref.values, new.values)
            drifted = pval < DRIFT_THRESHOLD
            return {
                "column": col, "test": "KS", "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4), "drifted": drifted,
                "severity": "HIGH" if pval < 0.01 else "MEDIUM" if pval < 0.05 else "LOW",
                "ref_mean": round(float(ref.mean()), 4),
                "new_mean": round(float(new.mean()), 4),
                "ref_std":  round(float(ref.std()), 4),
                "new_std":  round(float(new.std()), 4),
            }
        except Exception as e:
            return {"column": col, "test": "KS", "error": str(e), "drifted": False}

    def _chi2_test(self, col: str, ref: pd.Series, new: pd.Series) -> Dict:
        try:
            all_cats = set(ref.unique()) | set(new.unique())
            ref_counts = ref.value_counts().reindex(all_cats, fill_value=0)
            new_counts = new.value_counts().reindex(all_cats, fill_value=0)
            # Avoid zero cells
            ref_counts = ref_counts + 1
            new_counts = new_counts + 1
            stat, pval = stats.chisquare(new_counts.values,
                                          f_exp=ref_counts.values / ref_counts.sum() * new_counts.sum())
            drifted = pval < DRIFT_THRESHOLD
            return {
                "column": col, "test": "Chi²", "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4), "drifted": drifted,
                "severity": "HIGH" if pval < 0.01 else "MEDIUM" if pval < 0.05 else "LOW",
                "ref_top": ref.value_counts().index[0] if len(ref) > 0 else "N/A",
                "new_top": new.value_counts().index[0] if len(new) > 0 else "N/A",
            }
        except Exception as e:
            return {"column": col, "test": "Chi²", "error": str(e), "drifted": False}

    def summary(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = []
        for col, r in self.results.items():
            rows.append({
                "Column":    col,
                "Test":      r.get("test", "N/A"),
                "Statistic": r.get("statistic", "N/A"),
                "P-Value":   r.get("p_value", "N/A"),
                "Drifted":   "🔴 YES" if r.get("drifted") else "🟢 NO",
                "Severity":  r.get("severity", "N/A"),
            })
        return pd.DataFrame(rows).sort_values("Drifted", ascending=False)

    def n_drifted(self) -> int:
        return sum(1 for r in self.results.values() if r.get("drifted"))

    def plot_drift_summary(self) -> go.Figure:
        df = self.summary()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Run drift detection first",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        color_map = {"🔴 YES": "#ef4444", "🟢 NO": "#10b981"}
        fig = px.bar(df, x="P-Value", y="Column", color="Drifted",
                     color_discrete_map=color_map, orientation="h",
                     title="Drift Detection Results (p-value per column)",
                     template="plotly_white",
                     labels={"P-Value": "p-value (lower = more drift)"})
        fig.add_vline(x=DRIFT_THRESHOLD, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Threshold ({DRIFT_THRESHOLD})")
        fig.update_layout(height=max(350, 30 * len(df)))
        return fig

    def plot_distribution_comparison(self, col: str, new_df: pd.DataFrame) -> go.Figure:
        if col not in self.reference.columns or col not in new_df.columns:
            return go.Figure()
        fig = go.Figure()
        ref_col = self.reference[col].dropna()
        new_col = new_df[col].dropna()
        if pd.api.types.is_numeric_dtype(ref_col):
            fig.add_trace(go.Histogram(x=ref_col, name="Reference (train)",
                                        marker_color="#3b82f6", opacity=0.6,
                                        histnorm="probability"))
            fig.add_trace(go.Histogram(x=new_col, name="New data",
                                        marker_color="#ef4444", opacity=0.6,
                                        histnorm="probability"))
            fig.update_layout(barmode="overlay")
        else:
            ref_vc = ref_col.value_counts(normalize=True).rename("Reference")
            new_vc = new_col.value_counts(normalize=True).rename("New")
            combined = pd.concat([ref_vc, new_vc], axis=1).fillna(0)
            for col_name, color in [("Reference", "#3b82f6"), ("New", "#ef4444")]:
                if col_name in combined.columns:
                    fig.add_trace(go.Bar(name=col_name, x=combined.index.astype(str),
                                         y=combined[col_name], marker_color=color))
            fig.update_layout(barmode="group")
        result = self.results.get(col, {})
        pval = result.get("p_value", "N/A")
        drifted_txt = "🔴 DRIFT DETECTED" if result.get("drifted") else "🟢 No drift"
        fig.update_layout(
            title=f"Distribution: '{col}' — p={pval} {drifted_txt}",
            template="plotly_white", height=380,
        )
        return fig
