"""
SHAP explainability — robust version.
Handles sklearn pipeline models, feature_names_in_ conflicts, and multi-class.
"""
import io
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

from utils.logger import get_logger

logger = get_logger(__name__)
PLOTLY_TEMPLATE = "plotly_white"
MAX_BG = 150


def _strip_feature_names(model):
    """
    Remove feature_names_in_ from a fitted sklearn estimator (and sub-estimators)
    so SHAP's TreeExplainer doesn't trip over DataFrame column names vs array names.
    """
    for attr in ("feature_names_in_",):
        try:
            object.__delattr__(model, attr)
        except AttributeError:
            pass
    # Recurse into common ensemble attributes
    for sub_attr in ("estimators_", "base_estimator_", "estimator_"):
        subs = getattr(model, sub_attr, None)
        if subs is None:
            continue
        if hasattr(subs, "__iter__"):
            for s in subs:
                _strip_feature_names(s)
        else:
            _strip_feature_names(subs)
    return model


class SHAPExplainer:
    def __init__(self, model, X_train: pd.DataFrame, problem_type: str):
        self.problem_type = problem_type
        self.feature_names = list(X_train.columns)
        X_arr = X_train.values.astype(float)  # always pass numpy to SHAP

        logger.info("Building SHAP explainer for %s", type(model).__name__)

        # Attempt 1 — TreeExplainer on a clean copy
        try:
            import copy
            m_copy = copy.deepcopy(model)
            _strip_feature_names(m_copy)
            self.explainer = shap.TreeExplainer(m_copy)
            sv = self.explainer.shap_values(X_arr)
            self.shap_values = self._wrap(sv, X_arr)
            logger.info("TreeExplainer succeeded")
            return
        except Exception as e1:
            logger.warning("TreeExplainer failed (%s) — trying LinearExplainer", e1)

        # Attempt 2 — LinearExplainer (Logistic / Linear Regression)
        try:
            self.explainer = shap.LinearExplainer(model, X_arr)
            sv = self.explainer.shap_values(X_arr)
            self.shap_values = self._wrap(sv, X_arr)
            logger.info("LinearExplainer succeeded")
            return
        except Exception as e2:
            logger.warning("LinearExplainer failed (%s) — using KernelExplainer", e2)

        # Attempt 3 — KernelExplainer (model-agnostic, slower)
        n = min(MAX_BG, len(X_arr))
        bg = shap.sample(pd.DataFrame(X_arr), n)
        self.explainer = shap.KernelExplainer(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            bg.values,
        )
        sv = self.explainer.shap_values(X_arr[:n])
        self.shap_values = self._wrap(sv, X_arr[:n])
        logger.info("KernelExplainer succeeded")

    def _wrap(self, sv, X_arr: np.ndarray) -> shap.Explanation:
        """Normalise raw shap_values (list or array) into an Explanation object."""
        if isinstance(sv, list):
            # multi-class list → use positive class (index 1) for binary
            vals = sv[1] if len(sv) == 2 else sv[0]
        else:
            vals = sv

        if vals.ndim == 3:        # (samples, features, classes)
            vals = vals[:, :, 1]

        # base_values
        bv = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(bv, (list, np.ndarray)):
            bv = bv[1] if len(bv) == 2 else bv[0]

        return shap.Explanation(
            values=vals,
            base_values=np.full(len(vals), float(bv)),
            data=X_arr[:len(vals)],
            feature_names=self.feature_names,
        )

    def _vals_2d(self) -> np.ndarray:
        v = self.shap_values.values
        if v.ndim == 3:
            v = v[:, :, 1]
        return v

    def feature_importance_df(self) -> pd.DataFrame:
        v = self._vals_2d()
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": np.abs(v).mean(axis=0)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def summary_plot_bytes(self, max_display: int = 20) -> bytes:
        v = self._vals_2d()
        plt.figure(figsize=(10, max(5, max_display * 0.4)))
        shap.summary_plot(
            v, features=self.shap_values.data,
            feature_names=self.feature_names,
            max_display=max_display, show=False, plot_size=None,
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def waterfall_plot_bytes(self, instance_index: int = 0) -> bytes:
        sv = self.shap_values[instance_index]
        if sv.values.ndim == 2:
            bv = sv.base_values[1] if hasattr(sv.base_values, "__len__") else sv.base_values
            sv = shap.Explanation(
                values=sv.values[:, 1], base_values=float(bv),
                data=sv.data, feature_names=self.feature_names,
            )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(sv, max_display=15, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def feature_importance_plotly(self, top_n: int = 20) -> go.Figure:
        df = self.feature_importance_df().head(top_n).sort_values("importance")
        fig = px.bar(
            df, x="importance", y="feature", orientation="h",
            title=f"Top {top_n} Features — Mean |SHAP Value|",
            labels={"importance": "Mean |SHAP|", "feature": "Feature"},
            color="importance", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=max(350, top_n * 25), coloraxis_showscale=False)
        return fig

    def dependence_plot_plotly(self, feature: str) -> go.Figure:
        v = self._vals_2d()
        try:
            idx = self.feature_names.index(feature)
        except ValueError:
            return go.Figure()
        fig = px.scatter(
            x=self.shap_values.data[:, idx], y=v[:, idx],
            labels={"x": feature, "y": f"SHAP({feature})"},
            title=f"SHAP Dependence — {feature}",
            template=PLOTLY_TEMPLATE, opacity=0.6,
            color=v[:, idx], color_continuous_scale="RdBu_r",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig.update_layout(height=400, coloraxis_showscale=False)
        return fig
