"""
⚙️ Feature Engineering AI
Fixes:
- Uses shared gemini_client (Groq) instead of direct Gemini calls
- Correlation chart handles categorical/non-numeric targets correctly
"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from utils.config import GROQ_API_KEY, GOOGLE_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

HAS_LLM = bool(GROQ_API_KEY or GOOGLE_API_KEY)


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, schema=None, target_col: str = None):
        self.df = df.copy()
        self.schema = schema
        self.target_col = target_col
        self.engineered_df = df.copy()
        self.new_features: List[str] = []

    # ── Suggestions ───────────────────────────────────────────────────────────

    def suggest_features(self) -> Dict:
        num_cols = [c for c in (self.schema.numeric_cols if self.schema
                    else self.df.select_dtypes("number").columns.tolist())
                    if c != self.target_col]
        cat_cols = [c for c in (self.schema.categorical_cols if self.schema
                    else self.df.select_dtypes("object").columns.tolist())
                    if c != self.target_col]
        dt_cols  = self.schema.datetime_cols if self.schema else []

        return {
            "interactions":    self._suggest_interactions(num_cols),
            "polynomial":      self._suggest_polynomial(num_cols),
            "ratio_features":  self._suggest_ratios(num_cols),
            "binning":         self._suggest_binning(num_cols),
            "target_encoding": [
                {"column": c, "method": "target_encoding",
                 "reason": "High-cardinality categorical — encode with target mean."}
                for c in cat_cols if self.df[c].nunique() > 5
            ],
            "datetime": self._suggest_datetime(dt_cols),
        }

    def _target_numeric(self):
        """Return numeric version of target, or None if not possible."""
        if not self.target_col or self.target_col not in self.df.columns:
            return None
        try:
            t = pd.to_numeric(self.df[self.target_col], errors="coerce")
            if t.notna().sum() < 10:
                return None
            return t
        except Exception:
            return None

    def _suggest_interactions(self, num_cols: List[str]) -> List[Dict]:
        results = []
        pairs = list(combinations(num_cols[:8], 2))
        target_num = self._target_numeric()
        if target_num is not None:
            scored = []
            for a, b in pairs:
                try:
                    interaction = self.df[a] * self.df[b]
                    corr = abs(interaction.corr(target_num))
                    base_corr = max(abs(self.df[a].corr(target_num)),
                                    abs(self.df[b].corr(target_num)))
                    if corr > base_corr + 0.02:
                        scored.append((a, b, round(corr, 4), round(base_corr, 4)))
                except Exception:
                    pass
            for a, b, corr, base in sorted(scored, key=lambda x: x[2], reverse=True)[:5]:
                results.append({
                    "feature": f"{a} × {b}", "col_a": a, "col_b": b,
                    "method": "multiply", "corr_with_target": corr,
                    "reason": f"Interaction corr={corr:.3f} > individual max={base:.3f}."
                })
            if results:
                return results
        # No numeric target — suggest top pairs anyway
        for a, b in pairs[:5]:
            results.append({
                "feature": f"{a} × {b}", "col_a": a, "col_b": b,
                "method": "multiply", "corr_with_target": None,
                "reason": "Potential multiplicative interaction."
            })
        return results

    def _suggest_polynomial(self, num_cols: List[str]) -> List[Dict]:
        return [
            {"feature": f"{col}²", "column": col, "degree": 2,
             "reason": f"'{col}' ≥ 0 — square may capture non-linearity."}
            for col in num_cols[:6] if self.df[col].min() >= 0
        ]

    def _suggest_ratios(self, num_cols: List[str]) -> List[Dict]:
        results = []
        for a, b in list(combinations(num_cols[:6], 2))[:5]:
            if self.df[b].replace(0, np.nan).dropna().std() > 0:
                results.append({
                    "feature": f"{a} / {b}", "col_a": a, "col_b": b,
                    "method": "divide",
                    "reason": f"Ratio '{a}/{b}' captures relative magnitude."
                })
        return results

    def _suggest_binning(self, num_cols: List[str]) -> List[Dict]:
        return [
            {"feature": f"{col}_bin", "column": col, "n_bins": 5,
             "reason": f"Binning '{col}' into 5 quantile buckets reduces noise."}
            for col in num_cols[:5] if self.df[col].nunique() > 20
        ]

    def _suggest_datetime(self, dt_cols: List[str]) -> List[Dict]:
        results = []
        for col in dt_cols:
            for comp in ["year", "month", "dayofweek", "quarter"]:
                results.append({"feature": f"{col}_{comp}", "column": col,
                                "component": comp})
        return results

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply_features(self, suggestions: Dict, selected: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        df = self.df.copy()
        log = []
        sel = set(selected)

        for item in suggestions.get("interactions", []):
            if item["feature"] in sel:
                try:
                    df[item["feature"]] = df[item["col_a"]] * df[item["col_b"]]
                    log.append(f"✅ Created '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("polynomial", []):
            if item["feature"] in sel:
                try:
                    df[item["feature"]] = df[item["column"]] ** item["degree"]
                    log.append(f"✅ Created '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("ratio_features", []):
            if item["feature"] in sel:
                try:
                    denom = df[item["col_b"]].replace(0, np.nan)
                    df[item["feature"]] = (df[item["col_a"]] / denom).fillna(0)
                    log.append(f"✅ Created '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("binning", []):
            if item["feature"] in sel:
                try:
                    df[item["feature"]] = pd.qcut(
                        df[item["column"]], q=item["n_bins"],
                        labels=False, duplicates="drop").astype(float)
                    log.append(f"✅ Created '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("target_encoding", []):
            col = item["column"]
            feat_name = f"{col}_encoded"
            if feat_name in sel or col in sel:
                if self.target_col and self.target_col in df.columns:
                    try:
                        t_num = pd.to_numeric(df[self.target_col], errors="coerce")
                        mapping = df.groupby(col).apply(
                            lambda x: pd.to_numeric(x[self.target_col],
                                                     errors="coerce").mean())
                        df[feat_name] = df[col].map(mapping).fillna(t_num.mean())
                        log.append(f"✅ Target-encoded '{col}' → '{feat_name}'")
                        self.new_features.append(feat_name)
                    except Exception as e:
                        log.append(f"⚠️ Target encoding failed for '{col}': {e}")

        self.engineered_df = df
        return df, log

    # ── LLM suggestions ───────────────────────────────────────────────────────

    def llm_suggestions(self, suggestions: Dict) -> str:
        if not HAS_LLM:
            return self._rule_based_suggestions_text(suggestions)
        try:
            from utils.gemini_client import _gemini_generate
            top = {
                "interactions":    [s["feature"] for s in suggestions.get("interactions", [])[:3]],
                "polynomial":      [s["feature"] for s in suggestions.get("polynomial", [])[:3]],
                "ratios":          [s["feature"] for s in suggestions.get("ratio_features", [])[:3]],
                "target_encoding": [s["column"]  for s in suggestions.get("target_encoding", [])[:3]],
            }
            target = self.target_col or "the target"
            prompt = f"""You are a senior ML engineer reviewing feature engineering suggestions.
Target: {target}
Suggestions: {top}

Write 5 concise bullets:
- Which features are most promising and why
- Business intuition for the top 2 interaction features
- Any risks (overfitting, data leakage)
Be specific and actionable. Plain English."""
            return _gemini_generate(prompt, max_tokens=350, temperature=0.2)
        except Exception as e:
            logger.warning("FE LLM failed: %s", e)
            return self._rule_based_suggestions_text(suggestions)

    def _rule_based_suggestions_text(self, suggestions: Dict) -> str:
        lines = []
        for s in suggestions.get("interactions", [])[:3]:
            lines.append(f"• **{s['feature']}**: {s['reason']}")
        for s in suggestions.get("polynomial", [])[:2]:
            lines.append(f"• **{s['feature']}**: {s['reason']}")
        for s in suggestions.get("ratio_features", [])[:2]:
            lines.append(f"• **{s['feature']}**: {s['reason']}")
        return "\n".join(lines) if lines else "No feature engineering suggestions available."

    # ── Correlation chart (FIXED) ─────────────────────────────────────────────

    def plot_correlation_delta(self) -> go.Figure:
        """
        Correlation with target — handles both numeric and categorical targets.
        For categorical targets uses point-biserial or label-encoded correlation.
        """
        if not self.target_col or self.target_col not in self.df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Select a target column first",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False)
            return fig

        try:
            # Convert target to numeric for correlation (works for both num + cat)
            target_s = self.df[self.target_col]
            target_num = pd.to_numeric(target_s, errors="coerce")
            if target_num.isna().mean() > 0.5:
                # Categorical target — label encode it
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_num = pd.Series(
                    le.fit_transform(target_s.astype(str)),
                    index=target_s.index, name=self.target_col
                ).astype(float)

            rows = []

            # Original numeric features
            orig_num = self.df.select_dtypes("number").drop(
                columns=[self.target_col], errors="ignore")
            for col in orig_num.columns:
                try:
                    corr = abs(float(orig_num[col].corr(target_num)))
                    if not np.isnan(corr):
                        rows.append({"feature": col, "corr": corr,
                                     "type": "original"})
                except Exception:
                    pass

            # New engineered features
            if self.new_features and self.engineered_df is not None:
                for col in self.new_features:
                    if col in self.engineered_df.columns:
                        try:
                            col_s = pd.to_numeric(
                                self.engineered_df[col], errors="coerce")
                            corr = abs(float(col_s.corr(target_num)))
                            if not np.isnan(corr):
                                rows.append({"feature": col, "corr": corr,
                                             "type": "engineered"})
                        except Exception:
                            pass

            if not rows:
                fig = go.Figure()
                fig.add_annotation(
                    text="No numeric correlations available for this target.",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig

            plot_df = (pd.DataFrame(rows)
                       .sort_values("corr", ascending=True)
                       .tail(20))

            fig = px.bar(
                plot_df, x="corr", y="feature", color="type",
                color_discrete_map={"original": "#3b82f6", "engineered": "#10b981"},
                orientation="h",
                title=f"Feature Correlation with '{self.target_col}'",
                labels={"corr": "|Correlation|", "feature": "Feature"},
                template="plotly_white",
            )
            fig.update_layout(height=max(350, 28 * len(plot_df)))
            return fig

        except Exception as e:
            logger.warning("Correlation chart error: %s", e)
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart error: {e}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig