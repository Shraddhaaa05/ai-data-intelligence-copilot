"""
⚙️ Feature Engineering AI
Suggests and generates polynomial, interaction, time, and encoding features.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


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

        suggestions = {
            "interactions":  self._suggest_interactions(num_cols),
            "polynomial":    self._suggest_polynomial(num_cols),
            "ratio_features": self._suggest_ratios(num_cols),
            "binning":       self._suggest_binning(num_cols),
            "target_encoding": [{"column": c, "method": "target_encoding",
                                 "reason": "High-cardinality categorical — encode with target mean."}
                                 for c in cat_cols if self.df[c].nunique() > 5],
            "datetime":      self._suggest_datetime(dt_cols),
        }
        return suggestions

    def _suggest_interactions(self, num_cols: List[str]) -> List[Dict]:
        results = []
        pairs = list(combinations(num_cols[:8], 2))  # limit for speed
        if self.target_col and self.target_col in self.df.columns:
            try:
                target_num = pd.to_numeric(self.df[self.target_col], errors="coerce")
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
                        "method": "multiply",
                        "corr_with_target": corr, "base_corr": base,
                        "reason": f"Interaction corr={corr:.3f} > individual max={base:.3f}."
                    })
                return results
            except Exception:
                pass
        for a, b in pairs[:5]:
            results.append({"feature": f"{a} × {b}", "col_a": a, "col_b": b,
                            "method": "multiply", "corr_with_target": None,
                            "reason": "Potential multiplicative interaction."})
        return results

    def _suggest_polynomial(self, num_cols: List[str]) -> List[Dict]:
        results = []
        for col in num_cols[:6]:
            s = self.df[col].dropna()
            if s.min() >= 0:
                results.append({"feature": f"{col}²", "column": col, "degree": 2,
                                "reason": f"'{col}' has non-negative values; square may capture non-linearity."})
        return results

    def _suggest_ratios(self, num_cols: List[str]) -> List[Dict]:
        results = []
        for a, b in combinations(num_cols[:6], 2):
            if self.df[b].replace(0, np.nan).dropna().std() > 0:
                results.append({
                    "feature": f"{a} / {b}", "col_a": a, "col_b": b, "method": "divide",
                    "reason": f"Ratio '{a}/{b}' often captures relative magnitude better than raw values."
                })
        return results[:5]

    def _suggest_binning(self, num_cols: List[str]) -> List[Dict]:
        results = []
        for col in num_cols[:5]:
            s = self.df[col].dropna()
            if s.nunique() > 20:
                results.append({"feature": f"{col}_bin", "column": col, "n_bins": 5,
                               "reason": f"Binning '{col}' into 5 quantile buckets reduces noise."})
        return results

    def _suggest_datetime(self, dt_cols: List[str]) -> List[Dict]:
        results = []
        for col in dt_cols:
            results.extend([
                {"feature": f"{col}_year",    "column": col, "component": "year"},
                {"feature": f"{col}_month",   "column": col, "component": "month"},
                {"feature": f"{col}_dayofweek","column": col, "component": "dayofweek"},
                {"feature": f"{col}_quarter", "column": col, "component": "quarter"},
            ])
        return results

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply_features(self, suggestions: Dict, selected: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        df = self.df.copy()
        log = []
        selected_set = set(selected)

        for item in suggestions.get("interactions", []):
            if item["feature"] in selected_set:
                try:
                    df[item["feature"]] = df[item["col_a"]] * df[item["col_b"]]
                    log.append(f"✅ Created interaction feature '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("polynomial", []):
            if item["feature"] in selected_set:
                try:
                    df[item["feature"]] = df[item["column"]] ** item["degree"]
                    log.append(f"✅ Created polynomial feature '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("ratio_features", []):
            if item["feature"] in selected_set:
                try:
                    denom = df[item["col_b"]].replace(0, np.nan)
                    df[item["feature"]] = df[item["col_a"]] / denom
                    df[item["feature"]].fillna(0, inplace=True)
                    log.append(f"✅ Created ratio feature '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("binning", []):
            if item["feature"] in selected_set:
                try:
                    df[item["feature"]] = pd.qcut(
                        df[item["column"]], q=item["n_bins"],
                        labels=False, duplicates="drop")
                    log.append(f"✅ Created binned feature '{item['feature']}'")
                    self.new_features.append(item["feature"])
                except Exception as e:
                    log.append(f"⚠️ Skipped '{item['feature']}': {e}")

        for item in suggestions.get("target_encoding", []):
            if item["feature"] if "feature" in item else f"{item['column']}_encoded" in selected_set:
                col = item["column"]
                feat_name = f"{col}_encoded"
                if self.target_col and self.target_col in df.columns:
                    try:
                        target_mean = pd.to_numeric(df[self.target_col], errors="coerce")
                        mapping = df.groupby(col)[self.target_col].apply(
                            lambda x: pd.to_numeric(x, errors="coerce").mean())
                        df[feat_name] = df[col].map(mapping).fillna(target_mean.mean())
                        log.append(f"✅ Target-encoded '{col}' → '{feat_name}'")
                        self.new_features.append(feat_name)
                    except Exception as e:
                        log.append(f"⚠️ Target encoding failed for '{col}': {e}")

        self.engineered_df = df
        return df, log

    def llm_suggestions(self, suggestions: Dict) -> str:
        """Use Gemini to explain and rank the suggestions."""
        if not GOOGLE_API_KEY:
            return self._rule_based_suggestions_text(suggestions)
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            top = {
                "interactions":  [s["feature"] for s in suggestions.get("interactions", [])[:3]],
                "polynomial":    [s["feature"] for s in suggestions.get("polynomial", [])[:3]],
                "ratios":        [s["feature"] for s in suggestions.get("ratio_features", [])[:3]],
                "target_encoding": [s["column"] for s in suggestions.get("target_encoding", [])[:3]],
            }
            target = self.target_col or "the target"
            prompt = f"""You are a senior ML engineer reviewing feature engineering suggestions.
Target variable: {target}
Suggestions: {top}

Write 5 concise bullets explaining:
- Which features are most promising and why
- Business intuition for the top 2 interaction features
- Any risks (overfitting, data leakage)
Be specific and actionable. Plain English."""
            resp = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(max_output_tokens=350, temperature=0.2))
            return resp.text.strip()
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
        return "\n".join(lines) if lines else "No feature engineering suggestions for this dataset."

    def plot_correlation_delta(self) -> go.Figure:
        """Show correlation with target before and after feature engineering."""
        if not self.target_col or self.target_col not in self.df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Select a target column first",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        try:
            target_num = pd.to_numeric(self.df[self.target_col], errors="coerce")
            orig_num = self.df.select_dtypes("number").drop(
                columns=[self.target_col], errors="ignore")
            new_num = self.engineered_df[self.new_features].select_dtypes("number") \
                if self.new_features else pd.DataFrame()

            rows = []
            for col in orig_num.columns:
                try:
                    corr = abs(orig_num[col].corr(target_num))
                    rows.append({"feature": col, "corr": corr, "type": "original"})
                except Exception:
                    pass
            for col in new_num.columns:
                try:
                    corr = abs(new_num[col].corr(target_num))
                    rows.append({"feature": col, "corr": corr, "type": "engineered"})
                except Exception:
                    pass

            if not rows:
                return go.Figure()
            plot_df = pd.DataFrame(rows).sort_values("corr", ascending=True).tail(20)
            color_map = {"original": "#3b82f6", "engineered": "#10b981"}
            fig = px.bar(plot_df, x="corr", y="feature", color="type",
                         color_discrete_map=color_map, orientation="h",
                         title=f"Feature Correlation with '{self.target_col}'",
                         labels={"corr": "|Correlation|", "feature": "Feature"},
                         template="plotly_white")
            fig.update_layout(height=max(350, 28 * len(plot_df)))
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plot error: {e}", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig
