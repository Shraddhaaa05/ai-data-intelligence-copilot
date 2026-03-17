"""
🧹 Data Cleaning Assistant
Detects issues, suggests fixes, auto-applies them, explains each decision.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    def __init__(self, df: pd.DataFrame, schema=None):
        self.original_df = df.copy()
        self.df = df.copy()
        self.schema = schema

    def detect_all_issues(self) -> Dict:
        return {
            "missing":      self._detect_missing(),
            "outliers":     self._detect_outliers(),
            "duplicates":   self._detect_duplicates(),
            "dtype_issues": self._detect_dtype_issues(),
            "cardinality":  self._detect_high_cardinality(),
            "skewness":     self._detect_skewness(),
        }

    def _detect_missing(self) -> List[Dict]:
        results = []
        for col in self.df.columns:
            n = int(self.df[col].isna().sum())
            if n == 0:
                continue
            pct = round(n / len(self.df) * 100, 2)
            dtype = str(self.df[col].dtype)
            if pct > 50:
                strategy, reason = "drop_column", f"{pct}% missing — too much to impute reliably."
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                skew = abs(self.df[col].dropna().skew()) if len(self.df[col].dropna()) > 3 else 0
                strategy = "median" if skew > 1 else "mean"
                reason = f"Numeric; {'skewed → median' if skew > 1 else 'symmetric → mean'} imputation."
            else:
                strategy, reason = "most_frequent", "Categorical → most-frequent imputation."
            results.append({"column": col, "missing_count": n, "missing_pct": pct,
                            "dtype": dtype, "strategy": strategy, "reason": reason})
        return results

    def _detect_outliers(self) -> List[Dict]:
        results = []
        for col in self.df.select_dtypes(include="number").columns:
            s = self.df[col].dropna()
            if len(s) < 10:
                continue
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = int(((s < lo) | (s > hi)).sum())
            if n_out == 0:
                continue
            pct = round(n_out / len(s) * 100, 2)
            results.append({"column": col, "n_outliers": n_out, "pct": pct,
                            "lower_bound": round(float(lo), 4),
                            "upper_bound": round(float(hi), 4),
                            "strategy": "clip" if pct < 5 else "flag",
                            "reason": f"IQR method — {n_out} values outside bounds."})
        return results

    def _detect_duplicates(self) -> Dict:
        n_dup = int(self.df.duplicated().sum())
        return {"n_duplicates": n_dup, "pct": round(n_dup / len(self.df) * 100, 2),
                "strategy": "drop" if n_dup > 0 else "none"}

    def _detect_dtype_issues(self) -> List[Dict]:
        issues = []
        for col in self.df.select_dtypes("object").columns:
            try:
                pd.to_numeric(self.df[col].dropna().head(100), errors="raise")
                issues.append({"column": col, "issue": "stored_as_string",
                               "suggestion": "Convert to numeric",
                               "reason": f"'{col}' contains numeric values stored as strings."})
            except Exception:
                pass
        return issues

    def _detect_high_cardinality(self) -> List[Dict]:
        results = []
        for col in self.df.select_dtypes("object").columns:
            n_uniq = self.df[col].nunique()
            if n_uniq > 50 and n_uniq > 0.5 * len(self.df):
                results.append({"column": col, "n_unique": n_uniq,
                               "suggestion": "Target encoding or drop",
                               "reason": f"{n_uniq} unique values — likely ID/free-text."})
        return results

    def _detect_skewness(self) -> List[Dict]:
        results = []
        for col in self.df.select_dtypes("number").columns:
            s = self.df[col].dropna()
            if len(s) < 10:
                continue
            skew = float(s.skew())
            if abs(skew) > 2:
                results.append({"column": col, "skewness": round(skew, 3),
                               "suggestion": "log1p" if s.min() >= 0 else "yeo-johnson",
                               "reason": f"Skewness={skew:.2f} ({'right' if skew > 0 else 'left'}-skewed)."})
        return results

    def auto_clean(self, issues: Dict) -> Tuple[pd.DataFrame, List[str]]:
        df = self.original_df.copy()
        log = []

        dup = issues.get("duplicates", {})
        if dup.get("n_duplicates", 0) > 0:
            before = len(df)
            df = df.drop_duplicates()
            log.append(f"✅ Removed {before - len(df)} duplicate rows.")

        for issue in issues.get("dtype_issues", []):
            col = issue["column"]
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                log.append(f"✅ Converted '{col}' to numeric.")

        for m in issues.get("missing", []):
            col = m["column"]
            if col not in df.columns:
                continue
            if m["strategy"] == "drop_column":
                df = df.drop(columns=[col])
                log.append(f"✅ Dropped '{col}' ({m['missing_pct']}% missing).")
            elif m["strategy"] == "median":
                val = df[col].median()
                df[col] = df[col].fillna(val)
                log.append(f"✅ Imputed '{col}' with median={val:.4f}.")
            elif m["strategy"] == "mean":
                val = df[col].mean()
                df[col] = df[col].fillna(val)
                log.append(f"✅ Imputed '{col}' with mean={val:.4f}.")
            elif m["strategy"] == "most_frequent":
                val = df[col].mode()[0]
                df[col] = df[col].fillna(val)
                log.append(f"✅ Imputed '{col}' with mode='{val}'.")

        for o in issues.get("outliers", []):
            col = o["column"]
            if col in df.columns and o["strategy"] == "clip":
                df[col] = df[col].clip(o["lower_bound"], o["upper_bound"])
                log.append(f"✅ Clipped '{col}' outliers to [{o['lower_bound']}, {o['upper_bound']}].")

        self.df = df
        return df, log

    def llm_explain(self, issues: Dict) -> str:
        if not GOOGLE_API_KEY:
            return self._rule_based_explanation(issues)
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            summary = {
                "missing": [f"{m['column']} ({m['missing_pct']}% — {m['strategy']})"
                            for m in issues.get("missing", [])],
                "outliers": [f"{o['column']} ({o['n_outliers']} outliers)"
                             for o in issues.get("outliers", [])],
                "duplicates": issues.get("duplicates", {}).get("n_duplicates", 0),
                "skewed": [s["column"] for s in issues.get("skewness", [])],
            }
            prompt = f"""You are a senior data analyst reviewing data quality.
Automated scan results: {summary}

Write a concise 5-bullet data quality report for a business analyst.
Each bullet: column name + issue + why it matters + recommended fix.
Be specific. Plain English."""
            resp = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(max_output_tokens=400, temperature=0.2))
            return resp.text.strip()
        except Exception as e:
            logger.warning("LLM explain failed: %s", e)
            return self._rule_based_explanation(issues)

    def _rule_based_explanation(self, issues: Dict) -> str:
        lines = []
        for m in issues.get("missing", [])[:5]:
            lines.append(f"• **{m['column']}**: {m['missing_pct']}% missing → {m['reason']}")
        dup = issues.get("duplicates", {})
        if dup.get("n_duplicates", 0) > 0:
            lines.append(f"• **Duplicates**: {dup['n_duplicates']} duplicate rows found → will be dropped.")
        for o in issues.get("outliers", [])[:3]:
            lines.append(f"• **{o['column']}**: {o['n_outliers']} outliers ({o['pct']}%) → {o['reason']}")
        for s in issues.get("skewness", [])[:3]:
            lines.append(f"• **{s['column']}**: skewness={s['skewness']} → suggest {s['suggestion']} transform.")
        return "\n".join(lines) if lines else "✅ No major data quality issues detected."

    def plot_missing_heatmap(self) -> go.Figure:
        miss = (self.original_df.isna().sum() / len(self.original_df) * 100).sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            fig = go.Figure()
            fig.add_annotation(text="✅ No missing values", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=18))
            fig.update_layout(title="Missing Values", template="plotly_white", height=250)
            return fig
        colors = ["#ef4444" if v > 40 else "#f97316" if v > 20 else "#facc15" if v > 5
                  else "#3b82f6" for v in miss.values]
        fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h",
                               marker_color=colors,
                               text=[f"{v:.1f}%" for v in miss.values], textposition="outside"))
        fig.update_layout(title="Missing Values per Column", template="plotly_white",
                          height=max(300, 32 * len(miss)), xaxis_title="Missing %")
        return fig

    def plot_outlier_box(self, col: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Box(y=self.original_df[col].dropna(), name="Original",
                             marker_color="#ef4444", boxpoints="outliers"))
        if col in self.df.columns:
            fig.add_trace(go.Box(y=self.df[col].dropna(), name="After cleaning",
                                 marker_color="#10b981", boxpoints="outliers"))
        fig.update_layout(title=f"Outlier View — {col}", template="plotly_white", height=380)
        return fig
