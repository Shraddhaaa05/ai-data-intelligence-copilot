"""
LLM-powered business insight generator — Google Gemini edition.
Uses shared gemini_client with retry + fallback.
Falls back to rule-based insights on quota errors.
"""
import json
from typing import List
import pandas as pd
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


def _call_gemini(prompt: str, max_tokens: int = 600) -> str:
    from utils.gemini_client import _gemini_generate
    return _gemini_generate(prompt, max_tokens=max_tokens)


def generate_business_insights(
    df: pd.DataFrame, schema, shap_importance_df: pd.DataFrame,
    leaderboard: list, target_col: str, problem_type: str, n_insights: int = 8,
) -> List[str]:
    if GOOGLE_API_KEY:
        try:
            return _llm_insights(df, schema, shap_importance_df, leaderboard,
                                 target_col, problem_type, n_insights)
        except Exception as exc:
            logger.warning("LLM insights failed (%s) — using rule-based fallback", exc)
    return _rule_based_insights(df, schema, shap_importance_df, leaderboard,
                                target_col, problem_type)


def generate_dataset_summary(
    df: pd.DataFrame, schema, target_col: str, problem_type: str,
) -> str:
    if GOOGLE_API_KEY:
        try:
            return _llm_summary(df, schema, target_col, problem_type)
        except Exception as exc:
            logger.warning("LLM summary failed (%s) — using rule-based fallback", exc)
    return _rule_based_summary(df, schema, target_col, problem_type)


def _build_context(df, schema, shap_df, leaderboard, target_col, problem_type) -> str:
    best = leaderboard[0] if leaderboard else {}
    metric_k = "roc_auc" if problem_type == "classification" else "r2"
    target_dist = {}
    try:
        vc = df[target_col].value_counts(normalize=True).round(3).to_dict()
        target_dist = {str(k): v for k, v in vc.items()}
    except Exception: pass
    correlations = {}
    try:
        num_df = df[schema.numeric_cols]
        if target_col in num_df.columns:
            corr = num_df.corr()[target_col].drop(target_col, errors="ignore")
            correlations = corr.abs().sort_values(ascending=False).head(5).round(3).to_dict()
    except Exception: pass
    context = {
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "target_column": target_col, "problem_type": problem_type,
        "target_distribution": target_dist,
        "missing_columns": schema.missing_cols[:10],
        "top_features_by_shap": shap_df.head(8).to_dict(orient="records") if not shap_df.empty else [],
        "top_correlations_with_target": correlations,
        "best_model": {"name": best.get("model","N/A"), metric_k: best.get(metric_k,"N/A")},
    }
    return json.dumps(context, indent=2)


def _llm_insights(df, schema, shap_df, leaderboard, target_col, problem_type, n) -> List[str]:
    context = _build_context(df, schema, shap_df, leaderboard, target_col, problem_type)
    prompt = f"""You are a senior data scientist at a top analytics firm.
Generate exactly {n} concise, specific, actionable business insights.
Each insight: single sentence (max 30 words), include numbers where available,
business finding (not technical), avoid generic statements.
Return ONLY a valid JSON array of strings. No preamble, no markdown fences.
Dataset context:\n{context}"""
    raw = _call_gemini(prompt, max_tokens=600)
    raw = raw.replace("```json","").replace("```","").strip()
    insights = json.loads(raw)
    return [str(i) for i in insights[:n]]


def _llm_summary(df, schema, target_col, problem_type) -> str:
    prompt = f"""Write a 3-sentence plain English dataset summary for a business stakeholder.
Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns, target='{target_col}' ({problem_type})
Numeric: {', '.join(schema.numeric_cols[:8])}
Categorical: {', '.join(schema.categorical_cols[:8])}
Missing: {', '.join(schema.missing_cols[:5]) or 'None'}
3 sentences only. Plain text."""
    return _call_gemini(prompt, max_tokens=200)


def _rule_based_insights(df, schema, shap_df, leaderboard, target_col, problem_type) -> List[str]:
    insights = []
    insights.append(f"The dataset contains {df.shape[0]:,} records and {df.shape[1]} features available for analysis.")
    if schema.missing_cols:
        worst = max(schema.missing_cols, key=lambda c: df[c].isna().sum())
        pct = round(df[worst].isna().mean()*100, 1)
        insights.append(f"'{worst}' has the highest missingness at {pct}% — consider imputation or removal.")
    else:
        insights.append("The dataset is complete with no missing values across all columns.")
    if problem_type == "classification":
        try:
            vc = df[target_col].value_counts(normalize=True)*100
            majority_pct = round(vc.iloc[0], 1); minority_pct = round(vc.iloc[-1], 1)
            insights.append(f"Class imbalance detected: '{vc.index[0]}' accounts for {majority_pct}% vs {minority_pct}% for the minority class.")
        except Exception: pass
    if not shap_df.empty:
        top = shap_df.iloc[0]
        insights.append(f"'{top['feature']}' is the strongest predictor (SHAP={top['importance']:.4f}) — focus business attention here.")
        if len(shap_df) > 1:
            second = shap_df.iloc[1]
            insights.append(f"'{second['feature']}' is the second most important feature (SHAP={second['importance']:.4f}).")
    try:
        num_df = df[schema.numeric_cols]
        if target_col in num_df.columns and len(schema.numeric_cols) > 1:
            corr = num_df.corr()[target_col].drop(target_col, errors="ignore").abs()
            top_f = corr.idxmax(); top_v = round(corr.max(), 3)
            insights.append(f"'{top_f}' shows the highest linear correlation with '{target_col}' (r={top_v}).")
    except Exception: pass
    if leaderboard:
        best = leaderboard[0]
        mk = "roc_auc" if problem_type=="classification" else "r2"
        insights.append(f"The {best['model']} model achieved the best {'ROC-AUC' if problem_type=='classification' else 'R²'} = {best.get(mk,'N/A')}.")
    try:
        if schema.numeric_cols:
            col = schema.numeric_cols[0]; s = df[col].dropna()
            cv = round(s.std()/s.mean()*100, 1) if s.mean() != 0 else 0
            insights.append(f"'{col}' has a coefficient of variation of {cv}% (range: {s.min():.2f}–{s.max():.2f}).")
    except Exception: pass
    return insights[:8]


def _rule_based_summary(df, schema, target_col, problem_type) -> str:
    missing_note = f"{len(schema.missing_cols)} column(s) contain missing values." \
                   if schema.missing_cols else "No missing values detected."
    return (f"This dataset contains {df.shape[0]:,} records across {df.shape[1]} columns, "
            f"with '{target_col}' as the target for a {problem_type} task. "
            f"It includes {len(schema.numeric_cols)} numeric and "
            f"{len(schema.categorical_cols)} categorical features. {missing_note}")
