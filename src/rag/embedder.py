"""
Converts dataset facts, profiling stats, SHAP insights, and model
metrics into a list of plain-text chunks ready for vector indexing.
"""
from typing import List

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def build_corpus(
    df: pd.DataFrame,
    schema,
    shap_importance_df: pd.DataFrame,
    leaderboard: list,
    insights: list,
    target_col: str,
    problem_type: str,
) -> List[str]:
    """
    Build a list of text chunks that describe the dataset, model results,
    and AI insights. These chunks are embedded and stored in FAISS.

    Returns a list of strings, each representing one indexable document.
    """
    corpus: List[str] = []

    # ── 1. Dataset overview ───────────────────────────────────────────────────
    corpus.append(
        f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"The target variable is '{target_col}' and the problem type is {problem_type}."
    )

    corpus.append(
        f"Numeric columns: {', '.join(schema.numeric_cols[:20])}."
    )
    corpus.append(
        f"Categorical columns: {', '.join(schema.categorical_cols[:20])}."
    )

    # ── 2. Missing values ─────────────────────────────────────────────────────
    if schema.missing_cols:
        for col in schema.missing_cols:
            pct = round(df[col].isna().mean() * 100, 2)
            corpus.append(
                f"Column '{col}' has {df[col].isna().sum()} missing values ({pct}% of rows)."
            )
    else:
        corpus.append("The dataset has no missing values.")

    # ── 3. Statistical summaries ──────────────────────────────────────────────
    for col in schema.numeric_cols[:15]:
        try:
            s = df[col].dropna()
            corpus.append(
                f"Feature '{col}': mean={s.mean():.3f}, std={s.std():.3f}, "
                f"min={s.min():.3f}, max={s.max():.3f}, "
                f"median={s.median():.3f}."
            )
        except Exception:
            pass

    # ── 4. Target distribution ────────────────────────────────────────────────
    try:
        vc = df[target_col].value_counts(normalize=True) * 100
        for label, pct in vc.items():
            corpus.append(
                f"{pct:.1f}% of records have {target_col} = '{label}'."
            )
    except Exception:
        pass

    # ── 5. Correlations with target ───────────────────────────────────────────
    if problem_type == "regression" and target_col in df.select_dtypes("number").columns:
        try:
            corr = df[schema.numeric_cols].corr()[target_col].drop(target_col).sort_values(
                key=abs, ascending=False
            )
            for feat, val in corr.head(10).items():
                direction = "positively" if val > 0 else "negatively"
                corpus.append(
                    f"'{feat}' is {direction} correlated with '{target_col}' "
                    f"(Pearson r = {val:.3f})."
                )
        except Exception:
            pass

    # ── 6. SHAP feature importance ────────────────────────────────────────────
    if not shap_importance_df.empty:
        corpus.append(
            "The most important features for prediction (by mean SHAP value) are: "
            + ", ".join(
                f"'{r.feature}' ({r.importance:.4f})"
                for _, r in shap_importance_df.head(10).iterrows()
            ) + "."
        )
        for _, row in shap_importance_df.head(5).iterrows():
            corpus.append(
                f"'{row['feature']}' has a mean absolute SHAP value of {row['importance']:.5f}, "
                f"making it one of the strongest drivers of the model's predictions."
            )

    # ── 7. Model performance ──────────────────────────────────────────────────
    if leaderboard:
        best = leaderboard[0]
        metric_str = ", ".join(
            f"{k}={v}" for k, v in best.items()
            if k not in ("model", "estimator", "train_time_s")
        )
        corpus.append(
            f"The best performing model is {best['model']} with {metric_str}."
        )
        for entry in leaderboard:
            m_str = ", ".join(
                f"{k}={v}" for k, v in entry.items()
                if k not in ("model", "estimator", "train_time_s")
            )
            corpus.append(f"Model '{entry['model']}' achieved: {m_str}.")

    # ── 8. AI-generated insights ──────────────────────────────────────────────
    for insight in insights:
        corpus.append(insight)

    logger.info("RAG corpus built — %d text chunks", len(corpus))
    return corpus
