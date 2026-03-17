"""
Unit tests for the RAG corpus builder and schema components.
Tests that don't require an OpenAI API key are marked as unit tests.
Tests that require the API are marked and skipped without a key.
Run with: pytest tests/test_rag.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.ingestion.schema_detector import detect_schema
from src.rag.embedder import build_corpus


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "tenure":       np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "contract":     np.random.choice(["month-to-month", "one-year", "two-year"], n),
        "churn":        np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def schema_and_shap(sample_df):
    schema = detect_schema(sample_df)
    schema.problem_type = "classification"
    shap_df = pd.DataFrame({
        "feature": ["tenure", "monthly_charges", "contract"],
        "importance": [0.45, 0.35, 0.20],
    })
    return schema, shap_df


@pytest.fixture
def mock_leaderboard():
    return [
        {"model": "Random Forest", "accuracy": 0.82, "f1": 0.79, "roc_auc": 0.88},
        {"model": "XGBoost",       "accuracy": 0.80, "f1": 0.77, "roc_auc": 0.86},
    ]


# ── Corpus builder tests ──────────────────────────────────────────────────────

class TestCorpusBuilder:

    def test_corpus_not_empty(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=["Customers with high charges churn more."],
            target_col="churn",
            problem_type="classification",
        )
        assert isinstance(corpus, list)
        assert len(corpus) > 0

    def test_corpus_contains_strings(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        assert all(isinstance(c, str) for c in corpus)

    def test_corpus_mentions_target(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        full_text = " ".join(corpus)
        assert "churn" in full_text.lower()

    def test_corpus_mentions_top_shap_feature(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        full_text = " ".join(corpus)
        assert "tenure" in full_text

    def test_corpus_includes_provided_insights(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        custom_insight = "Customers with month-to-month contracts churn 3x more."
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[custom_insight], target_col="churn", problem_type="classification",
        )
        assert custom_insight in corpus

    def test_corpus_includes_model_performance(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        full_text = " ".join(corpus)
        assert "random forest" in full_text.lower()

    def test_corpus_handles_no_missing_cols(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        schema.missing_cols = []
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        full_text = " ".join(corpus)
        assert "no missing values" in full_text.lower()

    def test_corpus_handles_empty_shap(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, _ = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, pd.DataFrame(), mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        assert len(corpus) > 0  # Should still work without SHAP

    def test_corpus_regression_problem(self, sample_df, schema_and_shap, mock_leaderboard):
        schema, shap_df = schema_and_shap
        reg_leaderboard = [
            {"model": "Random Forest", "mae": 12.3, "rmse": 18.5, "r2": 0.78},
        ]
        corpus = build_corpus(
            sample_df, schema, shap_df, reg_leaderboard,
            insights=[], target_col="monthly_charges", problem_type="regression",
        )
        assert len(corpus) > 0


# ── Integration test (requires OPENAI_API_KEY) ────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable",
)
class TestRAGIntegration:

    def test_vector_store_builds(self, sample_df, schema_and_shap, mock_leaderboard):
        from src.rag.vector_store import build_vector_store
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=[], target_col="churn", problem_type="classification",
        )
        vectorstore = build_vector_store(corpus[:5])  # small subset for speed
        assert vectorstore is not None

    def test_rag_chain_answers_question(self, sample_df, schema_and_shap, mock_leaderboard):
        from src.rag.vector_store import build_vector_store
        from src.rag.qa_chain import build_rag_chain, ask
        schema, shap_df = schema_and_shap
        corpus = build_corpus(
            sample_df, schema, shap_df, mock_leaderboard,
            insights=["Customers with high charges churn more."],
            target_col="churn", problem_type="classification",
        )
        vectorstore = build_vector_store(corpus[:5])
        chain = build_rag_chain(vectorstore)
        result = ask(chain, "What is the churn rate in this dataset?")
        assert "answer" in result
        assert len(result["answer"]) > 10
