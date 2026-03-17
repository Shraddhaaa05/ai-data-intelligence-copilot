"""
Unit tests for the AutoML pipeline.
Run with: pytest tests/test_automl.py -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from src.ingestion.schema_detector import detect_schema
from src.automl.preprocessor import preprocess
from src.automl.trainer import train_all
from src.automl.evaluator import evaluate_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_df():
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    return df


@pytest.fixture
def regression_df():
    X, y = make_regression(
        n_samples=300, n_features=8, n_informative=4, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df["target"] = y
    return df


@pytest.fixture
def mixed_df():
    """DataFrame with both numeric and categorical columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age":      np.random.randint(18, 70, n),
        "income":   np.random.normal(50000, 15000, n),
        "tenure":   np.random.randint(0, 72, n),
        "plan":     np.random.choice(["basic", "standard", "premium"], n),
        "region":   np.random.choice(["north", "south", "east", "west"], n),
        "churn":    np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    return df


# ── Schema detection tests ────────────────────────────────────────────────────

class TestSchemaDetector:

    def test_numeric_columns_detected(self, classification_df):
        schema = detect_schema(classification_df)
        assert "target" not in schema.numeric_cols or len(schema.numeric_cols) > 0

    def test_problem_type_classification(self, classification_df):
        schema = detect_schema(classification_df)
        assert schema.problem_type == "classification"

    def test_problem_type_regression(self, regression_df):
        schema = detect_schema(regression_df)
        assert schema.problem_type == "regression"

    def test_mixed_df_categorical_detected(self, mixed_df):
        schema = detect_schema(mixed_df)
        assert "plan" in schema.categorical_cols
        assert "region" in schema.categorical_cols

    def test_suggested_targets_not_empty(self, classification_df):
        schema = detect_schema(classification_df)
        assert len(schema.suggested_targets) > 0

    def test_missing_cols_detected(self):
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        })
        schema = detect_schema(df)
        assert "a" in schema.missing_cols
        assert "b" not in schema.missing_cols


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessor:

    def test_output_shapes_match(self, mixed_df):
        schema = detect_schema(mixed_df)
        schema.problem_type = "classification"
        result = preprocess(mixed_df, "churn", schema)
        assert len(result.X_train) + len(result.X_test) == len(mixed_df)

    def test_no_nans_after_preprocessing(self, mixed_df):
        schema = detect_schema(mixed_df)
        result = preprocess(mixed_df, "churn", schema)
        assert not result.X_train.isna().any().any()
        assert not result.X_test.isna().any().any()

    def test_feature_names_consistent(self, mixed_df):
        schema = detect_schema(mixed_df)
        result = preprocess(mixed_df, "churn", schema)
        assert list(result.X_train.columns) == list(result.X_test.columns)
        assert list(result.X_train.columns) == result.feature_names

    def test_target_encoded_for_classification(self, mixed_df):
        schema = detect_schema(mixed_df)
        schema.problem_type = "classification"
        result = preprocess(mixed_df, "churn", schema)
        assert set(result.y_train.unique()).issubset({0, 1})

    def test_values_scaled_0_to_1(self, mixed_df):
        schema = detect_schema(mixed_df)
        result = preprocess(mixed_df, "churn", schema)
        numeric_cols = [c for c in result.feature_names
                        if c in schema.numeric_cols]
        if numeric_cols:
            for col in numeric_cols:
                assert result.X_train[col].min() >= -0.01
                assert result.X_train[col].max() <= 1.01


# ── Trainer tests ─────────────────────────────────────────────────────────────

class TestTrainer:

    def test_returns_leaderboard(self, classification_df):
        schema = detect_schema(classification_df)
        result = preprocess(classification_df, "target", schema)
        train_result = train_all(
            result.X_train, result.X_test,
            result.y_train, result.y_test,
            "classification",
        )
        assert "leaderboard" in train_result
        assert len(train_result["leaderboard"]) > 0

    def test_best_model_present(self, classification_df):
        schema = detect_schema(classification_df)
        result = preprocess(classification_df, "target", schema)
        train_result = train_all(
            result.X_train, result.X_test,
            result.y_train, result.y_test,
            "classification",
        )
        assert "best_model" in train_result
        assert "estimator" in train_result["best_model"]

    def test_leaderboard_sorted_by_roc_auc(self, classification_df):
        schema = detect_schema(classification_df)
        result = preprocess(classification_df, "target", schema)
        train_result = train_all(
            result.X_train, result.X_test,
            result.y_train, result.y_test,
            "classification",
        )
        lb = train_result["leaderboard"]
        scores = [m["roc_auc"] for m in lb]
        assert scores == sorted(scores, reverse=True)

    def test_regression_metrics_present(self, regression_df):
        schema = detect_schema(regression_df)
        result = preprocess(regression_df, "target", schema)
        train_result = train_all(
            result.X_train, result.X_test,
            result.y_train, result.y_test,
            "regression",
        )
        best = train_result["best_model"]
        assert "r2" in best
        assert "mae" in best


# ── Evaluator tests ───────────────────────────────────────────────────────────

class TestEvaluator:

    def test_classification_metrics_range(self, classification_df):
        from sklearn.ensemble import RandomForestClassifier
        schema = detect_schema(classification_df)
        result = preprocess(classification_df, "target", schema)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(result.X_train, result.y_train)
        metrics = evaluate_model(model, result.X_test, result.y_test, "classification")
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metrics[k] <= 1.0, f"{k} out of range: {metrics[k]}"

    def test_regression_metrics_present(self, regression_df):
        from sklearn.linear_model import LinearRegression
        schema = detect_schema(regression_df)
        result = preprocess(regression_df, "target", schema)
        model = LinearRegression()
        model.fit(result.X_train, result.y_train)
        metrics = evaluate_model(model, result.X_test, result.y_test, "regression")
        for k in ("mae", "rmse", "r2"):
            assert k in metrics
