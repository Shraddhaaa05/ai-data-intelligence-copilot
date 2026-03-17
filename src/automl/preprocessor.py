"""
Preprocessing pipeline: encoding, scaling, imputation, train/test split.
Fix: 'unseen labels' error — LabelEncoder now fits on full dataset before split.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder

from utils.config import RANDOM_STATE, TEST_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: List[str]
    label_encoder: LabelEncoder


def preprocess(df: pd.DataFrame, target: str, schema) -> PreprocessResult:
    """
    Full preprocessing pipeline.
    Key fix: LabelEncoder is fit on the FULL target column before splitting,
    so test set never contains unseen labels.
    """
    feature_cols = schema.feature_cols(target)
    feature_cols = [
        c for c in feature_cols
        if c not in schema.high_cardinality_cols or c in schema.numeric_cols
    ]

    X = df[feature_cols].copy()
    y_raw = df[target].copy()

    # ── Target encoding ───────────────────────────────────────────────────────
    le = LabelEncoder()
    if schema.problem_type == "classification":
        # FIT on ALL data first — prevents unseen labels in test set
        y_str = y_raw.astype(str).fillna("__missing__")
        le.fit(y_str)                           # ← fit on full column
        y = pd.Series(le.transform(y_str), name=target)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        y = y.fillna(y.median())
        le = LabelEncoder()  # empty encoder for regression

    # ── Column type detection ─────────────────────────────────────────────────
    num_cols = [c for c in feature_cols if c in schema.numeric_cols]
    cat_cols = [
        c for c in feature_cols
        if c in schema.categorical_cols and c not in schema.high_cardinality_cols
    ]

    # Also treat engineered numeric columns (not in original schema) as numeric
    extra_num = [
        c for c in feature_cols
        if c not in num_cols and c not in cat_cols
        and pd.api.types.is_numeric_dtype(X[c])
    ]
    num_cols = num_cols + extra_num

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    if not transformers:
        raise ValueError("No valid feature columns found after preprocessing.")

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    # ── Train/test split ──────────────────────────────────────────────────────
    # Use stratify only for binary/low-cardinality classification
    stratify = None
    if schema.problem_type == "classification":
        n_classes = y.nunique()
        min_class  = y.value_counts().min()
        # Stratify only if every class has at least 2 samples
        if n_classes <= 20 and min_class >= 2:
            stratify = y

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # ── Fit & transform ───────────────────────────────────────────────────────
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr  = preprocessor.transform(X_test_raw)

    feature_names = num_cols + cat_cols

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)

    logger.info(
        "Preprocessing done — train: %d, test: %d, features: %d",
        len(X_train), len(X_test), len(feature_names),
    )

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        preprocessor=preprocessor,
        feature_names=feature_names,
        label_encoder=le,
    )