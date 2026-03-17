"""
FIXED Preprocessing pipeline: encoding, scaling, imputation, train/test split.
Returns consistent DataFrames for training and evaluation.

✅ KEY FIXES:
1. Train/test split BEFORE fitting transformer (prevents leakage)
2. Added stratified split for classification (preserves class balance)
3. Proper imputation strategy selection
4. Better feature name handling
"""
from dataclasses import dataclass
from typing import List, Tuple

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
    label_encoder: LabelEncoder  # only used for classification targets


def preprocess(df: pd.DataFrame, target: str, schema) -> PreprocessResult:
    """
    ✅ FIXED: Full preprocessing pipeline with NO data leakage.

    Pipeline:
      1. Separate features and target
      2. ✅ Train/test split FIRST (prevents leakage)
      3. Encode the target (LabelEncoder for classification)
      4. Build a ColumnTransformer:
         - Numeric: median imputation → MinMaxScaler
         - Categorical (low-card): most_frequent imputation → OrdinalEncoder
      5. Fit transformer on TRAINING data only
      6. Transform both TRAIN and TEST using training statistics

    KEY FIX: Step 2 happens BEFORE fitting any transformers
    """
    feature_cols = schema.feature_cols(target)

    # Drop high-cardinality categorical columns that aren't useful
    feature_cols = [
        c for c in feature_cols
        if c not in schema.high_cardinality_cols or c in schema.numeric_cols
    ]

    X = df[feature_cols].copy()
    y = df[target].copy()

    logger.info(f"Starting preprocessing: target={target}, features={len(feature_cols)}")

    # ✅ FIX #1: Train/test split FIRST (before any transformations!)
    # This prevents data leakage - test set must be completely unseen
    # Safe stratification check
    stratify_col = None
    if schema.problem_type == "classification":
        class_counts = y.value_counts()
    if class_counts.min() >= 2:
        stratify_col = y

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_col,
)
    
    logger.info(f"Split: train={len(X_train_raw)}, test={len(X_test_raw)}")

    # ── Target encoding (on training data) ─────────────────────────────────────
    le = LabelEncoder()
    if schema.problem_type == "classification":
        y_train_encoded = pd.Series(
            le.fit_transform(y_train.astype(str)), 
            name=target,
            index=y_train.index
        )
        y_test_encoded = pd.Series(
            le.transform(y_test.astype(str)),
            name=target,
            index=y_test.index
        )
    else:
        y_train_encoded = pd.to_numeric(y_train, errors="coerce")
        y_train_encoded.fillna(y_train_encoded.median(), inplace=True)
        
        y_test_encoded = pd.to_numeric(y_test, errors="coerce")
        y_test_encoded.fillna(y_train_encoded.median(), inplace=True)  # Use train median for test
    
    logger.info(f"Target encoded: {len(le.classes_)} classes" if schema.problem_type == "classification" else "Regression target normalized")

    # ── Identify column types in the feature set ──────────────────────────────
    num_cols = [c for c in feature_cols if c in schema.numeric_cols]
    cat_cols = [
        c for c in feature_cols
        if c in schema.categorical_cols and c not in schema.high_cardinality_cols
    ]

    logger.info(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # ── Build transformer pipelines ───────────────────────────────────────────
    # Numeric pipeline: impute median → scale 0-1
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])

    # Categorical pipeline: impute mode → encode ordinal
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    if not transformers:
        raise ValueError("No valid feature columns found after preprocessing.")

    # ✅ FIX #2: Build transformer (don't fit yet)
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    # ✅ FIX #3: Fit ONLY on training data
    logger.info("Fitting preprocessor on training data only...")
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    
    # ✅ FIX #4: Transform test data using training statistics
    logger.info("Transforming test data using training statistics...")
    X_test_arr = preprocessor.transform(X_test_raw)

    # ── Reconstruct DataFrames with proper column names ───────────────────────
    feature_names = num_cols + cat_cols

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names)
    
    # Reset indices for alignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train_reset = y_train_encoded.reset_index(drop=True)
    y_test_reset = y_test_encoded.reset_index(drop=True)

    logger.info(
        f"✅ Preprocessing complete — train: {len(X_train)}, test: {len(X_test)}, "
        f"features: {len(feature_names)}, no data leakage"
    )

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_reset,
        y_test=y_test_reset,
        preprocessor=preprocessor,
        feature_names=feature_names,
        label_encoder=le,
    )