"""
Automatic schema detection for any uploaded dataset.
Detects column types, missing values, and suggests target columns.
"""
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSchema:
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    boolean_cols: List[str] = field(default_factory=list)
    high_cardinality_cols: List[str] = field(default_factory=list)  # cat with >50 unique
    missing_cols: List[str] = field(default_factory=list)
    suggested_targets: List[str] = field(default_factory=list)
    problem_type: str = "classification"  # "classification" | "regression"
    id_cols: List[str] = field(default_factory=list)  # likely ID columns to exclude

    def feature_cols(self, target: str) -> List[str]:
        """All usable feature columns excluding the target and ID columns."""
        exclude = set(self.id_cols + [target])
        return [
            c
            for c in self.numeric_cols + self.categorical_cols
            if c not in exclude
        ]


def detect_schema(df: pd.DataFrame) -> DatasetSchema:
    """
    Analyse a DataFrame and return a DatasetSchema with inferred types,
    missing columns, and target suggestions.
    """
    schema = DatasetSchema()

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique()
        n_rows = len(series)

        # Missing values
        if series.isna().sum() > 0:
            schema.missing_cols.append(col)

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            schema.datetime_cols.append(col)
            continue

        # Try parsing as datetime if object dtype
        if series.dtype == object:
            try:
                parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() > 0.8 * n_rows:
                    schema.datetime_cols.append(col)
                    continue
            except Exception:
                pass

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            # Likely an ID column: unique for almost every row AND integer-like
            if (n_unique > 0.95 * n_rows and n_unique > 100
                    and pd.api.types.is_integer_dtype(series)):
                schema.id_cols.append(col)
            else:
                schema.numeric_cols.append(col)
            continue

        # Boolean
        if series.dtype == bool or set(series.dropna().unique()).issubset({0, 1, True, False}):
            schema.boolean_cols.append(col)
            schema.categorical_cols.append(col)
            continue

        # Categorical
        schema.categorical_cols.append(col)
        if n_unique > 50:
            schema.high_cardinality_cols.append(col)

    # ── Target suggestions ────────────────────────────────────────────────────
    # Priority: binary > low-cardinality categorical > numeric (regression)
    binary_candidates = []
    low_card_candidates = []
    numeric_candidates = []

    for col in df.columns:
        if col in schema.id_cols or col in schema.datetime_cols:
            continue

        n_unique = df[col].nunique()

        if n_unique == 2:
            binary_candidates.append(col)
        elif n_unique <= 10 and col in schema.categorical_cols:
            low_card_candidates.append(col)
        elif col in schema.numeric_cols:
            numeric_candidates.append(col)

    schema.suggested_targets = binary_candidates + low_card_candidates + numeric_candidates

    # ── Problem type from top suggestion ──────────────────────────────────────
    if schema.suggested_targets:
        top = schema.suggested_targets[0]
        n_unique_top = df[top].nunique()
        if top in schema.numeric_cols and n_unique_top > 20:
            schema.problem_type = "regression"
        else:
            schema.problem_type = "classification"
    else:
        # Default: if most columns are numeric → regression
        if len(schema.numeric_cols) > len(schema.categorical_cols):
            schema.problem_type = "regression"

    logger.info(
        "Schema detected — numeric: %d, categorical: %d, datetime: %d, "
        "missing cols: %d, suggested targets: %s, problem type: %s",
        len(schema.numeric_cols),
        len(schema.categorical_cols),
        len(schema.datetime_cols),
        len(schema.missing_cols),
        schema.suggested_targets[:3],
        schema.problem_type,
    )
    return schema
