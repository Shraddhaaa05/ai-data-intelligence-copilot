"""
File upload handler.
Supports CSV, Excel (.xlsx / .xls), and JSON.
Performs size validation and basic sanity checks before returning a DataFrame.
"""
import io
from pathlib import Path

import pandas as pd

from utils.config import MAX_UPLOAD_BYTES
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
MIN_ROWS = 10
MIN_COLS = 2


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_file(file_obj) -> None:
    """
    Validate a Streamlit UploadedFile or file-like object.
    Raises UploadValidationError with a human-readable message on failure.
    """
    name = getattr(file_obj, "name", "unknown")
    suffix = Path(name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    size = getattr(file_obj, "size", None)
    if size is None:
        # Fall back: read and check
        data = file_obj.read()
        file_obj.seek(0)
        size = len(data)

    if size > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            f"File size {size / 1024 / 1024:.1f} MB exceeds the "
            f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB limit."
        )


def load_dataset(file_obj) -> pd.DataFrame:
    """
    Parse an uploaded file into a pandas DataFrame.

    Supports:
      - CSV  (.csv)
      - Excel (.xlsx, .xls)
      - JSON  (.json)

    Returns a cleaned DataFrame with whitespace stripped from column names.
    Raises UploadValidationError if the resulting DataFrame is too small.
    """
    validate_file(file_obj)

    name = getattr(file_obj, "name", "upload")
    suffix = Path(name).suffix.lower()

    try:
        if suffix == ".csv":
            # Try multiple common encodings
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding=enc, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UploadValidationError("Could not decode CSV — try saving as UTF-8.")

        elif suffix in (".xlsx", ".xls"):
            file_obj.seek(0)
            df = pd.read_excel(file_obj, engine="openpyxl" if suffix == ".xlsx" else "xlrd")

        elif suffix == ".json":
            file_obj.seek(0)
            df = pd.read_json(file_obj)

        else:
            raise UploadValidationError(f"Unhandled extension: {suffix}")

    except UploadValidationError:
        raise
    except Exception as exc:
        raise UploadValidationError(f"Failed to parse file: {exc}") from exc

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Remove fully-empty rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    if df.shape[0] < MIN_ROWS:
        raise UploadValidationError(
            f"Dataset has only {df.shape[0]} rows — minimum is {MIN_ROWS}."
        )
    if df.shape[1] < MIN_COLS:
        raise UploadValidationError(
            f"Dataset has only {df.shape[1]} columns — minimum is {MIN_COLS}."
        )

    logger.info("Loaded dataset '%s' — %d rows × %d cols", name, df.shape[0], df.shape[1])
    return df
