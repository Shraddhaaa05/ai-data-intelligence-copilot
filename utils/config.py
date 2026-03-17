import os
from dotenv import load_dotenv

# Always load .env from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Groq (primary — free, no daily limits) ────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Google Gemini (fallback — optional) ──────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# Legacy aliases used by older modules
OPENAI_API_KEY = GROQ_API_KEY or GOOGLE_API_KEY
OPENAI_MODEL   = GROQ_MODEL   or GEMINI_MODEL

# Upload limits
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_BYTES   = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Paths
DATA_DIR   = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample_datasets")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "saved")

# Model training
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# RAG
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_RETRIEVAL = 5

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure directories exist
for d in [UPLOAD_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
