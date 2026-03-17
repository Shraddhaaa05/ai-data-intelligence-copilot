"""
Vector store — tries FAISS + embeddings, falls back to simple keyword store.
Works with Groq (no embedding API needed for fallback).
"""
import os
import pickle
from typing import List
from utils.config import GROQ_API_KEY, GOOGLE_API_KEY, EMBEDDING_MODEL, MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.pkl")


def build_vector_store(corpus: List[str]):
    """
    Build vector store. Tries in order:
    1. FAISS + Google embeddings (if GOOGLE_API_KEY + packages available)
    2. FAISS + sentence-transformers (local, no API needed)
    3. Simple in-memory keyword store (always works)
    """
    # Try FAISS + Google embeddings
    if GOOGLE_API_KEY:
        try:
            return _build_faiss_google(corpus)
        except Exception as e:
            logger.warning("FAISS+Google failed (%s) — trying local embeddings", e)

    # Try FAISS + local sentence-transformers (no API key needed)
    try:
        return _build_faiss_local(corpus)
    except Exception as e:
        logger.warning("FAISS+local failed (%s) — using simple keyword store", e)

    # Always-works fallback
    return _build_simple_store(corpus)


def _build_faiss_google(corpus: List[str]):
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = [Document(page_content=c) for c in corpus]
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vs = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vs, f)
    logger.info("FAISS+Google store built — %d chunks", len(split_docs))
    return vs


def _build_faiss_local(corpus: List[str]):
    """FAISS with free local sentence-transformers embeddings — no API key needed."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = [Document(page_content=c) for c in corpus]
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vs, f)
    logger.info("FAISS+local embeddings store built — %d chunks", len(split_docs))
    return vs


def _build_simple_store(corpus: List[str]) -> dict:
    """No-dependency keyword store — always works."""
    logger.info("Simple keyword store built — %d chunks", len(corpus))
    return {"type": "simple", "chunks": corpus}


def load_vector_store():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"No FAISS index at {FAISS_INDEX_PATH}.")
    with open(FAISS_INDEX_PATH, "rb") as f:
        vs = pickle.load(f)
    logger.info("FAISS index loaded")
    return vs
