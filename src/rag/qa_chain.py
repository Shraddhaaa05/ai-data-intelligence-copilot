"""
RAG Q&A chain — Groq primary, Gemini fallback, keyword-only last resort.
"""
from typing import Dict
from utils.config import GROQ_API_KEY, GOOGLE_API_KEY, GROQ_MODEL, TOP_K_RETRIEVAL
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert data scientist and business analyst.
Answer questions about the user's dataset based on the provided context.
Use numbers and statistics from the context when available.
If the answer is not in the context, say so honestly.
Frame answers in plain business language. Be concise and specific."""


def build_rag_chain(vectorstore):
    """Build RAG chain — works with Groq, Gemini, or no LLM (keyword fallback)."""
    has_llm = bool(GROQ_API_KEY or GOOGLE_API_KEY)
    if not has_llm:
        raise EnvironmentError(
            "No API key found.\n"
            "Add GROQ_API_KEY to .env — get free key at https://console.groq.com"
        )

    # Try LangChain with Groq
    if GROQ_API_KEY:
        try:
            return _build_groq_chain(vectorstore)
        except Exception as e:
            logger.warning("LangChain Groq chain failed (%s) — using direct Groq", e)
            return {"type": "direct", "vectorstore": vectorstore}

    # Try LangChain with Gemini
    if GOOGLE_API_KEY:
        try:
            return _build_gemini_chain(vectorstore)
        except Exception as e:
            logger.warning("LangChain Gemini chain failed (%s) — using direct", e)
            return {"type": "direct", "vectorstore": vectorstore}

    return {"type": "direct", "vectorstore": vectorstore}


def _build_groq_chain(vectorstore):
    from langchain.chains import RetrievalQA
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K_RETRIEVAL})
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.1,
        max_tokens=600,
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:")
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt})
    logger.info("RAG chain built — LangChain + Groq (%s)", GROQ_MODEL)
    return {"type": "langchain", "chain": chain}


def _build_gemini_chain(vectorstore):
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from utils.config import GEMINI_MODEL

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K_RETRIEVAL})
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY,
        temperature=0.1, convert_system_message_to_human=True)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:")
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt})
    logger.info("RAG chain built — LangChain + Gemini")
    return {"type": "langchain", "chain": chain}


def ask(chain_obj, question: str) -> Dict:
    if chain_obj is None:
        return {"answer": "Chat not initialised.", "sources": []}

    ctype = chain_obj.get("type", "langchain")

    # LangChain path
    if ctype == "langchain":
        try:
            result = chain_obj["chain"].invoke({"query": question})
            return {
                "answer":  result.get("result", "No answer found."),
                "sources": [d.page_content for d in result.get("source_documents", [])]
            }
        except Exception as e:
            logger.warning("LangChain ask failed (%s) — falling back to direct", e)
            # Convert to direct and retry
            vs = chain_obj.get("chain").retriever.vectorstore if hasattr(
                chain_obj.get("chain",""), "retriever") else None
            if vs:
                return ask({"type": "direct", "vectorstore": vs}, question)
            return {"answer": f"Error: {e}", "sources": []}

    # Direct LLM + retrieval path
    if ctype == "direct":
        try:
            from utils.gemini_client import _gemini_generate
            vs = chain_obj["vectorstore"]
            try:
                if isinstance(vs, dict):
                    chunks = vs.get("chunks", [])
                    q_words = [w for w in question.lower().split() if len(w) > 3]
                    sources = [c for c in chunks if any(w in c.lower() for w in q_words)][:5] or chunks[:5]
                else:
                    docs = vs.similarity_search(question, k=TOP_K_RETRIEVAL)
                    sources = [d.page_content for d in docs]
            except Exception:
                sources = []
            context = "\n".join(sources) if sources else "No context available."
            prompt  = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            answer  = _gemini_generate(prompt, max_tokens=500, temperature=0.1)
            return {"answer": answer, "sources": sources[:3]}
        except Exception as e:
            return {"answer": f"Error: {e}", "sources": []}

    # Simple keyword path
    if ctype == "simple":
        try:
            from utils.gemini_client import _gemini_generate
            chunks  = chain_obj.get("chunks", [])
            q_words = [w for w in question.lower().split() if len(w) > 3]
            sources = [c for c in chunks if any(w in c.lower() for w in q_words)][:5] or chunks[:5]
            context = "\n".join(sources)
            prompt  = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            answer  = _gemini_generate(prompt, max_tokens=500, temperature=0.1)
            return {"answer": answer, "sources": sources[:3]}
        except Exception as e:
            return {"answer": f"Error: {e}", "sources": []}

    return {"answer": "Unknown chain type.", "sources": []}
