"""
Unified LLM client — Groq (primary, free) + Gemini (fallback if key set).
Install: pip install groq
Get free key: https://console.groq.com → API Keys
"""
import os
import time
import logging
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")   # best free Groq model

# Groq free models (fallback chain)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality — current
    "llama-3.1-8b-instant",      # fast + free
    "gemma2-9b-it",              # Google's model on Groq
    "mixtral-8x7b-32768",        # large context
]

# Gemini fallback chain
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]

_exhausted: set = set()


def get_active_provider() -> str:
    if GROQ_API_KEY:   return "groq"
    if GOOGLE_API_KEY: return "gemini"
    return "none"


def get_provider_display() -> str:
    if GROQ_API_KEY:   return f"Groq ({GROQ_MODEL})"
    if GOOGLE_API_KEY: return f"Gemini ({GEMINI_MODEL})"
    return "No API key"


def _gemini_generate(prompt: str, max_tokens: int = 600, temperature: float = 0.3) -> str:
    """
    Main LLM call — tries Groq first (free, unlimited), then Gemini.
    All modules call this function — no direct API calls anywhere else.
    """
    provider = get_active_provider()

    if provider == "groq":
        try:
            return _try_groq(prompt, max_tokens, temperature)
        except Exception as e:
            err = str(e)
            logger.warning("Groq failed: %s", err)
            # If Gemini key also set, try it as fallback
            if GOOGLE_API_KEY:
                logger.info("Groq failed — trying Gemini fallback")
                return _try_gemini(prompt, max_tokens, temperature)
            raise RuntimeError(
                f"Groq error: {err}\n"
                "Try: set GROQ_API_KEY in .env or restart the app."
            ) from e

    elif provider == "gemini":
        return _try_gemini(prompt, max_tokens, temperature)

    else:
        raise RuntimeError(
            "No LLM API key found.\n"
            "Add GROQ_API_KEY=your_key to .env\n"
            "Get free key at: https://console.groq.com"
        )


def _try_groq(prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Groq — free tier, generous limits, no daily quota."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "Groq package not installed.\n"
            "Run: pip install groq"
        )

    client = Groq(api_key=GROQ_API_KEY)

    # Try models in order
    candidates = [GROQ_MODEL] + [m for m in GROQ_MODELS if m != GROQ_MODEL]
    last_error = None

    for model_name in candidates:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content":
                     "You are an expert data scientist and business analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if model_name != GROQ_MODEL:
                logger.info("Used Groq fallback model: %s", model_name)
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            last_error = e
            if "model_not_found" in err or "404" in err:
                logger.warning("Groq model %s not found — trying next", model_name)
                continue
            elif "rate_limit" in err.lower() or "429" in err:
                logger.warning("Groq rate limit on %s — waiting 5s", model_name)
                time.sleep(5)
                continue
            else:
                raise

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


def _try_gemini(prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Gemini with model fallback."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    genai.configure(api_key=GOOGLE_API_KEY)
    candidates = [GEMINI_MODEL] + [m for m in GEMINI_MODELS if m != GEMINI_MODEL]
    candidates = [m for m in candidates if m not in _exhausted]

    if not candidates:
        raise RuntimeError("All Gemini models exhausted for today.")

    last_error = None
    for model_name in candidates:
        for attempt in range(2):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                last_error = e
                if "GenerateRequestsPerDay" in err or "limit: 0" in err:
                    _exhausted.add(model_name); break
                elif "429" in err:
                    time.sleep(15 * (attempt + 1))
                elif "404" in err or "not found" in err.lower():
                    _exhausted.add(model_name); break
                else:
                    raise

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def reset_quota_cache():
    _exhausted.clear()
