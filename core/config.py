"""
config.py — every setting for ConsultAgent-LLM lives here.
Values come from the .env file (copy .env.example to .env).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MEMORY_DIR  = DATA_DIR / "memory"
SAMPLES_DIR = BASE_DIR / "samples"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM provider ─────────────────────────────────────────────
# Groq only (cloud, free tier)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Model catalogues change often — check your provider console
# and override LLM_MODEL in .env if a default stops working.
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
}
LLM_MODEL       = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(LLM_PROVIDER, "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
}

# ── Web search ───────────────────────────────────────────────
# With a Tavily key search is more reliable; without one
# ConsultAgent-LLM falls back to DuckDuckGo (no key needed).
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "").strip()
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

# ── Second Brain (vector memory) ─────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150

# ── Agent limits (keep API usage and cost predictable) ───────
MAX_PLAN_STEPS       = 4
MAX_CLAIMS           = 8
MAX_COMPETITORS      = 4
MAX_PROPOSAL_ROUNDS  = 2
TARGET_WIN_PROB      = 75
PARALLEL_WORKERS     = 2   # small bursts stay under per-minute rate limits


def provider_ready() -> tuple[bool, str]:
    """Return (ok, message) describing whether the LLM can be called."""
    if LLM_PROVIDER not in PROVIDER_KEYS:
        return False, f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use groq."
    key_name = PROVIDER_KEYS[LLM_PROVIDER]
    if not os.getenv(key_name):
        return False, f"Add {key_name} to your .env file."
    return True, f"{LLM_PROVIDER} / {LLM_MODEL}"


def search_mode() -> str:
    return "Tavily" if TAVILY_API_KEY else "DuckDuckGo"