"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Global configuration loader for API keys, environment settings, and directory paths.
Dependencies: os, dotenv, pathlib
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b").strip()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper().strip()

# System and paths configuration
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Central RAG Pipeline Configurations
MAX_SEARCH_RESULTS = 8
MAX_SOURCES_TO_SCRAPE = 5
MAX_CONCURRENT_REQUESTS = 5
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB configurations
CHROMA_PERSIST_DIR = BASE_DIR / ".chroma_temp"

# ─── Semantic Cache + Redis + SQLite + MongoDB Configuration ─────────────────

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
REDIS_POOL_MAX_CONNECTIONS = int(os.getenv("REDIS_POOL_MAX_CONNECTIONS", "10"))

# MongoDB connection (L2 persistent user & query store)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "sourceiq_rag").strip()
JWT_SECRET = os.getenv("JWT_SECRET", "super_secret_jwt_key_sourceiq_2026").strip()

# SQLite persistent cache database
CACHE_DB_PATH = BASE_DIR / "cache" / "rag_cache.db"

# TTL values in seconds
EXACT_ANSWER_TTL    = int(os.getenv("EXACT_ANSWER_TTL",   "86400"))   # 24 hours
SEMANTIC_CACHE_TTL  = int(os.getenv("SEMANTIC_CACHE_TTL", "86400"))   # 24 hours
SEARCH_CACHE_TTL    = int(os.getenv("SEARCH_CACHE_TTL",   "86400"))   # 24 hours
PAGE_CACHE_TTL      = int(os.getenv("PAGE_CACHE_TTL",     "604800"))  # 7 days

# Freshness Sensitivity TTLs
DYNAMIC_QUERY_TTL = int(os.getenv("DYNAMIC_QUERY_TTL", "300"))      # 5 mins for real-time/finance
NEWS_QUERY_TTL    = int(os.getenv("NEWS_QUERY_TTL",    "3600"))     # 1 hour for news/current events
STABLE_QUERY_TTL  = int(os.getenv("STABLE_QUERY_TTL",  "2592000"))  # 30 days for educational/concepts

# Semantic similarity thresholds (configurable for benchmarking)
SEMANTIC_SIMILARITY_THRESHOLD  = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD",  "0.85"))
KNOWLEDGE_REUSE_MIN_SIMILARITY = float(os.getenv("KNOWLEDGE_REUSE_MIN_SIMILARITY", "0.75"))
SEMANTIC_CACHE_TOP_K           = int(os.getenv("SEMANTIC_CACHE_TOP_K", "5"))

# Embedding model identity for cache compatibility checks
CACHE_EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # Must match ChunkEmbedder
CACHE_EMBEDDING_DIMENSION = 384
CACHE_EMBEDDING_VERSION   = "1.0"


# Retrieve configurations validations
def validate_config() -> None:
    """
    Validates that essential configurations are present.
    Raises ValueError if API keys are missing.
    """
    errors = []
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is not set in the environment or .env file.")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set in the environment or .env file.")
        
    if errors:
        raise ValueError("Configuration Error: " + " | ".join(errors))

if __name__ == "__main__":
    print("Testing config.py...")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Logs Dir: {LOGS_DIR}")
    print(f"Tavily API Key Set: {bool(TAVILY_API_KEY)}")
    print(f"Gemini API Key Set: {bool(GEMINI_API_KEY)}")
    print(f"Groq API Key Set: {bool(GROQ_API_KEY)}")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Cache DB Path: {CACHE_DB_PATH}")
    try:
        validate_config()
        print("Success: Config is valid.")
    except ValueError as e:
        print(f"Config Validation Warning: {e}")
