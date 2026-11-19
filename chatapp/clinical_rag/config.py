"""
Settings for the Clinical RAG pipeline.

These values mirror the standalone scripts (ingest.py and query.py) and must stay in sync
with them: the same encoders, the same ChromaDB collection and the same Groq model.
"""

from pathlib import Path

# <project root>/chatapp/clinical_rag/config.py  ->  parents[2] is the folder with manage.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"

# ChromaDB
COLLECTION_NAME = "research_abstracts"

# MedCPT (asymmetric: articles use the Article Encoder in ingest.py, queries use the Query Encoder)
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
MAX_TOKENS = 512
TOP_K = 3

# Groq
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "You are a helpful medical assistant specializing in diabetes for children and adolescents. "
    "Use ONLY the provided context to answer the user's question. "
    "If the answer cannot be found in the context, say 'I cannot find the answer in the provided documents.'"
)
NO_ANSWER = "I cannot find the answer in the provided documents."

# Chat feature (not part of query.py): when a conversation already has messages, a follow-up such as
# "what about adolescents?" is first rewritten into a standalone question using the same Groq model.
# Set to False to search with exactly what the user typed.
CONDENSE_FOLLOW_UPS = True

# ============================================================
# Sufficiency Configuration (Batch 1)
# ============================================================
# Model C checkpoint path (relative to PROJECT_ROOT)
MODEL_C_CHECKPOINT = PROJECT_ROOT / "classification" / "model_c_v2" / "checkpoint" / "best"
MODEL_C_MAX_LENGTH = 512

# Combined scoring weights (sum to 1.0)
# Formula: combined = 0.10 * model_c + 0.45 * cosine + 0.45 * llm_judge
MODEL_C_WEIGHT = 0.10
COSINE_WEIGHT = 0.45
LLM_JUDGE_WEIGHT = 0.45

# Verify weights sum to 1.0 at module load
assert abs(MODEL_C_WEIGHT + COSINE_WEIGHT + LLM_JUDGE_WEIGHT - 1.0) < 1e-9, (
    "Sufficiency weights must sum to 1.0"
)

# Combined threshold (on 0-1 scale)
COMBINED_THRESHOLD = 0.70

# Whether the sufficiency gate is enabled
SUFFICIENCY_ENABLED = True

# LLM Judge configuration
LLM_JUDGE_ENABLED = True
LLM_JUDGE_TIMEOUT_SECONDS = 20  # Timeout for LLM judge API call
LLM_JUDGE_PROVIDER = "groq"  # "gemini" or "groq"
LLM_JUDGE_MODEL = "gemini-3.5-flash"  # Used when provider is "gemini"
LLM_JUDGE_GROQ_MODEL = "openai/gpt-oss-20b"  # Used when provider is "groq"

# ============================================================
# Scope Detection Configuration
# ============================================================
# Scope model path (relative to PROJECT_ROOT)
SCOPE_MODEL_PATH = PROJECT_ROOT / "classification" / "scope_detection" / "model" / "pubmedbert_best"

# Scope detection max sequence length
SCOPE_MAX_LENGTH = 64

# Whether the scope detection gate is enabled
# If disabled, all questions are treated as IN_SCOPE (for testing)
SCOPE_DETECTION_ENABLED = True
