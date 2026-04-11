"""
RAG Pipeline Configuration
==========================
Central constants for LLM, Phoenix, embedding model, and index paths.
"""

from pathlib import Path

# LLM (llama.cpp serving Qwen)
LLM_BASE_URL = "http://127.0.0.1:8002/v1"
MODEL_NAME = "qwen3-30b-instruct"
LLM_API_KEY = "not-needed"

# Phoenix observability
PHOENIX_URL = "http://localhost:6006"
PROJECT_NAME = "rag-pipeline"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# FAISS index
INDEX_DIR = Path("rag_index")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
DEFAULT_TOP_K = 5

# LLM generation
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.1
JUDGE_MAX_TOKENS = 512

# Rate limiting for LLM judge calls
MAX_CONCURRENT_JUDGE = 2

# QA data
QA_PAIRS_PATH = Path("rag_qa_pairs.jsonl")
BASELINE_PATH = Path("baseline.json")

# System prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the question using ONLY the "
    "provided context. If the context doesn't contain enough information, "
    "say so. Be concise and factual."
)
