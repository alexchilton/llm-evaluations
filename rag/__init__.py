"""
RAG Pipeline Package
=====================
Production RAG pipeline with Phoenix evaluation, OpenTelemetry tracing,
drift detection, and ReAct agent support.
"""

from rag.agent import AgentResult, AgentStep, RAGAgent
from rag.config import (
    DEFAULT_TOP_K,
    INDEX_DIR,
    LLM_BASE_URL,
    MODEL_NAME,
    PHOENIX_URL,
    PROJECT_NAME,
)
from rag.dataset import QADataset, QAPair
from rag.drift import DriftDetector, DriftReport
from rag.generator import Generator
from rag.indexer import Chunk, DocumentIndexer
from rag.pipeline import RAGPipeline, RAGResult
from rag.retriever import Retriever, RetrievalResult
from rag.runner import EvaluationRunner

__all__ = [
    # Core pipeline
    "RAGPipeline",
    "RAGResult",
    "Generator",
    "Retriever",
    "RetrievalResult",
    "DocumentIndexer",
    "Chunk",
    # Agent
    "RAGAgent",
    "AgentResult",
    "AgentStep",
    # Evaluation
    "EvaluationRunner",
    "QADataset",
    "QAPair",
    # Drift
    "DriftDetector",
    "DriftReport",
    # Config
    "LLM_BASE_URL",
    "MODEL_NAME",
    "PHOENIX_URL",
    "PROJECT_NAME",
    "INDEX_DIR",
    "DEFAULT_TOP_K",
]
