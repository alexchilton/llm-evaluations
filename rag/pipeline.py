"""
RAG Pipeline
=============
Wires together Retriever + Generator under a single traced operation.
Every query produces a trace with nested spans and returns trace_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from opentelemetry import trace

from rag.config import DEFAULT_TOP_K, INDEX_DIR, LLM_BASE_URL, MODEL_NAME
from rag.generator import Generator
from rag.indexer import DocumentIndexer
from rag.retriever import Retriever, RetrievalResult

tracer = trace.get_tracer("rag.pipeline")


@dataclass
class RAGResult:
    """Complete output from a single RAG query."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievalResult]
    context: str
    trace_id: str
    latency_ms: float
    retrieved_doc_ids: list[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    relevant_doc_ids: Optional[list[str]] = None
    cluster: Optional[str] = None


class RAGPipeline:
    """
    Retrieve-then-generate pipeline with full OpenTelemetry tracing.

    Every call to query() produces a trace with nested spans:
        rag_query
        ├── embed_query + faiss_retrieve
        └── llm_generate
    """

    def __init__(
        self,
        indexer: Optional[DocumentIndexer] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        if indexer is None:
            indexer = DocumentIndexer(index_dir=INDEX_DIR)
            indexer.build_all()
        self.indexer = indexer

        self.retriever = retriever or Retriever(indexer, top_k=top_k)
        self.generator = generator or Generator(
            llm_base_url=LLM_BASE_URL,
            model_name=MODEL_NAME,
        )
        self.top_k = top_k

    def query(self, question: str, ground_truth: Optional[str] = None,
              relevant_doc_ids: Optional[list[str]] = None,
              cluster: Optional[str] = None) -> RAGResult:
        """
        Full RAG pipeline: retrieve → build context → generate.
        Returns RAGResult with answer, chunks, and trace_id.
        """
        start = time.time()

        with tracer.start_as_current_span("rag_query") as root_span:
            root_span.set_attribute("rag.question", question[:200])
            root_span.set_attribute("rag.top_k", self.top_k)

            # Retrieve
            with tracer.start_as_current_span("embed_query_and_retrieve") as ret_span:
                retrieved = self.retriever.search(question, top_k=self.top_k)
                doc_ids = self.retriever.get_retrieved_doc_ids(retrieved)
                ret_span.set_attribute("rag.num_results", len(retrieved))
                ret_span.set_attribute("rag.doc_ids", str(doc_ids))
                if retrieved:
                    ret_span.set_attribute("rag.top_score", retrieved[0].score)

            # Build context
            context_parts = []
            for r in retrieved:
                context_parts.append(f"[Source: {r.source_article}]\n{r.chunk.text}")
            context = "\n\n---\n\n".join(context_parts)

            # Generate
            answer, gen_latency = self.generator.generate(question, context)

            total_latency = (time.time() - start) * 1000
            root_span.set_attribute("rag.latency_ms", total_latency)
            root_span.set_attribute("rag.answer_length", len(answer))

            trace_id = format(root_span.get_span_context().trace_id, "032x")

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved,
            context=context,
            trace_id=trace_id,
            latency_ms=total_latency,
            retrieved_doc_ids=doc_ids,
            ground_truth=ground_truth,
            relevant_doc_ids=relevant_doc_ids,
            cluster=cluster,
        )
