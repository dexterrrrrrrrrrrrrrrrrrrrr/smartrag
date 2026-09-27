"""Tests for the cost-aware query complexity router."""
from backend.models.llm import ModelRole


def test_simple_query_routes_to_small_model(query_router):
    decision = query_router.route("What is RAG?", num_retrieved_chunks=2)
    assert decision.role == ModelRole.SMALL
    assert decision.complexity_label == "LOW"


def test_complex_comparison_query_routes_to_large_model(query_router):
    decision = query_router.route(
        "Compare dense retrieval, sparse retrieval and hybrid retrieval, "
        "and explain their advantages and disadvantages.",
        num_retrieved_chunks=6,
    )
    assert decision.role == ModelRole.LARGE
    assert decision.complexity_label == "HIGH"
    assert "comparative reasoning" in decision.reason


def test_multi_step_query_routes_to_large_model(query_router):
    decision = query_router.route(
        "Walk me through, step by step, how to set up a complete RAG pipeline "
        "from document ingestion through chunking, embeddings, and retrieval, "
        "and also explain how the semantic cache and query router fit "
        "together, and what happens if a component fails?",
        num_retrieved_chunks=5,
    )
    assert decision.role == ModelRole.LARGE
    assert decision.signals.has_multi_step_keywords is True


def test_multiple_questions_increase_complexity(query_router):
    single = query_router.route("What is RAG?")
    multiple = query_router.route("What is RAG? What is a vector database? How does chunking work?")
    assert multiple.signals.score > single.signals.score


def test_long_query_increases_complexity(query_router):
    short = query_router.route("What is RAG?")
    long_query = query_router.route(
        "I've been reading about different approaches to retrieval augmented "
        "generation systems and I'm trying to understand the tradeoffs "
        "between various chunking strategies and how they affect downstream "
        "retrieval quality in production deployments"
    )
    assert long_query.signals.score > short.signals.score


def test_more_retrieved_chunks_increases_complexity(query_router):
    few_chunks = query_router.route("What is RAG?", num_retrieved_chunks=1)
    many_chunks = query_router.route("What is RAG?", num_retrieved_chunks=8)
    assert many_chunks.signals.score >= few_chunks.signals.score


def test_signals_are_exposed_for_explainability(query_router):
    decision = query_router.route("What is RAG?")
    assert decision.signals.word_count == 3
    assert decision.signals.has_comparison_keywords is False
    assert isinstance(decision.signals.score, float)
