from pathlib import Path

from src.query_parser import parse_query
from src.retriever import KnowledgeRetriever


def make_retriever() -> KnowledgeRetriever:
    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge"
    return KnowledgeRetriever(knowledge_dir)


def test_chill_study_request_retrieves_study_or_mood_knowledge():
    retriever = make_retriever()

    results = retriever.retrieve("I want chill songs for studying", top_k=3)

    assert len(results) == 3
    assert any(result.source_name == "listening_contexts.md" for result in results)
    assert any(
        result.source_name in {"listening_contexts.md", "mood_traits.md"}
        and ("study" in result.snippet.lower() or "chill" in result.snippet.lower())
        for result in results
    )


def test_acoustic_request_retrieves_acousticness_knowledge():
    retriever = make_retriever()
    parsed = parse_query("I want mellow acoustic songs for studying")

    results = retriever.retrieve(parsed.preferences, top_k=3)

    assert len(results) == 3
    assert any(result.source_name == "acousticness.md" for result in results)
    assert any("acoustic" in result.snippet.lower() for result in results)


def test_high_energy_dance_request_retrieves_feature_signals():
    retriever = make_retriever()

    results = retriever.retrieve("Give me high-energy dance music for a workout", top_k=3)

    assert len(results) == 3
    assert any(result.source_name == "feature_signals.md" for result in results)
    assert any(
        result.source_name == "feature_signals.md"
        and ("danceability" in result.snippet.lower() or "energy" in result.snippet.lower())
        for result in results
    )