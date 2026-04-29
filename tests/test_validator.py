from pathlib import Path

from src.query_parser import ParsedPreferences, parse_query
from src.recommender import Recommender, load_songs
from src.retriever import KnowledgeRetriever
from src.validator import RecommendationValidator


def make_system_components() -> tuple[Recommender, KnowledgeRetriever, RecommendationValidator]:
    repo_root = Path(__file__).resolve().parents[1]
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    recommender = Recommender(songs)
    retriever = KnowledgeRetriever(repo_root / "knowledge")
    validator = RecommendationValidator()
    return recommender, retriever, validator


def test_validator_returns_high_confidence_for_strong_fit_case():
    recommender, retriever, validator = make_system_components()
    parsed = parse_query("I want acoustic lofi songs for studying")
    evidence = retriever.retrieve(parsed, top_k=3)
    recommendations = recommender.recommend(parsed.preferences.to_user_profile(default_likes_acoustic=True), k=5)

    result = validator.validate(parsed, evidence, recommendations)

    assert result.confidence_score >= 0.80
    assert result.warnings == []
    assert any("Retrieved evidence count" in note for note in result.validation_notes)


def test_validator_warns_when_requested_genre_is_missing():
    recommender, retriever, validator = make_system_components()
    preferences = ParsedPreferences(
        favorite_genre="reggae",
        favorite_mood="happy",
        target_energy=0.60,
        likes_acoustic=True,
    )
    evidence = retriever.retrieve("I want happy reggae songs", top_k=3)
    recommendations = recommender.recommend(preferences.to_user_profile(default_likes_acoustic=True), k=5)

    result = validator.validate(preferences, evidence, recommendations)

    assert result.confidence_score < 0.70
    assert any("genre is missing" in warning.lower() for warning in result.warnings)


def test_validator_warns_for_contradictory_ambient_high_energy_request():
    recommender, retriever, validator = make_system_components()
    parsed = parse_query("I want ambient songs with very high energy and acoustic feel")
    evidence = retriever.retrieve(parsed, top_k=3)
    recommendations = recommender.recommend(parsed.preferences.to_user_profile(default_likes_acoustic=True), k=5)

    result = validator.validate(parsed, evidence, recommendations)

    assert result.confidence_score < 0.65
    assert any("contradiction" in warning.lower() for warning in result.warnings)
    assert any("weak fit" in warning.lower() for warning in result.warnings)