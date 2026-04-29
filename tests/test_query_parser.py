import pytest

from src.query_parser import parse_query
from src.recommender import UserProfile


def test_parse_chill_acoustic_study_request():
    result = parse_query("I want mellow acoustic songs for studying")
    preferences = result.preferences

    assert preferences.favorite_genre == ""
    assert preferences.favorite_mood == "chill"
    assert preferences.target_energy == pytest.approx(0.38, abs=0.01)
    assert preferences.likes_acoustic is True
    assert preferences.avoid_constraints == []
    assert isinstance(preferences.to_user_profile(), UserProfile)
    assert preferences.to_user_profile().likes_acoustic is True


def test_parse_high_energy_rock_request():
    result = parse_query("Recommend high-energy rock for the gym")
    preferences = result.preferences

    assert preferences.favorite_genre == "rock"
    assert preferences.favorite_mood == "intense"
    assert preferences.target_energy == pytest.approx(0.90, abs=0.01)
    assert "No explicit acoustic preference found." in result.assumptions


def test_parse_request_with_avoid_phrase():
    result = parse_query("I want mellow acoustic songs for studying but not anything too sad")
    preferences = result.preferences

    assert preferences.favorite_mood == "chill"
    assert preferences.likes_acoustic is True
    assert "melancholic" in preferences.avoid_constraints


def test_parse_ambiguous_request_adds_assumptions():
    result = parse_query("Play something for late night")
    preferences = result.preferences

    assert preferences.favorite_genre == ""
    assert preferences.favorite_mood == "chill"
    assert preferences.target_energy == pytest.approx(0.55, abs=0.01)
    assert preferences.likes_acoustic is None
    assert len(result.assumptions) >= 3


def test_parse_reggae_request_preserves_missing_coverage_genre_for_downstream_checks():
    result = parse_query("I want happy reggae songs")
    preferences = result.preferences
    user_profile = preferences.to_user_profile()

    assert preferences.favorite_genre == "reggae"
    assert preferences.favorite_mood == "happy"
    assert preferences.target_energy == pytest.approx(0.55, abs=0.01)
    assert preferences.likes_acoustic is None
    assert user_profile.favorite_genre == "reggae"
    assert user_profile.likes_acoustic is False