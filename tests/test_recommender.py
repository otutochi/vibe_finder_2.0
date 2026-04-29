from dataclasses import asdict

import pytest

from src.recommender import Recommender, Song, UserProfile, recommend_songs

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_score_song_returns_expected_score_breakdown():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    pop_score, pop_reasons = rec.score_song(user, rec.songs[0])
    lofi_score, lofi_reasons = rec.score_song(user, rec.songs[1])

    assert pop_score == pytest.approx(4.0)
    assert pop_reasons == [
        "genre match (+2.0)",
        "mood match (+1.0)",
        "energy closeness (+1.00)",
    ]
    assert lofi_score == pytest.approx(0.6)
    assert lofi_reasons == ["energy closeness (+0.60)"]


def test_recommend_sorts_by_score_even_when_best_song_is_not_first():
    user = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.35,
        likes_acoustic=True,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert [song.title for song in results] == ["Chill Lofi Loop", "Test Pop Track"]


def test_explain_recommendation_returns_full_breakdown():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation == "genre match (+2.0); mood match (+1.0); energy closeness (+1.00)"


def test_functional_and_oop_recommendations_match():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    user_prefs = {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }
    rec = make_small_recommender()
    songs = [asdict(song) for song in rec.songs]

    oop_results = rec.recommend_with_details(user, k=2)
    functional_results = recommend_songs(user_prefs, songs, k=2)

    assert len(oop_results) == len(functional_results) == 2

    for (oop_song, oop_score, oop_explanation), (func_song, func_score, func_explanation) in zip(
        oop_results,
        functional_results,
    ):
        assert oop_song.title == func_song["title"]
        assert oop_score == pytest.approx(func_score)
        assert oop_explanation == func_explanation
