import pytest

from src.recommender import Recommender, Song, UserProfile

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

    assert pop_score == pytest.approx(5.16, abs=0.01)
    assert pop_reasons == [
        "genre match (+2.0)",
        "mood match (+1.0)",
        "energy closeness (+1.00)",
        "tempo fit (+0.31)",
        "valence fit (+0.30)",
        "danceability fit (+0.25)",
        "acoustic preference fit (+0.30)",
    ]
    assert lofi_score == pytest.approx(1.27, abs=0.01)
    assert lofi_reasons == [
        "energy closeness (+0.60)",
        "tempo fit (+0.19)",
        "valence fit (+0.21)",
        "danceability fit (+0.17)",
        "acoustic preference fit (+0.09)",
    ]


def test_likes_acoustic_changes_scores_in_expected_direction():
    acoustic_song = make_small_recommender().songs[1]
    likes_acoustic_user = UserProfile(
        favorite_genre="",
        favorite_mood="",
        target_energy=0.4,
        likes_acoustic=True,
    )
    avoids_acoustic_user = UserProfile(
        favorite_genre="",
        favorite_mood="",
        target_energy=0.4,
        likes_acoustic=False,
    )
    rec = make_small_recommender()

    likes_score, likes_reasons = rec.score_song(likes_acoustic_user, acoustic_song)
    avoids_score, avoids_reasons = rec.score_song(avoids_acoustic_user, acoustic_song)

    assert likes_score == pytest.approx(2.08, abs=0.01)
    assert avoids_score == pytest.approx(1.90, abs=0.01)
    assert likes_score > avoids_score
    assert likes_reasons[-1] == "acoustic preference fit (+0.27)"
    assert avoids_reasons[-1] == "acoustic preference fit (+0.09)"


def test_recommend_returns_ranked_results_with_explanations():
    user = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.35,
        likes_acoustic=True,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert [song.title for song, _, _ in results] == ["Chill Lofi Loop", "Test Pop Track"]
    assert results[0][1] > results[1][1]
    assert "tempo fit" in results[0][2]
    assert "danceability fit" in results[0][2]


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
    assert explanation == (
        "genre match (+2.0); mood match (+1.0); energy closeness (+1.00); tempo fit (+0.31); "
        "valence fit (+0.30); danceability fit (+0.25); acoustic preference fit (+0.30)"
    )


def test_higher_energy_profile_prefers_faster_more_danceable_song():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    top_song, top_score, explanation = rec.recommend(user, k=1)[0]

    assert top_song.title == "Test Pop Track"
    assert top_score == pytest.approx(5.03, abs=0.01)
    assert "tempo fit" in explanation
    assert "valence fit" in explanation


def test_load_songs_returns_song_objects(tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,Sample Song,Sample Artist,pop,happy,0.8,120,0.9,0.8,0.2\n",
        encoding="utf-8",
    )

    from src.recommender import load_songs

    songs = load_songs(str(csv_path))

    assert len(songs) == 1
    assert isinstance(songs[0], Song)
    assert songs[0].tempo_bpm == pytest.approx(120.0)
