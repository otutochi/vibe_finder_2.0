from typing import List, Tuple
from dataclasses import dataclass


GENRE_MATCH_WEIGHT = 2.0
MOOD_MATCH_WEIGHT = 1.0
TEMPO_WEIGHT = 0.35
VALENCE_WEIGHT = 0.30
DANCEABILITY_WEIGHT = 0.25
ACOUSTIC_PREFERENCE_WEIGHT = 0.30

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _closeness(self, value: float, target: float, spread: float = 1.0) -> float:
        return max(0.0, 1.0 - abs(value - target) / spread)

    def _target_tempo(self, user: UserProfile) -> float:
        base_tempo = 60.0 + (user.target_energy * 80.0)
        mood_offsets = {
            "happy": 6.0,
            "chill": -8.0,
            "focused": -4.0,
            "intense": 12.0,
            "aggressive": 16.0,
            "relaxed": -12.0,
            "euphoric": 10.0,
            "moody": -4.0,
            "nostalgic": -2.0,
            "melancholic": -6.0,
            "dreamy": -10.0,
            "uplifting": 8.0,
            "groovy": 6.0,
            "romantic": -3.0,
        }
        genre_offsets = {
            "pop": 4.0,
            "lofi": -4.0,
            "rock": 8.0,
            "ambient": -12.0,
            "jazz": -2.0,
            "synthwave": 4.0,
            "indie pop": 2.0,
            "r&b": -2.0,
            "electronic": 10.0,
            "hip hop": 2.0,
            "classical": -10.0,
            "metal": 12.0,
            "funk": 6.0,
            "country": 0.0,
            "latin": 4.0,
        }
        target = base_tempo
        target += mood_offsets.get(user.favorite_mood.lower(), 0.0)
        target += genre_offsets.get(user.favorite_genre.lower(), 0.0)
        return min(180.0, max(60.0, target))

    def _target_valence(self, user: UserProfile) -> float:
        base_valence = 0.25 + (user.target_energy * 0.5)
        mood_targets = {
            "happy": 0.85,
            "chill": 0.58,
            "focused": 0.52,
            "intense": 0.45,
            "aggressive": 0.22,
            "relaxed": 0.68,
            "euphoric": 0.82,
            "moody": 0.40,
            "nostalgic": 0.62,
            "melancholic": 0.30,
            "dreamy": 0.56,
            "uplifting": 0.78,
            "groovy": 0.88,
            "romantic": 0.75,
        }
        genre_offsets = {
            "pop": 0.05,
            "lofi": 0.02,
            "rock": -0.05,
            "ambient": -0.05,
            "jazz": 0.02,
            "synthwave": -0.02,
            "indie pop": 0.04,
            "r&b": 0.04,
            "electronic": 0.03,
            "hip hop": -0.05,
            "classical": -0.02,
            "metal": -0.10,
            "funk": 0.05,
            "country": 0.04,
            "latin": 0.05,
        }
        target = mood_targets.get(user.favorite_mood.lower(), base_valence)
        target += genre_offsets.get(user.favorite_genre.lower(), 0.0)
        return min(1.0, max(0.0, target))

    def _target_danceability(self, user: UserProfile) -> float:
        base_danceability = 0.35 + (user.target_energy * 0.45)
        genre_targets = {
            "pop": 0.82,
            "lofi": 0.55,
            "rock": 0.62,
            "ambient": 0.35,
            "jazz": 0.52,
            "synthwave": 0.72,
            "indie pop": 0.76,
            "r&b": 0.74,
            "electronic": 0.89,
            "hip hop": 0.80,
            "classical": 0.25,
            "metal": 0.48,
            "funk": 0.92,
            "country": 0.62,
            "latin": 0.85,
        }
        mood_offsets = {
            "happy": 0.04,
            "chill": -0.03,
            "focused": -0.04,
            "intense": 0.02,
            "aggressive": -0.04,
            "relaxed": -0.05,
            "euphoric": 0.05,
            "moody": -0.03,
            "nostalgic": -0.02,
            "melancholic": -0.06,
            "dreamy": -0.08,
            "uplifting": 0.03,
            "groovy": 0.06,
            "romantic": 0.02,
        }
        genre_target = genre_targets.get(user.favorite_genre.lower(), base_danceability)
        target = (base_danceability + genre_target) / 2.0
        target += mood_offsets.get(user.favorite_mood.lower(), 0.0)
        return min(1.0, max(0.0, target))

    def feature_fit_scores(self, user: UserProfile, song: Song) -> dict[str, float]:
        return {
            "genre": 1.0 if song.genre.lower() == user.favorite_genre.lower() else 0.0,
            "mood": 1.0 if song.mood.lower() == user.favorite_mood.lower() else 0.0,
            "energy": self._closeness(song.energy, user.target_energy),
            "tempo": self._closeness(song.tempo_bpm, self._target_tempo(user), spread=120.0),
            "valence": self._closeness(song.valence, self._target_valence(user)),
            "danceability": self._closeness(song.danceability, self._target_danceability(user)),
            "acousticness": self._closeness(song.acousticness, 0.80 if user.likes_acoustic else 0.20),
        }

    def score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []
        fit_scores = self.feature_fit_scores(user, song)

        if fit_scores["genre"] == 1.0:
            score += GENRE_MATCH_WEIGHT
            reasons.append(f"genre match (+{GENRE_MATCH_WEIGHT:.1f})")

        if fit_scores["mood"] == 1.0:
            score += MOOD_MATCH_WEIGHT
            reasons.append(f"mood match (+{MOOD_MATCH_WEIGHT:.1f})")

        energy_score = fit_scores["energy"]
        score += energy_score
        reasons.append(f"energy closeness (+{energy_score:.2f})")

        tempo_score = TEMPO_WEIGHT * fit_scores["tempo"]
        score += tempo_score
        reasons.append(f"tempo fit (+{tempo_score:.2f})")

        valence_score = VALENCE_WEIGHT * fit_scores["valence"]
        score += valence_score
        reasons.append(f"valence fit (+{valence_score:.2f})")

        danceability_score = DANCEABILITY_WEIGHT * fit_scores["danceability"]
        score += danceability_score
        reasons.append(f"danceability fit (+{danceability_score:.2f})")

        acoustic_score = ACOUSTIC_PREFERENCE_WEIGHT * fit_scores["acousticness"]
        score += acoustic_score
        reasons.append(f"acoustic preference fit (+{acoustic_score:.2f})")

        return (score, reasons)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, str]]:
        scored = []
        for song in self.songs:
            score, reasons = self.score_song(user, song)
            explanation = "; ".join(reasons) if reasons else "no matching features"
            scored.append((song, score, explanation))

        scored.sort(key=lambda item: (-item[1], item[0].title))
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = self.score_song(user, song)
        return "; ".join(reasons) if reasons else "no matching features"


def load_songs(csv_path: str) -> List[Song]:
    """
    Loads songs from a CSV file into Song objects.
    Required by src/main.py
    """
    import csv

    songs = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                Song(
                    id=int(row["id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    mood=row["mood"],
                    energy=float(row["energy"]),
                    tempo_bpm=float(row["tempo_bpm"]),
                    valence=float(row["valence"]),
                    danceability=float(row["danceability"]),
                    acousticness=float(row["acousticness"]),
                )
            )

    return songs
