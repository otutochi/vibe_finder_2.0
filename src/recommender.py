from typing import List, Dict, Tuple
from dataclasses import asdict, dataclass

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

    def score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if song.genre.lower() == user.favorite_genre.lower():
            score += 2.0
            reasons.append("genre match (+2.0)")

        if song.mood.lower() == user.favorite_mood.lower():
            score += 1.0
            reasons.append("mood match (+1.0)")

        energy_score = 1.0 - abs(song.energy - user.target_energy)
        score += energy_score
        reasons.append(f"energy closeness (+{energy_score:.2f})")

        return (score, reasons)

    def recommend_with_details(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, str]]:
        scored = []
        for song in self.songs:
            score, reasons = self.score_song(user, song)
            explanation = "; ".join(reasons) if reasons else "no matching features"
            scored.append((song, score, explanation))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        recommendations = self.recommend_with_details(user, k=k)
        return [song for song, _, _ in recommendations]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = self.score_song(user, song)
        return "; ".join(reasons) if reasons else "no matching features"


def _dict_to_user_profile(user_prefs: Dict) -> UserProfile:
    return UserProfile(
        favorite_genre=user_prefs.get("genre", ""),
        favorite_mood=user_prefs.get("mood", ""),
        target_energy=float(user_prefs.get("energy", 0.0)),
        likes_acoustic=bool(user_prefs.get("likes_acoustic", False)),
    )


def _dict_to_song(song: Dict) -> Song:
    return Song(
        id=int(song.get("id", 0)),
        title=song.get("title", ""),
        artist=song.get("artist", ""),
        genre=song.get("genre", ""),
        mood=song.get("mood", ""),
        energy=float(song.get("energy", 0.0)),
        tempo_bpm=float(song.get("tempo_bpm", 0.0)),
        valence=float(song.get("valence", 0.0)),
        danceability=float(song.get("danceability", 0.0)),
        acousticness=float(song.get("acousticness", 0.0)),
    )

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv

    numeric_fields = {"id", "energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in numeric_fields:
                if field in row:
                    row[field] = int(row[field]) if field == "id" else float(row[field])
            songs.append(row)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    recommender = Recommender([_dict_to_song(song)])
    user = _dict_to_user_profile(user_prefs)
    return recommender.score_song(user, recommender.songs[0])

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    user = _dict_to_user_profile(user_prefs)
    recommender = Recommender([_dict_to_song(song) for song in songs])
    recommendations = recommender.recommend_with_details(user, k=k)
    return [(asdict(song), score, explanation) for song, score, explanation in recommendations]
