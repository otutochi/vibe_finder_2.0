"""Command line runner for the Music Recommender Simulation."""

from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .recommender import Recommender, UserProfile, load_songs


def print_recommendations(label: str, user_profile: UserProfile, recommender: Recommender, k: int = 5) -> None:
    """Print a labeled block of recommendations for one user profile."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Prefs: {asdict(user_profile)}")
    print(f"{'='*50}\n")

    recommendations = recommender.recommend(user_profile, k=k)
    for i, (song, score, explanation) in enumerate(recommendations, 1):
        print(f"  {i}. {song.title} by {song.artist} — Score: {score:.2f}")
        print(f"     Because: {explanation}")
    print()


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "songs.csv"
    songs = load_songs(str(csv_path))
    recommender = Recommender(songs)
    print(f"Loaded {len(songs)} songs.\n")

    # --- Core profiles ---
    profiles: Dict[str, UserProfile] = {
        "Danceable Pop Commute": UserProfile(
            favorite_genre="pop",
            favorite_mood="happy",
            target_energy=0.85,
            likes_acoustic=False,
        ),
        "Acoustic Study Lofi": UserProfile(
            favorite_genre="lofi",
            favorite_mood="focused",
            target_energy=0.35,
            likes_acoustic=True,
        ),
        "Deep Intense Rock": UserProfile(
            favorite_genre="rock",
            favorite_mood="intense",
            target_energy=0.90,
            likes_acoustic=False,
        ),
    }

    # --- Adversarial / edge-case profiles ---
    edge_cases: Dict[str, UserProfile] = {
        "Conflicting: Fast Acoustic Ambient": UserProfile(
            favorite_genre="ambient",
            favorite_mood="chill",
            target_energy=0.95,
            likes_acoustic=True,
        ),
        "Genre Outsider (Reggae Request)": UserProfile(
            favorite_genre="reggae",
            favorite_mood="happy",
            target_energy=0.60,
            likes_acoustic=True,
        ),
        "Extreme Low Energy Metal": UserProfile(
            favorite_genre="metal",
            favorite_mood="aggressive",
            target_energy=0.10,
            likes_acoustic=False,
        ),
    }

    for label, profile in profiles.items():
        print_recommendations(label, profile, recommender)

    print("\n--- ADVERSARIAL / EDGE-CASE PROFILES ---")
    for label, profile in edge_cases.items():
        print_recommendations(label, profile, recommender)


if __name__ == "__main__":
    main()
