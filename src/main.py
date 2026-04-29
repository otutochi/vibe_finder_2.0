"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from pathlib import Path
from typing import Dict, List

from .recommender import load_songs, recommend_songs


def print_recommendations(label: str, user_prefs: Dict, songs: List[Dict], k: int = 5) -> None:
    """Print a labeled block of recommendations for one user profile."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Prefs: {user_prefs}")
    print(f"{'='*50}\n")

    recommendations = recommend_songs(user_prefs, songs, k=k)
    for i, (song, score, explanation) in enumerate(recommendations, 1):
        print(f"  {i}. {song['title']} by {song['artist']} — Score: {score:.2f}")
        print(f"     Because: {explanation}")
    print()


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "songs.csv"
    songs = load_songs(str(csv_path))
    print(f"Loaded {len(songs)} songs.\n")

    # --- Core profiles ---
    profiles = {
        "High-Energy Pop Fan": {"genre": "pop", "mood": "happy", "energy": 0.85},
        "Chill Lofi Listener": {"genre": "lofi", "mood": "chill", "energy": 0.35},
        "Deep Intense Rock":   {"genre": "rock", "mood": "intense", "energy": 0.90},
    }

    # --- Adversarial / edge-case profiles ---
    edge_cases = {
        "Conflicting: High Energy + Chill":  {"genre": "ambient", "mood": "chill", "energy": 0.95},
        "Genre Outsider (no match in CSV)":  {"genre": "reggae", "mood": "happy", "energy": 0.60},
        "Extreme Low Energy Metal":          {"genre": "metal", "mood": "aggressive", "energy": 0.10},
    }

    for label, prefs in profiles.items():
        print_recommendations(label, prefs, songs)

    print("\n--- ADVERSARIAL / EDGE-CASE PROFILES ---")
    for label, prefs in edge_cases.items():
        print_recommendations(label, prefs, songs)


if __name__ == "__main__":
    main()
