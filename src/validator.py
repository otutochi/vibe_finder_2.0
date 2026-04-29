from dataclasses import dataclass, field
from typing import List, Tuple, Union

from .query_parser import ParsedPreferences, QueryParseResult
from .recommender import (
    ACOUSTIC_PREFERENCE_WEIGHT,
    DANCEABILITY_WEIGHT,
    GENRE_MATCH_WEIGHT,
    MOOD_MATCH_WEIGHT,
    TEMPO_WEIGHT,
    VALENCE_WEIGHT,
    Recommender,
    Song,
    UserProfile,
)
from .retriever import RetrievedSnippet


MAX_RECOMMENDER_SCORE = (
    GENRE_MATCH_WEIGHT
    + MOOD_MATCH_WEIGHT
    + 1.0
    + TEMPO_WEIGHT
    + VALENCE_WEIGHT
    + DANCEABILITY_WEIGHT
    + ACOUSTIC_PREFERENCE_WEIGHT
)

GENRE_EXPECTED_ENERGY = {
    "ambient": 0.25,
    "classical": 0.22,
    "country": 0.60,
    "electronic": 0.88,
    "funk": 0.78,
    "hip hop": 0.62,
    "indie pop": 0.76,
    "jazz": 0.37,
    "latin": 0.68,
    "lofi": 0.40,
    "metal": 0.96,
    "pop": 0.82,
    "r&b": 0.55,
    "rock": 0.91,
    "synthwave": 0.75,
}


@dataclass
class ValidationResult:
    confidence_score: float
    warnings: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)


class RecommendationValidator:
    def __init__(self):
        self._recommender = Recommender([])

    def _normalize_preferences(
        self,
        parsed_preferences: Union[ParsedPreferences, QueryParseResult],
    ) -> ParsedPreferences:
        if isinstance(parsed_preferences, QueryParseResult):
            return parsed_preferences.preferences
        return parsed_preferences

    def _build_user_profile(self, preferences: ParsedPreferences) -> UserProfile:
        return UserProfile(
            favorite_genre=preferences.favorite_genre,
            favorite_mood=preferences.favorite_mood,
            target_energy=preferences.target_energy,
            likes_acoustic=False if preferences.likes_acoustic is None else preferences.likes_acoustic,
        )

    def _relevant_overall_features(self, preferences: ParsedPreferences) -> List[str]:
        features = ["energy", "tempo", "valence", "danceability"]
        if preferences.favorite_genre:
            features.insert(0, "genre")
        if preferences.favorite_mood:
            features.insert(1 if preferences.favorite_genre else 0, "mood")
        if preferences.likes_acoustic is not None:
            features.append("acousticness")
        return features

    def _supporting_features(self, preferences: ParsedPreferences) -> List[str]:
        features = ["energy", "tempo", "valence", "danceability"]
        if preferences.likes_acoustic is not None:
            features.append("acousticness")
        return features

    def validate(
        self,
        parsed_preferences: Union[ParsedPreferences, QueryParseResult],
        retrieved_evidence: List[RetrievedSnippet],
        recommendations: List[Tuple[Song, float, str]],
    ) -> ValidationResult:
        preferences = self._normalize_preferences(parsed_preferences)
        warnings: List[str] = []
        notes: List[str] = []

        if not recommendations:
            warnings.append("No recommendations were available to validate.")
            return ValidationResult(confidence_score=0.0, warnings=warnings, validation_notes=notes)

        user_profile = self._build_user_profile(preferences)
        top_song, top_score, _ = recommendations[0]
        fit_scores = self._recommender.feature_fit_scores(user_profile, top_song)

        overall_features = self._relevant_overall_features(preferences)
        supporting_features = self._supporting_features(preferences)
        overall_fit = sum(fit_scores[feature] for feature in overall_features) / len(overall_features)
        supporting_fit = sum(fit_scores[feature] for feature in supporting_features) / len(supporting_features)
        normalized_top_score = min(1.0, top_score / MAX_RECOMMENDER_SCORE)
        evidence_support = min(1.0, len(retrieved_evidence) / 3.0)

        notes.append(f"Top recommendation normalized score: {normalized_top_score:.2f}.")
        notes.append(f"Average overall fit: {overall_fit:.2f}; supporting feature fit: {supporting_fit:.2f}.")
        notes.append(f"Retrieved evidence count: {len(retrieved_evidence)}.")

        penalty = 0.0

        if preferences.favorite_genre:
            genre_present = any(song.genre.lower() == preferences.favorite_genre.lower() for song, _, _ in recommendations)
            if not genre_present:
                warnings.append("Requested genre is missing from the current catalog coverage.")
                penalty += 0.25

        expected_energy = GENRE_EXPECTED_ENERGY.get(preferences.favorite_genre.lower()) if preferences.favorite_genre else None
        if expected_energy is not None and abs(preferences.target_energy - expected_energy) >= 0.45:
            warnings.append("The request contains a likely contradiction between genre expectations and target energy.")
            penalty += 0.20

        if supporting_fit < 0.72 or fit_scores["energy"] < 0.60:
            warnings.append("Top recommendation is only a weak fit on supporting features like energy, tempo, valence, danceability, or acousticness.")
            penalty += 0.15

        if preferences.avoid_constraints:
            top_attributes = {top_song.genre.lower(), top_song.mood.lower()}
            if preferences.likes_acoustic is True:
                top_attributes.add("acoustic")
            if top_attributes & {constraint.lower() for constraint in preferences.avoid_constraints}:
                warnings.append("Top recommendation conflicts with an explicit avoid constraint.")
                penalty += 0.20

        confidence = (0.55 * normalized_top_score) + (0.30 * overall_fit) + (0.15 * evidence_support) - penalty
        confidence = max(0.0, min(1.0, confidence))

        return ValidationResult(
            confidence_score=confidence,
            warnings=warnings,
            validation_notes=notes,
        )