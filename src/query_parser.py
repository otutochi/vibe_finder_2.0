import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .recommender import UserProfile


GENRE_ALIASES = {
    "ambient": ("ambient",),
    "classical": ("classical",),
    "country": ("country",),
    "electronic": ("electronic", "edm"),
    "funk": ("funk", "funky"),
    "hip hop": ("hip hop", "hip-hop", "hiphop", "rap"),
    "indie pop": ("indie pop", "indie-pop"),
    "jazz": ("jazz",),
    "latin": ("latin",),
    "lofi": ("lofi", "lo-fi"),
    "metal": ("metal",),
    "pop": ("pop",),
    "reggae": ("reggae",),
    "r&b": ("r&b", "rnb", "r and b"),
    "rock": ("rock",),
    "synthwave": ("synthwave", "synth-wave"),
}

MOOD_ALIASES = {
    "aggressive": ("aggressive", "angry"),
    "chill": ("chill", "mellow", "calm", "laid back", "laid-back"),
    "dreamy": ("dreamy", "floaty"),
    "euphoric": ("euphoric",),
    "focused": ("focused", "focus", "study", "studying", "concentrate", "concentrating"),
    "groovy": ("groovy",),
    "happy": ("happy", "cheerful", "bright"),
    "intense": ("intense", "powerful", "high-energy", "high energy"),
    "melancholic": ("melancholic", "sad", "heartbroken"),
    "moody": ("moody", "brooding"),
    "nostalgic": ("nostalgic", "throwback"),
    "relaxed": ("relaxed", "easygoing", "easy going"),
    "romantic": ("romantic", "love", "lovey"),
    "uplifting": ("uplifting", "inspiring"),
}

ENERGY_HINTS: List[Tuple[float, Tuple[str, ...]]] = [
    (0.95, ("very high energy", "maximum energy", "adrenaline")),
    (0.90, ("high energy", "high-energy", "energetic", "gym", "workout", "running")),
    (0.80, ("upbeat", "dance", "dancing")),
    (0.65, ("midtempo", "mid-tempo", "moderate")),
    (0.40, ("focus", "focused", "study", "studying", "concentrating", "concentration")),
    (0.35, ("chill", "mellow", "calm", "relaxed", "ambient", "soft", "gentle")),
    (0.25, ("quiet", "sleepy", "sleep", "slow")),
]

NEGATIVE_ACOUSTIC_ALIASES = ("not acoustic", "avoid acoustic", "without acoustic", "anything but acoustic")
POSITIVE_ACOUSTIC_ALIASES = ("acoustic", "unplugged")
DEFAULT_TARGET_ENERGY = 0.55


@dataclass
class ParsedPreferences:
    favorite_genre: str = ""
    favorite_mood: str = ""
    target_energy: float = DEFAULT_TARGET_ENERGY
    likes_acoustic: Optional[bool] = None
    avoid_constraints: List[str] = field(default_factory=list)

    def to_user_profile(self, default_likes_acoustic: bool = False) -> UserProfile:
        return UserProfile(
            favorite_genre=self.favorite_genre,
            favorite_mood=self.favorite_mood,
            target_energy=self.target_energy,
            likes_acoustic=default_likes_acoustic if self.likes_acoustic is None else self.likes_acoustic,
        )


@dataclass
class QueryParseResult:
    preferences: ParsedPreferences
    assumptions: List[str] = field(default_factory=list)


def _match_alias(text: str, alias: str) -> Optional[re.Match[str]]:
    pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
    return re.search(pattern, text)


def _find_first_label(text: str, aliases_by_label: dict[str, Tuple[str, ...]]) -> Tuple[str, str]:
    best_label = ""
    best_alias = ""
    best_position: Optional[int] = None

    for label, aliases in aliases_by_label.items():
        for alias in aliases:
            match = _match_alias(text, alias)
            if not match:
                continue
            if best_position is None or match.start() < best_position:
                best_label = label
                best_alias = alias
                best_position = match.start()
            elif match.start() == best_position and len(alias) > len(best_alias):
                best_label = label
                best_alias = alias

    return best_label, best_alias


def _contains_any_alias(text: str, aliases: Tuple[str, ...]) -> bool:
    return any(_match_alias(text, alias) for alias in aliases)


def _detect_target_energy(text: str) -> Tuple[float, bool]:
    explicit_energy = re.search(r"(?:energy\s*(?:around|about)?\s*|around\s*|about\s*)(0(?:\.\d+)?|1(?:\.0+)?)", text)
    if explicit_energy:
        return float(explicit_energy.group(1)), False

    matched_scores = []
    for score, aliases in ENERGY_HINTS:
        if _contains_any_alias(text, aliases):
            matched_scores.append(score)

    if matched_scores:
        return round(sum(matched_scores) / len(matched_scores), 2), False

    return DEFAULT_TARGET_ENERGY, True


def _detect_likes_acoustic(text: str) -> Optional[bool]:
    if _contains_any_alias(text, NEGATIVE_ACOUSTIC_ALIASES):
        return False
    if _contains_any_alias(text, POSITIVE_ACOUSTIC_ALIASES):
        return True
    return None


def _detect_avoid_constraints(text: str) -> List[str]:
    avoidable_terms = {**GENRE_ALIASES, **MOOD_ALIASES, "acoustic": POSITIVE_ACOUSTIC_ALIASES}
    found: List[Tuple[int, str]] = []

    for label, aliases in avoidable_terms.items():
        for alias in aliases:
            pattern = rf"(?:avoid|without|anything but|not)\s+(?:anything\s+)?(?:too\s+)?{re.escape(alias)}(?!\w)"
            match = re.search(pattern, text)
            if match:
                found.append((match.start(), label))
                break

    found.sort(key=lambda item: item[0])

    ordered_labels: List[str] = []
    for _, label in found:
        if label not in ordered_labels:
            ordered_labels.append(label)
    return ordered_labels


def parse_query(query: str) -> QueryParseResult:
    text = query.lower().strip()
    assumptions: List[str] = []

    genre, _ = _find_first_label(text, GENRE_ALIASES)
    if not genre:
        assumptions.append("No supported genre found; leaving genre blank.")

    target_energy, defaulted_energy = _detect_target_energy(text)
    if defaulted_energy:
        assumptions.append(f"No explicit energy clue found; defaulted target energy to {DEFAULT_TARGET_ENERGY:.2f}.")

    mood, _ = _find_first_label(text, MOOD_ALIASES)
    if not mood:
        if _contains_any_alias(text, MOOD_ALIASES["focused"]):
            mood = "focused"
            assumptions.append("Mapped study language to the focused mood.")
        elif target_energy >= 0.75:
            mood = "intense"
            assumptions.append("Inferred intense mood from high-energy language.")
        elif target_energy <= 0.40:
            mood = "chill"
            assumptions.append("Inferred chill mood from low-energy language.")
        else:
            mood = "chill"
            assumptions.append("No explicit mood found; defaulted to chill.")

    likes_acoustic = _detect_likes_acoustic(text)
    if likes_acoustic is None:
        assumptions.append("No explicit acoustic preference found.")

    preferences = ParsedPreferences(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=target_energy,
        likes_acoustic=likes_acoustic,
        avoid_constraints=_detect_avoid_constraints(text),
    )

    return QueryParseResult(preferences=preferences, assumptions=assumptions)