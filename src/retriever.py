import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from .query_parser import ParsedPreferences, QueryParseResult


@dataclass(frozen=True)
class RetrievedSnippet:
    source_name: str
    snippet: str
    relevance_score: float


@dataclass(frozen=True)
class _KnowledgeSnippet:
    source_name: str
    heading: str
    snippet: str


class KnowledgeRetriever:
    def __init__(self, knowledge_dir: Union[str, Path]):
        self.knowledge_dir = Path(knowledge_dir)
        self.snippets = self._load_snippets()

    def _load_snippets(self) -> List[_KnowledgeSnippet]:
        snippets: List[_KnowledgeSnippet] = []

        for path in sorted(self.knowledge_dir.glob("*.md")):
            current_heading = ""
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    current_heading = line.lstrip("# ")
                    continue
                if line.startswith("- "):
                    snippets.append(
                        _KnowledgeSnippet(
                            source_name=path.name,
                            heading=current_heading,
                            snippet=line[2:].strip(),
                        )
                    )

        return snippets

    def _normalize_token(self, token: str) -> str:
        token = token.lower()
        aliases = {
            "studying": "study",
            "study": "study",
            "focused": "focus",
            "focusing": "focus",
            "focus": "focus",
            "dance": "danceability",
            "danceable": "danceability",
            "dancing": "danceability",
            "danceability": "danceability",
            "acoustic": "acousticness",
            "unplugged": "acousticness",
            "energetic": "energy",
            "energy": "energy",
            "tempo": "tempo",
            "valence": "valence",
            "groove": "danceability",
            "groovy": "danceability",
            "workout": "workout",
            "gym": "workout",
            "commuting": "commuting",
            "commute": "commuting",
            "relaxing": "relaxing",
            "relaxed": "relaxing",
        }
        return aliases.get(token, token)

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9&]+", text.lower())
        return {self._normalize_token(token) for token in tokens if len(token) > 1}

    def _preference_text(self, preferences: ParsedPreferences) -> str:
        parts = [preferences.favorite_genre, preferences.favorite_mood]

        if preferences.likes_acoustic is True:
            parts.append("acoustic")
        elif preferences.likes_acoustic is False:
            parts.append("not acoustic")

        if preferences.target_energy >= 0.75:
            parts.extend(["high energy", "workout", "tempo", "danceability"])
        elif preferences.target_energy <= 0.40:
            parts.extend(["low energy", "study", "focus"])
        else:
            parts.append("moderate energy")

        parts.extend(preferences.avoid_constraints)
        return " ".join(part for part in parts if part)

    def _query_text(self, query_or_preferences: Union[str, ParsedPreferences, QueryParseResult]) -> str:
        if isinstance(query_or_preferences, str):
            return query_or_preferences
        if isinstance(query_or_preferences, QueryParseResult):
            return self._preference_text(query_or_preferences.preferences)
        return self._preference_text(query_or_preferences)

    def _score_snippet(self, query_tokens: set[str], snippet: _KnowledgeSnippet) -> float:
        snippet_tokens = self._tokenize(f"{snippet.source_name} {snippet.heading} {snippet.snippet}")
        overlap = query_tokens & snippet_tokens
        if not overlap:
            return 0.0

        score = float(len(overlap))
        if snippet.source_name == "feature_signals.md" and {"tempo", "valence", "danceability", "energy"} & query_tokens:
            score += 1.25
        if snippet.source_name == "acousticness.md" and "acousticness" in query_tokens:
            score += 0.5
        if snippet.source_name == "listening_contexts.md" and {"study", "workout", "relaxing", "commuting"} & query_tokens:
            score += 0.5
        return score

    def retrieve(self, query_or_preferences: Union[str, ParsedPreferences, QueryParseResult], top_k: int = 3) -> List[RetrievedSnippet]:
        query_text = self._query_text(query_or_preferences)
        query_tokens = self._tokenize(query_text)
        scored_snippets = []

        for snippet in self.snippets:
            score = self._score_snippet(query_tokens, snippet)
            if score <= 0:
                continue
            scored_snippets.append(
                RetrievedSnippet(
                    source_name=snippet.source_name,
                    snippet=snippet.snippet,
                    relevance_score=score,
                )
            )

        scored_snippets.sort(key=lambda item: (-item.relevance_score, item.source_name, item.snippet))
        return scored_snippets[:top_k]