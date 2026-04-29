"""Command line runner for the retrieval-aware music recommendation assistant."""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

from .query_parser import QueryParseResult, parse_query
from .recommender import Recommender, Song, UserProfile, load_songs
from .retriever import KnowledgeRetriever, RetrievedSnippet
from .validator import RecommendationValidator, ValidationResult


DEFAULT_DEMO_QUERIES = [
    "I want acoustic lofi songs for studying",
    "Give me high-energy rock for the gym",
    "I want ambient songs with very high energy and acoustic feel",
]


@dataclass
class AssistantResult:
    raw_query: str
    parsed_result: QueryParseResult
    user_profile: UserProfile
    retrieved_evidence: List[RetrievedSnippet]
    recommendations: List[Tuple[Song, float, str]]
    validation: ValidationResult


def build_assistant_components() -> Tuple[Recommender, KnowledgeRetriever, RecommendationValidator]:
    repo_root = Path(__file__).resolve().parents[1]
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    recommender = Recommender(songs)
    retriever = KnowledgeRetriever(repo_root / "knowledge")
    validator = RecommendationValidator()
    return recommender, retriever, validator


def run_assistant_query(
    raw_query: str,
    recommender: Recommender,
    retriever: KnowledgeRetriever,
    validator: RecommendationValidator,
    k: int = 3,
) -> AssistantResult:
    parsed_result = parse_query(raw_query)
    user_profile = parsed_result.preferences.to_user_profile(default_likes_acoustic=False)
    retrieved_evidence = retriever.retrieve(parsed_result, top_k=3)
    recommendations = recommender.recommend(user_profile, k=k)
    validation = validator.validate(parsed_result, retrieved_evidence, recommendations)

    return AssistantResult(
        raw_query=raw_query,
        parsed_result=parsed_result,
        user_profile=user_profile,
        retrieved_evidence=retrieved_evidence,
        recommendations=recommendations,
        validation=validation,
    )


def _print_lines(label: str, lines: Iterable[str]) -> None:
    line_list = list(lines)
    print(f"{label}:")
    if not line_list:
        print("  None")
        return
    for line in line_list:
        print(f"  - {line}")


def print_assistant_result(result: AssistantResult) -> None:
    """Print one end-to-end assistant run."""
    print(f"\n{'='*72}")
    print(f"Query: {result.raw_query}")
    print(f"{'='*72}")
    print(f"Parsed Preferences: {asdict(result.parsed_result.preferences)}")
    print(f"User Profile: {asdict(result.user_profile)}")

    _print_lines("Assumptions", result.parsed_result.assumptions)
    _print_lines(
        "Retrieved Evidence",
        [
            f"[{snippet.source_name}] score={snippet.relevance_score:.2f} {snippet.snippet}"
            for snippet in result.retrieved_evidence
        ],
    )

    print("Recommendations:")
    for index, (song, score, explanation) in enumerate(result.recommendations, start=1):
        print(f"  {index}. {song.title} by {song.artist} — Score: {score:.2f}")
        print(f"     Because: {explanation}")

    print(f"Confidence: {result.validation.confidence_score:.2f}")
    _print_lines("Warnings", result.validation.warnings)
    _print_lines("Validation Notes", result.validation.validation_notes)


def _resolve_queries(argv: List[str]) -> List[str]:
    if argv:
        return [" ".join(argv).strip()]
    return DEFAULT_DEMO_QUERIES


def main() -> None:
    recommender, retriever, validator = build_assistant_components()
    print(f"Loaded {len(recommender.songs)} songs.")

    queries = _resolve_queries(sys.argv[1:])
    if queries == DEFAULT_DEMO_QUERIES:
        print("Running predefined demo requests because no query was provided.")

    for query in queries:
        result = run_assistant_query(query, recommender, retriever, validator)
        print_assistant_result(result)


if __name__ == "__main__":
    main()
