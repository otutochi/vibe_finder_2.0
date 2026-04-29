"""Command line runner for the retrieval-aware music recommendation assistant."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
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

SELF_CHECK_CONSTRAINT_PENALTY = 1.05


@dataclass
class SelfCheckResult:
    triggered: bool = False
    reranked: bool = False
    penalty_applied: float = 0.0
    violated_constraints: List[str] = field(default_factory=list)
    original_top_title: str = ""
    final_top_title: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class AssistantResult:
    raw_query: str
    parsed_result: QueryParseResult
    user_profile: UserProfile
    retrieved_evidence: List[RetrievedSnippet]
    recommendations: List[Tuple[Song, float, str]]
    self_check: SelfCheckResult
    validation: ValidationResult


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_assistant_components() -> Tuple[Recommender, KnowledgeRetriever, RecommendationValidator]:
    repo_root = _repo_root()
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    recommender = Recommender(songs)
    retriever = KnowledgeRetriever(repo_root / "knowledge")
    validator = RecommendationValidator()
    return recommender, retriever, validator


def _apply_self_check(
    parsed_result: QueryParseResult,
    recommendations: List[Tuple[Song, float, str]],
    validator: RecommendationValidator,
    k: int,
) -> Tuple[List[Tuple[Song, float, str]], SelfCheckResult]:
    if not recommendations:
        return [], SelfCheckResult()

    original_top_song, _, _ = recommendations[0]
    top_conflicts = validator.constraint_conflicts(parsed_result, original_top_song)
    if not top_conflicts:
        return recommendations[:k], SelfCheckResult(
            original_top_title=original_top_song.title,
            final_top_title=original_top_song.title,
        )

    adjusted_recommendations: List[Tuple[Song, float, str]] = []
    for song, score, explanation in recommendations:
        conflicts = validator.constraint_conflicts(parsed_result, song)
        adjusted_score = score
        adjusted_explanation = explanation
        if conflicts:
            penalty = SELF_CHECK_CONSTRAINT_PENALTY * len(conflicts)
            adjusted_score -= penalty
            adjusted_explanation = (
                f"{explanation}; self-check penalty (-{penalty:.2f} for avoid constraints: {', '.join(conflicts)})"
            )
        adjusted_recommendations.append((song, adjusted_score, adjusted_explanation))

    adjusted_recommendations.sort(key=lambda item: (-item[1], item[0].title))
    final_top_song, _, _ = adjusted_recommendations[0]
    reranked = final_top_song.title != original_top_song.title
    top_penalty = SELF_CHECK_CONSTRAINT_PENALTY * len(top_conflicts)

    notes = [
        "Initial top recommendation conflicted with avoid constraints: "
        f"{', '.join(top_conflicts)}.",
    ]
    if reranked:
        notes.append(
            f"Applied a self-check penalty of {SELF_CHECK_CONSTRAINT_PENALTY:.2f} per violated avoid constraint "
            f"and reranked '{final_top_song.title}' ahead of '{original_top_song.title}'."
        )
    else:
        notes.append(
            f"Applied a self-check penalty of {SELF_CHECK_CONSTRAINT_PENALTY:.2f} per violated avoid constraint, "
            "but the original top recommendation remained the best available match."
        )

    return adjusted_recommendations[:k], SelfCheckResult(
        triggered=True,
        reranked=reranked,
        penalty_applied=top_penalty,
        violated_constraints=top_conflicts,
        original_top_title=original_top_song.title,
        final_top_title=final_top_song.title,
        notes=notes,
    )


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
    ranked_recommendations = recommender.recommend(user_profile, k=len(recommender.songs))
    recommendations, self_check = _apply_self_check(parsed_result, ranked_recommendations, validator, k)
    validation = validator.validate(parsed_result, retrieved_evidence, recommendations)

    if self_check.triggered:
        validation.warnings.insert(
            0,
            "Self-check detected an explicit avoid-constraint conflict on the initial top recommendation.",
        )
        validation.validation_notes.extend(self_check.notes)

    return AssistantResult(
        raw_query=raw_query,
        parsed_result=parsed_result,
        user_profile=user_profile,
        retrieved_evidence=retrieved_evidence,
        recommendations=recommendations,
        self_check=self_check,
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

    if result.self_check.notes:
        _print_lines("Self-Check", result.self_check.notes)

    print(f"Confidence: {result.validation.confidence_score:.2f}")
    _print_lines("Warnings", result.validation.warnings)
    _print_lines("Validation Notes", result.validation.validation_notes)


def serialize_assistant_result(result: AssistantResult) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_query": result.raw_query,
        "parsed_preferences": asdict(result.parsed_result.preferences),
        "assumptions": result.parsed_result.assumptions,
        "user_profile": asdict(result.user_profile),
        "retrieved_evidence": [
            {
                "source_name": snippet.source_name,
                "snippet": snippet.snippet,
                "relevance_score": snippet.relevance_score,
            }
            for snippet in result.retrieved_evidence
        ],
        "top_recommendations": [
            {
                "song": asdict(song),
                "score": score,
                "explanation": explanation,
            }
            for song, score, explanation in result.recommendations
        ],
        "self_check": asdict(result.self_check),
        "confidence": result.validation.confidence_score,
        "warnings": result.validation.warnings,
        "validation_notes": result.validation.validation_notes,
    }


def append_run_log(result: AssistantResult, log_path: Path | None = None) -> bool:
    target_path = log_path or (_repo_root() / "logs" / "runs.jsonl")

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serialize_assistant_result(result), ensure_ascii=True) + "\n")
    except OSError:
        return False

    return True


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
        append_run_log(result)
        print_assistant_result(result)


if __name__ == "__main__":
    main()
