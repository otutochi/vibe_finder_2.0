from collections import Counter
from dataclasses import dataclass, field
from typing import List, Sequence

from .main import AssistantResult, build_assistant_components, run_assistant_query


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    query: str
    expected_top_genre: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    required_warning_substrings: tuple[str, ...] = ()
    required_evidence_sources: tuple[str, ...] = ()
    expect_no_warnings: bool = False
    forbidden_top_moods: tuple[str, ...] = ()


@dataclass
class EvaluationOutcome:
    case: EvaluationCase
    result: AssistantResult
    passed: bool
    failures: List[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    outcomes: List[EvaluationOutcome]
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_confidence: float
    common_warning_types: dict[str, int]


def default_evaluation_cases() -> List[EvaluationCase]:
    return [
        EvaluationCase(
            name="Study Lofi Strong Fit",
            query="I want acoustic lofi songs for studying",
            expected_top_genre="lofi",
            min_confidence=0.85,
            required_evidence_sources=("genre_traits.md", "listening_contexts.md"),
            expect_no_warnings=True,
        ),
        EvaluationCase(
            name="Rock Workout Strong Fit",
            query="Give me high-energy rock for the gym",
            expected_top_genre="rock",
            min_confidence=0.85,
            required_evidence_sources=("feature_signals.md",),
            expect_no_warnings=True,
        ),
        EvaluationCase(
            name="Pop Commute Strong Fit",
            query="I want happy pop for my commute",
            expected_top_genre="pop",
            min_confidence=0.85,
            expect_no_warnings=True,
        ),
        EvaluationCase(
            name="Classical Relax Strong Fit",
            query="I want dreamy classical music for relaxing",
            expected_top_genre="classical",
            min_confidence=0.80,
            required_evidence_sources=("listening_contexts.md", "feature_signals.md"),
            expect_no_warnings=True,
        ),
        EvaluationCase(
            name="Funk Dance Fit",
            query="I want funk songs that make me want to dance",
            expected_top_genre="funk",
            min_confidence=0.70,
            required_evidence_sources=("feature_signals.md",),
            expect_no_warnings=True,
        ),
        EvaluationCase(
            name="Missing Coverage Reggae",
            query="I want happy reggae songs",
            max_confidence=0.80,
            required_warning_substrings=("missing from the current catalog coverage",),
        ),
        EvaluationCase(
            name="Contradictory Ambient Energy",
            query="I want ambient songs with very high energy and acoustic feel",
            expected_top_genre="ambient",
            max_confidence=0.65,
            required_warning_substrings=("contradiction", "weak fit"),
        ),
        EvaluationCase(
            name="Electronic Non-Acoustic Workout",
            query="I want electronic workout songs without acoustic sounds",
            expected_top_genre="electronic",
            min_confidence=0.75,
            required_evidence_sources=("feature_signals.md", "acousticness.md"),
            expect_no_warnings=True,
        ),
        EvaluationCase(
            name="Avoid Sad Lofi",
            query="I want acoustic lofi songs for studying but not anything sad",
            expected_top_genre="lofi",
            min_confidence=0.85,
            expect_no_warnings=True,
            forbidden_top_moods=("melancholic",),
        ),
    ]


def evaluate_case(case: EvaluationCase) -> EvaluationOutcome:
    recommender, retriever, validator = build_assistant_components()
    result = run_assistant_query(case.query, recommender, retriever, validator)
    failures: List[str] = []

    top_song = result.recommendations[0][0] if result.recommendations else None
    top_genre = top_song.genre.lower() if top_song else ""
    top_mood = top_song.mood.lower() if top_song else ""
    warnings = result.validation.warnings
    warning_text = " ".join(warnings).lower()
    evidence_sources = {snippet.source_name for snippet in result.retrieved_evidence}

    if case.expected_top_genre and top_genre != case.expected_top_genre.lower():
        failures.append(
            f"Expected top genre '{case.expected_top_genre}' but got '{top_genre or 'none'}'."
        )

    if case.min_confidence is not None and result.validation.confidence_score < case.min_confidence:
        failures.append(
            f"Expected confidence >= {case.min_confidence:.2f} but got {result.validation.confidence_score:.2f}."
        )

    if case.max_confidence is not None and result.validation.confidence_score > case.max_confidence:
        failures.append(
            f"Expected confidence <= {case.max_confidence:.2f} but got {result.validation.confidence_score:.2f}."
        )

    for required_warning in case.required_warning_substrings:
        if required_warning.lower() not in warning_text:
            failures.append(f"Missing expected warning containing '{required_warning}'.")

    if case.expect_no_warnings and warnings:
        failures.append(f"Expected no warnings but got {len(warnings)} warning(s).")

    if case.required_evidence_sources and not (evidence_sources & set(case.required_evidence_sources)):
        failures.append(
            "Expected retrieved evidence from at least one of "
            f"{sorted(case.required_evidence_sources)} but got {sorted(evidence_sources)}."
        )

    if case.forbidden_top_moods and top_mood in {mood.lower() for mood in case.forbidden_top_moods}:
        failures.append(f"Top recommendation mood '{top_mood}' violates a forbidden mood expectation.")

    return EvaluationOutcome(
        case=case,
        result=result,
        passed=not failures,
        failures=failures,
    )


def run_evaluation(cases: Sequence[EvaluationCase] | None = None) -> EvaluationSummary:
    selected_cases = list(cases) if cases is not None else default_evaluation_cases()
    outcomes = [evaluate_case(case) for case in selected_cases]
    warning_counts = Counter(
        warning
        for outcome in outcomes
        for warning in outcome.result.validation.warnings
    )
    average_confidence = (
        sum(outcome.result.validation.confidence_score for outcome in outcomes) / len(outcomes)
        if outcomes
        else 0.0
    )
    passed_cases = sum(1 for outcome in outcomes if outcome.passed)

    return EvaluationSummary(
        outcomes=outcomes,
        total_cases=len(outcomes),
        passed_cases=passed_cases,
        failed_cases=len(outcomes) - passed_cases,
        average_confidence=average_confidence,
        common_warning_types=dict(warning_counts.most_common()),
    )


def print_evaluation_report(summary: EvaluationSummary) -> None:
    print("Evaluation Results")
    print("==================")
    for outcome in summary.outcomes:
        top_song = outcome.result.recommendations[0][0] if outcome.result.recommendations else None
        top_label = f"{top_song.title} ({top_song.genre})" if top_song else "No recommendation"
        status = "PASS" if outcome.passed else "FAIL"
        print(
            f"{status} | {outcome.case.name} | confidence={outcome.result.validation.confidence_score:.2f} | "
            f"top={top_label}"
        )
        if outcome.failures:
            for failure in outcome.failures:
                print(f"  - {failure}")

    print("\nSummary")
    print("-------")
    print(f"Pass/Fail Totals: {summary.passed_cases}/{summary.total_cases} passed")
    print(f"Average Confidence: {summary.average_confidence:.2f}")
    print("Common Warning Types:")
    if not summary.common_warning_types:
        print("  None")
    else:
        for warning, count in summary.common_warning_types.items():
            print(f"  - {count}x {warning}")


def main() -> None:
    summary = run_evaluation()
    print_evaluation_report(summary)


if __name__ == "__main__":
    main()