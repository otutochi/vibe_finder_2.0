import json

from src.main import append_run_log, build_assistant_components, run_assistant_query


def test_run_assistant_query_returns_structured_pipeline_result():
    recommender, retriever, validator = build_assistant_components()

    result = run_assistant_query(
        "I want acoustic lofi songs for studying",
        recommender,
        retriever,
        validator,
    )

    assert result.parsed_result.preferences.favorite_genre == "lofi"
    assert result.user_profile.favorite_genre == "lofi"
    assert len(result.retrieved_evidence) == 3
    assert len(result.recommendations) == 3
    assert result.recommendations[0][0].genre == "lofi"
    assert result.self_check.triggered is False
    assert result.validation.confidence_score >= 0.80
    assert result.validation.warnings == []


def test_run_assistant_query_surfaces_warning_for_contradictory_request():
    recommender, retriever, validator = build_assistant_components()

    result = run_assistant_query(
        "I want ambient songs with very high energy and acoustic feel",
        recommender,
        retriever,
        validator,
    )

    assert result.validation.confidence_score < 0.65
    assert any("contradiction" in warning.lower() for warning in result.validation.warnings)


def test_run_assistant_query_warns_for_missing_catalog_coverage():
    recommender, retriever, validator = build_assistant_components()

    result = run_assistant_query(
        "I want happy reggae songs",
        recommender,
        retriever,
        validator,
    )

    assert result.parsed_result.preferences.favorite_genre == "reggae"
    assert result.recommendations[0][0].genre != "reggae"
    assert result.validation.confidence_score < 0.80
    assert any("missing from the current catalog coverage" in warning.lower() for warning in result.validation.warnings)


def test_run_assistant_query_self_check_reranks_explicit_avoid_conflict():
    recommender, retriever, validator = build_assistant_components()

    result = run_assistant_query(
        "I want lofi songs but not chill",
        recommender,
        retriever,
        validator,
    )

    assert result.self_check.triggered is True
    assert result.self_check.reranked is True
    assert result.self_check.original_top_title == "Library Rain"
    assert result.recommendations[0][0].title == "Focus Flow"
    assert result.recommendations[0][0].mood != "chill"
    assert any("self-check detected" in warning.lower() for warning in result.validation.warnings)


def test_append_run_log_writes_json_safe_entry(tmp_path):
    recommender, retriever, validator = build_assistant_components()
    result = run_assistant_query(
        "I want lofi songs but not chill",
        recommender,
        retriever,
        validator,
    )
    log_path = tmp_path / "logs" / "runs.jsonl"

    wrote_log = append_run_log(result, log_path=log_path)

    assert wrote_log is True
    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 1

    payload = json.loads(log_lines[0])
    assert payload["raw_query"] == "I want lofi songs but not chill"
    assert payload["parsed_preferences"]["favorite_genre"] == "lofi"
    assert payload["retrieved_evidence"]
    assert payload["top_recommendations"][0]["song"]["title"] == "Focus Flow"
    assert payload["self_check"]["triggered"] is True
    assert payload["self_check"]["reranked"] is True
    assert payload["self_check"]["original_top_title"] == "Library Rain"
    assert payload["self_check"]["final_top_title"] == "Focus Flow"
    assert isinstance(payload["confidence"], float)