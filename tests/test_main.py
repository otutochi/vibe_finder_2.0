from src.main import build_assistant_components, run_assistant_query


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