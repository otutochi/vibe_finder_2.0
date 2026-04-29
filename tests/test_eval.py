from src.eval import default_evaluation_cases, run_evaluation


def test_default_evaluation_cases_cover_required_scenarios():
    cases = default_evaluation_cases()

    assert 8 <= len(cases) <= 12
    assert any("missing from the current catalog coverage" in " ".join(case.required_warning_substrings) for case in cases)
    assert any("contradiction" in " ".join(case.required_warning_substrings) for case in cases)
    assert any(case.min_confidence is not None and case.expect_no_warnings for case in cases)


def test_run_evaluation_passes_default_cases():
    summary = run_evaluation()

    assert summary.total_cases == len(default_evaluation_cases())
    assert summary.failed_cases == 0
    assert summary.passed_cases == summary.total_cases
    assert 0.0 <= summary.average_confidence <= 1.0
    assert any("contradiction" in warning.lower() for warning in summary.common_warning_types)