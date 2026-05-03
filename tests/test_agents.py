# tests/test_agents.py — renamed module is now foreign_whispers.reranking
from foreign_whispers.reranking import (
    get_shorter_translations,
    analyze_failures,
    TranslationCandidate,
    FailureAnalysis,
)


def test_get_shorter_returns_candidates():
    """Verify that rule-based shortenings are applied and return candidates."""
    baseline = "hola sin embargo en este momento"
    # Expected transformations:
    # "sin embargo" -> "pero"
    # "en este momento" -> "ahora"
    # Result should be "hola pero ahora"
    
    candidates = get_shorter_translations("hello however right now", baseline, 1.0)
    
    assert len(candidates) > 1, "Should return baseline plus at least one shortened candidate"
    
    shortest = candidates[0]
    assert shortest.text == "hola pero ahora", f"Shortest text was {shortest.text}"
    assert shortest.char_count == len("hola pero ahora")
    assert "pero" in shortest.brevity_rationale
    assert "ahora" in shortest.brevity_rationale
    
    baseline_candidate = candidates[-1]
    assert baseline_candidate.text == baseline
    assert baseline_candidate.brevity_rationale == "baseline"


def test_analyze_failures_returns_dataclass():
    result = analyze_failures({"mean_abs_duration_error_s": 0.5})
    assert isinstance(result, FailureAnalysis)
    assert result.failure_category == "ok"


def test_analyze_failures_detects_overflow():
    result = analyze_failures({"pct_severe_stretch": 30})
    assert result.failure_category == "duration_overflow"


def test_analyze_failures_detects_drift():
    result = analyze_failures({"total_cumulative_drift_s": 5.0})
    assert result.failure_category == "cumulative_drift"


def test_analyze_failures_detects_stretch_quality():
    result = analyze_failures({"mean_abs_duration_error_s": 1.2})
    assert result.failure_category == "stretch_quality"
